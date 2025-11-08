/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_matcher.cc
 * \brief This source file implement the matcher class, especially the logic related to LLM tokens,
 * like accepting tokens, leveraging the token mask cache to generate the mask, etc. matcher_base.cc
 * implements the basic matching algorithm from strings to grammar.
 */

#include <dlpack/dlpack.h>
#include <xgrammar/matcher.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "compiled_grammar_impl.h"
#include "earley_parser.h"
#include "grammar_impl.h"
#include "support/dynamic_bitset.h"
#include "support/encoding.h"
#include "support/int_set.h"
#include "support/logging.h"
#include "support/thread_pool.h"
#include "testing.h"

namespace xgrammar {

/******************* Tool functions for token mask *******************/
using GrammarExprType = Grammar::Impl::GrammarExprType;

int32_t GetBitmaskSize(int vocab_size) { return DynamicBitset::GetBufferSize(vocab_size); }

DLDataType GetBitmaskDLType() { return DLDataType{kDLInt, 32, 1}; }

int32_t* CheckAndGetBitmaskPtr(const DLTensor& token_bitmask, int vocab_size, int index) {
  XGRAMMAR_CHECK(token_bitmask.dtype.code == kDLInt && token_bitmask.dtype.bits == 32)
      << "The provied bitmask's dtype is not valid: should be int32";

  int32_t buffer_size = GetBitmaskSize(vocab_size);
  if (token_bitmask.ndim == 1) {
    XGRAMMAR_CHECK(token_bitmask.shape[0] == buffer_size)
        << "The provided bitmask's shape is not valid: should be (" << buffer_size << ", )";
    XGRAMMAR_CHECK(index == 0) << "The index should be 0 when the bitmask is 1D";
  } else {
    XGRAMMAR_CHECK(token_bitmask.ndim == 2)
        << "The provided bitmask's shape is not valid: should be (batch_size, " << buffer_size
        << ")";
    XGRAMMAR_CHECK(token_bitmask.shape[1] == buffer_size)
        << "The provided bitmask's shape is not valid: should be (batch_size, " << buffer_size
        << ")";
    XGRAMMAR_CHECK(index >= 0 && index < token_bitmask.shape[0])
        << "The provided index is out of bounds";
  }

  XGRAMMAR_CHECK(
      token_bitmask.device.device_type == kDLCPU ||
      token_bitmask.device.device_type == kDLCUDAHost ||
      token_bitmask.device.device_type == kDLROCMHost
  ) << "The provided bitmask's device is not valid: should be CPU";

  return reinterpret_cast<int32_t*>(token_bitmask.data) + index * buffer_size;
}

void _DebugGetMaskedTokensFromBitmask(
    std::vector<int>* rejected_tokens, const DLTensor& token_bitmask, int vocab_size, int index
) {
  int32_t* data_ptr = CheckAndGetBitmaskPtr(token_bitmask, vocab_size, index);
  DynamicBitset bitset(vocab_size, reinterpret_cast<uint32_t*>(data_ptr));
  rejected_tokens->clear();
  for (int i = bitset.FindFirstZero(); i != -1; i = bitset.FindNextZero(i)) {
    rejected_tokens->push_back(i);
  }
}

std::pair<bool, int> _IsSingleTokenBitmask(const DLTensor& bitmask, int vocab_size, int index) {
  int32_t* data_ptr = CheckAndGetBitmaskPtr(bitmask, vocab_size, index);
  DynamicBitset bitset(vocab_size, reinterpret_cast<uint32_t*>(data_ptr));
  if (bitset.Count() == 1) {
    return std::make_pair(true, bitset.FindFirstOne());
  } else {
    return std::make_pair(false, -1);
  }
}

void ApplyMask32Bits(
    DLTensor* logits,
    const DLTensor& bitmask,
    int vocab_size,
    std::optional<std::vector<int>> indices
) {
  XGRAMMAR_CHECK(logits->dtype.code == kDLFloat && logits->dtype.bits == 32)
      << "The provided logits's dtype is not valid: should be float32";
  std::pair<int, int> logits_shape =
      logits->ndim == 2
          ? std::make_pair(static_cast<int>(logits->shape[0]), static_cast<int>(logits->shape[1]))
          : std::make_pair(1, static_cast<int>(logits->shape[0]));
  int logits_stride0 = logits->strides[0];
  int bitmask_stride0 = bitmask.strides[0];
  if (indices.has_value()) {
    for (auto idx : indices.value()) {
      uint32_t* data_ptr = reinterpret_cast<uint32_t*>(bitmask.data) + idx * bitmask_stride0;
      DynamicBitset bitset(vocab_size, data_ptr);
      auto logits_ptr = reinterpret_cast<float*>(logits->data) + idx * logits_stride0;
      for (int i = bitset.FindFirstZero(); i != -1; i = bitset.FindNextZero(i)) {
        logits_ptr[i] = -std::numeric_limits<float>::infinity();
      }
    }
  } else {
    for (int idx = 0; idx < logits_shape.first; ++idx) {
      uint32_t* data_ptr = reinterpret_cast<uint32_t*>(bitmask.data) + idx * bitmask_stride0;
      DynamicBitset bitset(vocab_size, data_ptr);
      auto logits_ptr = reinterpret_cast<float*>(logits->data) + idx * logits_stride0;
      for (int i = bitset.FindFirstZero(); i != -1; i = bitset.FindNextZero(i)) {
        logits_ptr[i] = -std::numeric_limits<float>::infinity();
      }
    }
  }
}

void ApplyMask16Bits(
    DLTensor* logits,
    const DLTensor& bitmask,
    int vocab_size,
    std::optional<std::vector<int>> indices
) {
  XGRAMMAR_CHECK(logits->dtype.bits == 16)
      << "The provided logits's dtype is not valid: should be bfloat16 or float16";
  uint16_t kMinusInfinity;
  const uint16_t kMinusInfinityBf16 = 0xff80;
  const uint16_t kMinusInfinityFp16 = 0xfc00;
  switch (logits->dtype.code) {
    case kDLBfloat:
      kMinusInfinity = kMinusInfinityBf16;
      break;
    case kDLFloat:
      kMinusInfinity = kMinusInfinityFp16;
      break;
    default:
      XGRAMMAR_LOG(FATAL
      ) << "The provided logits's dtype is not valid: should be bfloat16 or float16";
  }
  std::pair<int, int> logits_shape =
      logits->ndim == 2
          ? std::make_pair(static_cast<int>(logits->shape[0]), static_cast<int>(logits->shape[1]))
          : std::make_pair(1, static_cast<int>(logits->shape[0]));
  int logits_stride0 = logits->strides[0];
  int bitmask_stride0 = bitmask.strides[0];
  if (indices.has_value()) {
    for (auto idx : indices.value()) {
      uint32_t* data_ptr = reinterpret_cast<uint32_t*>(bitmask.data) + idx * bitmask_stride0;
      DynamicBitset bitset(vocab_size, data_ptr);
      auto logits_ptr = reinterpret_cast<uint16_t*>(logits->data) + idx * logits_stride0;
      for (int i = bitset.FindFirstZero(); i != -1; i = bitset.FindNextZero(i)) {
        logits_ptr[i] = kMinusInfinity;
      }
    }
  } else {
    for (int idx = 0; idx < logits_shape.first; ++idx) {
      uint32_t* data_ptr = reinterpret_cast<uint32_t*>(bitmask.data) + idx * bitmask_stride0;
      DynamicBitset bitset(vocab_size, data_ptr);
      auto logits_ptr = reinterpret_cast<uint16_t*>(logits->data) + idx * logits_stride0;
      for (int i = bitset.FindFirstZero(); i != -1; i = bitset.FindNextZero(i)) {
        logits_ptr[i] = kMinusInfinity;
      }
    }
  }
}

void ApplyTokenBitmaskInplaceCPU(
    DLTensor* logits,
    const DLTensor& bitmask,
    int vocab_size,
    std::optional<std::vector<int>> indices
) {
  // Check device and dim
  XGRAMMAR_CHECK(
      logits->device.device_type == kDLCPU || logits->device.device_type == kDLCUDAHost ||
      logits->device.device_type == kDLROCMHost
  ) << "The provided logits's device is not valid: should be CPU";
  XGRAMMAR_CHECK(
      bitmask.device.device_type == kDLCPU || bitmask.device.device_type == kDLCUDAHost ||
      bitmask.device.device_type == kDLROCMHost
  ) << "The provided bitmask's device is not valid: should be CPU";
  XGRAMMAR_CHECK(logits->ndim == 2 || logits->ndim == 1)
      << "The provided logits's shape is not valid: should be 2D or 1D";
  XGRAMMAR_CHECK(bitmask.ndim == 2 || bitmask.ndim == 1)
      << "The provided bitmask's shape is not valid: should be 2D or 1D";

  // Check type
  XGRAMMAR_CHECK(logits->dtype.lanes == 1)
      << "The provided logits's dtype is not valid: lanes should be 1";
  XGRAMMAR_CHECK(
      bitmask.dtype.code == kDLInt && bitmask.dtype.bits == 32 && bitmask.dtype.lanes == 1
  ) << "The provided bitmask's dtype is not valid: should be int32";

  // Check shape
  std::pair<int, int> logits_shape =
      logits->ndim == 2
          ? std::make_pair(static_cast<int>(logits->shape[0]), static_cast<int>(logits->shape[1]))
          : std::make_pair(1, static_cast<int>(logits->shape[0]));
  std::pair<int, int> bitmask_shape =
      bitmask.ndim == 2
          ? std::make_pair(static_cast<int>(bitmask.shape[0]), static_cast<int>(bitmask.shape[1]))
          : std::make_pair(1, static_cast<int>(bitmask.shape[0]));

  XGRAMMAR_CHECK(
      vocab_size <= bitmask_shape.second * DynamicBitset::BITS_PER_BLOCK &&
      vocab_size <= logits_shape.second
  );

  if (!indices.has_value()) {
    XGRAMMAR_CHECK(logits_shape.first == bitmask_shape.first)
        << "When indices is not provided, the logits's batch size should be equal to the "
           "bitmask's batch size, but got "
        << logits_shape.first << " vs " << bitmask_shape.first;
  }

  // Apply mask
  if (logits->dtype.bits == 32) {
    ApplyMask32Bits(logits, bitmask, vocab_size, indices);
  } else if (logits->dtype.bits == 16) {
    ApplyMask16Bits(logits, bitmask, vocab_size, indices);
  } else {
    XGRAMMAR_LOG(FATAL
    ) << "The provided logits's dtype is not valid: should be float32 or float16/bfloat16";
  }
}

/******************* Grammar Matcher with Adaptive Token Mask *******************/

/*
 * Note on the matching algorithm (this is the old description for the matching algorithm, please
 * refer to https://arxiv.org/pdf/2411.15100 for the latest description)
 *
 * Given a context-free grammar, we match the characters in a string one by one.
 *
 * We adopt a non-deterministic pushdown automata (NPDA) in matching. To be specific, we maintain
 * several stacks, each of which represents a possible path in the NPDA, and update the stacks
 * during matching.
 *
 * ## Stack Structure (see grammar_matcher_state.h)
 * The element of every stack is a StackElement object, referring a position in the grammar. If a
 * StackElement points to a RuleRef element (referring to another rule), the next element of the
 * stack will be a position in this rule. If a StackElement is a CharacterClass element, it will be
 * the last in the stack, meaning *the next* character to match.
 *
 * ## Matching Process (see grammar_matcher_base.h)
 * When accepting a new character and it is accepted by a stack, the last element of the stack will
 * be advanced to the next position in the grammar. If it gets to the end of the rule, several
 * elements at the end may be popped out, and the last element of the stack will be advanced.
 *
 * One stack may split since there may be multiple possible next positions. In this case, similar
 * stacks with different top elements will be added. When one stack cannot accept the new character,
 * it will be removed from the stacks.
 *
 * ## Storage of Stacks (see grammar_matcher_state.h)
 * Note these stacks form a tree structure as when splitting, the new stacks share the same prefix.
 * We store all StackElements as a tree, where every path from tree root to a node represents a
 * stack. To represent stack tops, we attach additional pointers pointing the stack top nodes.
 * Also, We maintain a history of the stack top pointers, so we can rollback to the previous state.
 *
 * All tree nodes are maintained by a buffer, and utilize reference counting to recycle. If a node
 * is neither pointed by a stack top pointer, not pointed by some child nodes, it will be freed.
 *
 * ## Example
 * ### Grammar
 * root ::= [a] R
 * R ::= [b] S [c] | [b] [c] T
 * S ::= "" | [c] [d]
 * T ::= [e]
 *
 * ### The previous step
 * Previous accepted string: ab
 * Previous stack tree:
 * A------
 * |  \   \
 * B   D<  E<
 * |
 * C<
 *
 * A: (rule root, choice 0, element 1)
 * B: (rule R, choice 0, element 1)
 * C: (rule S, choice 1, element 0)
 * D: (rule R, choice 0, element 2)
 * E: (rule R, choice 1, element 1)
 * < means the stack top pointers in the previous step.
 * The stacks in the previous step is: (A, B, C), (A, D), (A, E)
 *
 * ### The current step
 * Current accepted string: abc
 * Current stack tree:
 * A-----------------      G<<
 * |     \     \     \
 * B---   D<    E<    H
 * |   \              |
 * C<   F<<           I<<
 *
 * F: (rule S, choice 1, element 1)
 * G: (rule root, choice 0, element 2) (means the matching process has finished, and will be deleted
 * when the next char comes)
 * H: (rule R, choice 1, element 2)
 * I: (rule T, choice 0, element 0)
 * << means the stack top pointers in the current step.
 * The stacks in the current step is: (A, B, F), (A, H, I), (G,)
 *
 * ## Preprocess (see grammar_matcher_preproc.h)
 * We will store all information about tokens that needed in matching in a CompiledGrammar
 * object. Tokens are sorted by codepoint, allowing us to reuse the repeated prefixes between
 * different tokens.
 *
 * For a given position in a rule, if we only consider this rule and its sub-rules during matching,
 * without considering its parent rules (in actual matching, we also need to consider its parent
 * rules), we can already determine that some tokens are acceptable while others are definitely
 * rejected. Therefore, for a position in a rule, we can divide the token set into three categories:
 * - accepted_indices: If a token is accepted by this rule
 * - rejected_indices: If a token is rejected by this rule
 * - uncertain_indices: Whether it can be accepted depends on the information from the parent
 * level during actual matching. To be specific, If this token has a prefix that has not been
 * rejected and has reached the end of this rule, then it is possible for it to be further accepted
 * by the parent rule.
 *
 * During actual matching, we will directly accept or reject the tokens in accepted_indices and
 * rejected_indices, and only consider the tokens in uncertain_indices. That speeds up the matching
 * process.
 */

/* \brief The concrete implementation of GrammarMatcherNode. */
class GrammarMatcher::Impl : public EarleyParser {
 public:
  Impl(
      const CompiledGrammar& compiled_grammar,
      std::optional<std::vector<int>> override_stop_tokens = std::nullopt,
      bool terminate_without_stop_token = false,
      // max_rollback_tokens_ is deprecated and not used.
      int max_rollback_tokens = -1
  )
      : EarleyParser(compiled_grammar->grammar, ParserState::GetInvalidState()),
        compiled_grammar_(compiled_grammar),
        tokenizer_info_(compiled_grammar->tokenizer_info),
        stop_token_ids_(override_stop_tokens.value_or(tokenizer_info_.GetStopTokenIds())),
        terminate_without_stop_token_(terminate_without_stop_token),
        tmp_accepted_bitset_(tokenizer_info_.GetVocabSize()) {
    XGRAMMAR_CHECK(!override_stop_tokens.has_value() || !override_stop_tokens->empty())
        << "The override_stop_tokens should not be empty";
  }

  bool AcceptToken(int32_t token_id, bool debug_print = false);

  bool AcceptString(const std::string& input_str, bool debug_print = false);

  bool FillNextTokenBitmask(DLTensor* next_token_bitmask, int index, bool debug_print = false);

  std::string FindJumpForwardString();

  void Rollback(int num_tokens);

  bool IsTerminated() const;

  void Reset() { EarleyParser::Reset(); }

  int GetMaxRollbackTokens() const { return -1; }

  const std::vector<int>& GetStopTokenIds() const { return stop_token_ids_; }

  std::string _DebugPrintInternalState() const { return PrintStates(); }

 private:
  using StoreType = AdaptiveTokenMask::StoreType;

  /*!
   * \brief If is_uncertain_saved is true, find the next token in uncertain_indices. Otherwise,
   * find the next token that is set to true in uncertain_tokens_bitset.
   * \param iterator_uncertain The helper iterator to iterate over uncertain_indices or
   * uncertain_tokens_bitset.
   * \returns The index of the next token, or -1 if no more token.
   */
  int GetNextUncertainToken(
      bool is_uncertain_saved,
      int* iterator_uncertain,
      const std::vector<int>& uncertain_indices,
      const std::vector<bool>& uncertain_tokens_bitset
  );

  /*! \brief Set the acceptable next token in next_token_bitmask. */
  void SetTokenBitmask(
      int32_t* bitmask_data_ptr,
      const DynamicBitset& accepted_bitset,
      const std::vector<int32_t>& rejected_indices,
      bool can_reach_end,
      bool allow_special_token = false
  );

  /*!
   * \brief Accept the stop token and terminates the matcher.
   * \returns Whether the stop token can be accepted.
   */
  bool AcceptStopToken();

  bool IsStopTokenAccepted() const;

  /*! \brief Check if the token bitmask is all-true. */
  bool IsTokenBitmaskAllTrue(int32_t* bitmask_data_ptr);

  std::string PrintBitmask(int32_t* bitmask_data_ptr, const TokenizerInfo& tokenizer_info);

  CompiledGrammar compiled_grammar_;
  TokenizerInfo tokenizer_info_;
  std::vector<int> stop_token_ids_;
  bool terminate_without_stop_token_;
  std::deque<int> token_length_history;

  // Temporary data for FillNextTokenBitmask. They are stored here to avoid repeated allocation.
  DynamicBitset tmp_accepted_bitset_;
  std::vector<int32_t> tmp_rejected_indices_;
  std::vector<int32_t> tmp_rejected_indices_delta_;
};

class BatchGrammarMatcher::Impl {
 public:
  Impl(std::variant<std::string, int32_t> max_threads) {
    if (std::holds_alternative<int32_t>(max_threads)) {
      int32_t num_threads = std::get<int32_t>(max_threads);
      XGRAMMAR_CHECK(num_threads >= 1)
          << "The num_threads should be at least 1, but got " << num_threads;
      if (num_threads > 1) {
        if (num_threads > static_cast<int32_t>(std::thread::hardware_concurrency())) {
          XGRAMMAR_LOG(WARNING) << "The num_threads " << num_threads << " is larger than the "
                                << "number of hardware threads. Using "
                                << static_cast<int32_t>(std::thread::hardware_concurrency())
                                << " instead.";
        }
        max_threads_ =
            std::min(num_threads, static_cast<int32_t>(std::thread::hardware_concurrency()));
      }
    } else {
      std::string str = std::get<std::string>(max_threads);
      XGRAMMAR_CHECK(str == "auto");
      max_threads_ = std::thread::hardware_concurrency() / 2;
    }
  }

  void BatchFillNextTokenBitmask(
      std::vector<GrammarMatcher>* matchers,
      DLTensor* next_token_bitmask,
      const std::optional<std::vector<int32_t>>& indices,
      bool debug_print
  );

  static std::vector<uint8_t> BatchAcceptToken(
      std::vector<GrammarMatcher>* matchers, const std::vector<int32_t>& token_ids, bool debug_print
  );

  static std::vector<uint8_t> BatchAcceptString(
      std::vector<GrammarMatcher>* matchers,
      const std::vector<std::string>& input_strs,
      bool debug_print
  );

 private:
  std::optional<ThreadPool> thread_pool_ = std::nullopt;
  int32_t max_threads_ = 1;
};

bool GrammarMatcher::Impl::AcceptStopToken() {
  if (terminate_without_stop_token_) {
    return false;
  }
  if (!IsCompleted()) {
    return false;
  }
  XGRAMMAR_DCHECK(!stop_token_is_accepted_);
  token_length_history.push_back(0);
  stop_token_is_accepted_ = true;
  return true;
}

bool GrammarMatcher::Impl::IsTerminated() const {
  if (terminate_without_stop_token_) {
    return IsCompleted();
  }
  return IsStopTokenAccepted();
}

bool GrammarMatcher::Impl::IsStopTokenAccepted() const { return stop_token_is_accepted_; }

// TODO(yixin): Polish verbose logging
bool GrammarMatcher::Impl::AcceptToken(int32_t token_id, bool debug_print) {
  if (IsStopTokenAccepted()) {
    XGRAMMAR_LOG(WARNING) << "The matcher has terminated after accepting the stop token, but is "
                          << "trying to accept new token with id " << token_id << ".";
    return false;
  }

  if (token_id < 0 || token_id >= tokenizer_info_.GetVocabSize()) {
    XGRAMMAR_LOG(WARNING) << "The token id " << token_id << " is out of range [0, "
                          << tokenizer_info_.GetVocabSize() << "). Rejecting the token.";
    return false;
  }

  if (debug_print) {
    std::string states_str;
    for (const auto& state : GetLatestScanableStates()) {
      states_str += "  " + state.ToString() + "\n";
    }
    XGRAMMAR_LOG(INFO) << "Accepting token id " << token_id << ", string: \""
                       << EscapeString(tokenizer_info_.GetDecodedVocab()[token_id])
                       << "\", current state:\n"
                       << states_str;
  }
  // Handle the stop token
  if (std::find(stop_token_ids_.begin(), stop_token_ids_.end(), token_id) !=
      stop_token_ids_.end()) {
    bool accepted = AcceptStopToken();
    if (debug_print) {
      XGRAMMAR_LOG(INFO) << "The token is an end token. Is accepted: " << accepted;
    }
    return accepted;
  }

  const auto& special_token_ids = tokenizer_info_.GetSpecialTokenIds();
  if (std::find(special_token_ids.begin(), special_token_ids.end(), token_id) !=
      special_token_ids.end()) {
    XGRAMMAR_LOG(WARNING) << "GrammarMatcher cannot accept special token id " << token_id << ": "
                          << tokenizer_info_.GetDecodedVocab()[token_id]
                          << ". Rejecting the token.";
    return false;
  }

  const auto& token = tokenizer_info_.GetDecodedVocab()[token_id];
  int pos = 0;
  for (auto char_value : token) {
    if (!Advance(char_value, debug_print)) {
      if (debug_print) {
        XGRAMMAR_LOG(INFO) << "Token #" << token_id << "<" << EscapeString(token)
                           << "> rejected at position " << pos << ", char "
                           << EscapeString(char_value);
      }
      PopLastStates(pos);
      return false;
    }
    ++pos;
  }
  token_length_history.push_back(token.size());

  if (debug_print) {
    XGRAMMAR_LOG(INFO) << "Token #" << token_id << "<"
                       << EscapeString(tokenizer_info_.GetDecodedVocab()[token_id])
                       << "> accepted.";
  }
  return true;
}

bool GrammarMatcher::Impl::AcceptString(const std::string& input_str, bool debug_print) {
  if (IsStopTokenAccepted()) {
    XGRAMMAR_LOG(WARNING) << "The matcher has terminated after accepting the stop token, but is "
                          << "trying to accept new string \"" << EscapeString(input_str) << "\".";
    return false;
  }

  if (debug_print) {
    XGRAMMAR_LOG(INFO) << "Trying to accept string \"" << EscapeString(input_str)
                       << "\". Current state:\n"
                       << PrintStates();
  }

  int accepted_cnt = 0;
  for (auto char_value : input_str) {
    if (!Advance(char_value, debug_print)) {
      if (debug_print) {
        XGRAMMAR_LOG(INFO) << "String \"" << EscapeString(input_str) << "\" is rejected at "
                           << "position " << accepted_cnt << ", char " << EscapeString(char_value);
      }
      PopLastStates(accepted_cnt);
      return false;
    }
    if (debug_print) {
      XGRAMMAR_LOG(INFO) << "Char " << EscapeString(char_value) << " is accepted. Current state:\n"
                         << PrintStates();
    }
    ++accepted_cnt;
  }
  token_length_history.push_back(input_str.size());

  if (debug_print) {
    XGRAMMAR_LOG(INFO) << "String \"" << EscapeString(input_str) << "\" is accepted.";
  }
  return true;
}

std::string GrammarMatcher::Impl::PrintBitmask(
    int32_t* bitmask_data_ptr, const TokenizerInfo& tokenizer_info
) {
  constexpr int kMaxPrintTokens = 100;
  std::vector<int32_t> accepted_ids;
  std::vector<int32_t> rejected_ids;
  auto bitset =
      DynamicBitset(tokenizer_info.GetVocabSize(), reinterpret_cast<uint32_t*>(bitmask_data_ptr));
  for (int i = 0; i < tokenizer_info.GetVocabSize(); ++i) {
    if (bitset[i]) {
      accepted_ids.push_back(i);
    } else {
      rejected_ids.push_back(i);
    }
  }
  std::stringstream ss;
  ss << "TokenBitmask(num_tokens=" << tokenizer_info.GetVocabSize()
     << ", accepted_num=" << accepted_ids.size() << ", rejected_num=" << rejected_ids.size()
     << ",\naccepted_ids=" << PrintTokenByIds(accepted_ids, tokenizer_info, kMaxPrintTokens)
     << ",\nrejected_ids=" << PrintTokenByIds(rejected_ids, tokenizer_info, kMaxPrintTokens) << ")";
  return ss.str();
}

bool GrammarMatcher::Impl::IsTokenBitmaskAllTrue(int32_t* bitmask_data_ptr) {
  DynamicBitset next_token_bitset(
      tokenizer_info_.GetVocabSize(), reinterpret_cast<uint32_t*>(bitmask_data_ptr)
  );
  return next_token_bitset.All();
}

bool GrammarMatcher::Impl::FillNextTokenBitmask(
    DLTensor* next_token_bitmask, int index, bool debug_print
) {
  XGRAMMAR_CHECK(!IsStopTokenAccepted())
      << "GrammarMatcher has terminated after accepting the stop token, but is trying to "
         "find the next token mask";
  int32_t* bitmask_data_ptr =
      CheckAndGetBitmaskPtr(*next_token_bitmask, tokenizer_info_.GetVocabSize(), index);
  const auto& sorted_decoded_vocab = tokenizer_info_.GetSortedDecodedVocab();
  const auto& subtree_range = tokenizer_info_.GetTrieSubtreeNodesRange();
  const auto& adaptive_token_mask_cache = compiled_grammar_->adaptive_token_mask_cache;
  // We need to have a copy, because scanable_state_history_ will be modified during the
  // FillNextTokenBitmask process, which can lead to undefined behavior.
  auto latest_states = GetLatestScanableStates();

  // We check all the latest states of the earley parser, and check all the masks of the leaf
  // states. The final accepted token set is the union of the accepted token sets of all leaf
  // states. The final rejected token set is the intersection of the rejected token sets of all leaf
  // states.

  // Note these indices store the indices in sorted_decoded_vocab, instead of the token ids.
  tmp_accepted_bitset_.Reset();
  // {-1} means the universal set, i.e. all tokens initially
  tmp_rejected_indices_.assign({-1});

  if (debug_print) {
    XGRAMMAR_LOG(INFO) << "FillNextTokenBitmask: index=" << index
                       << ", num of states=" << latest_states.size();
  }

  std::vector<std::pair<ParserState, decltype(adaptive_token_mask_cache.cbegin())>>
      latest_states_with_masks;

  for (const auto& state : latest_states) {
    auto adaptive_token_mask_it = adaptive_token_mask_cache.find(state);
    XGRAMMAR_CHECK(adaptive_token_mask_it != adaptive_token_mask_cache.end()) << state;
    const auto& adaptive_token_mask = adaptive_token_mask_it->second;
    latest_states_with_masks.push_back(std::make_pair(state, adaptive_token_mask_it));
    if (adaptive_token_mask.store_type == StoreType::kAcceptedBitset) {
      tmp_accepted_bitset_ |= adaptive_token_mask.accepted_bitset;
    } else if (adaptive_token_mask.store_type == StoreType::kAccepted) {
      for (auto idx : adaptive_token_mask.accepted_indices) {
        tmp_accepted_bitset_.Set(sorted_decoded_vocab[idx].first, true);
      }
    }
  }

  for (const auto& [state, adaptive_token_mask_it] : latest_states_with_masks) {
    const auto& adaptive_token_mask = adaptive_token_mask_it->second;

    // For each ParserState, we will check every uncertain token and put them into the accepted or
    // rejected list.

    // Step 2. Update the accepted tokens in accepted_indices_delta, or the rejected tokens in
    // rejected_indices_delta.

    // If the accepted tokens are saved, it means it is likely to be smaller than the rejected
    // tokens, so we will just find the accepted tokens, and vice versa.

    tmp_rejected_indices_delta_.clear();

    // Examine only the current one ParserState
    PushOneStateToCheck(state);

    const std::string* prev_token = nullptr;
    int prev_matched_size = 0;
    if (debug_print) {
      XGRAMMAR_LOG(INFO) << "The ParserState is " << state << ", the mask is "
                         << adaptive_token_mask.Print(tokenizer_info_);
    }
    int last_rejected_uncertain_range = 0;
    for (const auto& cur_token_idx : adaptive_token_mask.uncertain_indices) {
      // Check if the current token is already accepted. If it is, we can skip it.
      if (tmp_accepted_bitset_[sorted_decoded_vocab[cur_token_idx].first]) {
        continue;
      }

      // Check if the current token is in the rejected range. i.e. check if the current token
      // is on the subtree of the rejected token.
      if (cur_token_idx < last_rejected_uncertain_range) {
        if (adaptive_token_mask.store_type == StoreType::kRejected) {
          tmp_rejected_indices_delta_.push_back(cur_token_idx);
        }
        continue;
      }

      const auto& cur_token = sorted_decoded_vocab[cur_token_idx].second;
      bool accepted = true;

      // Step 2.1. Find the longest common prefix with the accepted part of the previous token.
      // We can reuse the previous matched size to avoid unnecessary matching.
      if (prev_token) {
        int lcp_len = std::mismatch(
                          cur_token.begin(), cur_token.end(), prev_token->begin(), prev_token->end()
                      )
                          .first -
                      cur_token.begin();
        if (lcp_len > prev_matched_size) {
          last_rejected_uncertain_range = subtree_range[cur_token_idx];
          accepted = false;
        } else if (lcp_len < prev_matched_size) {
          PopLastStates(prev_matched_size - lcp_len);
        }
        prev_matched_size = std::min(prev_matched_size, lcp_len);
      }

      // Step 2.2. Find if the current token is accepted or rejected.
      if (accepted) {
        for (int j = prev_matched_size; j < static_cast<int>(cur_token.size()); ++j) {
          if (!Advance(cur_token[j])) {
            last_rejected_uncertain_range = subtree_range[cur_token_idx];
            accepted = false;
            break;
          }
          prev_matched_size = j + 1;
        }
      }

      // Step 2.3. Push the result to the delta list.
      if (adaptive_token_mask.store_type == StoreType::kAcceptedBitset ||
          adaptive_token_mask.store_type == StoreType::kAccepted) {
        if (accepted) {
          tmp_accepted_bitset_.Set(sorted_decoded_vocab[cur_token_idx].first, true);
        }
      } else {
        if (!accepted) {
          tmp_rejected_indices_delta_.push_back(cur_token_idx);
        }
      }

      prev_token = &cur_token;
    }

    PopLastStates(prev_matched_size + 1);
    // Step 3. Update the accepted_indices or rejected_indices
    if (adaptive_token_mask.store_type == StoreType::kRejected) {
      // rejected_indices = Intersect(
      //     rejected_indices,
      //     adaptive_token_mask.rejected_indices + rejected_indices_delta)
      IntsetUnion(&tmp_rejected_indices_delta_, adaptive_token_mask.rejected_indices);
      IntsetIntersection(&tmp_rejected_indices_, tmp_rejected_indices_delta_);
    }
  }

  // Finally update the rejected_ids bitset
  bool can_reach_end = IsCompleted();
  SetTokenBitmask(
      bitmask_data_ptr, tmp_accepted_bitset_, tmp_rejected_indices_, can_reach_end, false
  );
  if (debug_print) {
    XGRAMMAR_LOG(INFO) << "Filled bitmask: " << PrintBitmask(bitmask_data_ptr, tokenizer_info_);
  }
  return !IsTokenBitmaskAllTrue(bitmask_data_ptr);
}

std::string GrammarMatcher::Impl::FindJumpForwardString() {
  XGRAMMAR_CHECK(!IsStopTokenAccepted())
      << "GrammarMatcher has terminated after accepting the stop token, but is trying to "
         "get the jump forward string";

  std::string result;
  int num_accepted_chars = 0;
  bool can_find_next_char = true;

  while (can_find_next_char) {
    const auto& states = scanable_state_history_[scanable_state_history_.size() - 1];

    // The state comes to the end of the grammar
    if (IsCompleted()) {
      can_find_next_char = false;
      break;
    }

    // 1. Check that for every leaf ParserState, the next possible char is unique and the same
    // -1 means not found yet; 0~255 means the next char
    int next_char = -1;
    for (const auto& state : states) {
      if (state.rule_id != -1 && grammar_->per_rule_fsms[state.rule_id].has_value()) {
        const auto& fsm = grammar_->per_rule_fsms[state.rule_id].value();
        const auto& current_edges = fsm.GetFsm().GetEdges(state.element_id);
        for (const auto& edge : current_edges) {
          if (!edge.IsCharRange()) {
            continue;
          }
          if (edge.min != edge.max) {
            can_find_next_char = false;
            break;
          }
          if (next_char == -1) {
            next_char = edge.min;
          } else if (next_char != edge.min) {
            can_find_next_char = false;
            break;
          }
        }
        continue;
      }

      auto cur_sequence = grammar_->GetGrammarExpr(state.sequence_id);

      // We cannot deduce the next char for tag dispatch
      if (cur_sequence.type == GrammarExprType::kTagDispatch) {
        can_find_next_char = false;
        break;
      }
      // The ParserState comes to the end of the grammar
      XGRAMMAR_DCHECK(state.element_id != cur_sequence.size());
      XGRAMMAR_DCHECK(
          cur_sequence.type != GrammarExprType::kChoices &&
          cur_sequence.type != GrammarExprType::kEmptyStr
      );
      const auto& cur_element = grammar_->GetGrammarExpr(cur_sequence[state.element_id]);
      if (cur_element.type == GrammarExprType::kByteString) {
        XGRAMMAR_DCHECK(state.sub_element_id < cur_element.size());
        if (next_char == -1) {
          next_char = cur_element[state.sub_element_id];
        } else if (next_char != cur_element[state.sub_element_id]) {
          can_find_next_char = false;
          break;
        }
        continue;
      }
      if (cur_element.type == GrammarExprType::kRuleRef) {
        continue;
      }

      XGRAMMAR_DCHECK(
          cur_element.type == GrammarExprType::kCharacterClass ||
          cur_element.type == GrammarExprType::kCharacterClassStar
      );
      if (state.sub_element_id > 0 || cur_element.size() != 3 || cur_element[0] != 0 ||
          cur_element[1] != cur_element[2]) {
        can_find_next_char = false;
        break;
      } else if (next_char == -1) {
        next_char = cur_element[1];
      } else if (next_char != cur_element[1]) {
        can_find_next_char = false;
        break;
      }
    }

    if (next_char == -1) {
      can_find_next_char = false;
    }

    // 2. If found, accept the char and iterate to the next position
    if (can_find_next_char) {
      result += static_cast<uint8_t>(next_char);
      Advance(next_char);
      ++num_accepted_chars;
    }
  }

  // Rollback all chars accepted
  PopLastStates(num_accepted_chars);
  return result;
}

void GrammarMatcher::Impl::Rollback(int num_tokens) {
  XGRAMMAR_CHECK(num_tokens <= static_cast<int>(token_length_history.size()))
      << "Intended to rollback " << num_tokens << " tokens, but only the last "
      << token_length_history.size() << " steps of history are saved";
  while (num_tokens > 0) {
    int steps = token_length_history.back();
    PopLastStates(steps);
    token_length_history.pop_back();
    --num_tokens;
  }
}

void GrammarMatcher::Impl::SetTokenBitmask(
    int32_t* bitmask_data_ptr,
    const DynamicBitset& accepted_bitset,
    const std::vector<int32_t>& rejected_indices,
    bool can_reach_end,
    bool allow_special_token
) {
  // next_token_bitmask = set(all accepted tokens) =
  // 1. all_tokens - (rejected_ids / accepted_ids)
  //    (when rejected_ids != {-1}, i.e. rejected_ids is not the universal set)
  // 2. accepted_ids
  //    (otherwise, when rejected_ids is the universal set)
  DynamicBitset next_token_bitset(
      tokenizer_info_.GetVocabSize(), reinterpret_cast<uint32_t*>(bitmask_data_ptr)
  );
  const auto& sorted_decoded_vocab = tokenizer_info_.GetSortedDecodedVocab();

  if (rejected_indices.size() == 1 && rejected_indices[0] == -1) {
    // If rejected_indices is the universal set, the final accepted token set is just
    // accepted_indices
    next_token_bitset = accepted_bitset;

    if (allow_special_token) {
      for (int id : tokenizer_info_.GetSpecialTokenIds()) {
        next_token_bitset.Set(id, true);
      }
    }

    if (can_reach_end) {
      // add end tokens
      for (int id : stop_token_ids_) {
        next_token_bitset.Set(id, true);
      }
    }
  } else {
    // Otherwise, the final rejected token set is (rejected_indices \ accepted_indices)
    next_token_bitset.Set();

    for (auto i : rejected_indices) {
      auto id = sorted_decoded_vocab[i].first;
      if (!accepted_bitset[id]) {
        next_token_bitset.Set(id, false);
      }
    }
    if (!allow_special_token) {
      for (int id : tokenizer_info_.GetSpecialTokenIds()) {
        next_token_bitset.Set(id, false);
      }
    }
    if (!can_reach_end) {
      for (int id : stop_token_ids_) {
        next_token_bitset.Set(id, false);
      }
    }
  }
}

int GrammarMatcher::Impl::GetNextUncertainToken(
    bool is_uncertain_saved,
    int* iterator_uncertain,
    const std::vector<int>& uncertain_indices,
    const std::vector<bool>& uncertain_tokens_bitset
) {
  if (is_uncertain_saved) {
    ++*iterator_uncertain;
    if (*iterator_uncertain == static_cast<int>(uncertain_indices.size())) {
      return -1;
    }
    return uncertain_indices[*iterator_uncertain];
  } else {
    ++*iterator_uncertain;
    while (*iterator_uncertain < static_cast<int>(uncertain_tokens_bitset.size()) &&
           !uncertain_tokens_bitset[*iterator_uncertain]) {
      ++*iterator_uncertain;
    }
    if (*iterator_uncertain == static_cast<int>(uncertain_tokens_bitset.size())) {
      return -1;
    }
    return *iterator_uncertain;
  }
}

void BatchGrammarMatcher::Impl::BatchFillNextTokenBitmask(
    std::vector<GrammarMatcher>* matchers,
    DLTensor* next_token_bitmask,
    const std::optional<std::vector<int32_t>>& indices,
    bool debug_print
) {
  XGRAMMAR_CHECK(!indices.has_value() || indices->size() == matchers->size())
      << "The size of indices (" << (indices.has_value() ? indices->size() : 0)
      << ") should be the same as the size of matchers (" << matchers->size() << ").";
  // Initialize the thread pool if needed. It should be initialized each time,
  // because ThreadPool cannot be reused after Join().
  if (max_threads_ > 1) {
    thread_pool_.emplace(max_threads_);
  }
  if (!thread_pool_.has_value()) {
    for (int i = 0; i < static_cast<int32_t>(matchers->size()); i++) {
      auto& matcher = (*matchers)[i];
      int index = indices.has_value() ? (*indices)[i] : i;
      XGRAMMAR_CHECK(index >= 0 && index < next_token_bitmask->shape[0])
          << "The index " << index << " is out of range [0, " << next_token_bitmask->shape[0]
          << ") for batch_id " << i << ".";
      matcher->FillNextTokenBitmask(next_token_bitmask, index, debug_print);
    }
  } else {
    auto fill_next_token_mask = [&](int32_t batch_id) {
      auto& matcher = (*matchers)[batch_id];
      int index = indices.has_value() ? (*indices)[batch_id] : batch_id;
      XGRAMMAR_CHECK(index >= 0 && index < next_token_bitmask->shape[0])
          << "The index " << index << " is out of range [0, " << next_token_bitmask->shape[0]
          << ") for batch_id " << batch_id << ".";
      matcher->FillNextTokenBitmask(next_token_bitmask, index, debug_print);
    };
    for (int i = 0; i < static_cast<int32_t>(matchers->size()); i++) {
      thread_pool_->Execute([fill_next_token_mask, i]() { fill_next_token_mask(i); });
    }
    thread_pool_->Join();
  }
}

std::vector<uint8_t> BatchGrammarMatcher::Impl::BatchAcceptString(
    std::vector<GrammarMatcher>* matchers,
    const std::vector<std::string>& input_strs,
    bool debug_print
) {
  XGRAMMAR_CHECK(matchers->size() == input_strs.size())
      << "The size of matchers (" << matchers->size() << ") and input_strs (" << input_strs.size()
      << ") should be the same.";
  std::vector<uint8_t> accepted(matchers->size());
  for (int i = 0; i < static_cast<int32_t>(matchers->size()); i++) {
    auto& matcher = (*matchers)[i];
    accepted[i] = matcher->AcceptString(input_strs[i], debug_print);
  }
  return accepted;
}

std::vector<uint8_t> BatchGrammarMatcher::Impl::BatchAcceptToken(
    std::vector<GrammarMatcher>* matchers, const std::vector<int32_t>& token_ids, bool debug_print
) {
  XGRAMMAR_CHECK(matchers->size() == token_ids.size())
      << "The size of matchers (" << matchers->size() << ") and token_ids (" << token_ids.size()
      << ") should be the same.";
  std::vector<uint8_t> accepted(matchers->size());
  for (int i = 0; i < static_cast<int32_t>(matchers->size()); i++) {
    auto& matcher = (*matchers)[i];
    accepted[i] = matcher->AcceptToken(token_ids[i], debug_print);
  }
  return accepted;
}

GrammarMatcher::GrammarMatcher(
    const CompiledGrammar& compiled_grammar,
    std::optional<std::vector<int>> override_stop_tokens,
    bool terminate_without_stop_token,
    int max_rollback_tokens
)
    : pimpl_(std::make_shared<GrammarMatcher::Impl>(
          compiled_grammar, override_stop_tokens, terminate_without_stop_token, max_rollback_tokens
      )) {}

bool GrammarMatcher::AcceptToken(int32_t token_id, bool debug_print) {
  return pimpl_->AcceptToken(token_id, debug_print);
}

bool GrammarMatcher::AcceptString(const std::string& input_str, bool debug_print) {
  return pimpl_->AcceptString(input_str, debug_print);
}

bool GrammarMatcher::FillNextTokenBitmask(
    DLTensor* next_token_bitmask, int index, bool debug_print
) {
  return pimpl_->FillNextTokenBitmask(next_token_bitmask, index, debug_print);
}

std::string GrammarMatcher::FindJumpForwardString() { return pimpl_->FindJumpForwardString(); }

void GrammarMatcher::Rollback(int num_tokens) { pimpl_->Rollback(num_tokens); }

bool GrammarMatcher::IsTerminated() const { return pimpl_->IsTerminated(); }

void GrammarMatcher::Reset() { pimpl_->Reset(); }

int GrammarMatcher::GetMaxRollbackTokens() const { return pimpl_->GetMaxRollbackTokens(); }

const std::vector<int>& GrammarMatcher::GetStopTokenIds() const {
  return pimpl_->GetStopTokenIds();
}

std::string GrammarMatcher::_DebugPrintInternalState() const {
  return pimpl_->_DebugPrintInternalState();
}

void BatchGrammarMatcher::BatchFillNextTokenBitmask(
    std::vector<GrammarMatcher>* matchers,
    DLTensor* next_token_bitmask,
    const std::optional<std::vector<int32_t>>& indices,
    bool debug_print
) {
  return pimpl_->BatchFillNextTokenBitmask(matchers, next_token_bitmask, indices, debug_print);
}

std::vector<uint8_t> BatchGrammarMatcher::BatchAcceptString(
    std::vector<GrammarMatcher>* matchers,
    const std::vector<std::string>& input_strs,
    bool debug_print
) {
  return Impl::BatchAcceptString(matchers, input_strs, debug_print);
}

std::vector<uint8_t> BatchGrammarMatcher::BatchAcceptToken(
    std::vector<GrammarMatcher>* matchers, const std::vector<int32_t>& token_ids, bool debug_print
) {
  return Impl::BatchAcceptToken(matchers, token_ids, debug_print);
}

BatchGrammarMatcher::BatchGrammarMatcher(std::variant<std::string, int32_t> max_threads)
    : pimpl_(std::make_shared<BatchGrammarMatcher::Impl>(max_threads)) {}

}  // namespace xgrammar
