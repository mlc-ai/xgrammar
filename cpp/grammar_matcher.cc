/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_matcher.cc
 * \brief This source file implement the matcher class, especially the logic related to LLM tokens,
 * like accepting tokens, leveraging the token mask cache to generate the mask, etc. matcher_base.cc
 * implements the basic matching algorithm from strings to grammar.
 */

#include <xgrammar/matcher.h>

#include <chrono>
#include <queue>

#include "compiled_grammar_data_structure.h"
#include "grammar_data_structure.h"
#include "grammar_matcher_base.h"
#include "grammar_serializer.h"
#include "persistent_stack.h"
#include "support/dynamic_bitset.h"
#include "support/encoding.h"
#include "support/int_set.h"
#include "support/logging.h"

namespace xgrammar {

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

  XGRAMMAR_CHECK(token_bitmask.device.device_type == kDLCPU)
      << "The provided bitmask's device is not valid: should be CPU";

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

void ApplyTokenBitmaskInplaceCPU(
    DLTensor* logits, const DLTensor& bitmask, std::optional<std::vector<int>> indices
) {
  XGRAMMAR_CHECK(logits->device.device_type == kDLCPU)
      << "The provided logits's device is not valid: should be CPU";
  XGRAMMAR_CHECK(bitmask.device.device_type == kDLCPU)
      << "The provided bitmask's device is not valid: should be CPU";
  int batch_size;
  int vocab_size;
  if (logits->ndim == 2) {
    batch_size = logits->shape[0];
    vocab_size = logits->shape[1];
  } else {
    batch_size = 1;
    vocab_size = logits->shape[0];
  }
  int bitmask_size = GetBitmaskSize(vocab_size);
  if (bitmask.ndim == 2) {
    XGRAMMAR_CHECK(bitmask.shape[0] == batch_size)
        << "The provided bitmask's batch size is not consistent with logits";
    XGRAMMAR_CHECK(bitmask.shape[1] == bitmask_size)
        << "The provided bitmask's bitmask size is not consistent with logits";
  } else {
    XGRAMMAR_CHECK(bitmask.ndim == 1)
        << "The provided bitmask's shape is not valid: should be (batch_size, vocab_size)";
    XGRAMMAR_CHECK(bitmask.shape[0] == bitmask_size)
        << "The provided bitmask's bitmask size is not consistent with logits";
  }
  XGRAMMAR_CHECK(
      logits->dtype.code == kDLFloat && logits->dtype.bits == 32 && logits->dtype.lanes == 1
  ) << "The provided logits's dtype is not valid: should be float32";
  XGRAMMAR_CHECK(
      bitmask.dtype.code == kDLInt && bitmask.dtype.bits == 32 && bitmask.dtype.lanes == 1
  ) << "The provided bitmask's dtype is not valid: should be int32";

  std::vector<int> indices_value;
  if (indices.has_value()) {
    indices_value = indices.value();
    std::sort(indices_value.begin(), indices_value.end());
    indices_value.erase(
        std::unique(indices_value.begin(), indices_value.end()), indices_value.end()
    );
    XGRAMMAR_CHECK(indices_value.back() < batch_size)
        << "The provided indices is out of bounds: " << indices_value.back()
        << " >= " << batch_size;
  } else {
    indices_value.resize(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      indices_value[i] = i;
    }
  }

  for (auto idx : indices_value) {
    uint32_t* data_ptr = reinterpret_cast<uint32_t*>(bitmask.data) + idx * bitmask_size;
    DynamicBitset bitset(vocab_size, data_ptr);
    auto logits_ptr = reinterpret_cast<float*>(logits->data) + idx * vocab_size;
    for (int i = bitset.FindFirstZero(); i != -1; i = bitset.FindNextZero(i)) {
      logits_ptr[i] = -std::numeric_limits<float>::infinity();
    }
  }
}

/*
 * Note on the matching algorithm
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
class GrammarMatcher::Impl : public GrammarMatcherBase {
 public:
  Impl(
      const CompiledGrammar& compiled_grammar,
      std::optional<std::vector<int>> override_stop_tokens = std::nullopt,
      bool terminate_without_stop_token = false,
      int max_rollback_tokens = 0
  )
      : GrammarMatcherBase(compiled_grammar->grammar),
        compiled_grammar_(compiled_grammar),
        tokenizer_info_(compiled_grammar->tokenizer_info),
        stop_token_ids_(override_stop_tokens.value_or(tokenizer_info_.GetStopTokenIds())),
        terminate_without_stop_token_(terminate_without_stop_token),
        max_rollback_tokens_(max_rollback_tokens),
        tmp_accepted_bitset_(tokenizer_info_.GetVocabSize()) {
    XGRAMMAR_CHECK(!override_stop_tokens.has_value() || !override_stop_tokens->empty())
        << "The override_stop_tokens should not be empty";
  }

  bool AcceptToken(int32_t token_id, bool debug_print = false);

  void FillNextTokenBitmask(DLTensor* next_token_bitmask, int index);

  std::string FindJumpForwardString();

  void Rollback(int num_tokens);

  bool IsTerminated() const;

  void Reset() {
    stack_tops_history_.Reset();
    token_length_history.clear();
    PushInitialState(kInvalidStackElement, true);
  }

  int GetMaxRollbackTokens() const { return max_rollback_tokens_; }

  const std::vector<int>& GetStopTokenIds() const { return stop_token_ids_; }

  bool _DebugAcceptString(const std::string& input_str, bool debug_print = false);

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
      bool can_reach_end
  );

  /*!
   * \brief Accept the stop token and terminates the matcher.
   * \returns Whether the stop token can be accepted.
   */
  bool AcceptStopToken();

  CompiledGrammar compiled_grammar_;
  TokenizerInfo tokenizer_info_;
  std::vector<int> stop_token_ids_;
  bool terminate_without_stop_token_;
  int max_rollback_tokens_;
  std::deque<int> token_length_history;

  // Temporary data for FillNextTokenBitmask. They are stored here to avoid repeated allocation.
  DynamicBitset tmp_accepted_bitset_;
  std::vector<int32_t> tmp_rejected_indices_;
  std::vector<int32_t> tmp_rejected_indices_delta_;
};

bool GrammarMatcher::Impl::AcceptStopToken() {
  if (terminate_without_stop_token_) {
    return false;
  }
  if (!CanReachEnd()) {
    return false;
  }
  stack_tops_history_.PushHistory({});  // Terminate the matcher by setting the stack to empty
  token_length_history.push_back(1);  // When rolling back a stop token, we need to rollback 1 state
  return true;
}

bool GrammarMatcher::Impl::IsTerminated() const {
  if (terminate_without_stop_token_) {
    return CanReachEnd();
  }
  return stack_tops_history_.GetLatest().empty();
}

// TODO(yixin): Polish verbose logging
bool GrammarMatcher::Impl::AcceptToken(int32_t token_id, bool debug_print) {
  if (IsTerminated()) {
    if (debug_print) {
      XGRAMMAR_LOG(INFO) << "The matcher has terminated after accepting the stop token, but is "
                            "trying to accept new token with id "
                         << token_id;
    }
    return false;
  }

  XGRAMMAR_CHECK(token_id >= 0 && token_id < tokenizer_info_.GetVocabSize())
      << "Invalid token id " << token_id << " for GrammarMatcher";

  if (debug_print) {
    XGRAMMAR_LOG(INFO) << "Accepting token id " << token_id << ", string: \""
                       << PrintAsEscapedUTF8(tokenizer_info_.GetDecodedVocab()[token_id])
                       << "\", state state:\n"
                       << PrintStackState();
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
    XGRAMMAR_LOG(FATAL) << "Token id " << token_id << ": "
                        << tokenizer_info_.GetDecodedVocab()[token_id]
                        << " is regarded as a special token, and cannot be accepted by the "
                           "GrammarMatcher";
  }

  const auto& token = tokenizer_info_.GetDecodedVocab()[token_id];
  int pos = 0;
  for (auto char_value : token) {
    if (!AcceptChar(char_value, debug_print)) {
      if (debug_print) {
        XGRAMMAR_LOG(INFO) << "The token is rejected at position " << pos << ", character "
                           << PrintAsEscapedUTF8(char_value);
      }
      return false;
    }
    ++pos;
  }
  token_length_history.push_back(token.size());
  if (static_cast<int>(token_length_history.size()) > max_rollback_tokens_) {
    DiscardEarliestChars(token_length_history.front());
    token_length_history.pop_front();
  }
  if (debug_print) {
    XGRAMMAR_LOG(INFO) << "The token is accepted. State after accepting:\n" << PrintStackState();
  }
  return true;
}

bool GrammarMatcher::Impl::_DebugAcceptString(const std::string& input_str, bool debug_print) {
  if (IsTerminated()) {
    if (debug_print) {
      XGRAMMAR_LOG(INFO) << "The matcher has terminated after accepting the stop token, but is "
                            "trying to accept new string "
                         << PrintAsEscapedUTF8(input_str);
    }
    return false;
  }

  int accepted_cnt = 0;
  for (auto char_value : input_str) {
    if (!AcceptChar(char_value, debug_print)) {
      if (debug_print) {
        XGRAMMAR_LOG(INFO) << "Matching failed after accepting " << accepted_cnt << " characters";
      }
      RollbackChars(accepted_cnt);
      return false;
    }
    ++accepted_cnt;
  }
  token_length_history.push_back(input_str.size());
  if (static_cast<int>(token_length_history.size()) > max_rollback_tokens_) {
    DiscardEarliestChars(token_length_history.front());
    token_length_history.pop_front();
  }
  if (debug_print) {
    XGRAMMAR_LOG(INFO) << "String \"" << PrintAsEscapedUTF8(input_str)
                       << "\" is accepted. State after accepting:\n"
                       << PrintStackState();
  }
  return true;
}

void GrammarMatcher::Impl::FillNextTokenBitmask(DLTensor* next_token_bitmask, int index) {
  XGRAMMAR_CHECK(!IsTerminated()
  ) << "GrammarMatcher has terminated after accepting the stop token, but is trying to "
       "find the next token mask";
  int32_t* bitmask_data_ptr =
      CheckAndGetBitmaskPtr(*next_token_bitmask, tokenizer_info_.GetVocabSize(), index);
  const auto& sorted_decoded_vocab = tokenizer_info_.GetSortedDecodedVocab();
  const auto& adaptive_token_mask_cache = compiled_grammar_->adaptive_token_mask_cache;
  const auto& latest_stack_tops = stack_tops_history_.GetLatest();

  // We check all the stacks one by one, and find the accepted token set or the rejected token set
  // for each stack. We will try to find the small one of the two sets.
  // The final accepted token set is the union of the accepted token sets of all stacks.
  // The final rejected token set is the intersection of the rejected token sets of all stacks.

  // Note these indices store the indices in sorted_decoded_vocab, instead of the token ids.
  tmp_accepted_bitset_.Reset();
  // {-1} means the universal set, i.e. all tokens initially
  tmp_rejected_indices_.assign({-1});

  for (auto top : latest_stack_tops) {
    auto cur_stack_element = persistent_stack_[top];
    if (persistent_stack_.IsEndOfGrammar(cur_stack_element)) {
      continue;
    }

    const auto& adaptive_token_mask = adaptive_token_mask_cache.at(cur_stack_element);

    // For each stack, we will check every uncertain token and put them into the accepted or
    // rejected list.

    // Step 2. Update the accepted tokens in accepted_indices_delta, or the rejected tokens in
    // rejected_indices_delta.

    // If the accepted tokens are saved, it means it is likely to be smaller than the rejected
    // tokens, so we will just find the accepted tokens, and vice versa.

    tmp_rejected_indices_delta_.clear();

    // Examine only the current one stack
    stack_tops_history_.PushHistory({persistent_stack_.NewNode(cur_stack_element)});

    const std::string* prev_token = nullptr;
    int prev_matched_size = 0;

    for (auto cur_token_idx : adaptive_token_mask.uncertain_indices) {
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
          accepted = false;
        } else if (lcp_len < prev_matched_size) {
          RollbackChars(prev_matched_size - lcp_len);
        }
        prev_matched_size = std::min(prev_matched_size, lcp_len);
      }

      // Step 2.2. Find if the current token is accepted or rejected.
      if (accepted) {
        for (int j = prev_matched_size; j < static_cast<int>(cur_token.size()); ++j) {
          if (!AcceptChar(cur_token[j], false)) {
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

    RollbackChars(prev_matched_size + 1);

    // Step 3. Update the accepted_indices or rejected_indices
    if (adaptive_token_mask.store_type == StoreType::kAcceptedBitset) {
      tmp_accepted_bitset_ |= adaptive_token_mask.accepted_bitset;
    } else if (adaptive_token_mask.store_type == StoreType::kAccepted) {
      for (auto idx : adaptive_token_mask.accepted_indices) {
        tmp_accepted_bitset_.Set(sorted_decoded_vocab[idx].first, true);
      }
    } else {
      // rejected_indices = Intersect(
      //     rejected_indices,
      //     adaptive_token_mask.rejected_indices + rejected_indices_delta)
      IntsetUnion(&tmp_rejected_indices_delta_, adaptive_token_mask.rejected_indices);
      IntsetIntersection(&tmp_rejected_indices_, tmp_rejected_indices_delta_);
    }
  }

  // Finally update the rejected_ids bitset
  bool can_reach_end = CanReachEnd();
  SetTokenBitmask(bitmask_data_ptr, tmp_accepted_bitset_, tmp_rejected_indices_, can_reach_end);
}

std::string GrammarMatcher::Impl::FindJumpForwardString() {
  XGRAMMAR_CHECK(!IsTerminated()
  ) << "GrammarMatcher has terminated after accepting the stop token, but is trying to "
       "get the jump forward string";

  std::string result;
  int num_accepted_chars = 0;
  bool can_find_next_char = true;

  while (can_find_next_char) {
    const auto& stack_tops = stack_tops_history_.GetLatest();

    // 1. Check that for every stack top, the next possible char is unique and the same
    // -1 means not found yet; 0~255 means the next char
    int next_char = -1;
    for (auto stack_top : stack_tops) {
      auto stack_element = persistent_stack_[stack_top];
      auto cur_sequence = grammar_->GetRuleExpr(stack_element.sequence_id);
      if (stack_element.parent_id == StackElement::kNoParent &&
          stack_element.element_id == cur_sequence.size()) {
        can_find_next_char = false;
        break;
      }

      auto cur_element = grammar_->GetRuleExpr(cur_sequence[stack_element.element_id]);

      if (cur_element.type == RuleExprType::kByteString) {
        XGRAMMAR_DCHECK(stack_element.element_in_string < cur_element.size());
        if (next_char == -1) {
          next_char = cur_element[stack_element.element_in_string];
        } else if (next_char != cur_element[stack_element.element_in_string]) {
          can_find_next_char = false;
          break;
        }
      } else {
        XGRAMMAR_DCHECK(
            cur_element.type == RuleExprType::kCharacterClass ||
            cur_element.type == RuleExprType::kCharacterClassStar
        );
        if (stack_element.left_utf8_bytes > 0 || cur_element.size() != 3 || cur_element[0] != 0 ||
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
    }

    if (next_char == -1) {
      can_find_next_char = false;
    }

    // 2. If found, accept the char and iterate to the next position
    if (can_find_next_char) {
      result += static_cast<uint8_t>(next_char);

      tmp_new_stack_tops_.clear();
      for (auto stack_top : stack_tops) {
        auto cur_stack_element = persistent_stack_[stack_top];
        auto new_stack_element = AdvanceStackElementWithChar(cur_stack_element, next_char);

        if (new_stack_element == cur_stack_element) {
          ExpandEquivalentStackElements(new_stack_element, &tmp_new_stack_tops_, stack_top);
        } else {
          ExpandEquivalentStackElements(new_stack_element, &tmp_new_stack_tops_);
        }
      }
      stack_tops_history_.PushHistory(tmp_new_stack_tops_);
      ++num_accepted_chars;
    }
  }

  // Rollback all chars accepted
  RollbackChars(num_accepted_chars);
  return result;
}

void GrammarMatcher::Impl::Rollback(int num_tokens) {
  XGRAMMAR_CHECK(num_tokens <= static_cast<int>(token_length_history.size()))
      << "Intended to rollback " << num_tokens << " tokens, but only the last "
      << token_length_history.size() << " steps of history are saved";
  while (num_tokens > 0) {
    int steps = token_length_history.back();
    RollbackChars(steps);
    token_length_history.pop_back();
    --num_tokens;
  }
}

void GrammarMatcher::Impl::SetTokenBitmask(
    int32_t* bitmask_data_ptr,
    const DynamicBitset& accepted_bitset,
    const std::vector<int32_t>& rejected_indices,
    bool can_reach_end
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

    for (int id : tokenizer_info_.GetSpecialTokenIds()) {
      next_token_bitset.Set(id, false);
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

void GrammarMatcher::FillNextTokenBitmask(DLTensor* next_token_bitmask, int index) {
  pimpl_->FillNextTokenBitmask(next_token_bitmask, index);
}

std::string GrammarMatcher::FindJumpForwardString() { return pimpl_->FindJumpForwardString(); }

void GrammarMatcher::Rollback(int num_tokens) { pimpl_->Rollback(num_tokens); }

bool GrammarMatcher::IsTerminated() const { return pimpl_->IsTerminated(); }

void GrammarMatcher::Reset() { pimpl_->Reset(); }

int GrammarMatcher::GetMaxRollbackTokens() const { return pimpl_->GetMaxRollbackTokens(); }

const std::vector<int>& GrammarMatcher::GetStopTokenIds() const {
  return pimpl_->GetStopTokenIds();
}

bool GrammarMatcher::_DebugAcceptString(const std::string& input_str, bool debug_print) {
  return pimpl_->_DebugAcceptString(input_str, debug_print);
}
}  // namespace xgrammar
