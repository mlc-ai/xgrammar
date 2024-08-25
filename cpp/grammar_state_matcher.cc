/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_state_matcher.cc
 */
// #define XGRAMMAR_ENABLE_LOG_DEBUG 1
#include <xgrammar/xgrammar.h>

#include <chrono>
#include <queue>

// #include "../tokenizers/tokenizers.h"
#include "grammar_ast.h"
#include "grammar_serializer.h"
#include "grammar_state_matcher_base.h"
#include "grammar_state_matcher_preproc.h"
#include "grammar_state_matcher_state.h"
#include "support/dynamic_bitset.h"
#include "support/int_set.h"

namespace xgrammar {

/*
 * Note on the matching algorithm
 *
 * Given a context-free grammar, we match the characters in a string one by one.
 *
 * We adopt a non-deterministic pushdown automata (NPDA) in matching. To be specific, we maintain
 * several stacks, each of which represents a possible path in the NPDA, and update the stacks
 * during matching.
 *
 * ## Stack Structure (see grammar_state_matcher_state.h)
 * The element of every stack is a RulePosition object, referring a position in the grammar. If a
 * RulePosition is a RuleRef element (referring to another rule), the next element of the stack will
 * be a position in this rule. If a RulePosition is a CharacterClass element, it will be the last
 * in the stack, meaning *the next* character to match.
 *
 * ## Matching Process (see grammar_state_matcher_base.h)
 * When accepting a new character and it is accepted by a stack, the last element of the stack will
 * be advanced to the next position in the grammar. If it gets to the end of the rule, several
 * elements at the end may be popped out, and the last element of the stack will be advanced.
 *
 * One stack may split since there may be multiple possible next positions. In this case, similar
 * stacks with different top elements will be added. When one stack cannot accept the new character,
 * it will be removed from the stacks.
 *
 * ## Storage of Stacks (see grammar_state_matcher_state.h)
 * Note these stacks form a tree structure as when splitting, the new stacks share the same prefix.
 * We store all RulePositions as a tree, where every path from tree root to a node represents a
 * stack. To represent stack tops, we attach additional pointers pointing the stack top nodes.
 * Also, We maintain a history of the stack top pointers, so we can rollback to the previous state.
 *
 * All tree nodes are maintained by a buffer, and utilize reference counting to recycle. If a node
 * is neither pointed by a stack top pointer, not pointed by some child nodes, it will be freed.
 *
 * ## Example
 * ### Grammar
 * main ::= [a] R
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
 * A: (rule main, choice 0, element 1)
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
 * G: (rule main, choice 0, element 2) (means the matching process has finished, and will be deleted
 * when the next char comes)
 * H: (rule R, choice 1, element 2)
 * I: (rule T, choice 0, element 0)
 * << means the stack top pointers in the current step.
 * The stacks in the current step is: (A, B, F), (A, H, I), (G,)
 *
 * ## Preprocess (see grammar_state_matcher_preproc.h)
 * We will store all information about tokens that needed in matching in a GrammarStateInitContext
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

/* \brief The concrete implementation of GrammarStateMatcherNode. */
class GrammarStateMatcher::Impl : public GrammarStateMatcherBase {
 private:
  using RuleExpr = BNFGrammar::Impl::RuleExpr;
  using RuleExprType = BNFGrammar::Impl::RuleExprType;
  using SaveType = CatagorizedTokens::SaveType;

 public:
  Impl(
      std::shared_ptr<GrammarStateInitContext> init_ctx,
      std::optional<std::vector<int>> stop_token_ids = std::nullopt,
      bool terminate_without_stop_token = false,
      int max_rollback_steps = 0
  )
      : GrammarStateMatcherBase(init_ctx->grammar),
        init_ctx_(init_ctx),
        stop_token_ids_(stop_token_ids.value_or(init_ctx->detected_stop_token_ids)),
        terminate_without_stop_token_(terminate_without_stop_token),
        max_rollback_steps_(max_rollback_steps),
        tmp_accepted_bitset_(init_ctx_->vocab_size) {
    XGRAMMAR_CHECK(!stop_token_ids.has_value() || !stop_token_ids->empty())
        << "The stop_token_ids should not be empty";
  }

  bool AcceptToken(int32_t token_id, bool verbose = false);

  bool _AcceptString(const std::string& input_str, bool verbose = false);

  void FindNextTokenBitmask(DLTensor* next_token_bitmask);

  static void GetRejectedTokensFromBitMask(
      const DLTensor& token_bitmask, size_t vocab_size, std::vector<int>* rejected_tokens
  );

  std::string FindJumpForwardString();

  void Rollback(int num_tokens);

  int GetMaxRollbackSteps() const { return max_rollback_steps_; }

  size_t GetVocabSize() const { return init_ctx_->vocab_size; }

  bool IsTerminated() const;

  void Reset() {
    stack_tops_history_.Reset();
    token_length_history.clear();
    PushInitialState(kInvalidRulePosition, true);
  }

 private:
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

  static void CheckTokenBitmaskValidity(const DLTensor& token_bitmask, size_t vocab_size);

  /*! \brief Set the acceptable next token in next_token_bitmask. */
  void SetTokenBitmask(
      DLTensor* next_token_bitmask,
      const DynamicBitset& accepted_bitset,
      const std::vector<int32_t>& rejected_indices,
      bool can_reach_end
  );

  /*!
   * \brief Accept the stop token and terminates the matcher.
   * \returns Whether the stop token can be accepted.
   */
  bool AcceptStopToken();

  // friend IntTuple FindNextRejectedTokens(GrammarStateMatcher matcher, bool verbose);
  // friend NDArray FindNextTokenBitmaskAsNDArray(GrammarStateMatcher matcher);

  std::shared_ptr<GrammarStateInitContext> init_ctx_;
  std::vector<int> stop_token_ids_;
  bool terminate_without_stop_token_;
  int max_rollback_steps_;
  std::deque<int> token_length_history;

  // Temporary data for FindNextTokenBitmask. They are stored here to avoid repeated allocation.
  DynamicBitset tmp_accepted_bitset_;
  std::vector<int32_t> tmp_rejected_indices_;
  std::vector<int32_t> tmp_rejected_indices_delta_;
};

bool GrammarStateMatcher::Impl::AcceptStopToken() {
  if (terminate_without_stop_token_) {
    return false;
  }
  if (!CanReachEnd()) {
    return false;
  }
  stack_tops_history_.PushHistory({});  // Terminate the matcher by setting the stack to empty
  return true;
}

bool GrammarStateMatcher::Impl::IsTerminated() const {
  if (terminate_without_stop_token_) {
    return CanReachEnd();
  }
  return stack_tops_history_.GetLatest().empty();
}

// TODO(yixin): Polish verbose logging
bool GrammarStateMatcher::Impl::AcceptToken(int32_t token_id, bool verbose) {
  if (IsTerminated()) {
    if (verbose) {
      XGRAMMAR_LOG(INFO) << "The matcher has terminated after accepting the stop token, but is "
                            "trying to accept new token with id "
                         << token_id;
    }
    return false;
  }

  XGRAMMAR_CHECK(token_id >= 0 && token_id < static_cast<int>(init_ctx_->vocab_size))
      << "Invalid token id " << token_id << " for GrammarStateMatcher";

  if (verbose) {
    XGRAMMAR_LOG(INFO) << "Accepting token id " << token_id << ", string: \""
                       << PrintAsEscapedUTF8(init_ctx_->token_table[token_id])
                       << "\", state state:\n"
                       << PrintStackState();
  }

  // Handle the stop token
  if (std::find(stop_token_ids_.begin(), stop_token_ids_.end(), token_id) !=
      stop_token_ids_.end()) {
    bool accepted = AcceptStopToken();
    if (verbose) {
      XGRAMMAR_LOG(INFO) << "The token is an end token. Is accepted: " << accepted;
    }
    return accepted;
  }

  if (init_ctx_->special_token_ids.count(token_id) > 0) {
    XGRAMMAR_LOG(FATAL
    ) << "Token id "
      << token_id << ": " << init_ctx_->token_table[token_id]
      << " is regarded as a special token, and cannot be accepted by the GrammarStateMatcher";
  }

  const auto& token = init_ctx_->token_table[token_id];
  int pos = 0;
  for (auto char_value : token) {
    if (!AcceptChar(char_value, verbose)) {
      if (verbose) {
        XGRAMMAR_LOG(INFO) << "The token is rejected at position " << pos << ", character "
                           << PrintAsEscapedUTF8(char_value);
      }
      return false;
    }
    ++pos;
  }
  token_length_history.push_back(token.size());
  if (static_cast<int>(token_length_history.size()) > max_rollback_steps_) {
    DiscardEarliestChars(token_length_history.front());
    token_length_history.pop_front();
  }
  if (verbose) {
    XGRAMMAR_LOG(INFO) << "The token is accepted. State after accepting:\n" << PrintStackState();
  }
  return true;
}

bool GrammarStateMatcher::Impl::_AcceptString(const std::string& input_str, bool verbose) {
  if (IsTerminated()) {
    if (verbose) {
      XGRAMMAR_LOG(INFO) << "The matcher has terminated after accepting the stop token, but is "
                            "trying to accept new string "
                         << PrintAsEscapedUTF8(input_str);
    }
    return false;
  }

  int accepted_cnt = 0;
  for (auto char_value : input_str) {
    if (!AcceptChar(char_value, verbose)) {
      if (verbose) {
        XGRAMMAR_LOG(INFO) << "Matching failed after accepting " << accepted_cnt << " characters";
      }
      RollbackChars(accepted_cnt);
      return false;
    }
    ++accepted_cnt;
  }
  token_length_history.push_back(input_str.size());
  if (static_cast<int>(token_length_history.size()) > max_rollback_steps_) {
    DiscardEarliestChars(token_length_history.front());
    token_length_history.pop_front();
  }
  if (verbose) {
    XGRAMMAR_LOG(INFO) << "String \"" << PrintAsEscapedUTF8(input_str)
                       << "\" is accepted. State after accepting:\n"
                       << PrintStackState();
  }
  return true;
}

void GrammarStateMatcher::Impl::CheckTokenBitmaskValidity(
    const DLTensor& token_bitmask, size_t vocab_size
) {
  XGRAMMAR_CHECK(
      token_bitmask.dtype.code == kDLUInt && token_bitmask.dtype.bits == 32 && token_bitmask.data &&
      token_bitmask.ndim == 1 && token_bitmask.shape
  ) << "The provied bitmask's shape or dtype is not valid.";
  XGRAMMAR_CHECK(token_bitmask.shape[0] >= DynamicBitset::CalculateBufferSize(vocab_size))
      << "The provided bitmask is not large enough to store the token set. The length should be "
      << DynamicBitset::CalculateBufferSize(vocab_size) << " at least";
}

void GrammarStateMatcher::Impl::FindNextTokenBitmask(DLTensor* next_token_bitmask) {
  XGRAMMAR_CHECK(!IsTerminated()
  ) << "GrammarStateMatcher has terminated after accepting the stop token, but is trying to "
       "find the next token mask";
  CheckTokenBitmaskValidity(*next_token_bitmask, init_ctx_->vocab_size);
  const auto& sorted_token_table = init_ctx_->sorted_token_table;
  const auto& catagorized_tokens_for_grammar = init_ctx_->catagorized_tokens_for_grammar;
  const auto& latest_stack_tops = stack_tops_history_.GetLatest();

  // We check all the stacks one by one, and find the accepted token set or the rejected token set
  // for each stack. We will try to find the small one of the two sets.
  // The final accepted token set is the union of the accepted token sets of all stacks.
  // The final rejected token set is the intersection of the rejected token sets of all stacks.

  // Note these indices store the indices in sorted_token_table, instead of the token ids.
  tmp_accepted_bitset_.Reset();
  // {-1} means the universal set, i.e. all tokens initially
  tmp_rejected_indices_.assign({-1});

  for (auto top : latest_stack_tops) {
    auto cur_rule_position = tree_[top];
    if (tree_.IsEndPosition(cur_rule_position)) {
      continue;
    }

    const auto& catagorized_tokens = catagorized_tokens_for_grammar.at(cur_rule_position);

    // For each stack, we will check every uncertain token and put them into the accepted or
    // rejected list.

    // Step 2. Update the accepted tokens in accepted_indices_delta, or the rejected tokens in
    // rejected_indices_delta.

    // If the accepted tokens are saved, it means it is likely to be smaller than the rejected
    // tokens, so we will just find the accepted tokens, and vice versa.

    tmp_rejected_indices_delta_.clear();

    // Examine only the current one stack
    stack_tops_history_.PushHistory({tree_.NewNode(cur_rule_position)});

    const std::string* prev_token = nullptr;
    int prev_matched_size = 0;

    for (auto cur_token_idx : catagorized_tokens.uncertain_indices) {
      const auto& cur_token = sorted_token_table[cur_token_idx].second;
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
      if (catagorized_tokens.save_type == SaveType::kAcceptedBitset ||
          catagorized_tokens.save_type == SaveType::kAccepted) {
        if (accepted) {
          tmp_accepted_bitset_.Set(sorted_token_table[cur_token_idx].first, true);
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
    if (catagorized_tokens.save_type == SaveType::kAcceptedBitset) {
      tmp_accepted_bitset_ |= catagorized_tokens.accepted_bitset;
    } else if (catagorized_tokens.save_type == SaveType::kAccepted) {
      for (auto idx : catagorized_tokens.accepted_indices) {
        tmp_accepted_bitset_.Set(sorted_token_table[idx].first, true);
      }
    } else {
      // rejected_indices = Intersect(
      //     rejected_indices,
      //     catagorized_tokens.rejected_indices + rejected_indices_delta)
      IntsetUnion(&tmp_rejected_indices_delta_, catagorized_tokens.rejected_indices);
      IntsetIntersection(&tmp_rejected_indices_, tmp_rejected_indices_delta_);
    }
  }

  // Finally update the rejected_ids bitset
  bool can_reach_end = CanReachEnd();
  SetTokenBitmask(next_token_bitmask, tmp_accepted_bitset_, tmp_rejected_indices_, can_reach_end);
}

void GrammarStateMatcher::Impl::GetRejectedTokensFromBitMask(
    const DLTensor& token_bitmask, size_t vocab_size, std::vector<int>* rejected_tokens
) {
  CheckTokenBitmaskValidity(token_bitmask, vocab_size);
  DynamicBitset bitset(vocab_size, reinterpret_cast<uint32_t*>(token_bitmask.data));
  rejected_tokens->clear();
  for (int i = bitset.FindFirstZero(); i != -1; i = bitset.FindNextZero(i)) {
    std::cout << i << std::endl;
    rejected_tokens->push_back(i);
  }
}

std::string GrammarStateMatcher::Impl::FindJumpForwardString() {
  XGRAMMAR_CHECK(!IsTerminated()
  ) << "GrammarStateMatcher has terminated after accepting the stop token, but is trying to "
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
      auto rule_position = tree_[stack_top];
      auto cur_sequence = grammar_->GetRuleExpr(rule_position.sequence_id);
      if (rule_position.parent_id == RulePosition::kNoParent &&
          rule_position.element_id == cur_sequence.size()) {
        can_find_next_char = false;
        break;
      }

      auto cur_element = grammar_->GetRuleExpr(cur_sequence[rule_position.element_id]);

      if (cur_element.type == RuleExprType::kByteString) {
        XGRAMMAR_DCHECK(rule_position.element_in_string < cur_element.size());
        if (next_char == -1) {
          next_char = cur_element[rule_position.element_in_string];
        } else if (next_char != cur_element[rule_position.element_in_string]) {
          can_find_next_char = false;
          break;
        }
      } else {
        XGRAMMAR_DCHECK(
            cur_element.type == RuleExprType::kCharacterClass ||
            cur_element.type == RuleExprType::kCharacterClassStar
        );
        if (rule_position.left_utf8_bytes > 0 || cur_element.size() != 3 || cur_element[0] != 0 ||
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
        auto cur_rule_position = tree_[stack_top];
        auto new_rule_position = UpdatePositionWithChar(cur_rule_position, next_char);

        if (new_rule_position == cur_rule_position) {
          ExpandRulePosition(new_rule_position, &tmp_new_stack_tops_, true, stack_top);
        } else {
          ExpandRulePosition(new_rule_position, &tmp_new_stack_tops_, true);
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

void GrammarStateMatcher::Impl::Rollback(int num_tokens) {
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

void GrammarStateMatcher::Impl::SetTokenBitmask(
    DLTensor* next_token_bitmask,
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
      init_ctx_->vocab_size, reinterpret_cast<uint32_t*>(next_token_bitmask->data)
  );
  const auto& sorted_token_table = init_ctx_->sorted_token_table;

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
      auto id = sorted_token_table[i].first;
      if (!accepted_bitset[id]) {
        next_token_bitset.Set(id, false);
      }
    }

    for (int id : init_ctx_->special_token_ids) {
      next_token_bitset.Set(id, false);
    }
    if (!can_reach_end) {
      for (int id : stop_token_ids_) {
        next_token_bitset.Set(id, false);
      }
    }
  }
}

int GrammarStateMatcher::Impl::GetNextUncertainToken(
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

GrammarStateMatcher::GrammarStateMatcher(
    std::shared_ptr<GrammarStateInitContext> init_ctx,
    std::optional<std::vector<int>> stop_token_ids,
    bool terminate_without_stop_token,
    int max_rollback_steps
)
    : pimpl_(std::make_shared<GrammarStateMatcher::Impl>(
          init_ctx, stop_token_ids, terminate_without_stop_token, max_rollback_steps
      )) {}

bool GrammarStateMatcher::AcceptToken(int32_t token_id, bool verbose) {
  return pimpl_->AcceptToken(token_id, verbose);
}

bool GrammarStateMatcher::_AcceptString(const std::string& input_str, bool verbose) {
  return pimpl_->_AcceptString(input_str, verbose);
}

uint32_t GrammarStateMatcher::GetBufferSize(size_t vocab_size) {
  return DynamicBitset::CalculateBufferSize(vocab_size);
}

void GrammarStateMatcher::FindNextTokenBitmask(DLTensor* next_token_bitmask) {
  pimpl_->FindNextTokenBitmask(next_token_bitmask);
}

void GrammarStateMatcher::GetRejectedTokensFromBitMask(
    const DLTensor& token_bitmask, size_t vocab_size, std::vector<int>* rejected_tokens
) {
  return Impl::GetRejectedTokensFromBitMask(token_bitmask, vocab_size, rejected_tokens);
}

std::string GrammarStateMatcher::FindJumpForwardString() { return pimpl_->FindJumpForwardString(); }

void GrammarStateMatcher::Rollback(int num_tokens) { pimpl_->Rollback(num_tokens); }

int GrammarStateMatcher::GetMaxRollbackSteps() const { return pimpl_->GetMaxRollbackSteps(); }

size_t GrammarStateMatcher::GetVocabSize() const { return pimpl_->GetVocabSize(); }

bool GrammarStateMatcher::IsTerminated() const { return pimpl_->IsTerminated(); }

void GrammarStateMatcher::Reset() { pimpl_->Reset(); }

// #ifndef COMPILE_MLC_WASM_RUNTIME
// // This creates tokenizer dependency issue in WASM building for web, hence skipped
// TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherFromTokenizer")
//     .set_body_typed([](BNFGrammar grammar, Optional<Tokenizer> tokenizer, int max_rollback_steps)
//     {
//       auto preproc_start = std::chrono::high_resolution_clock::now();
//       std::shared_ptr<mlc::llm::serve::GrammarStateInitContext> init_ctx;
//       if (tokenizer) {
//         init_ctx = GrammarStateMatcher::CreateInitContext(
//             grammar, tokenizer.value()->PostProcessedTokenTable()
//         );
//       } else {
//         init_ctx = GrammarStateMatcher::CreateInitContext(grammar, {});
//       }

//       auto preproc_end = std::chrono::high_resolution_clock::now();
//       XGRAMMAR_LOG(INFO
//       ) << "GrammarStateMatcher preprocess takes "
//         << std::chrono::duration_cast<std::chrono::microseconds>(preproc_end - preproc_start)
//                .count()
//         << "us";
//       return GrammarStateMatcher(init_ctx, max_rollback_steps);
//     });
// #endif

// TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherFromTokenTable")
//     .set_body([](TVMArgs args, TVMRetValue* rv) {
//       BNFGrammar grammar = args[0];
//       Array<String> token_table_arr = args[1];
//       std::vector<std::string> token_table;
//       for (int i = 0; i < token_table_arr.size(); ++i) {
//         token_table.push_back(token_table_arr[i]);
//       }
//       int max_rollback_steps = args[args.size() - 1];
//       auto init_ctx = GrammarStateMatcher::CreateInitContext(grammar, token_table);
//       *rv = GrammarStateMatcher(init_ctx, max_rollback_steps);
//     });

// TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherDebugAcceptChar")
//     .set_body_typed([](GrammarStateMatcher matcher, int32_t codepoint, bool verbose) {
//       auto mutable_node =
//           const_cast<GrammarStateMatcher::Impl*>(matcher.as<GrammarStateMatcher::Impl>());
//       return mutable_node->AcceptChar(codepoint, verbose);
//     });

// TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherAcceptToken")
//     .set_body_typed([](GrammarStateMatcher matcher, int32_t token_id, bool verbose) {
//       return matcher->AcceptToken(token_id, verbose);
//     });

// TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherFindJumpForwardString")
//     .set_body_typed([](GrammarStateMatcher matcher) { return matcher->FindJumpForwardString();
//     });

// TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherRollback")
//     .set_body_typed([](GrammarStateMatcher matcher, int num_tokens) {
//       matcher->Rollback(num_tokens);
//     });

// TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherMaxRollbackSteps")
//     .set_body_typed([](GrammarStateMatcher matcher) { return matcher->MaxRollbackSteps(); });

// TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherIsTerminated")
//     .set_body_typed([](GrammarStateMatcher matcher) { return matcher->IsTerminated(); });

// TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherResetState")
//     .set_body_typed([](GrammarStateMatcher matcher) { matcher->ResetState(); });

// /*! \brief Check if a matcher can accept the complete string, and then reach the end of the
//  * grammar. Does not change the state of the GrammarStateMatcher. For test purpose. */
// bool MatchCompleteString(GrammarStateMatcher matcher, String str, bool verbose) {
//   auto mutable_node =
//       const_cast<GrammarStateMatcher::Impl*>(matcher.as<GrammarStateMatcher::Impl>());
//   int accepted_cnt = 0;
//   for (auto char_value : str.operator std::string()) {
//     if (!mutable_node->AcceptChar(char_value, verbose)) {
//       if (verbose) {
//         XGRAMMAR_LOG(INFO) << "Matching failed after accepting " << accepted_cnt << "
//         characters";
//       }
//       mutable_node->RollbackChars(accepted_cnt);
//       return false;
//     }
//     ++accepted_cnt;
//   }
//   auto accepted = mutable_node->CanReachEnd();
//   if (verbose) {
//     if (accepted) {
//       XGRAMMAR_LOG(INFO) << "Matching succeed after accepting " << accepted_cnt << " characters";
//     } else {
//       XGRAMMAR_LOG(INFO) << "Matching failed due to the end state not reached after all "
//                          << accepted_cnt << " characters are accepted";
//     }
//   }
//   mutable_node->RollbackChars(accepted_cnt);
//   return accepted;
// }

// TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherDebugMatchCompleteString")
//     .set_body_typed([](GrammarStateMatcher matcher, String str, bool verbose) {
//       return MatchCompleteString(matcher, str, verbose);
//     });

/*! \brief Print the accepted and rejected tokens stored in the bitset. For debug purposes. */
// std::string PrintAcceptedRejectedTokens(
//     const std::shared_ptr<mlc::llm::serve::GrammarStateInitContext>& init_ctx,
//     const DynamicBitset& bitset,
//     int threshold = 300
// ) {
//   std::stringstream ss;
//   auto vocab_size = init_ctx->vocab_size;
//   std::vector<int64_t> accepted_ids;
//   std::vector<int64_t> rejected_ids;
//   for (int i = 0; i < vocab_size; i++) {
//     if (bitset[i]) {
//       accepted_ids.push_back(i);
//     } else {
//       rejected_ids.push_back(i);
//     }
//   }

//   ss << "Accepted: ";
//   auto end_it =
//       accepted_ids.size() > threshold ? accepted_ids.begin() + threshold : accepted_ids.end();
//   for (auto it = accepted_ids.begin(); it != end_it; ++it) {
//     ss << "<" << PrintAsEscapedUTF8(init_ctx->token_table[*it]) << "> ";
//   }
//   if (accepted_ids.size() > threshold) {
//     ss << "...";
//   }
//   ss << "\n";

//   ss << "Rejected: ";
//   end_it = rejected_ids.size() > threshold ? rejected_ids.begin() + threshold :
//   rejected_ids.end(); for (auto it = rejected_ids.begin(); it != end_it; ++it) {
//     ss << "<" << PrintAsEscapedUTF8(init_ctx->token_table[*it]) << "> ";
//   }
//   if (rejected_ids.size() > threshold) {
//     ss << "...";
//   }
//   ss << "\n";
//   return ss.str();
// }

/*!
 * \brief Find the ids of the rejected tokens for the next step. For debug purposes.
 * \param matcher The matcher to test.
 * \param verbose Whether to print information about the timing and results to stderr.
 * \returns A tuple of rejected token ids.
 */
// IntTuple FindNextRejectedTokens(GrammarStateMatcher matcher, bool verbose = false) {
//   auto init_ctx = matcher.as<GrammarStateMatcher::Impl>()->init_ctx_;
//   auto vocab_size = init_ctx->vocab_size;
//   auto bitset_size = DynamicBitset::CalculateBufferSize(vocab_size);
//   auto ndarray = NDArray::Empty(
//       ShapeTuple{static_cast<long>(bitset_size)}, DLDataType{kDLUInt, 32, 1}, DLDevice{kDLCPU, 0}
//   );
//   auto dltensor = const_cast<DLTensor*>(ndarray.operator->());

//   std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
//   if (verbose) {
//     start = std::chrono::high_resolution_clock::now();
//   }
//   matcher->FindNextTokenBitmask(dltensor);
//   if (verbose) {
//     end = std::chrono::high_resolution_clock::now();
//   }

//   auto bitset = DynamicBitset(vocab_size, reinterpret_cast<uint32_t*>(dltensor->data));
//   std::vector<int64_t> rejected_ids;
//   for (int i = 0; i < vocab_size; i++) {
//     if (bitset[i] == 0) {
//       rejected_ids.push_back(i);
//     }
//   }

//   if (verbose) {
//     XGRAMMAR_LOG(INFO) << "FindNextTokenBitmask takes "
//                        << std::chrono::duration_cast<std::chrono::microseconds>(end -
//                        start).count()
//                        << "us"
//                        << ", found accepted: " << vocab_size - rejected_ids.size()
//                        << ", rejected: " << rejected_ids.size();
//   }

//   auto ret = IntTuple(rejected_ids);
//   return ret;
// }

// TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherFindNextRejectedTokens")
//     .set_body_typed(FindNextRejectedTokens);

// /*!
//  * \brief Find the bitmask for the next token as an NDArray.
//  * \returns An NDArray of the bitmask for the next token of shape (bitmask_size,).
//  */
// NDArray FindNextTokenBitmaskAsNDArray(GrammarStateMatcher matcher) {
//   auto init_ctx = matcher.as<GrammarStateMatcher::Impl>()->init_ctx_;
//   auto vocab_size = init_ctx->vocab_size;
//   auto bitset_size = DynamicBitset::CalculateBufferSize(vocab_size);
//   auto bitmask = NDArray::Empty(
//       ShapeTuple{static_cast<long>(bitset_size)}, DLDataType{kDLUInt, 32, 1}, DLDevice{kDLCPU, 0}
//   );
//   auto dltensor = const_cast<DLTensor*>(bitmask.operator->());
//   matcher->FindNextTokenBitmask(dltensor);
//   return bitmask;
// }

// TVM_REGISTER_GLOBAL("mlc.grammar.GrammarStateMatcherFindNextTokenBitmaskAsNDArray")
//     .set_body_typed(FindNextTokenBitmaskAsNDArray);

}  // namespace xgrammar
