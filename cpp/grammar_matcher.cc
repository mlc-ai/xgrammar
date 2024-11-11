/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_matcher.cc
 */
#include <xgrammar/xgrammar.h>

#include <chrono>
#include <queue>

#include "grammar_data_structure.h"
#include "grammar_matcher_base.h"
#include "grammar_matcher_preproc.h"
#include "grammar_matcher_state.h"
#include "grammar_serializer.h"
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
 * ## Stack Structure (see grammar_matcher_state.h)
 * The element of every stack is a RulePosition object, referring a position in the grammar. If a
 * RulePosition is a RuleRef element (referring to another rule), the next element of the stack will
 * be a position in this rule. If a RulePosition is a CharacterClass element, it will be the last
 * in the stack, meaning *the next* character to match.
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
 * We store all RulePositions as a tree, where every path from tree root to a node represents a
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
 private:
  using RuleExpr = BNFGrammar::Impl::RuleExpr;
  using RuleExprType = BNFGrammar::Impl::RuleExprType;
  using SaveType = CatagorizedTokens::SaveType;

 public:
  Impl(
      const CompiledGrammar& compiled_grammar,
      std::optional<std::vector<int>> override_stop_tokens = std::nullopt,
      bool terminate_without_stop_token = false,
      std::optional<int> mask_vocab_size = std::nullopt,
      int max_rollback_tokens = 0
  )
      : GrammarMatcherBase(compiled_grammar->grammar),
        compiled_grammar_(compiled_grammar),
        stop_token_ids_(override_stop_tokens.value_or(compiled_grammar->detected_stop_token_ids)),
        terminate_without_stop_token_(terminate_without_stop_token),
        mask_vocab_size_(mask_vocab_size.value_or(compiled_grammar_->vocab_size)),
        max_rollback_tokens_(max_rollback_tokens),
        tmp_accepted_bitset_(mask_vocab_size_) {
    XGRAMMAR_CHECK(!override_stop_tokens.has_value() || !override_stop_tokens->empty())
        << "The override_stop_tokens should not be empty";
  }

  bool AcceptToken(int32_t token_id, bool verbose = false);

  bool AcceptString(const std::string& input_str, bool verbose = false);

  void GetNextTokenBitmask(DLTensor* next_token_bitmask);

  static void GetRejectedTokensFromBitMask(
      const DLTensor& token_bitmask, size_t mask_vocab_size, std::vector<int>* rejected_tokens
  );

  std::string FindJumpForwardString();

  void Rollback(int num_tokens);

  int GetMaxRollbackTokens() const { return max_rollback_tokens_; }

  size_t GetMaskVocabSize() const { return mask_vocab_size_; }

  const std::vector<int>& GetStopTokenIds() const { return stop_token_ids_; }

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

  static void CheckTokenBitmaskValidity(const DLTensor& token_bitmask, size_t mask_vocab_size);

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

  // friend IntTuple FindNextRejectedTokens(GrammarMatcher matcher, bool verbose);
  // friend NDArray GetNextTokenBitmaskAsNDArray(GrammarMatcher matcher);

  CompiledGrammar compiled_grammar_;
  std::vector<int> stop_token_ids_;
  bool terminate_without_stop_token_;
  size_t mask_vocab_size_;
  int max_rollback_tokens_;
  std::deque<int> token_length_history;

  // Temporary data for GetNextTokenBitmask. They are stored here to avoid repeated allocation.
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
  return true;
}

bool GrammarMatcher::Impl::IsTerminated() const {
  if (terminate_without_stop_token_) {
    return CanReachEnd();
  }
  return stack_tops_history_.GetLatest().empty();
}

// TODO(yixin): Polish verbose logging
bool GrammarMatcher::Impl::AcceptToken(int32_t token_id, bool verbose) {
  if (IsTerminated()) {
    if (verbose) {
      XGRAMMAR_LOG(INFO) << "The matcher has terminated after accepting the stop token, but is "
                            "trying to accept new token with id "
                         << token_id;
    }
    return false;
  }

  XGRAMMAR_CHECK(token_id >= 0 && token_id < static_cast<int>(compiled_grammar_->vocab_size))
      << "Invalid token id " << token_id << " for GrammarMatcher";

  if (verbose) {
    XGRAMMAR_LOG(INFO) << "Accepting token id " << token_id << ", string: \""
                       << PrintAsEscapedUTF8(compiled_grammar_->decoded_vocab[token_id])
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

  if (compiled_grammar_->special_token_ids.count(token_id) > 0) {
    XGRAMMAR_LOG(FATAL
    ) << "Token id "
      << token_id << ": " << compiled_grammar_->decoded_vocab[token_id]
      << " is regarded as a special token, and cannot be accepted by the GrammarMatcher";
  }

  const auto& token = compiled_grammar_->decoded_vocab[token_id];
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
  if (static_cast<int>(token_length_history.size()) > max_rollback_tokens_) {
    DiscardEarliestChars(token_length_history.front());
    token_length_history.pop_front();
  }
  if (verbose) {
    XGRAMMAR_LOG(INFO) << "The token is accepted. State after accepting:\n" << PrintStackState();
  }
  return true;
}

bool GrammarMatcher::Impl::AcceptString(const std::string& input_str, bool verbose) {
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
  if (static_cast<int>(token_length_history.size()) > max_rollback_tokens_) {
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

void GrammarMatcher::Impl::CheckTokenBitmaskValidity(
    const DLTensor& token_bitmask, size_t mask_vocab_size
) {
  XGRAMMAR_CHECK(
      token_bitmask.dtype.code == kDLInt && token_bitmask.dtype.bits == 32 && token_bitmask.data &&
      token_bitmask.ndim == 1 && token_bitmask.shape
  ) << "The provied bitmask's shape or dtype is not valid.";
  XGRAMMAR_CHECK(token_bitmask.shape[0] >= DynamicBitset::CalculateBufferSize(mask_vocab_size))
      << "The provided bitmask is not large enough to store the token set. The length should be "
      << DynamicBitset::CalculateBufferSize(mask_vocab_size) << " at least";
}

void GrammarMatcher::Impl::GetNextTokenBitmask(DLTensor* next_token_bitmask) {
  XGRAMMAR_CHECK(!IsTerminated()
  ) << "GrammarMatcher has terminated after accepting the stop token, but is trying to "
       "find the next token mask";
  CheckTokenBitmaskValidity(*next_token_bitmask, mask_vocab_size_);
  const auto& sorted_decoded_vocab = compiled_grammar_->sorted_decoded_vocab;
  const auto& catagorized_tokens_for_grammar = compiled_grammar_->catagorized_tokens_for_grammar;
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
      if (catagorized_tokens.save_type == SaveType::kAcceptedBitset ||
          catagorized_tokens.save_type == SaveType::kAccepted) {
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
    if (catagorized_tokens.save_type == SaveType::kAcceptedBitset) {
      tmp_accepted_bitset_ |= catagorized_tokens.accepted_bitset;
    } else if (catagorized_tokens.save_type == SaveType::kAccepted) {
      for (auto idx : catagorized_tokens.accepted_indices) {
        tmp_accepted_bitset_.Set(sorted_decoded_vocab[idx].first, true);
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

void GrammarMatcher::Impl::GetRejectedTokensFromBitMask(
    const DLTensor& token_bitmask, size_t mask_vocab_size, std::vector<int>* rejected_tokens
) {
  CheckTokenBitmaskValidity(token_bitmask, mask_vocab_size);
  DynamicBitset bitset(mask_vocab_size, reinterpret_cast<uint32_t*>(token_bitmask.data));
  rejected_tokens->clear();
  for (int i = bitset.FindFirstZero(); i != -1; i = bitset.FindNextZero(i)) {
    rejected_tokens->push_back(i);
  }
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
      mask_vocab_size_, reinterpret_cast<uint32_t*>(next_token_bitmask->data)
  );
  const auto& sorted_decoded_vocab = compiled_grammar_->sorted_decoded_vocab;

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

    for (int id : compiled_grammar_->special_token_ids) {
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
    std::optional<int> mask_vocab_size,
    int max_rollback_tokens
)
    : pimpl_(std::make_shared<GrammarMatcher::Impl>(
          compiled_grammar,
          override_stop_tokens,
          terminate_without_stop_token,
          mask_vocab_size,
          max_rollback_tokens
      )) {}

bool GrammarMatcher::AcceptToken(int32_t token_id, bool verbose) {
  return pimpl_->AcceptToken(token_id, verbose);
}

bool GrammarMatcher::AcceptString(const std::string& input_str, bool verbose) {
  return pimpl_->AcceptString(input_str, verbose);
}

uint32_t GrammarMatcher::GetBufferSize(size_t mask_vocab_size) {
  return DynamicBitset::CalculateBufferSize(mask_vocab_size);
}

void GrammarMatcher::GetNextTokenBitmask(DLTensor* next_token_bitmask) {
  pimpl_->GetNextTokenBitmask(next_token_bitmask);
}

void GrammarMatcher::GetRejectedTokensFromBitMask(
    const DLTensor& token_bitmask, size_t mask_vocab_size, std::vector<int>* rejected_tokens
) {
  return Impl::GetRejectedTokensFromBitMask(token_bitmask, mask_vocab_size, rejected_tokens);
}

std::string GrammarMatcher::FindJumpForwardString() { return pimpl_->FindJumpForwardString(); }

void GrammarMatcher::Rollback(int num_tokens) { pimpl_->Rollback(num_tokens); }

int GrammarMatcher::GetMaxRollbackTokens() const { return pimpl_->GetMaxRollbackTokens(); }

size_t GrammarMatcher::GetMaskVocabSize() const { return pimpl_->GetMaskVocabSize(); }

const std::vector<int>& GrammarMatcher::GetStopTokenIds() const {
  return pimpl_->GetStopTokenIds();
}

bool GrammarMatcher::IsTerminated() const { return pimpl_->IsTerminated(); }

void GrammarMatcher::Reset() { pimpl_->Reset(); }

}  // namespace xgrammar
