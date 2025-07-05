/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/compiler.cc
 */

#include <xgrammar/compiler.h>

#include <algorithm>
#include <bitset>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include "compiled_grammar_data_structure.h"
#include "earley_parser.h"
#include "fsm.h"
#include "fsm_builder.h"
#include "grammar_data_structure.h"
#include "grammar_functor.h"
#include "support/logging.h"
#include "support/thread_pool.h"
#include "support/thread_safe_cache.h"
#include "support/utils.h"
#include "testing.h"

namespace std {

/*! \brief Define the hash function for StructuralTagItem. */
template <>
struct hash<xgrammar::StructuralTagItem> {
  size_t operator()(const xgrammar::StructuralTagItem& tag) const {
    return xgrammar::HashCombine(
        std::hash<std::string>{}(tag.begin),
        std::hash<std::string>{}(tag.schema),
        std::hash<std::string>{}(tag.end)
    );
  }
};

}  // namespace std

namespace xgrammar {

/******************* MemorySize *******************/

std::size_t MemorySize(const Grammar::Impl& impl) {
  // we assume strings are not long, so we don't iterate through all the rules
  std::size_t result = impl.rules_.size() * sizeof(impl.rules_[0]) +
                       MemorySize(impl.grammar_expr_data_) + MemorySize(impl.grammar_expr_indptr_) +
                       MemorySize(impl.allow_empty_rule_ids) + MemorySize(impl.complete_fsm) +
                       impl.per_rule_fsms.size() * sizeof(impl.per_rule_fsms[0]);
  return result;
}

std::size_t Grammar::Impl::MemorySize() const { return xgrammar::MemorySize(*this); }

std::size_t MemorySize(const AdaptiveTokenMask& mask) {
  return MemorySize(mask.uncertain_indices) + MemorySize(mask.accepted_indices) +
         MemorySize(mask.rejected_indices) + MemorySize(mask.accepted_bitset);
}

std::size_t AdaptiveTokenMask::MemorySize() const { return xgrammar::MemorySize(*this); }

std::size_t CompiledGrammar::Impl::MemorySize() const {
  std::size_t sum = 0;
  sum += grammar->MemorySize();
  sum += adaptive_token_mask_cache.size() * sizeof(*adaptive_token_mask_cache.begin());
  for (auto& [_, mask] : adaptive_token_mask_cache) sum += mask.MemorySize();
  return sum;
}

std::size_t CompiledGrammar::MemorySizeBytes() const { return pimpl_->MemorySize(); }

/******************* AdaptiveTokenMask and CompiledGrammar *******************/

AdaptiveTokenMask::AdaptiveTokenMask(
    size_t vocab_size,
    const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
    const std::vector<int32_t>& accepted_indices,
    const std::vector<int32_t>& rejected_indices,
    const std::vector<int32_t>& uncertain_indices
) {
  auto size_acc = accepted_indices.size();
  auto size_rej = rejected_indices.size();

  store_type = size_acc >= USE_BITSET_THRESHOLD && size_rej >= USE_BITSET_THRESHOLD
                   ? StoreType::kAcceptedBitset
               : size_acc < size_rej ? StoreType::kAccepted
                                     : StoreType::kRejected;

  if (store_type == StoreType::kAcceptedBitset) {
    accepted_bitset = DynamicBitset(vocab_size);
    for (auto idx : accepted_indices) {
      accepted_bitset.Set(sorted_decoded_vocab[idx].first, true);
    }
  } else if (store_type == StoreType::kAccepted) {
    this->accepted_indices = accepted_indices;
  } else {
    this->rejected_indices = rejected_indices;
  }

  this->uncertain_indices = uncertain_indices;
}

AdaptiveTokenMask::AdaptiveTokenMask(
    size_t vocab_size,
    const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
    const std::vector<int32_t>& accepted_indices,
    const std::vector<int32_t>& uncertain_indices
) {
  auto size_acc = accepted_indices.size();

  store_type = size_acc >= USE_BITSET_THRESHOLD ? StoreType::kAcceptedBitset : StoreType::kAccepted;

  if (store_type == StoreType::kAcceptedBitset) {
    accepted_bitset = DynamicBitset(vocab_size);
    for (auto idx : accepted_indices) {
      accepted_bitset.Set(sorted_decoded_vocab[idx].first, true);
    }
  } else {
    XGRAMMAR_DCHECK(store_type == StoreType::kAccepted);
    this->accepted_indices = accepted_indices;
  }
  this->uncertain_indices = uncertain_indices;
}

std::string AdaptiveTokenMask::Print(const TokenizerInfo& tokenizer_info) const {
  constexpr int kMaxPrintTokens = 100;
  std::stringstream ss;
  const auto& sorted_decoded_vocab = tokenizer_info.GetSortedDecodedVocab();
  std::vector<int32_t> accepted_indices;
  std::vector<int32_t> rejected_indices;
  std::unordered_set<int32_t> uncertain_indices_set(
      uncertain_indices.begin(), uncertain_indices.end()
  );

  accepted_indices.reserve(sorted_decoded_vocab.size());
  rejected_indices.reserve(sorted_decoded_vocab.size());

  if (store_type == StoreType::kAcceptedBitset) {
    for (int i = 0; i < static_cast<int>(sorted_decoded_vocab.size()); ++i) {
      if (uncertain_indices_set.count(i)) {
        continue;
      }
      if (accepted_bitset[i]) {
        accepted_indices.push_back(i);
      } else {
        rejected_indices.push_back(i);
      }
    }
  } else if (store_type == StoreType::kAccepted) {
    accepted_indices = this->accepted_indices;
    // Reject indices = [0, sorted_decoded_vocab.size()) \ accepted_indices \ uncertain_indices
    int acc_ptr = 0;
    for (int i = 0; i < static_cast<int>(sorted_decoded_vocab.size()); ++i) {
      while (acc_ptr < static_cast<int>(accepted_indices.size()) && accepted_indices[acc_ptr] < i) {
        ++acc_ptr;
      }
      if (acc_ptr < static_cast<int>(accepted_indices.size()) && accepted_indices[acc_ptr] == i) {
        continue;
      }
      if (uncertain_indices_set.count(i)) {
        continue;
      }
      rejected_indices.push_back(i);
    }
  } else {
    XGRAMMAR_DCHECK(store_type == StoreType::kRejected);
    rejected_indices = this->rejected_indices;
    // Accepted indices = [0, sorted_decoded_vocab.size()) \ rejected_indices \ uncertain_indices
    int rej_ptr = 0;
    for (int i = 0; i < static_cast<int>(sorted_decoded_vocab.size()); ++i) {
      while (rej_ptr < static_cast<int>(rejected_indices.size()) && rejected_indices[rej_ptr] < i) {
        ++rej_ptr;
      }
      if (rej_ptr < static_cast<int>(rejected_indices.size()) && rejected_indices[rej_ptr] == i) {
        continue;
      }
      if (uncertain_indices_set.count(i)) {
        continue;
      }
      accepted_indices.push_back(i);
    }
  }

  std::string storage_type_str = store_type == StoreType::kAcceptedBitset ? "AcceptedBitset"
                                 : store_type == StoreType::kAccepted     ? "Accepted"
                                                                          : "Rejected";

  ss << "AdaptiveTokenMask(num_tokens=" << sorted_decoded_vocab.size()
     << ", accepted_num=" << accepted_indices.size() << ", rejected_num=" << rejected_indices.size()
     << ", uncertain_num=" << uncertain_indices.size() << ", storage_type=" << storage_type_str
     << ",\n";

  // Convert indices to token ids for printing
  std::vector<int32_t> accepted_token_ids;
  std::vector<int32_t> rejected_token_ids;
  std::vector<int32_t> uncertain_token_ids;
  accepted_token_ids.reserve(accepted_indices.size());
  rejected_token_ids.reserve(rejected_indices.size());
  uncertain_token_ids.reserve(uncertain_indices.size());

  for (auto idx : accepted_indices) {
    accepted_token_ids.push_back(sorted_decoded_vocab[idx].first);
  }
  std::sort(accepted_token_ids.begin(), accepted_token_ids.end());
  for (auto idx : rejected_indices) {
    rejected_token_ids.push_back(sorted_decoded_vocab[idx].first);
  }
  std::sort(rejected_token_ids.begin(), rejected_token_ids.end());
  for (auto idx : uncertain_indices) {
    uncertain_token_ids.push_back(sorted_decoded_vocab[idx].first);
  }
  std::sort(uncertain_token_ids.begin(), uncertain_token_ids.end());

  ss << "accepted=" << PrintTokenByIds(accepted_token_ids, tokenizer_info, kMaxPrintTokens)
     << ",\nrejected=" << PrintTokenByIds(rejected_token_ids, tokenizer_info, kMaxPrintTokens)
     << ",\nuncertain=" << PrintTokenByIds(uncertain_token_ids, tokenizer_info, kMaxPrintTokens)
     << "\n)";
  return ss.str();
}

Grammar CompiledGrammar::GetGrammar() const { return pimpl_->GetGrammar(); }

TokenizerInfo CompiledGrammar::GetTokenizerInfo() const { return pimpl_->GetTokenizerInfo(); }

/************** Use GrammarMatcher to generate the AdaptiveTokenMaskCache **************/

/*! \brief The concrete implementation of GrammarMatcherNode. */
class GrammarMatcherForTokenMaskCache : public EarleyParser {
 public:
  GrammarMatcherForTokenMaskCache(
      const Grammar& grammar, const ParserState& init_state, const bool& need_expand = true
  )
      : EarleyParser(grammar, init_state),
        init_rule_id(init_state.rule_id),
        initial_state(init_state) {}
  /*!
   * \brief Get the adaptive token mask for the given ParserState.
   * \param is_root_rule Whether to consider the parent rule. If false, there will be
   * no uncertain tokens. Useful for the root rule.
   */
  AdaptiveTokenMask GetAdaptiveTokenMask(
      size_t vocab_size,
      const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
      const std::vector<int32_t>& subtree_nodes_range,
      bool is_root_rule
  );

  /*!
   * \brief Get the token mask for the given ParserState.
   * \param sorted_decoded_vocab The sorted decoded vocabulary.
   * \param first_char_mask The first character mask.
   * \param is_root_rule Whether to consider the parent rule. If false, there will be
   * no uncertain tokens. Useful for the root rule.
   * \returns True if the rejected indices are filled as usual, False otherwise.
   * It's used to determine which construction function will be used.
   */
  bool GetTokenMaskWithFirstCharacterCheck(
      const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
      const std::bitset<256>& first_char_mask,
      const std::vector<int>& subtree_nodes_range,
      bool is_root_rule
  );

 private:
  /*! \brief Check if a token can pass the lookahead assertion. */
  bool IsTokenPassLookaheadAssertion(
      const std::string& token, const std::vector<bool>& can_reach_end_stack
  );

  /*! \brief Check if speculative calculation will be applied.*/
  bool IsSpeculativeCalculationApplied(
      const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
      int possible_token_num
  );

  // The id of the initial rule.
  int32_t init_rule_id;

  // The initial state of the parser.
  ParserState initial_state;

  // Temporary data for GetAdaptiveTokenMask.
  std::vector<int32_t> tmp_accepted_indices_;
  std::vector<int32_t> tmp_rejected_indices_;
  std::vector<int32_t> tmp_uncertain_indices_;
  std::vector<bool> tmp_can_reach_end_stack_;
  std::vector<bool> tmp_can_reach_end_prefix_or_stack_;
};

bool GrammarMatcherForTokenMaskCache::IsTokenPassLookaheadAssertion(
    const std::string& token, const std::vector<bool>& can_reach_end_stack
) {
  auto lookahead_assertion_id = grammar_->GetRule(init_rule_id).lookahead_assertion_id;
  if (lookahead_assertion_id == -1) {
    return true;
  }
  auto lookahead_state =
      ParserState(/*rule_id*/ -1, lookahead_assertion_id, 0, ParserState::kNoPrevInputPos, 0);
  PushStateAndExpand(lookahead_state);
  int token_len = token.size();

  // Find all positions that can come to and end. Then check if the suffix from that position
  // can be accepted by the lookahead assertion.
  for (int i = static_cast<int>(can_reach_end_stack.size()) - 1; i >= 0; --i) {
    if (!can_reach_end_stack[i]) {
      continue;
    }
    int last_accept_pos = i - 1;
    for (int pos = i; pos < token_len; ++pos) {
      if (!Advance(token[pos])) {
        break;
      }
      last_accept_pos = pos;
      // Case 1. The whole rule is finished.
      if (IsCompleted()) {
        // accepted chars: pos - i + 1
        // we need to rollback the pushed initial state as well
        PopLastStates(pos - i + 2);
        return true;
      }
    }
    // Case 2. The whole token is accepted
    if (last_accept_pos == token_len - 1) {
      PopLastStates(last_accept_pos - i + 2);
      return true;
    }
    // Case 3. The token is not accepted. Check the next position.
    PopLastStates(last_accept_pos - i + 1);
  }

  PopLastStates(1);
  return false;
}

// Comparator for std::pair<int32_t, std::string> based on the string value.
class IntStringPairComparator {
 public:
  bool operator()(
      const std::pair<int32_t, std::string>& lhs, const std::pair<int32_t, std::string>& rhs
  ) const {
    return lhs.second < rhs.second;
  }
};

int GetPossibleTokenIntervals(
    const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
    const std::bitset<256>& first_char_mask,
    std::vector<std::pair<int32_t, int32_t>>& possible_intervals
) {
  int possible_token_num = 0;
  int matched_size = 0;
  int last_interval_end = -1;
  for (int32_t i = 0; i < 256; i++) {
    if (first_char_mask[i]) {
      if (last_interval_end == -1) {
        last_interval_end = i;
      }
    } else {
      if (last_interval_end != -1) {
        int32_t interval_left_end =
            std::lower_bound(
                sorted_decoded_vocab.begin() + matched_size,
                sorted_decoded_vocab.end(),
                std::make_pair(0, std::string(1, static_cast<uint8_t>(last_interval_end))),
                IntStringPairComparator()
            ) -
            sorted_decoded_vocab.begin();
        int32_t interval_right_end = std::lower_bound(
                                         sorted_decoded_vocab.begin() + interval_left_end,
                                         sorted_decoded_vocab.end(),
                                         std::make_pair(0, std::string(1, static_cast<uint8_t>(i))),
                                         IntStringPairComparator()
                                     ) -
                                     sorted_decoded_vocab.begin();
        possible_intervals.emplace_back(interval_left_end, interval_right_end);
        possible_token_num += interval_right_end - interval_left_end;
        last_interval_end = -1;
        matched_size = interval_right_end;
      }
    }
  }

  if (last_interval_end != -1) {
    // If the last interval is not closed, we need to close it.
    int32_t interval_left_end =
        std::lower_bound(
            sorted_decoded_vocab.begin() + matched_size,
            sorted_decoded_vocab.end(),
            std::make_pair(0, std::string(1, static_cast<uint8_t>(last_interval_end))),
            IntStringPairComparator()
        ) -
        sorted_decoded_vocab.begin();
    possible_intervals.emplace_back(interval_left_end, sorted_decoded_vocab.size());
    possible_token_num += sorted_decoded_vocab.size() - interval_left_end;
  }
  return possible_token_num;
}

bool GrammarMatcherForTokenMaskCache::IsSpeculativeCalculationApplied(
    const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab, int possible_token_num
) {
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  // Check if the initial state is self-recursive-like. If the state is self-recursive-like,
  // and it covers a large part of the vocabulary, we will do speculative calculation in compiling.
  if (initial_state.sub_element_id == 0 &&
      possible_token_num > static_cast<int>(sorted_decoded_vocab.size() / 4)) {
    const auto& sequence_expr = grammar_->GetGrammarExpr(initial_state.sequence_id);
    // A self-recursive-like rule must be a sequence.
    if (sequence_expr.type == GrammarExprType::kSequence) {
      const auto& current_element_expr =
          grammar_->GetGrammarExpr(sequence_expr[initial_state.element_id]);
      // If the current element is a character class star, then it's self-recursive without doubt.
      if (current_element_expr.type == GrammarExprType::kCharacterClassStar) {
        return true;
        // If the current element is a character class, and the next element is a rule ref to
        // itself, and the rule only has 2 elements, then it's self-recursive-like.
      } else if (current_element_expr.type == GrammarExprType::kCharacterClass &&
                 sequence_expr.size() == 2 && initial_state.element_id == 0) {
        const auto& end_element_expr = grammar_->GetGrammarExpr(sequence_expr[1]);
        if (end_element_expr.type == GrammarExprType::kRuleRef &&
            end_element_expr[0] == initial_state.rule_id) {
          return true;
        }
      }
    }
  }
  return false;
}

bool GrammarMatcherForTokenMaskCache::GetTokenMaskWithFirstCharacterCheck(
    const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
    const std::bitset<256>& first_char_mask,
    const std::vector<int>& subtree_nodes_range,
    bool is_root_rule
) {
  // the pair (a, b) means [a, b). Intialize the possible intervals.
  std::vector<std::pair<int32_t, int32_t>> possible_intervals;
  int possible_token_num =
      GetPossibleTokenIntervals(sorted_decoded_vocab, first_char_mask, possible_intervals);

  // Check if the type of the mask can be krejected.
  bool fill_reject_indices =
      (sorted_decoded_vocab.size() - possible_token_num) < AdaptiveTokenMask::USE_BITSET_THRESHOLD;

  XGRAMMAR_DCHECK(possible_intervals.size() > 0)
      << "There should be at least one possible interval for the first character mask.";

  if (possible_intervals[0].first != 0 && fill_reject_indices) {
    for (int i = 0; i < possible_intervals[0].first; ++i) {
      tmp_rejected_indices_.push_back(i);
    }
  }

  bool speculative_calculation =
      IsSpeculativeCalculationApplied(sorted_decoded_vocab, possible_token_num);

  int prev_matched_size = 0;
  int last_rejected_range = 0;
  const std::string* prev_token = nullptr;
  for (size_t interval_idx = 0; interval_idx < possible_intervals.size(); ++interval_idx) {
    const auto& interval = possible_intervals[interval_idx];
    for (int i = interval.first; i < interval.second; ++i) {
      // Check if the current token is in the rejected range. i.e. check if the current token
      // is on the subtree of the rejected token.
      if (i < last_rejected_range) {
        if (fill_reject_indices) {
          tmp_rejected_indices_.push_back(i);
          fill_reject_indices =
              tmp_rejected_indices_.size() < AdaptiveTokenMask::USE_BITSET_THRESHOLD;
        } else {
          i = last_rejected_range - 1;
        }
        continue;
      }

      const auto& token = sorted_decoded_vocab[i].second;
      // This optimization is useful for simple self-recursive rules, like string content.
      if (speculative_calculation) {
        bool all_accepted = true;
        for (char ch : token) {
          // If the first character is not the ascii character or can't be accepted by the
          // first character mask, we need to check them in the parser.
          if (isascii(ch) == 0 || !first_char_mask[static_cast<uint8_t>(ch)]) {
            all_accepted = false;
            break;
          }
        }
        if (all_accepted) {
          tmp_accepted_indices_.push_back(i);
          continue;
        }
      }
      // Many tokens may contain the same prefix, so we will avoid unnecessary matching
      // by finding the longest common prefix with the previous token.
      bool accepted = true;
      if (prev_token != nullptr) {
        int lcp_len =
            std::mismatch(token.begin(), token.end(), prev_token->begin(), prev_token->end())
                .first -
            token.begin();
        if (lcp_len > prev_matched_size) {
          // Case 1. The common prefix is rejected by the matcher in the last token. Reject
          // directly.
          accepted = false;
        } else if (lcp_len < prev_matched_size) {
          // Case 2. The common prefix is shorter than the previous matched size. Rollback
          // the non-common part.
          PopLastStates(prev_matched_size - lcp_len);
          tmp_can_reach_end_stack_.erase(
              tmp_can_reach_end_stack_.end() - (prev_matched_size - lcp_len),
              tmp_can_reach_end_stack_.end()
          );
          tmp_can_reach_end_prefix_or_stack_.erase(
              tmp_can_reach_end_prefix_or_stack_.end() - (prev_matched_size - lcp_len),
              tmp_can_reach_end_prefix_or_stack_.end()
          );
        }
        prev_matched_size = std::min(prev_matched_size, lcp_len);
      }

      prev_token = &token;

      if (accepted) {
        // Accept the rest chars one by one.
        for (int j = prev_matched_size; j < static_cast<int>(token.size()); ++j) {
          if (!Advance(token[j])) {
            accepted = false;
            break;
          }
          tmp_can_reach_end_stack_.push_back(IsCompleted());
          tmp_can_reach_end_prefix_or_stack_.push_back(
              tmp_can_reach_end_stack_.back() || tmp_can_reach_end_prefix_or_stack_.back()
          );
          prev_matched_size = j + 1;
        }
      }

      bool can_reach_end = tmp_can_reach_end_prefix_or_stack_.back();

      if (accepted) {
        tmp_accepted_indices_.push_back(i);
      } else if (can_reach_end && !is_root_rule &&
                 IsTokenPassLookaheadAssertion(token, tmp_can_reach_end_stack_) &&
                 prev_matched_size > 0) {
        // 1. If the current rule is the root rule (is_root_rule=true), there are no
        // uncertain tokens. Not accepted tokens are just rejected.
        // 2. If a token cannot pass the lookahead assertion, it is rejected.
        tmp_uncertain_indices_.push_back(i);
      } else {
        tmp_rejected_indices_.push_back(i);
        last_rejected_range = subtree_nodes_range[i];
        fill_reject_indices =
            tmp_rejected_indices_.size() < AdaptiveTokenMask::USE_BITSET_THRESHOLD;
      }
    }
    if (interval_idx != possible_intervals.size() - 1 && fill_reject_indices) {
      const auto& next_interval = possible_intervals[interval_idx + 1];
      for (int i = interval.second; i < next_interval.first; ++i) {
        tmp_rejected_indices_.push_back(i);
      }
      fill_reject_indices = tmp_rejected_indices_.size() < AdaptiveTokenMask::USE_BITSET_THRESHOLD;
    }
  }

  // Rollback the last matched part.
  PopLastStates(prev_matched_size);

  if (possible_intervals.back().second != static_cast<int>(sorted_decoded_vocab.size()) &&
      fill_reject_indices) {
    // If the last interval is not closed, we need to reject the rest tokens.
    for (int i = possible_intervals.back().second;
         i < static_cast<int>(sorted_decoded_vocab.size());
         ++i) {
      tmp_rejected_indices_.push_back(i);
    }
  }

  return fill_reject_indices;
}

AdaptiveTokenMask GrammarMatcherForTokenMaskCache::GetAdaptiveTokenMask(
    size_t vocab_size,
    const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
    const std::vector<int32_t>& subtree_nodes_range,
    bool is_root_rule
) {
  tmp_accepted_indices_.clear();
  tmp_rejected_indices_.clear();
  tmp_uncertain_indices_.clear();
  // For every character in the current token, stores whether it is possible to reach the end of
  // the rule when matching until this character. Store it in a stack for later rollback.
  tmp_can_reach_end_stack_.assign({IsCompleted()});
  tmp_can_reach_end_prefix_or_stack_.assign({tmp_can_reach_end_stack_.back()});
  std::bitset<256> first_character_mask;
  const auto& sequence = grammar_->GetGrammarExpr(initial_state.sequence_id);
  if (sequence.type == Grammar::Impl::GrammarExprType::kSequence) {
    const auto& sub_sequence = grammar_->GetGrammarExpr(sequence[initial_state.element_id]);
    switch (sub_sequence.type) {
      case Grammar::Impl::GrammarExprType::kByteString: {
        first_character_mask[sub_sequence[initial_state.sub_element_id]] = true;
        break;
      }
      case xgrammar::Grammar::Impl::GrammarExprType::kCharacterClass:
      case xgrammar::Grammar::Impl::GrammarExprType::kCharacterClassStar: {
        if (initial_state.sub_element_id == 0) {
          bool is_negative = sub_sequence[0];
          for (int i = 1; i < sub_sequence.size(); i += 2) {
            int left_char = static_cast<uint8_t>(sub_sequence[i]);
            int right_char = static_cast<uint8_t>(sub_sequence[i + 1]);
            for (int c = left_char; c <= right_char; ++c) {
              first_character_mask[c] = true;
            }
          }
          if (is_negative) {
            first_character_mask = ~first_character_mask;
          }
          break;
        }
        // Otherwise, it's matching a UTF-8 character. We can optimize the matching process
        // here.
        for (size_t i = 0x80; i < 0xC0; ++i) {
          first_character_mask[i] = true;
        }
        break;
      }
      default: {
        XGRAMMAR_LOG(FATAL) << "Unsupported grammar expr type: " << static_cast<int>(sequence.type);
      }
    }
  } else {
    XGRAMMAR_DCHECK(sequence.type == Grammar::Impl::GrammarExprType::kTagDispatch);
    first_character_mask.set();
  }
  bool rejected_indices_are_filled = GetTokenMaskWithFirstCharacterCheck(
      sorted_decoded_vocab, first_character_mask, subtree_nodes_range, is_root_rule
  );
  if (rejected_indices_are_filled) {
    return AdaptiveTokenMask(
        vocab_size,
        sorted_decoded_vocab,
        tmp_accepted_indices_,
        tmp_rejected_indices_,
        tmp_uncertain_indices_
    );
  } else {
    return AdaptiveTokenMask(
        vocab_size, sorted_decoded_vocab, tmp_accepted_indices_, tmp_uncertain_indices_
    );
  }
}

/******************* GrammarCompiler::Impl *******************/

using SchemaKey =
    std::tuple<std::string, bool, std::optional<int>, std::pair<std::string, std::string>, bool>;
using StructuralTagKey = std::tuple<std::vector<StructuralTagItem>, std::vector<std::string>>;
using GrammarKey = std::pair<std::string, std::string>;

class GrammarCompiler::Impl {
 public:
  Impl(
      const TokenizerInfo& tokenizer_info,
      int max_threads,
      bool cache_enabled,
      long long max_memory_bytes
  )
      : tokenizer_info_(tokenizer_info),
        max_threads_(max_threads),
        cache_enabled_(cache_enabled),
        compile_builtin_json_grammar_cache_([&] { return CompileJson(); }),
        compile_cache_(static_cast<std::size_t>(max_memory_bytes), *this) {}

  /*!
   * \brief Build the fsm for each rule and store in the grammar.
   */
  void BuildFSM(Grammar grammar);

  /*! \brief Multi-thread compile the grammar. */
  CompiledGrammar MultiThreadCompileGrammar(Grammar grammar);

  /*! \brief Compile the built-in JSON grammar. */
  CompiledGrammar CompileJson();

  /*!
   * \brief Compile different types of grammars.
   * \attention This template function is marked as deleted.
   * User must explicitly specialize the template to support new key types.
   */
  template <typename KeyType>
  CompiledGrammar Compute(const KeyType& key) = delete;

  /*! \brief Forwards the key to the corresponding compile function. */
  template <typename KeyType>
  CompiledGrammar operator()(const KeyType& key) {
    return Compute<KeyType>(key);
  }

  CompiledGrammar CompileBuiltinJSONGrammar();

  CompiledGrammar CompileJSONSchema(
      const std::string& schema,
      bool any_whitespace,
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool strict_mode = true
  );

  CompiledGrammar CompileStructuralTag(
      const std::vector<StructuralTagItem>& tags, const std::vector<std::string>& triggers
  );

  CompiledGrammar CompileRegex(const std::string& regex);

  CompiledGrammar CompileGrammar(const Grammar& grammar);

  void ClearCache();

  long long GetCacheSizeBytes() const;
  long long CacheLimitBytes() const;

 private:
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  using MultipleKey = std::variant<SchemaKey, StructuralTagKey, std::string, GrammarKey>;

  struct Computer {
    Computer(Impl& compiler) : compiler(compiler) {}
    // dispatch the key to the corresponding compile function
    CompiledGrammar operator()(const MultipleKey& key) const { return std::visit(compiler, key); }
    GrammarCompiler::Impl& compiler;
  };

  struct SizeEstimator {
    std::size_t operator()(const CompiledGrammar& value) const { return value.MemorySizeBytes(); }
  };

  /*! \brief The vocabulary associated with this storage class. */
  const TokenizerInfo tokenizer_info_;
  /*! \brief The maximum number of threads to use. */
  const int max_threads_;
  /*! \brief Whether the cache is enabled. */
  const bool cache_enabled_;

  ThreadSafeCache<CompiledGrammar> compile_builtin_json_grammar_cache_;
  ThreadSafeLRUCache<MultipleKey, CompiledGrammar, Computer, SizeEstimator> compile_cache_;
};

CompiledGrammar GrammarCompiler::Impl::MultiThreadCompileGrammar(Grammar grammar) {
  auto compiled_grammar_impl = std::make_shared<CompiledGrammar::Impl>();

  compiled_grammar_impl->grammar = grammar;
  compiled_grammar_impl->tokenizer_info = tokenizer_info_;

  // Step 1. Compute the ids of rules that can be empty
  compiled_grammar_impl->grammar->allow_empty_rule_ids = AllowEmptyRuleAnalyzer::Apply(grammar);

  // Step 2. Build the fsm for each rule
  GrammarFSMBuilder::Apply(&compiled_grammar_impl->grammar);

  if (tokenizer_info_.GetVocabSize() == 0) {
    return CompiledGrammar(compiled_grammar_impl);
  }

  // Step 3. Compute the adaptive token mask cache
  // The token mask cache is computed for these positions in the grammar:
  // 1. All character class or character class star (with last_utf8_bytes=0, 1, 2, 3)
  // 2. All byte strings (with element_in_string=0, 1, 2, ...)
  // since other positions will be expanded to the above positions

  // TODO(Charlie): Figure out how to support ThreadPool and std::mutex in WebAssembly.
  // Only declare ThreadPool and mutex if max_threads > 1, so when max_threads = 1, we do
  // not need ThreadPool or std::mutex, which throws error in runtime in WebAssembly.
  std::optional<ThreadPool> thread_pool;
  std::optional<std::mutex> adaptive_token_mask_cache_mutex;

  if (max_threads_ > 1) {
    thread_pool.emplace(max_threads_);
    adaptive_token_mask_cache_mutex.emplace();
  }

  auto add_adaptive_token_mask = [&](const ParserState& state, bool is_root_rule) {
    auto grammar_matcher = GrammarMatcherForTokenMaskCache(grammar, state, false);
    auto cur_adaptive_token_mask_cache = grammar_matcher.GetAdaptiveTokenMask(
        tokenizer_info_.GetVocabSize(),
        tokenizer_info_.GetSortedDecodedVocab(),
        tokenizer_info_.GetTrieSubtreeNodesRange(),
        is_root_rule
    );
    if (max_threads_ > 1) {
      std::lock_guard<std::mutex> lock(adaptive_token_mask_cache_mutex.value());
      compiled_grammar_impl->adaptive_token_mask_cache[state] = cur_adaptive_token_mask_cache;
    } else {
      compiled_grammar_impl->adaptive_token_mask_cache[state] = cur_adaptive_token_mask_cache;
    }
  };

  auto add_task_adaptive_token_mask = [&](const ParserState& state, bool is_root_rule) {
    // Execute depending on whether we use thread_pool
    if (max_threads_ > 1) {
      thread_pool->Execute([add_adaptive_token_mask, state, is_root_rule]() {
        add_adaptive_token_mask(state, is_root_rule);
      });
    } else {
      add_adaptive_token_mask(state, is_root_rule);
    }
  };

  auto root_rule_id = grammar->GetRootRuleId();

  for (int32_t rule_id = 0; rule_id < static_cast<int>(grammar->NumRules()); ++rule_id) {
    auto rule = grammar->GetRule(rule_id);
    auto rule_body = grammar->GetGrammarExpr(rule.body_expr_id);
    const auto& rule_fsm = grammar->per_rule_fsms[rule_id];

    if (rule_fsm.has_value()) {
      auto cur_stack_element =
          ParserState(rule_id, rule.body_expr_id, 0, ParserState::kNoPrevInputPos, 0);
      std::unordered_set<int> reachable_states;
      rule_fsm->GetReachableStates(&reachable_states);
      for (int i : reachable_states) {
        cur_stack_element.element_id = i;
        add_task_adaptive_token_mask(cur_stack_element, rule_id == root_rule_id);
      }
      continue;
    }

    XGRAMMAR_DCHECK(rule_body.type == GrammarExprType::kChoices);
    for (auto sequence_id : rule_body) {
      const auto& sequence = grammar->GetGrammarExpr(sequence_id);
      if (sequence.type == GrammarExprType::kEmptyStr) {
        continue;
      }
      XGRAMMAR_DCHECK(sequence.type == GrammarExprType::kSequence);
      auto state = ParserState(rule_id, sequence_id, 0, ParserState::kNoPrevInputPos, 0);
      for (int element_id = 0; element_id < sequence.size(); ++element_id) {
        state.element_id = element_id;
        auto element = grammar->GetGrammarExpr(sequence[element_id]);
        if (element.type == GrammarExprType::kRuleRef) {
          continue;
        }
        if (element.type == GrammarExprType::kByteString) {
          for (int idx = 0; idx < element.size(); ++idx) {
            state.sub_element_id = idx;
            add_task_adaptive_token_mask(state, rule_id == root_rule_id);
          }
        } else {
          XGRAMMAR_DCHECK(
              element.type == GrammarExprType::kCharacterClassStar ||
              element.type == GrammarExprType::kCharacterClass
          );
          for (int left_utf8_bytes = 0; left_utf8_bytes <= 3; ++left_utf8_bytes) {
            state.sub_element_id = left_utf8_bytes;
            add_task_adaptive_token_mask(state, rule_id == root_rule_id);
          }
        }
      }
    }
  }

  if (max_threads_ > 1) {
    thread_pool->Join();
  }

  return CompiledGrammar(compiled_grammar_impl);
}

CompiledGrammar GrammarCompiler::Impl::CompileJson() {
  return MultiThreadCompileGrammar(Grammar::BuiltinJSONGrammar());
}

template <>
CompiledGrammar GrammarCompiler::Impl::Compute<SchemaKey>(const SchemaKey& key) {
  const auto& [schema, any_whitespace, indent, separators, strict_mode] = key;
  auto grammar = Grammar::FromJSONSchema(schema, any_whitespace, indent, separators, strict_mode);
  return MultiThreadCompileGrammar(grammar);
}

template <>
CompiledGrammar GrammarCompiler::Impl::Compute<StructuralTagKey>(const StructuralTagKey& key) {
  const auto& [tags, triggers] = key;
  return MultiThreadCompileGrammar(Grammar::FromStructuralTag(tags, triggers));
}

template <>
CompiledGrammar GrammarCompiler::Impl::Compute<std::string>(const std::string& key) {
  return MultiThreadCompileGrammar(Grammar::FromRegex(key));
}

template <>
CompiledGrammar GrammarCompiler::Impl::Compute<GrammarKey>(const GrammarKey& key) {
  const auto& [grammar_str, root_rule_name] = key;
  return MultiThreadCompileGrammar(Grammar::FromEBNF(grammar_str, root_rule_name));
}

CompiledGrammar GrammarCompiler::Impl::CompileBuiltinJSONGrammar() {
  if (!cache_enabled_) {
    return MultiThreadCompileGrammar(Grammar::BuiltinJSONGrammar());
  }
  return compile_builtin_json_grammar_cache_.Get();
}

CompiledGrammar GrammarCompiler::Impl::CompileJSONSchema(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode
) {
  if (!cache_enabled_) {
    return MultiThreadCompileGrammar(
        Grammar::FromJSONSchema(schema, any_whitespace, indent, separators, strict_mode)
    );
  }
  auto separators_value = separators.value_or(
      (indent == std::nullopt) ? std::make_pair(", ", ": ") : std::make_pair(",", ": ")
  );
  auto key = std::make_tuple(schema, any_whitespace, indent, separators_value, strict_mode);
  return compile_cache_.Get(key);
}

CompiledGrammar GrammarCompiler::Impl::CompileStructuralTag(
    const std::vector<StructuralTagItem>& tags, const std::vector<std::string>& triggers
) {
  if (!cache_enabled_) {
    return MultiThreadCompileGrammar(Grammar::FromStructuralTag(tags, triggers));
  }
  auto key = std::make_tuple(tags, triggers);
  return compile_cache_.Get(key);
}

CompiledGrammar GrammarCompiler::Impl::CompileRegex(const std::string& regex) {
  if (!cache_enabled_) {
    return MultiThreadCompileGrammar(Grammar::FromRegex(regex));
  }
  return compile_cache_.Get(regex);
}

CompiledGrammar GrammarCompiler::Impl::CompileGrammar(const Grammar& grammar) {
  if (!cache_enabled_) {
    return MultiThreadCompileGrammar(grammar);
  }
  auto key = std::make_pair(grammar.ToString(), grammar->GetRootRule().name);
  return compile_cache_.Get(key);
}

void GrammarCompiler::Impl::ClearCache() {
  compile_builtin_json_grammar_cache_.Clear();
  compile_cache_.Clear();
}

long long GrammarCompiler::Impl::GetCacheSizeBytes() const {
  return static_cast<long long>(compile_cache_.MemorySize());
}

long long GrammarCompiler::Impl::CacheLimitBytes() const {
  const auto size = compile_cache_.MaxMemorySize();
  if (size == compile_cache_.UNLIMITED_SIZE) return -1;
  return static_cast<long long>(size);
}

/******************* GrammarCompiler *******************/

GrammarCompiler::GrammarCompiler(
    const TokenizerInfo& tokenizer_info,
    int max_threads,
    bool cache_enabled,
    long long max_memory_bytes
)
    : pimpl_(std::make_shared<Impl>(tokenizer_info, max_threads, cache_enabled, max_memory_bytes)) {
  if (max_memory_bytes < -1) {
    XGRAMMAR_LOG(FATAL) << "Invalid max_memory_bytes: " << max_memory_bytes << ". "
                        << "It should be -1 (unlimited) or a non-negative integer.";
  }
}

CompiledGrammar GrammarCompiler::CompileJSONSchema(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode
) {
  return pimpl_->CompileJSONSchema(schema, any_whitespace, indent, separators, strict_mode);
}

CompiledGrammar GrammarCompiler::CompileBuiltinJSONGrammar() {
  return pimpl_->CompileBuiltinJSONGrammar();
}

CompiledGrammar GrammarCompiler::CompileStructuralTag(
    const std::vector<StructuralTagItem>& tags, const std::vector<std::string>& triggers
) {
  return pimpl_->CompileStructuralTag(tags, triggers);
}

CompiledGrammar GrammarCompiler::CompileRegex(const std::string& regex) {
  return pimpl_->CompileRegex(regex);
}

CompiledGrammar GrammarCompiler::CompileGrammar(const Grammar& grammar) {
  return pimpl_->CompileGrammar(grammar);
}

void GrammarCompiler::ClearCache() { pimpl_->ClearCache(); }

long long GrammarCompiler::GetCacheSizeBytes() const { return pimpl_->GetCacheSizeBytes(); }

long long GrammarCompiler::CacheLimitBytes() const { return pimpl_->CacheLimitBytes(); }

}  // namespace xgrammar
