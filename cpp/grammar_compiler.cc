/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/compiler.cc
 */

#include <xgrammar/compiler.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "character_class_token_summary.h"
#include "compiled_grammar_impl.h"
#include "earley_parser.h"
#include "fsm.h"
#include "grammar_functor.h"
#include "grammar_impl.h"
#include "json_schema_converter.h"
#include "structural_tag.h"
#include "support/dynamic_bitset.h"
#include "support/int_set.h"
#include "support/logging.h"
#include "support/thread_pool.h"
#include "support/thread_safe_cache.h"
#include "support/utils.h"
#include "tokenizer_info_impl.h"
#include "xgrammar/grammar.h"
#include "xgrammar/tokenizer_info.h"

namespace xgrammar {

/************** AdaptiveTokenMaskCache Generator **************/

std::vector<uint8_t> GetRuleLevelCacheableRules(const Grammar& grammar) {
  const int32_t num_rules = grammar->NumRules();
  std::vector<uint8_t> context_dependent(num_rules, 0);
  std::vector<uint8_t> can_enter_json_string_length_rule(num_rules, 0);
  std::vector<std::vector<int32_t>> referenced_rules(num_rules);
  std::vector<std::vector<int32_t>> referencing_rules(num_rules);
  std::vector<uint32_t> state_epochs(grammar->complete_fsm.NumStates(), 0);
  uint32_t state_epoch = 0;
  std::vector<int32_t> reachable_states;
  for (int32_t rule_id = 0; rule_id < num_rules; ++rule_id) {
    const auto& rule = grammar->GetRule(rule_id);
    context_dependent[rule_id] = rule.max_tokens >= 0 || rule.max_chars >= 0 || rule.is_lazy ||
                                 rule.temperature.has_value() ||
                                 grammar->GetSuffixStopInfo(rule_id) != nullptr;
    can_enter_json_string_length_rule[rule_id] = rule.json_string_min_length >= 0;
    const auto& fsm = grammar->per_rule_fsms[rule_id]->GetFsm();
    ++state_epoch;
    if (state_epoch == 0) {
      std::fill(state_epochs.begin(), state_epochs.end(), 0);
      ++state_epoch;
    }
    reachable_states.clear();
    reachable_states.push_back(fsm.GetStart());
    state_epochs[fsm.GetStart()] = state_epoch;
    for (size_t state_index = 0; state_index < reachable_states.size(); ++state_index) {
      int32_t state_id = reachable_states[state_index];
      for (const auto& edge : fsm.GetFsm().GetEdges(state_id)) {
        if (edge.IsRuleRef()) {
          referenced_rules[rule_id].push_back(edge.GetRefRuleId());
          referencing_rules[edge.GetRefRuleId()].push_back(rule_id);
        } else if (edge.IsRepeatRef()) {
          const int32_t referenced_rule_id =
              grammar->complete_fsm.GetRepeatEdgeInfo(edge.GetAuxIndex()).RuleId();
          referenced_rules[rule_id].push_back(referenced_rule_id);
          referencing_rules[referenced_rule_id].push_back(rule_id);
        }
        if (state_epochs[edge.target] != state_epoch) {
          state_epochs[edge.target] = state_epoch;
          reachable_states.push_back(edge.target);
        }
      }
    }
  }

  // Propagate context dependence through rule references once, instead of rescanning the complete
  // graph until a fixed point is reached.
  std::vector<int32_t> worklist;
  worklist.reserve(num_rules);
  for (int32_t rule_id = 0; rule_id < num_rules; ++rule_id) {
    if (context_dependent[rule_id]) {
      worklist.push_back(rule_id);
    }
  }
  for (size_t work_index = 0; work_index < worklist.size(); ++work_index) {
    for (int32_t referenced_rule_id : referenced_rules[worklist[work_index]]) {
      if (!context_dependent[referenced_rule_id]) {
        context_dependent[referenced_rule_id] = 1;
        worklist.push_back(referenced_rule_id);
      }
    }
  }
  // A mask built inside a JSON-length rule is length-independent because the cache parser disables
  // enforcement and the runtime filters its accepted tokens against the current deadline. A rule
  // that can *enter* such a rule is different: a token may cross the boundary while its current
  // state has no deadline, so that caller cannot reuse a mask whose grammar lacked the annotation.
  worklist.clear();
  for (int32_t rule_id = 0; rule_id < num_rules; ++rule_id) {
    if (can_enter_json_string_length_rule[rule_id]) {
      worklist.push_back(rule_id);
    }
  }
  for (size_t work_index = 0; work_index < worklist.size(); ++work_index) {
    for (int32_t referencing_rule_id : referencing_rules[worklist[work_index]]) {
      if (!can_enter_json_string_length_rule[referencing_rule_id]) {
        can_enter_json_string_length_rule[referencing_rule_id] = 1;
        worklist.push_back(referencing_rule_id);
      }
    }
  }
  for (int32_t rule_id = 0; rule_id < num_rules; ++rule_id) {
    context_dependent[rule_id] =
        !(context_dependent[rule_id] || can_enter_json_string_length_rule[rule_id]);
  }
  return context_dependent;
}

namespace {

bool IsPotentiallyValidUTF8Prefix(std::string_view value) {
  for (size_t offset = 0; offset < value.size();) {
    const uint8_t first = static_cast<uint8_t>(value[offset]);
    int32_t total_bytes;
    uint8_t second_min = 0x80;
    uint8_t second_max = 0xbf;
    if (first <= 0x7f) {
      ++offset;
      continue;
    } else if (first >= 0xc2 && first <= 0xdf) {
      total_bytes = 2;
    } else if (first >= 0xe0 && first <= 0xef) {
      total_bytes = 3;
      second_min = first == 0xe0 ? 0xa0 : 0x80;
      second_max = first == 0xed ? 0x9f : 0xbf;
    } else if (first >= 0xf0 && first <= 0xf4) {
      total_bytes = 4;
      second_min = first == 0xf0 ? 0x90 : 0x80;
      second_max = first == 0xf4 ? 0x8f : 0xbf;
    } else {
      return false;
    }
    for (int32_t continuation = 1; continuation < total_bytes; ++continuation) {
      if (offset + continuation == value.size()) {
        return true;
      }
      const uint8_t byte = static_cast<uint8_t>(value[offset + continuation]);
      if (continuation == 1) {
        if (byte < second_min || byte > second_max) {
          return false;
        }
      } else if (byte < 0x80 || byte > 0xbf) {
        return false;
      }
    }
    offset += total_bytes;
  }
  return true;
}

std::vector<uint8_t> GetRedundantLookaheadRules(const Grammar& grammar) {
  struct LookaheadInfo {
    bool disqualified{false};
    int32_t non_last_occurrences{0};
    std::vector<int32_t> suffix;
  };
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  std::vector<LookaheadInfo> infos(grammar->NumRules());
  for (int32_t parent_rule_id = 0; parent_rule_id < grammar->NumRules(); ++parent_rule_id) {
    const auto body = grammar->GetGrammarExpr(grammar->GetRule(parent_rule_id).body_expr_id);
    if (body.type == GrammarExprType::kTagDispatch) {
      for (const auto& [tag, rule_id] : grammar->GetTagDispatch(body).tag_rule_pairs) {
        infos[rule_id].disqualified = true;
      }
      continue;
    }
    if (body.type == GrammarExprType::kTokenTagDispatch) {
      for (const auto& [token_id, rule_id] :
           grammar->GetTokenTagDispatch(body).trigger_rule_pairs) {
        infos[rule_id].disqualified = true;
      }
      continue;
    }
    if (body.type != GrammarExprType::kChoices) {
      continue;
    }
    for (int32_t sequence_id : body) {
      const auto sequence = grammar->GetGrammarExpr(sequence_id);
      if (sequence.type != GrammarExprType::kSequence) {
        continue;
      }
      for (int32_t position = 0; position < sequence.size(); ++position) {
        const auto element = grammar->GetGrammarExpr(sequence[position]);
        int32_t referenced_rule_id = -1;
        if (element.type == GrammarExprType::kRuleRef) {
          referenced_rule_id = element[0];
        } else if (element.type == GrammarExprType::kRepeat) {
          if (element[2] == 0) {
            continue;
          }
          referenced_rule_id = element[0];
          if (element[2] != 1) {
            infos[referenced_rule_id].disqualified = true;
            continue;
          }
        } else {
          continue;
        }
        auto& info = infos[referenced_rule_id];
        if (position + 1 == sequence.size()) {
          if (parent_rule_id != referenced_rule_id) {
            info.disqualified = true;
          }
          continue;
        }
        if (info.non_last_occurrences++ == 0) {
          info.suffix.assign(sequence.begin() + position + 1, sequence.end());
        }
      }
    }
  }

  std::vector<uint8_t> result(grammar->NumRules(), false);
  for (int32_t rule_id = 0; rule_id < grammar->NumRules(); ++rule_id) {
    const auto& rule = grammar->GetRule(rule_id);
    if (rule.lookahead_assertion_id == -1) {
      result[rule_id] = true;
      continue;
    }
    const auto lookahead = grammar->GetGrammarExpr(rule.lookahead_assertion_id);
    const auto& info = infos[rule_id];
    result[rule_id] = rule.is_exact_lookahead && !info.disqualified &&
                      info.non_last_occurrences == 1 &&
                      lookahead.type == GrammarExprType::kSequence &&
                      lookahead.size() == static_cast<int32_t>(info.suffix.size()) &&
                      std::equal(lookahead.begin(), lookahead.end(), info.suffix.begin());
  }
  return result;
}

std::vector<uint8_t> GetPureFSMSemanticsRules(const Grammar& grammar) {
  const auto redundant_lookahead = GetRedundantLookaheadRules(grammar);
  std::vector<uint8_t> result(grammar->NumRules(), true);
  std::vector<std::vector<int32_t>> referencing_rules(grammar->NumRules());
  std::vector<int32_t> impure_rules;
  for (int32_t rule_id = 0; rule_id < grammar->NumRules(); ++rule_id) {
    const auto& rule = grammar->GetRule(rule_id);
    result[rule_id] = rule.max_tokens == -1 && rule.max_chars == -1 &&
                      rule.json_string_min_length == -1 && rule.json_string_max_length == -1 &&
                      !rule.is_lazy && !rule.temperature.has_value() && rule.capture_name.empty() &&
                      grammar->GetSuffixStopInfo(rule_id) == nullptr &&
                      grammar->per_rule_fsms[rule_id].has_value();
    if (!result[rule_id]) {
      impure_rules.push_back(rule_id);
    }
    if (!grammar->per_rule_fsms[rule_id].has_value()) {
      continue;
    }
    const auto& local_fsm = grammar->per_rule_fsms[rule_id]->GetFsm();
    std::unordered_set<int32_t> reachable_states;
    local_fsm.GetReachableStates(&reachable_states);
    for (int32_t state : reachable_states) {
      for (const auto& edge : local_fsm.GetFsm().GetEdges(state)) {
        int32_t referenced_rule_id = -1;
        if (edge.IsRuleRef()) {
          referenced_rule_id = edge.GetRefRuleId();
        } else if (edge.IsRepeatRef()) {
          referenced_rule_id = grammar->complete_fsm.GetRepeatEdgeInfo(edge.GetAuxIndex()).RuleId();
        }
        if (referenced_rule_id >= 0) {
          referencing_rules[referenced_rule_id].push_back(rule_id);
        }
      }
    }
  }
  // Lookahead assertions are only enforced during compile-time mask adaptation, never when the
  // online parser advances bytes. A candidate rule's own non-redundant lookahead is instead
  // enforced by filtering the crossing tokens when its local-completion summary is built. A
  // non-redundant lookahead in a referenced rule would still be skipped with the local prefix,
  // so conservatively invalidate each of its callers. Self-recursive tail calls complete at the
  // same boundary and do not introduce another lookahead position.
  for (int32_t rule_id = 0; rule_id < grammar->NumRules(); ++rule_id) {
    if (redundant_lookahead[rule_id]) {
      continue;
    }
    for (int32_t referencing_rule_id : referencing_rules[rule_id]) {
      if (referencing_rule_id != rule_id && result[referencing_rule_id]) {
        result[referencing_rule_id] = false;
        impure_rules.push_back(referencing_rule_id);
      }
    }
  }
  for (size_t position = 0; position < impure_rules.size(); ++position) {
    for (int32_t referencing_rule_id : referencing_rules[impure_rules[position]]) {
      if (result[referencing_rule_id]) {
        result[referencing_rule_id] = false;
        impure_rules.push_back(referencing_rule_id);
      }
    }
  }
  return result;
}

}  // namespace

std::shared_ptr<const LocalCompletionTokenSummary> BuildLocalCompletionTokenSummary(
    const Grammar& grammar,
    int32_t rule_id,
    const TokenizerInfo& tokenizer_info,
    bool has_pure_fsm_semantics,
    bool enforce_own_lookahead
) {
  if (rule_id < 0 || rule_id >= grammar->NumRules() || tokenizer_info.GetVocabSize() == 0 ||
      !grammar->per_rule_fsms[rule_id].has_value()) {
    return nullptr;
  }
  if (!has_pure_fsm_semantics) {
    return nullptr;
  }

  auto flattened_result = GrammarFSMBuilder::FlattenRuleFSMs(grammar, rule_id, 256);
  if (flattened_result.IsErr()) {
    return nullptr;
  }
  auto dfa_result = std::move(flattened_result).Unwrap().MinimizeDFA(64);
  if (dfa_result.IsErr()) {
    return nullptr;
  }
  const auto dfa = std::move(dfa_result).Unwrap();
  const auto& fsm = dfa.GetFsm();
  const int32_t num_states = dfa.NumStates();
  const int32_t start = dfa.GetStart();
  if (num_states == 0 || dfa.IsEndState(start) || dfa.GetEnds().empty()) {
    return nullptr;
  }

  std::vector<std::array<int32_t, 256>> transitions(num_states);
  for (auto& state_transitions : transitions) {
    state_transitions.fill(FSM::kNoNextState);
  }
  int32_t completion_byte = -1;
  bool found_completion = false;
  std::vector<std::vector<int32_t>> predecessors(num_states);
  for (int32_t state = 0; state < num_states; ++state) {
    if (dfa.IsEndState(state) && !fsm.GetEdges(state).empty()) {
      return nullptr;
    }
    for (const auto& edge : fsm.GetEdges(state)) {
      if (!edge.IsCharRange() || edge.min < 0 || edge.max > 255) {
        return nullptr;
      }
      predecessors[edge.target].push_back(state);
      for (int32_t byte = edge.min; byte <= edge.max; ++byte) {
        auto& target = transitions[state][byte];
        if (target != FSM::kNoNextState && target != edge.target) {
          return nullptr;
        }
        target = edge.target;
        if (!dfa.IsEndState(edge.target)) {
          continue;
        }
        if (state != start || (completion_byte != -1 && completion_byte != byte)) {
          return nullptr;
        }
        completion_byte = byte;
        found_completion = true;
      }
    }
  }
  // The optimization is useful only for a repeatable local prefix. Since every DFA state is
  // reachable from start, an incoming edge to start proves that start belongs to a non-trivial
  // SCC (including a self-loop), rather than describing a one-shot delimiter rule.
  if (!found_completion || predecessors[start].empty()) {
    return nullptr;
  }

  // A decoded token ending in a non-accepting state is a valid local prefix only when some
  // completion remains reachable from that state.
  std::vector<uint8_t> can_reach_end(num_states, false);
  std::vector<int32_t> pending(dfa.GetEnds().begin(), dfa.GetEnds().end());
  for (int32_t end : pending) {
    can_reach_end[end] = true;
  }
  for (size_t position = 0; position < pending.size(); ++position) {
    for (int32_t predecessor : predecessors[pending[position]]) {
      if (!can_reach_end[predecessor]) {
        can_reach_end[predecessor] = true;
        pending.push_back(predecessor);
      }
    }
  }

  const auto& sorted_vocab = tokenizer_info.GetSortedDecodedVocab();
  const auto& lcp_with_previous = tokenizer_info.ImplPtr()->GetSortedVocabLCPWithPrevious();
  auto summary = std::make_shared<LocalCompletionTokenSummary>();
  summary->completion_byte = static_cast<uint8_t>(completion_byte);
  summary->accepted_prefix_tokens = DynamicBitset(tokenizer_info.GetVocabSize());
  summary->crossing_tokens = DynamicBitset(tokenizer_info.GetVocabSize());
  summary->crossing_indices.reserve(sorted_vocab.size() / 16);
  summary->crossings_by_suffix.reserve(sorted_vocab.size() / 16);

  enum class PreviousDecision : uint8_t { kNone, kCrossing, kRejected };
  PreviousDecision previous_decision = PreviousDecision::kNone;
  int32_t previous_decision_prefix = -1;
  std::vector<int32_t> prefix_states{start};
  for (int32_t token_index = 0; token_index < static_cast<int32_t>(sorted_vocab.size());
       ++token_index) {
    const auto& token = sorted_vocab[token_index].second;
    const int32_t lcp = token_index == 0 ? 0 : lcp_with_previous[token_index];
    if (previous_decision != PreviousDecision::kNone && lcp >= previous_decision_prefix) {
      if (previous_decision == PreviousDecision::kCrossing) {
        summary->crossing_tokens.Set(sorted_vocab[token_index].first);
        summary->crossing_indices.push_back(token_index);
        summary->crossings_by_suffix.push_back(
            LocalCompletionCrossingToken{token_index, previous_decision_prefix}
        );
      }
      continue;
    }

    XGRAMMAR_DCHECK(lcp < static_cast<int32_t>(prefix_states.size()));
    prefix_states.resize(lcp + 1);
    int32_t state = prefix_states.back();
    previous_decision = PreviousDecision::kNone;
    previous_decision_prefix = -1;
    for (int32_t offset = lcp; offset < static_cast<int32_t>(token.size()); ++offset) {
      const int32_t next = transitions[state][static_cast<uint8_t>(token[offset])];
      if (next == FSM::kNoNextState) {
        previous_decision = PreviousDecision::kRejected;
        previous_decision_prefix = offset + 1;
        break;
      }
      state = next;
      prefix_states.push_back(state);
      if (dfa.IsEndState(state)) {
        previous_decision = PreviousDecision::kCrossing;
        previous_decision_prefix = offset + 1;
        summary->crossing_tokens.Set(sorted_vocab[token_index].first);
        summary->crossing_indices.push_back(token_index);
        summary->crossings_by_suffix.push_back(
            LocalCompletionCrossingToken{token_index, previous_decision_prefix}
        );
        break;
      }
    }
    if (previous_decision == PreviousDecision::kNone && can_reach_end[state]) {
      summary->accepted_prefix_tokens.Set(sorted_vocab[token_index].first);
    }
  }
  if (summary->crossing_indices.empty()) {
    return nullptr;
  }

  std::stable_sort(
      summary->crossings_by_suffix.begin(),
      summary->crossings_by_suffix.end(),
      [&](const auto& lhs, const auto& rhs) {
        const std::string_view lhs_token = sorted_vocab[lhs.sorted_vocab_index].second;
        const std::string_view rhs_token = sorted_vocab[rhs.sorted_vocab_index].second;
        return lhs_token.substr(lhs.suffix_offset) < rhs_token.substr(rhs.suffix_offset);
      }
  );

  const auto* tokenizer_impl = tokenizer_info.ImplPtr();
  summary->json_string_length_compatible =
      summary->completion_byte == static_cast<uint8_t>('"') &&
      summary->crossing_indices == tokenizer_impl->GetJSONStringCrossingIndices();
  if (summary->json_string_length_compatible) {
    const auto& quote_offsets = tokenizer_impl->GetJSONStringClosingQuoteOffsets();
    for (const auto& crossing : summary->crossings_by_suffix) {
      if (crossing.suffix_offset != quote_offsets[crossing.sorted_vocab_index] + 1) {
        summary->json_string_length_compatible = false;
        break;
      }
    }
  }
  if (summary->json_string_length_compatible) {
    const auto& json_prefix_tokens = tokenizer_impl->GetJSONStringContentPrefixBitset();
    for (const auto& [token_id, token] : sorted_vocab) {
      const bool local_prefix = summary->accepted_prefix_tokens[token_id];
      const bool json_prefix = json_prefix_tokens[token_id];
      if ((json_prefix && !local_prefix) ||
          (local_prefix && !json_prefix && IsPotentiallyValidUTF8Prefix(token))) {
        summary->json_string_length_compatible = false;
        break;
      }
    }
  }
  if (summary->json_string_length_compatible) {
    // The local grammar machinery deliberately preserves a few malformed UTF-8 prefixes, while
    // JSON strings require RFC 3629. Project those orthogonal byte prefixes out only after the
    // finite vocabulary has proven otherwise identical to JSON lexical behavior.
    summary->accepted_prefix_tokens = tokenizer_impl->GetJSONStringContentPrefixBitset();
  }
  // A non-redundant lookahead on the candidate rule itself must keep filtering crossing tokens
  // exactly like the fallback path's lookahead adaptation: a crossing token whose suffix cannot
  // start the assertion is rejected. JSON-length-compatible summaries are exempt: their
  // machine-generated closing-context lookahead matches the rule's true successors, which the
  // per-fill suffix reparse already enforces exactly, and the quote-suffix machinery requires
  // the full crossing set. If the assertion cannot become a byte-level DFA, skip the
  // optimization so the fallback path enforces it.
  if (enforce_own_lookahead && !summary->json_string_length_compatible) {
    const int32_t lookahead_id = grammar->GetRule(rule_id).lookahead_assertion_id;
    XGRAMMAR_DCHECK(lookahead_id != -1);
    auto lookahead_fsm =
        GrammarFSMBuilder::Sequence(grammar->GetGrammarExpr(lookahead_id), grammar);
    if (!lookahead_fsm.has_value()) {
      return nullptr;
    }
    const auto simplified = lookahead_fsm->SimplifyEpsilon();
    for (int32_t state = 0; state < simplified.NumStates(); ++state) {
      for (const auto& edge : simplified.GetFsm().GetEdges(state)) {
        if (!edge.IsCharRange() && !edge.IsEpsilon()) {
          return nullptr;
        }
      }
    }
    auto lookahead_dfa_result = simplified.MinimizeDFA(64);
    if (lookahead_dfa_result.IsErr()) {
      return nullptr;
    }
    const auto lookahead_dfa = std::move(lookahead_dfa_result).Unwrap();
    std::vector<std::array<int32_t, 256>> lookahead_transitions(lookahead_dfa.NumStates());
    for (auto& state_transitions : lookahead_transitions) {
      state_transitions.fill(FSM::kNoNextState);
    }
    for (int32_t state = 0; state < lookahead_dfa.NumStates(); ++state) {
      for (const auto& edge : lookahead_dfa.GetFsm().GetEdges(state)) {
        if (!edge.IsCharRange() || edge.min < 0 || edge.max > 255) {
          return nullptr;
        }
        for (int32_t byte = edge.min; byte <= edge.max; ++byte) {
          lookahead_transitions[state][byte] = edge.target;
        }
      }
    }
    const auto lookahead_allows = [&](const LocalCompletionCrossingToken& crossing) {
      const std::string& token = sorted_vocab[crossing.sorted_vocab_index].second;
      int32_t state = lookahead_dfa.GetStart();
      if (lookahead_dfa.IsEndState(state)) {
        return true;
      }
      for (size_t offset = crossing.suffix_offset; offset < token.size(); ++offset) {
        state = lookahead_transitions[state][static_cast<uint8_t>(token[offset])];
        if (state == FSM::kNoNextState) {
          return false;
        }
        if (lookahead_dfa.IsEndState(state)) {
          return true;
        }
      }
      return true;
    };
    std::vector<LocalCompletionCrossingToken> kept_crossings;
    kept_crossings.reserve(summary->crossings_by_suffix.size());
    for (const auto& crossing : summary->crossings_by_suffix) {
      if (lookahead_allows(crossing)) {
        kept_crossings.push_back(crossing);
      } else {
        summary->crossing_tokens.Reset(sorted_vocab[crossing.sorted_vocab_index].first);
      }
    }
    if (kept_crossings.size() != summary->crossings_by_suffix.size()) {
      summary->crossings_by_suffix = std::move(kept_crossings);
      summary->crossing_indices.clear();
      for (const auto& crossing : summary->crossings_by_suffix) {
        summary->crossing_indices.push_back(crossing.sorted_vocab_index);
      }
      std::sort(summary->crossing_indices.begin(), summary->crossing_indices.end());
    }
    if (summary->crossing_indices.empty()) {
      return nullptr;
    }
  }
  return summary;
}

std::vector<std::shared_ptr<const LocalCompletionTokenSummary>> BuildLocalCompletionTokenSummaries(
    const Grammar& grammar, const TokenizerInfo& tokenizer_info
) {
  const auto pure_fsm_semantics = GetPureFSMSemanticsRules(grammar);
  const auto redundant_lookahead = GetRedundantLookaheadRules(grammar);
  std::vector<std::shared_ptr<const LocalCompletionTokenSummary>> result(grammar->NumRules());
  for (int32_t rule_id = 0; rule_id < grammar->NumRules(); ++rule_id) {
    result[rule_id] = BuildLocalCompletionTokenSummary(
        grammar, rule_id, tokenizer_info, pure_fsm_semantics[rule_id], !redundant_lookahead[rule_id]
    );
  }
  return result;
}

class CharacterClassTokenSummaryCache {
 private:
  struct KeyHash {
    size_t operator()(const std::vector<int32_t>& key) const {
      uint64_t result = 0;
      for (int32_t value : key) {
        HashCombineBinary(result, static_cast<uint64_t>(value));
      }
      return result;
    }
  };

  struct RepeatKey {
    std::vector<int32_t> character_class;
    int32_t max_characters;

    bool operator==(const RepeatKey& other) const {
      return max_characters == other.max_characters && character_class == other.character_class;
    }
  };

  struct RepeatKeyHash {
    size_t operator()(const RepeatKey& key) const {
      uint64_t result = KeyHash{}(key.character_class);
      HashCombineBinary(result, static_cast<uint64_t>(key.max_characters));
      return result;
    }
  };

  using ASCIIByteLoopKey = std::array<uint64_t, 2>;

  struct ASCIIByteLoopKeyHash {
    size_t operator()(const ASCIIByteLoopKey& key) const {
      uint64_t result = key[0];
      HashCombineBinary(result, key[1]);
      return result;
    }
  };

 public:
  struct Result {
    std::vector<CharacterClassTokenSummary> summaries;
    DynamicBitset consumed_whole_token_bitset;
    std::vector<int32_t> small_consumed_whole_token_indices;
    std::vector<int32_t> completed_prefix_unconsumed_indices;
    int32_t consumed_whole_token_count;
  };

  struct ASCIIByteLoopTokenMask {
    DynamicBitset accepted_bitset;
    std::vector<int32_t> unaccepted_indices;
  };

  std::shared_ptr<const Result> GetOrCreate(
      const Grammar::Impl::GrammarExpr& character_class,
      const std::vector<std::pair<int32_t, std::string>>& sorted_vocab,
      const std::vector<int32_t>& lcp_with_previous,
      const std::vector<int32_t>& ascii_string_safe_indices,
      size_t vocab_size
  ) {
    std::vector<int32_t> key(character_class.begin(), character_class.end());
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto existing = cache_.find(key);
      if (existing != cache_.end()) {
        return existing->second;
      }
    }

    auto summaries = BuildCharacterClassTokenSummaries(
        character_class, sorted_vocab, lcp_with_previous, ascii_string_safe_indices
    );
    DynamicBitset consumed_whole_token_bitset(vocab_size);
    std::vector<int32_t> small_consumed_whole_token_indices;
    std::vector<int32_t> completed_prefix_unconsumed_indices;
    small_consumed_whole_token_indices.reserve(AdaptiveTokenMask::USE_BITSET_THRESHOLD);
    completed_prefix_unconsumed_indices.reserve(summaries.size() / 16);
    int32_t consumed_whole_token_count = 0;
    for (const auto& summary : summaries) {
      if (summary.consumed_whole_token) {
        consumed_whole_token_bitset.Set(sorted_vocab[summary.sorted_vocab_index].first, true);
        ++consumed_whole_token_count;
        if (consumed_whole_token_count <= AdaptiveTokenMask::USE_BITSET_THRESHOLD) {
          small_consumed_whole_token_indices.push_back(summary.sorted_vocab_index);
        }
      } else if (summary.has_completed_character_prefix) {
        completed_prefix_unconsumed_indices.push_back(summary.sorted_vocab_index);
      }
    }
    if (consumed_whole_token_count >= AdaptiveTokenMask::USE_BITSET_THRESHOLD) {
      small_consumed_whole_token_indices.clear();
    }
    auto computed = std::make_shared<const Result>(Result{
        std::move(summaries),
        std::move(consumed_whole_token_bitset),
        std::move(small_consumed_whole_token_indices),
        std::move(completed_prefix_unconsumed_indices),
        consumed_whole_token_count
    });
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.emplace(std::move(key), computed).first->second;
  }

  std::shared_ptr<const CharacterClassRepeatTokenMask> GetOrCreateRepeatMask(
      const Grammar::Impl::GrammarExpr& character_class,
      const std::vector<std::pair<int32_t, std::string>>& sorted_vocab,
      const std::vector<int32_t>& lcp_with_previous,
      const std::vector<int32_t>& ascii_string_safe_indices,
      const std::vector<uint8_t>& json_string_crossing_flags,
      size_t vocab_size,
      int32_t max_characters
  ) {
    RepeatKey key{
        std::vector<int32_t>(character_class.begin(), character_class.end()), max_characters
    };
    {
      std::lock_guard<std::mutex> lock(repeat_mutex_);
      const auto existing = repeat_cache_.find(key);
      if (existing != repeat_cache_.end()) {
        if (auto retained = existing->second.lock()) {
          return retained;
        }
      }
    }

    const auto summaries = GetOrCreate(
        character_class, sorted_vocab, lcp_with_previous, ascii_string_safe_indices, vocab_size
    );
    if (max_characters < 0) {
      AdaptiveTokenMask adaptive_token_mask =
          summaries->consumed_whole_token_count >= AdaptiveTokenMask::USE_BITSET_THRESHOLD
              ? AdaptiveTokenMask(
                    summaries->consumed_whole_token_bitset,
                    sorted_vocab,
                    /*additional_accepted_indices=*/{},
                    summaries->completed_prefix_unconsumed_indices
                )
              : AdaptiveTokenMask(
                    vocab_size,
                    sorted_vocab,
                    summaries->small_consumed_whole_token_indices,
                    summaries->completed_prefix_unconsumed_indices
                );
      auto computed =
          std::make_shared<const CharacterClassRepeatTokenMask>(CharacterClassRepeatTokenMask{
              std::move(adaptive_token_mask),
              DynamicBitset(vocab_size),
              std::all_of(
                  summaries->completed_prefix_unconsumed_indices.begin(),
                  summaries->completed_prefix_unconsumed_indices.end(),
                  [&](int32_t index) { return json_string_crossing_flags[index]; }
              )
          });
      std::lock_guard<std::mutex> lock(repeat_mutex_);
      auto& cached = repeat_cache_[std::move(key)];
      if (auto retained = cached.lock()) {
        return retained;
      }
      cached = computed;
      return computed;
    }
    std::vector<int32_t> accepted_indices;
    std::vector<int32_t> uncertain_indices;
    accepted_indices.reserve(summaries->summaries.size());
    uncertain_indices.reserve(summaries->summaries.size());
    DynamicBitset accepted_prefix_tokens(vocab_size);
    for (const auto& summary : summaries->summaries) {
      if (!summary.consumed_whole_token || summary.locally_consumed_characters > max_characters) {
        uncertain_indices.push_back(summary.sorted_vocab_index);
      } else {
        accepted_prefix_tokens.Set(sorted_vocab[summary.sorted_vocab_index].first);
      }
    }
    auto computed =
        std::make_shared<const CharacterClassRepeatTokenMask>(CharacterClassRepeatTokenMask{
            AdaptiveTokenMask(vocab_size, sorted_vocab, accepted_indices, uncertain_indices),
            std::move(accepted_prefix_tokens),
            std::all_of(
                uncertain_indices.begin(),
                uncertain_indices.end(),
                [&](int32_t index) { return json_string_crossing_flags[index]; }
            )
        });
    std::lock_guard<std::mutex> lock(repeat_mutex_);
    auto& cached = repeat_cache_[std::move(key)];
    if (auto retained = cached.lock()) {
      return retained;
    }
    cached = computed;
    return computed;
  }

  std::shared_ptr<const ASCIIByteLoopTokenMask> GetOrCreateASCIIByteLoopMask(
      const std::bitset<256>& byte_mask,
      const std::vector<std::pair<int32_t, std::string>>& sorted_vocab,
      const std::vector<int32_t>& lcp_with_previous,
      size_t vocab_size
  ) {
    ASCIIByteLoopKey key{0, 0};
    for (int32_t byte = 0; byte < 128; ++byte) {
      if (byte_mask[byte]) {
        key[byte / 64] |= uint64_t{1} << (byte % 64);
      }
    }
    {
      std::lock_guard<std::mutex> lock(ascii_byte_loop_mutex_);
      const auto existing = ascii_byte_loop_cache_.find(key);
      if (existing != ascii_byte_loop_cache_.end()) {
        return existing->second;
      }
    }

    DynamicBitset accepted_bitset(vocab_size);
    std::vector<int32_t> unaccepted_indices;
    unaccepted_indices.reserve(sorted_vocab.size() / 4);
    int32_t previous_rejected_offset = -1;
    for (int32_t index = 0; index < static_cast<int32_t>(sorted_vocab.size()); ++index) {
      const std::string& token = sorted_vocab[index].second;
      bool token_accepted = !token.empty();
      const int32_t common_prefix = lcp_with_previous[index];
      int32_t offset = common_prefix;
      if (previous_rejected_offset >= 0 && common_prefix > previous_rejected_offset) {
        token_accepted = false;
        offset = previous_rejected_offset;
      }
      for (; token_accepted && offset < static_cast<int32_t>(token.size()); ++offset) {
        const uint8_t byte = static_cast<uint8_t>(token[offset]);
        if (byte >= 0x80 || !byte_mask[byte]) {
          token_accepted = false;
          break;
        }
      }
      if (token_accepted) {
        accepted_bitset.Set(sorted_vocab[index].first, true);
      } else {
        unaccepted_indices.push_back(index);
      }
      previous_rejected_offset = token_accepted ? -1 : offset;
    }
    auto computed = std::make_shared<const ASCIIByteLoopTokenMask>(
        ASCIIByteLoopTokenMask{std::move(accepted_bitset), std::move(unaccepted_indices)}
    );
    std::lock_guard<std::mutex> lock(ascii_byte_loop_mutex_);
    return ascii_byte_loop_cache_.emplace(key, computed).first->second;
  }

  void Clear() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cache_.clear();
    }
    {
      std::lock_guard<std::mutex> lock(repeat_mutex_);
      repeat_cache_.clear();
    }
    {
      std::lock_guard<std::mutex> lock(ascii_byte_loop_mutex_);
      ascii_byte_loop_cache_.clear();
    }
  }

 private:
  std::mutex mutex_;
  std::unordered_map<std::vector<int32_t>, std::shared_ptr<const Result>, KeyHash> cache_;
  std::mutex repeat_mutex_;
  // Compiled grammars retain shared ownership. Weak values keep evicted grammars from making the
  // compiler-wide index retain their token bitmasks indefinitely.
  std::unordered_map<RepeatKey, std::weak_ptr<const CharacterClassRepeatTokenMask>, RepeatKeyHash>
      repeat_cache_;
  std::mutex ascii_byte_loop_mutex_;
  std::unordered_map<
      ASCIIByteLoopKey,
      std::shared_ptr<const ASCIIByteLoopTokenMask>,
      ASCIIByteLoopKeyHash>
      ascii_byte_loop_cache_;
};

struct JSONStringSinkInfo {
  std::bitset<256> ascii_transition_mask;
  std::bitset<256> content_prefix_transition_mask;
  bool current_state_accepts_content_prefixes{false};
  bool current_state_accepts_ascii_safe_prefixes{false};
};

/*!
 * \brief Minimum number of quote-crossing tokens before a recognized standard JSON string state
 * defers them to the runtime matcher instead of resolving them here. Below this, one compile-time
 * lookahead walk over the set is cheap and keeps every later runtime mask fill metadata-only;
 * above it (quote-heavy adversarial tokenizers), the compile-time walk itself would dominate.
 */
constexpr size_t kJSONStringDeferCrossingTokensThreshold = 16384;

/*! \brief The concrete implementation of GrammarMatcherNode. */
class GrammarMatcherForTokenMaskCache : public EarleyParser {
 public:
  GrammarMatcherForTokenMaskCache(
      const Grammar& grammar,
      const ParserState& init_state,
      const std::unordered_map<int32_t, DynamicBitset>&
          tag_dispatch_rule_id_to_second_slicing_bitset,
      const TokenizerInfo& tokenizer_info,
      std::optional<RuleLevelCache>& rule_level_cache,
      const std::shared_ptr<CharacterClassTokenSummaryCache>& character_class_token_summary_cache,
      const std::vector<std::shared_ptr<const LocalCompletionTokenSummary>>&
          local_completion_summaries,
      bool enable_direct_character_class_mask,
      const std::shared_ptr<const EarleyParserGrammarFeatures>& grammar_features
  )
      : EarleyParser(
            grammar,
            init_state,
            grammar_features,
            /*enforce_json_string_lengths=*/false
        ),
        init_rule_id_(init_state.rule_id),
        initial_state_(init_state),
        tag_dispatch_rule_id_to_second_slicing_bitset_(tag_dispatch_rule_id_to_second_slicing_bitset
        ),
        tokenizer_info_(tokenizer_info),
        rule_level_cache_(rule_level_cache),
        character_class_token_summary_cache_(character_class_token_summary_cache),
        local_completion_summaries_(local_completion_summaries),
        enable_direct_character_class_mask_(enable_direct_character_class_mask) {
    // A context-independent rule has already been proven unable to enter a JSON string length
    // rule by GetRuleLevelCacheableRules. Avoid maintaining the grammar-wide JSON counter while
    // this isolated parser walks the vocabulary trie; no reachable state can observe it.
    if (enable_direct_character_class_mask_ && has_json_string_length_rules_) {
      has_json_string_length_rules_ = false;
      json_string_char_count_history_.clear();
      tmp_json_string_length_entered_on_latest_advance_ = false;
      tmp_json_string_length_suspended_ = false;
    }
    if (has_json_string_length_rules_) {
      XGRAMMAR_DCHECK(!json_string_char_count_history_.empty());
      json_string_char_count_history_.back().length_entered = false;
    }
  }
  /*!
   * \brief Get the adaptive token mask for the given ParserState.
   * \param is_root_rule Whether to consider the parent rule. If false, there will be
   * no uncertain tokens. Useful for the root rule.
   */
  AdaptiveTokenMask GetAdaptiveTokenMask(bool is_root_rule);

  /*!
   * \brief Get the token mask for the given ParserState.
   * \param first_char_mask The first character mask.
   * \param is_root_rule Whether to consider the parent rule. If false, there will be
   * no uncertain tokens. Useful for the root rule.
   * \returns True if the rejected indices are filled as usual, False otherwise.
   * It's used to determine which construction function will be used.
   */
  bool GetTokenMaskWithFirstCharacterCheck(
      const std::bitset<256>& first_char_mask,
      bool is_root_rule,
      const std::vector<int32_t>& token_edge_accepted
  );

  /*!
   * \brief Adapt the cache with lookahead assertion.
   * \param cache The adaptive token mask to be adapted.
   * \param is_root_rule Whether to consider the parent rule.
   */
  void AdaptCacheWithLookahead(AdaptiveTokenMask* cache, bool is_root_rule);

 private:
  /*! \brief Build a token mask directly for a context-independent single character class. */
  std::optional<AdaptiveTokenMask> GetSingleCharacterClassDirectMask(bool is_root_rule) const;

  /*! \brief Build a direct mask for a rule with a mechanically proven stable completion. */
  std::optional<AdaptiveTokenMask> GetLocalCompletionDirectMask(bool is_root_rule) const;

  /*! \brief Reuse tokenizer metadata for a deterministic ASCII alphanumeric run. */
  std::optional<AdaptiveTokenMask> GetAsciiAlphanumericRunDirectMask(bool is_root_rule);

  /*! \brief Classify one segment of an exact delimiter-separated recursive character run. */
  std::optional<AdaptiveTokenMask> GetDelimitedRecursiveRunDirectMask(bool is_root_rule) const;

  /*! \brief Classify one prefix of a delimiter-separated recursive ASCII label. */
  std::optional<AdaptiveTokenMask> GetDelimitedRecursiveLabelDirectMask(bool is_root_rule) const;

  /*! \brief Classify the ASCII-safe subset of a fixed-width JSON wildcard path. */
  std::optional<AdaptiveTokenMask> GetFixedWidthJSONStringDirectMask(bool is_root_rule) const;

  /*! \brief Classify a broad absorbing accepting state with its local byte DFA. */
  std::optional<AdaptiveTokenMask> GetAbsorbingEndStateDirectMask(bool is_root_rule) const;

  /*! \brief Classify a deterministic byte FSM state that has a stable character loop. */
  std::optional<AdaptiveTokenMask> GetDeterministicByteLoopDirectMask(bool is_root_rule) const;

  /*! \brief Build a token mask directly for a deterministic byte path from the current state. */
  std::optional<AdaptiveTokenMask> GetDeterministicBytePathDirectMask(bool is_root_rule) const;

  /*! \brief Build a token mask for a long byte prefix before a non-byte FSM transition. */
  std::optional<AdaptiveTokenMask> GetDeterministicBytePrefixDirectMask(bool is_root_rule) const;

  /*! \brief Find absorbing JSON string states and the first bytes that enter them. */
  JSONStringSinkInfo GetJSONStringSinkInfo() const;

  AdaptiveTokenMask BuildAdaptiveTokenMask(
      bool rejected_filled,
      const std::vector<int32_t>& accepted_indices,
      const std::vector<int32_t>& rejected_indices,
      const std::vector<int32_t>& uncertain_indices
  ) const;

  /*! \brief Check if a token can pass the lookahead assertion. */
  std::pair</*acceptable*/ bool, /*can reach end*/ bool> IsTokenPassLookaheadAssertion(
      const std::string& token, const std::vector<bool>& can_reach_end_stack
  );

  /*!
   * \brief Check if speculative calculation will be applied.
   * \return first: whether speculative calculation is applicable.
   * \return second: part of the first character mask,
   * which can be used in speculative calculation.
   */
  std::pair<bool, std::bitset<256>> GetSpeculativeCalculation();

  /*! \brief Return a character class that loops directly back to this rule's start state. */
  std::optional<int32_t> GetSpeculativeCharacterClassExprId() const;

  /*!
   * \brief Get the first character mask.
   * \param first_character_mask the bitset to store the first character mask.
   */
  void GetFirstCharacterMask(std::bitset<256>& first_character_mask);

  /*!
   * \brief Compute sorted vocab indices accepted by token edges at the current FSM state.
   * Token(ids) edges accept listed token IDs.
   * ExcludeToken(ids) edges accept all tokens except listed IDs.
   * \return Sorted, deduplicated vector of accepted sorted vocab indices.
   */
  const std::vector<int32_t>& GetTokenEdgeAcceptedIndices();

  // The id of the initial rule.
  int32_t init_rule_id_;

  // The initial state of the parser.
  ParserState initial_state_;

  /*!
   * \brief This is a mapping from TagDispatch rule id to the bitset used for second slicing.
   * \note If a rule is a TagDispatch rule, then there will be an AC automaton for its triggers.
   *  Which means that it can accept a lot of tokens. However, it will be slow to check a lot of
   *  tokens. The DynamicBitset here is used to do a second slicing: if a token's substr(1, n - 1)
   *  can be accepted by the start state of the AC automaton, then it will be True in the bitset.
   *  When we check a token, we first check if its first character can transit to the start state.
   *  If yes, then we check if it is in the bitset. If yes, then we accept it directly.
   */
  const std::unordered_map<int32_t, DynamicBitset>& tag_dispatch_rule_id_to_second_slicing_bitset_;

  const TokenizerInfo& tokenizer_info_;

  std::optional<RuleLevelCache> rule_level_cache_;

  std::shared_ptr<CharacterClassTokenSummaryCache> character_class_token_summary_cache_;

  const std::vector<std::shared_ptr<const LocalCompletionTokenSummary>>&
      local_completion_summaries_;

  bool enable_direct_character_class_mask_;

  // Temporary data for GetAdaptiveTokenMask.
  std::vector<int32_t> tmp_accepted_indices_;
  std::vector<int32_t> tmp_rejected_indices_;
  std::vector<int32_t> tmp_uncertain_indices_;
  std::vector<int32_t> tmp_rejected_by_lookahead_indices_;
  std::vector<int32_t> tmp_accepted_by_lookahead_indices_;
  std::vector<bool> tmp_can_reach_end_stack_;
  std::vector<bool> tmp_can_reach_end_prefix_or_stack_;
  std::optional<DynamicBitset> tmp_base_accepted_bitset_;
  std::shared_ptr<const CharacterClassTokenSummaryCache::Result>
      speculative_character_class_summary_;
  // Temporary data for GetTokenEdgeAcceptedIndices.
  std::vector<int32_t> tmp_token_edge_accepted_;
  std::vector<int32_t> tmp_token_edge_excluded_;
};

void GrammarMatcherForTokenMaskCache::AdaptCacheWithLookahead(
    AdaptiveTokenMask* cache_ptr, bool is_root_rule
) {
  AdaptiveTokenMask& cache = *cache_ptr;
  const auto& sorted_decoded_vocab = tokenizer_info_.GetSortedDecodedVocab();
  const auto& subtree_nodes_range = tokenizer_info_.GetTrieSubtreeNodesRange();
  const auto& lcp_with_previous = tokenizer_info_.ImplPtr()->GetSortedVocabLCPWithPrevious();
  const std::string* prev_token = nullptr;
  int32_t prev_token_idx = -1;
  bool is_exact_lookahead = grammar_->GetRule(init_rule_id_).is_exact_lookahead;
  int prev_matched_size = 0;
  int last_rejected_range = 0;
  int last_uncertain_range = 0;
  if (is_root_rule) {
    tmp_rejected_indices_ = cache.uncertain_indices;
  } else {
    const auto& lookahead_id = grammar_->GetRule(init_rule_id_).lookahead_assertion_id;
    if (lookahead_id == -1) {
      return;
    }
    for (const auto& uncertain_index : cache.uncertain_indices) {
      const auto& token = sorted_decoded_vocab[uncertain_index].second;
      // Many tokens may contain the same prefix, so we will avoid unnecessary matching
      // by finding the longest common prefix with the previous token.
      bool accepted = true;
      if (uncertain_index < last_rejected_range) {
        tmp_rejected_indices_.push_back(uncertain_index);
        continue;
      }
      if (uncertain_index < last_uncertain_range) {
        // This token is already marked as uncertain.
        continue;
      }
      if (prev_token != nullptr) {
        XGRAMMAR_DCHECK(prev_token_idx < uncertain_index);
        int lcp_len = static_cast<int>(token.size());
        for (int32_t lcp_index = prev_token_idx + 1; lcp_index <= uncertain_index; ++lcp_index) {
          lcp_len = std::min(lcp_len, lcp_with_previous[lcp_index]);
        }
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
      prev_token_idx = uncertain_index;

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

      XGRAMMAR_DCHECK(!tmp_can_reach_end_prefix_or_stack_.empty());
      bool can_reach_end = tmp_can_reach_end_prefix_or_stack_.back();

      XGRAMMAR_DCHECK(!accepted) << "All the tokens are at least uncertain!";
      if (can_reach_end && prev_matched_size > 0) {
        auto [lookahead_accepted, lookahead_completed] =
            IsTokenPassLookaheadAssertion(token, tmp_can_reach_end_stack_);
        if ((!is_root_rule) && lookahead_accepted) {
          if (lookahead_completed || !is_exact_lookahead) {
            tmp_uncertain_indices_.push_back(uncertain_index);
          } else {
            tmp_accepted_indices_.push_back(uncertain_index);
          }
        } else {
          tmp_rejected_indices_.push_back(uncertain_index);
          last_rejected_range = subtree_nodes_range[uncertain_index];
        }
      } else {
        tmp_rejected_indices_.push_back(uncertain_index);
        last_rejected_range = subtree_nodes_range[uncertain_index];
      }
    }
  }

  // This strategy ensures the consistency of the cache storage type in most cases.
  // However, in this case, the storage type is inconsistent:
  // 1. The original cache is accepted_indices, and rejected_indices is also small.
  // After adapting with lookahead, |accepted_indices| + |accepted_by_lookahead_indices| >
  // |rejected_indices| + |rejected_by_lookahead_indices|, and |rejected_indices| +
  // |rejected_by_lookahead_indices| < AdaptiveTokenMask::USE_BITSET_THRESHOLD. In this case, it
  // should be kRejected, but ignored.
  // 2. The original cache is rejected_indices, and accepted_indices is also small.
  // After adapting with lookahead, |accepted_indices| + |accepted_by_lookahead_indices| <
  // |rejected_indices| + |rejected_by_lookahead_indices|, and |accepted_indices| +
  // |accepted_by_lookahead_indices| < AdaptiveTokenMask::USE_BITSET_THRESHOLD. In this case, it
  // should be kAccepted, but ignored. These two cases are very rare in practice, and the impact is
  // very limited, so we ignore them for simplicity.
  cache.uncertain_indices = tmp_uncertain_indices_;
  switch (cache.store_type) {
    case AdaptiveTokenMask::StoreType::kAccepted: {
      if (cache.accepted_indices.size() + tmp_accepted_indices_.size() <
          AdaptiveTokenMask::USE_BITSET_THRESHOLD) {
        IntsetUnion(&cache.accepted_indices, tmp_accepted_indices_);
        break;
      }
      // Transform to bitset.
      cache.store_type = AdaptiveTokenMask::StoreType::kAcceptedBitset;
      cache.accepted_bitset = DynamicBitset(tokenizer_info_.GetVocabSize());
      for (const auto& accepted_index : cache.accepted_indices) {
        cache.accepted_bitset.Set(sorted_decoded_vocab[accepted_index].first);
      }
      for (const auto& accepted_index : tmp_accepted_indices_) {
        cache.accepted_bitset.Set(sorted_decoded_vocab[accepted_index].first);
      }
      cache.accepted_indices.clear();
      break;
    }
    case AdaptiveTokenMask::StoreType::kRejected: {
      if (cache.rejected_indices.size() + tmp_rejected_indices_.size() <
          AdaptiveTokenMask::USE_BITSET_THRESHOLD) {
        IntsetUnion(&cache.rejected_indices, tmp_rejected_indices_);
        break;
      }
      // Transform to bitset.
      cache.store_type = AdaptiveTokenMask::StoreType::kAcceptedBitset;
      cache.accepted_bitset = DynamicBitset(tokenizer_info_.GetVocabSize());
      cache.accepted_bitset.Set();
      for (const auto& special_index : tokenizer_info_.GetSpecialTokenIds()) {
        cache.accepted_bitset.Reset(special_index);
      }
      for (const auto& uncertain_index : cache.uncertain_indices) {
        cache.accepted_bitset.Reset(sorted_decoded_vocab[uncertain_index].first);
      }
      for (const auto& rejected_index : cache.rejected_indices) {
        cache.accepted_bitset.Reset(sorted_decoded_vocab[rejected_index].first);
      }
      for (const auto& rejected_index : tmp_rejected_indices_) {
        cache.accepted_bitset.Reset(sorted_decoded_vocab[rejected_index].first);
      }
      cache.rejected_indices.clear();
      break;
    }
    case AdaptiveTokenMask::StoreType::kAcceptedBitset: {
      for (const auto& accepted_index : tmp_accepted_indices_) {
        cache.accepted_bitset.Set(sorted_decoded_vocab[accepted_index].first);
      }
      break;
    }
  }
  cache.RecomputeAcceptedCount(sorted_decoded_vocab.size());
}

std::pair<bool, bool> GrammarMatcherForTokenMaskCache::IsTokenPassLookaheadAssertion(
    const std::string& token, const std::vector<bool>& can_reach_end_stack
) {
  bool accepted = true;
  bool can_reach_end = true;
  auto lookahead_assertion_id = grammar_->GetRule(init_rule_id_).lookahead_assertion_id;
  if (lookahead_assertion_id == -1) {
    return {accepted, can_reach_end};
  }
  auto lookahead_state =
      ParserState(/*rule_id*/ -1, lookahead_assertion_id, 0, ParserState::kNoPrevInputPos, 0);
  PushStateAndExpand(lookahead_state);
  int token_len = token.size();
  if (IsCompleted()) {
    // If the lookahead assertion is already completed, we can accept the token.
    PopLastStates(1);
    return {accepted, can_reach_end};
  }

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
        return {accepted, can_reach_end};
      }
    }
    // Case 2. The whole token is accepted
    if (last_accept_pos == token_len - 1) {
      PopLastStates(last_accept_pos - i + 2);
      can_reach_end = false;
      return {accepted, can_reach_end};
    }
    // Case 3. The token is not accepted. Check the next position.
    PopLastStates(last_accept_pos - i + 1);
  }

  PopLastStates(1);
  can_reach_end = false;
  accepted = false;
  return {accepted, can_reach_end};
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

std::pair<bool, std::bitset<256>> GrammarMatcherForTokenMaskCache::GetSpeculativeCalculation() {
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  // If the initial rule is a tag dispatch, we will check if it can achieve its initial state.
  const auto& rule = grammar_->GetRule(init_rule_id_);
  if (rule.is_lazy) {
    // The fast path assumes greedy self-loop extension is always legal, which does not hold for
    // committed-shortest rules; they must go through the full per-token simulation.
    return {false, std::bitset<256>()};
  }
  const auto& rule_body = grammar_->GetGrammarExpr(rule.body_expr_id);
  if (rule_body.type == GrammarExprType::kTagDispatch) {
    std::bitset<256> speculative_mask;
    XGRAMMAR_DCHECK(grammar_->per_rule_fsms[init_rule_id_].has_value());
    const auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
    for (const auto& edge : fsm.GetFsm().GetFsm().GetEdges(initial_state_.element_id)) {
      if (edge.target != fsm.GetFsm().GetStart()) {
        continue;
      }
      if (!edge.IsCharRange()) {
        continue;
      }
      for (int32_t ch = edge.min; ch <= edge.max; ++ch) {
        speculative_mask.set(ch);
      }
    }
    return {true, speculative_mask};
  }

  // Check if the initial state is self-recursive-like via FSM.
  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[init_rule_id_].has_value());
  bool can_be_applied = false;
  std::bitset<256> speculative_mask;
  const auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
  XGRAMMAR_DCHECK(initial_state_.element_id < fsm.GetFsm().NumStates())
      << "Initial State's element id cannot exceed the whole FSM's number of states.";
  for (const auto& edge : fsm.GetFsm().GetFsm().GetEdges(initial_state_.element_id)) {
    if (edge.IsCharRange()) {
      // Case A: The edge is towards itself.
      if (edge.target == initial_state_.element_id) {
        can_be_applied = true;
        for (int ch = edge.min; ch <= edge.max; ++ch) {
          speculative_mask.set(ch);
        }
        continue;
      }

      // Case B: The state is the start state, and there's an edge to another state,
      // which calls the fsm itself.
      if (fsm.GetFsm().GetStart() == initial_state_.element_id) {
        for (const auto& next_edge : fsm.GetFsm().GetFsm().GetEdges(edge.target)) {
          if ((next_edge.IsRuleRef() && next_edge.GetRefRuleId() == init_rule_id_) ||
              (next_edge.IsRepeatRef() &&
               fsm.GetFsm().GetFsm().GetRepeatEdgeInfo(next_edge.GetAuxIndex()).RuleId() ==
                   init_rule_id_)) {
            can_be_applied = true;
            for (int ch = edge.min; ch <= edge.max; ++ch) {
              speculative_mask.set(ch);
            }
            break;
          }
        }
      }
    }
  }
  return {can_be_applied, speculative_mask};
}

std::optional<int32_t> GrammarMatcherForTokenMaskCache::GetSpeculativeCharacterClassExprId() const {
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  const auto& rule = grammar_->GetRule(init_rule_id_);
  if (rule.is_lazy || initial_state_.sub_element_id != 0) {
    return std::nullopt;
  }
  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[init_rule_id_].has_value());
  if (initial_state_.element_id != grammar_->per_rule_fsms[init_rule_id_]->GetFsm().GetStart()) {
    return std::nullopt;
  }
  const auto& body = grammar_->GetGrammarExpr(rule.body_expr_id);
  if (body.type != GrammarExprType::kChoices) {
    return std::nullopt;
  }
  for (int32_t sequence_id : body) {
    const auto& sequence = grammar_->GetGrammarExpr(sequence_id);
    if (sequence.type != GrammarExprType::kSequence || sequence.size() != 2) {
      continue;
    }
    const auto& character_class = grammar_->GetGrammarExpr(sequence[0]);
    const auto& recursive_reference = grammar_->GetGrammarExpr(sequence[1]);
    if (character_class.type == GrammarExprType::kCharacterClass &&
        recursive_reference.type == GrammarExprType::kRuleRef &&
        recursive_reference[0] == init_rule_id_) {
      return sequence[0];
    }
  }
  return std::nullopt;
}

bool GrammarMatcherForTokenMaskCache::GetTokenMaskWithFirstCharacterCheck(
    const std::bitset<256>& first_char_mask,
    bool is_root_rule,
    const std::vector<int32_t>& token_edge_accepted
) {
  const auto& sorted_decoded_vocab = tokenizer_info_.GetSortedDecodedVocab();
  const auto& subtree_nodes_range = tokenizer_info_.GetTrieSubtreeNodesRange();
  const auto& lcp_with_previous = tokenizer_info_.ImplPtr()->GetSortedVocabLCPWithPrevious();
  // the pair (a, b) means [a, b). Intialize the possible intervals.
  std::vector<std::pair<int32_t, int32_t>> possible_intervals;
  int possible_token_num =
      GetPossibleTokenIntervals(sorted_decoded_vocab, first_char_mask, possible_intervals);

  // Check if the type of the mask can be rejected.
  tmp_accepted_indices_.reserve(possible_token_num);
  bool fill_reject_indices =
      (sorted_decoded_vocab.size() - possible_token_num) < AdaptiveTokenMask::USE_BITSET_THRESHOLD;

  XGRAMMAR_DCHECK(possible_intervals.size() > 0)
      << "There should be at least one possible interval for the first character mask.";

  if (possible_intervals[0].first != 0 && fill_reject_indices) {
    for (int i = 0; i < possible_intervals[0].first; ++i) {
      tmp_rejected_indices_.push_back(i);
    }
  }

  XGRAMMAR_DCHECK(init_rule_id_ != -1 && grammar_->per_rule_fsms[init_rule_id_].has_value());
  auto [speculative_calculation, speculative_mask] = GetSpeculativeCalculation();
  // A speculative byte stays on the current rule's self-loop. JSON length enforcement can
  // therefore filter that definitely accepted token at runtime, while tokens that leave the loop
  // still take the ordinary path below and become uncertain if they enter a length rule.
  if (has_char_budget_rules_) {
    speculative_calculation = false;
  }

  int prev_matched_size = 0;
  int last_rejected_range = 0;
  const bool& is_exact_lookahead = grammar_->GetRule(init_rule_id_).is_exact_lookahead;
  std::optional<const DynamicBitset*> definite_accepted_bitset = std::nullopt;
  const bool is_tag_dispatch_rule =
      grammar_->GetGrammarExpr(grammar_->GetRule(init_rule_id_).body_expr_id).type ==
      Grammar::Impl::GrammarExprType::kTagDispatch;
  if (is_tag_dispatch_rule) {
    XGRAMMAR_DCHECK(tag_dispatch_rule_id_to_second_slicing_bitset_.count(init_rule_id_) > 0);
    definite_accepted_bitset = &tag_dispatch_rule_id_to_second_slicing_bitset_.at(init_rule_id_);
  }

  if (speculative_calculation && !definite_accepted_bitset.has_value()) {
    if (auto character_class_expr_id = GetSpeculativeCharacterClassExprId()) {
      XGRAMMAR_DCHECK(character_class_token_summary_cache_ != nullptr);
      speculative_character_class_summary_ = character_class_token_summary_cache_->GetOrCreate(
          grammar_->GetGrammarExpr(*character_class_expr_id),
          sorted_decoded_vocab,
          tokenizer_info_.ImplPtr()->GetSortedVocabLCPWithPrevious(),
          tokenizer_info_.ImplPtr()->GetAsciiStringSafeIndices(),
          tokenizer_info_.GetVocabSize()
      );
    }
  }

  std::shared_ptr<const CharacterClassTokenSummaryCache::ASCIIByteLoopTokenMask>
      speculative_ascii_byte_loop_mask;
  if (speculative_calculation && !definite_accepted_bitset.has_value() &&
      !speculative_character_class_summary_) {
    XGRAMMAR_DCHECK(character_class_token_summary_cache_ != nullptr);
    speculative_ascii_byte_loop_mask =
        character_class_token_summary_cache_->GetOrCreateASCIIByteLoopMask(
            speculative_mask,
            sorted_decoded_vocab,
            tokenizer_info_.ImplPtr()->GetSortedVocabLCPWithPrevious(),
            tokenizer_info_.GetVocabSize()
        );
    tmp_base_accepted_bitset_ = speculative_ascii_byte_loop_mask->accepted_bitset;
  }

  const std::string* prev_token = nullptr;
  int32_t prev_token_idx = -1;
  int32_t skip_ptr = 0;
  const int32_t skip_size = static_cast<int32_t>(token_edge_accepted.size());
  bool accepts_ascii_string_safe_slice =
      speculative_calculation && !definite_accepted_bitset.has_value() &&
      !speculative_character_class_summary_ && !speculative_ascii_byte_loop_mask;
  if (accepts_ascii_string_safe_slice) {
    for (int32_t byte = 0x20; byte < 0x7f; ++byte) {
      if (byte != '"' && byte != '\\' && !speculative_mask[byte]) {
        accepts_ascii_string_safe_slice = false;
        break;
      }
    }
  }
  const JSONStringSinkInfo string_sink_info = GetJSONStringSinkInfo();
  std::bitset<256> directly_accepted_ascii_first_bytes = string_sink_info.ascii_transition_mask;
  if (accepts_ascii_string_safe_slice) {
    for (int32_t byte = 0x20; byte < 0x7f; ++byte) {
      if (byte != '"' && byte != '\\') {
        directly_accepted_ascii_first_bytes.set(byte);
      }
    }
  }
  const auto tokenizer_impl = tokenizer_info_.ImplPtr();
  if (string_sink_info.current_state_accepts_content_prefixes) {
    tmp_base_accepted_bitset_ = tokenizer_impl->GetJSONStringContentPrefixBitset();
    // The converter marks only canonical JSON string states whose ordinary content language is
    // represented exactly by the tokenizer metadata above. Tokens crossing a closing quote still
    // depend on the parent parser states, so leave that small fixed set for the runtime matcher
    // instead of replaying it through an isolated Earley parser to rediscover the same boundary.
    tmp_uncertain_indices_ = tokenizer_impl->GetJSONStringCrossingIndices();
    return false;
  } else if (string_sink_info.current_state_accepts_ascii_safe_prefixes &&
             tokenizer_impl->GetJSONStringCrossingIndices().size() >=
                 kJSONStringDeferCrossingTokensThreshold) {
    tmp_base_accepted_bitset_ = tokenizer_impl->GetAsciiStringSafeBitset();
    const auto& content_prefix_bitset = tokenizer_impl->GetJSONStringContentPrefixBitset();
    const auto& ascii_safe_bitset = tokenizer_impl->GetAsciiStringSafeBitset();
    const auto& crossing_indices = tokenizer_impl->GetJSONStringCrossingIndices();
    size_t crossing_ptr = 0;
    tmp_uncertain_indices_.reserve(
        crossing_indices.size() + tokenizer_impl->GetJSONStringEscapedTokenIndices().size()
    );
    for (int32_t index = 0; index < static_cast<int32_t>(sorted_decoded_vocab.size()); ++index) {
      while (crossing_ptr < crossing_indices.size() && crossing_indices[crossing_ptr] < index) {
        ++crossing_ptr;
      }
      const int32_t token_id = sorted_decoded_vocab[index].first;
      if ((crossing_ptr < crossing_indices.size() && crossing_indices[crossing_ptr] == index) ||
          (content_prefix_bitset[token_id] && !ascii_safe_bitset[token_id])) {
        tmp_uncertain_indices_.push_back(index);
      }
    }
    return false;
  } else if (string_sink_info.content_prefix_transition_mask.any() ||
             directly_accepted_ascii_first_bytes.any()) {
    if (!tmp_base_accepted_bitset_.has_value()) {
      tmp_base_accepted_bitset_.emplace(tokenizer_info_.GetVocabSize());
    }
  }
  if (tmp_base_accepted_bitset_.has_value()) {
    const auto& content_prefix_by_first_byte =
        tokenizer_impl->GetJSONStringContentPrefixIndicesByFirstByte();
    const auto& ascii_by_first_byte = tokenizer_impl->GetAsciiStringSafeIndicesByFirstByte();
    if (!string_sink_info.current_state_accepts_content_prefixes) {
      for (int32_t byte = 0; byte < 256; ++byte) {
        if (!string_sink_info.content_prefix_transition_mask[byte]) {
          continue;
        }
        for (int32_t index : content_prefix_by_first_byte[byte]) {
          tmp_base_accepted_bitset_->Set(sorted_decoded_vocab[index].first, true);
        }
      }
    }
    for (int32_t byte = 0x20; byte < 0x7f; ++byte) {
      if (!directly_accepted_ascii_first_bytes[byte] ||
          string_sink_info.content_prefix_transition_mask[byte]) {
        continue;
      }
      for (int32_t index : ascii_by_first_byte[byte]) {
        tmp_base_accepted_bitset_->Set(sorted_decoded_vocab[index].first, true);
      }
    }
  }
  const std::vector<int32_t>* token_candidate_indices = nullptr;
  if (string_sink_info.current_state_accepts_content_prefixes) {
    token_candidate_indices = &tokenizer_impl->GetJSONStringCrossingIndices();
  } else if (speculative_ascii_byte_loop_mask) {
    token_candidate_indices = &speculative_ascii_byte_loop_mask->unaccepted_indices;
  }
  size_t token_candidate_ptr = 0;
  for (size_t interval_idx = 0; interval_idx < possible_intervals.size(); ++interval_idx) {
    const auto& interval = possible_intervals[interval_idx];
    for (int i = interval.first; i < interval.second; ++i) {
      if (token_candidate_indices != nullptr) {
        while (token_candidate_ptr < token_candidate_indices->size() &&
               (*token_candidate_indices)[token_candidate_ptr] < i) {
          ++token_candidate_ptr;
        }
        if (token_candidate_ptr == token_candidate_indices->size() ||
            (*token_candidate_indices)[token_candidate_ptr] >= interval.second) {
          break;
        }
        i = (*token_candidate_indices)[token_candidate_ptr++];
      }
      // Skip tokens already accepted by token edges (avoid expensive Earley simulation).
      while (skip_ptr < skip_size && token_edge_accepted[skip_ptr] < i) ++skip_ptr;
      if (skip_ptr < skip_size && token_edge_accepted[skip_ptr] == i) continue;

      // Check if the current token is in the rejected range. i.e. check if the current token
      // is on the subtree of the rejected token.
      if (i < last_rejected_range) {
        if (fill_reject_indices) {
          tmp_rejected_indices_.push_back(i);
          fill_reject_indices =
              tmp_rejected_indices_.size() >= AdaptiveTokenMask::USE_BITSET_THRESHOLD
                  ? false
                  : fill_reject_indices;
        } else {
          i = last_rejected_range - 1;
        }
        continue;
      }
      const auto& token = sorted_decoded_vocab[i].second;
      if (tmp_base_accepted_bitset_.has_value() &&
          (*tmp_base_accepted_bitset_)[sorted_decoded_vocab[i].first]) {
        continue;
      }
      if (speculative_character_class_summary_ &&
          speculative_character_class_summary_
              ->consumed_whole_token_bitset[sorted_decoded_vocab[i].first]) {
        continue;
      }
      // This optimization is useful for simple self-recursive rules, like string content.
      if (speculative_calculation) {
        // Optimization for tag dispatch rules.
        if (definite_accepted_bitset.has_value()) {
          // If the token is empty, it must be accepted.
          if (token.empty()) {
            tmp_accepted_indices_.push_back(i);
            continue;
          }
          // If the token doesn't contain tags or stop strings since the second character, and it
          // will transit to the start state after consuming the first character, it must be
          // accepted.
          if (speculative_mask[static_cast<uint8_t>(token[0])] &&
              (*definite_accepted_bitset.value())[i]) {
            tmp_accepted_indices_.push_back(i);
            continue;
          }
        } else if (!speculative_character_class_summary_) {
          // A character-class summary has already classified every candidate token. Whole-token
          // matches were accepted above; rescanning the remaining tokens can only rediscover the
          // mismatch recorded by the summary, so send them directly to the continuation parser.
          bool all_accepted = true;
          for (char ch : token) {
            // If the first character is not the ascii character or can't be accepted by the
            // first character mask, we need to check them in the parser.
            if (isascii(ch) == 0 || !speculative_mask[static_cast<uint8_t>(ch)]) {
              all_accepted = false;
              break;
            }
          }
          if (all_accepted) {
            tmp_accepted_indices_.push_back(i);
            continue;
          }
        }
      }
      // Many tokens may contain the same prefix, so we will avoid unnecessary matching
      // by finding the longest common prefix with the previous token.
      bool accepted = true;
      if (prev_token != nullptr) {
        XGRAMMAR_DCHECK(prev_token_idx < i);
        int lcp_len = static_cast<int>(token.size());
        for (int32_t lcp_index = prev_token_idx + 1; lcp_index <= i; ++lcp_index) {
          lcp_len = std::min(lcp_len, lcp_with_previous[lcp_index]);
        }
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
      prev_token_idx = i;

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
        if (HasEnteredCharBudget() || HasEnteredJSONStringLengthRule()) {
          tmp_uncertain_indices_.push_back(i);
        } else {
          tmp_accepted_indices_.push_back(i);
        }
      } else if (can_reach_end && prev_matched_size > 0) {
        auto [lookahead_accepted, lookahead_completed] =
            IsTokenPassLookaheadAssertion(token, tmp_can_reach_end_stack_);
        if ((!is_root_rule) && lookahead_accepted) {
          if (lookahead_completed || !is_exact_lookahead) {
            tmp_uncertain_indices_.push_back(i);
          } else if (HasEnteredCharBudget() || HasEnteredJSONStringLengthRule()) {
            tmp_uncertain_indices_.push_back(i);
          } else {
            tmp_accepted_indices_.push_back(i);
            tmp_accepted_by_lookahead_indices_.push_back(i);
          }
        } else {
          for (int j = i; j < subtree_nodes_range[i]; j++) {
            tmp_rejected_indices_.push_back(j);
            tmp_rejected_by_lookahead_indices_.push_back(j);
          }
          i = subtree_nodes_range[i] - 1;  // Skip the subtree nodes.
        }
      } else {
        tmp_rejected_indices_.push_back(i);
        last_rejected_range = subtree_nodes_range[i];
        fill_reject_indices =
            tmp_rejected_indices_.size() >= AdaptiveTokenMask::USE_BITSET_THRESHOLD
                ? false
                : fill_reject_indices;
      }
    }
    if (interval_idx != possible_intervals.size() - 1 && fill_reject_indices) {
      const auto& next_interval = possible_intervals[interval_idx + 1];
      for (int i = interval.second; i < next_interval.first; ++i) {
        tmp_rejected_indices_.push_back(i);
      }
      fill_reject_indices = tmp_rejected_indices_.size() >= AdaptiveTokenMask::USE_BITSET_THRESHOLD
                                ? false
                                : fill_reject_indices;
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

void GrammarMatcherForTokenMaskCache::GetFirstCharacterMask(std::bitset<256>& first_character_mask
) {
  first_character_mask.reset();
  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[init_rule_id_].has_value());
  const auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
  const auto& edges = fsm.GetFsm().GetFsm().GetEdges(initial_state_.element_id);
  for (const auto& edge : edges) {
    if (edge.IsCharRange()) {
      for (int c = edge.min; c <= edge.max; ++c) {
        first_character_mask[c] = true;
      }
    }
  }
}

const std::vector<int32_t>& GrammarMatcherForTokenMaskCache::GetTokenEdgeAcceptedIndices() {
  // Compute sorted vocab indices accepted by Token(ids) and ExcludeToken(ids) edges.
  // Result is stored in tmp_token_edge_accepted_.

  tmp_token_edge_accepted_.clear();
  tmp_token_edge_excluded_.clear();

  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[init_rule_id_].has_value());
  const auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
  const auto& edges = fsm.GetFsm().GetFsm().GetEdges(initial_state_.element_id);

  const auto& sorted_decoded_vocab = tokenizer_info_.GetSortedDecodedVocab();
  int32_t sorted_size = static_cast<int32_t>(sorted_decoded_vocab.size());
  const auto& tid_to_sorted = tokenizer_info_.ImplPtr()->GetTokenIdToSortedVocabIndex();

  bool has_exclude_token = false;

  for (const auto& edge : edges) {
    if (edge.IsToken()) {
      auto info = fsm.GetFsm().GetFsm().GetTokenEdgeInfo(edge.GetAuxIndex());
      for (int32_t i = 0; i < info.Count(); ++i) {
        int32_t tid = info.TokenIds()[i];
        XGRAMMAR_DCHECK(tid >= 0 && tid < static_cast<int32_t>(tid_to_sorted.size()));
        if (tid_to_sorted[tid] >= 0) {
          tmp_token_edge_accepted_.push_back(tid_to_sorted[tid]);
        }
      }
    } else if (edge.IsExcludeToken()) {
      has_exclude_token = true;
      auto info = fsm.GetFsm().GetFsm().GetExcludeTokenEdgeInfo(edge.GetAuxIndex());
      for (int32_t i = 0; i < info.Count(); ++i) {
        int32_t tid = info.TokenIds()[i];
        XGRAMMAR_DCHECK(tid >= 0 && tid < static_cast<int32_t>(tid_to_sorted.size()));
        if (tid_to_sorted[tid] >= 0) {
          tmp_token_edge_excluded_.push_back(tid_to_sorted[tid]);
        }
      }
    }
  }

  // Token-only: result = token_accepted
  if (!has_exclude_token) {
    if (!tmp_token_edge_accepted_.empty()) {
      std::sort(tmp_token_edge_accepted_.begin(), tmp_token_edge_accepted_.end());
      tmp_token_edge_accepted_.erase(
          std::unique(tmp_token_edge_accepted_.begin(), tmp_token_edge_accepted_.end()),
          tmp_token_edge_accepted_.end()
      );
    }
    return tmp_token_edge_accepted_;
  }

  // ExcludeToken: result = [0, sorted_size) - (excluded - token_accepted)
  // Token(ids) overrides ExcludeToken(ids) when both present.
  if (!tmp_token_edge_accepted_.empty()) {
    std::sort(tmp_token_edge_accepted_.begin(), tmp_token_edge_accepted_.end());
    tmp_token_edge_accepted_.erase(
        std::unique(tmp_token_edge_accepted_.begin(), tmp_token_edge_accepted_.end()),
        tmp_token_edge_accepted_.end()
    );
  }
  std::sort(tmp_token_edge_excluded_.begin(), tmp_token_edge_excluded_.end());
  tmp_token_edge_excluded_.erase(
      std::unique(tmp_token_edge_excluded_.begin(), tmp_token_edge_excluded_.end()),
      tmp_token_edge_excluded_.end()
  );
  IntsetDifference(&tmp_token_edge_excluded_, tmp_token_edge_accepted_);
  IntsetComplement(&tmp_token_edge_accepted_, sorted_size, tmp_token_edge_excluded_);
  return tmp_token_edge_accepted_;
}

std::optional<AdaptiveTokenMask> GrammarMatcherForTokenMaskCache::GetSingleCharacterClassDirectMask(
    bool is_root_rule
) const {
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  if (!enable_direct_character_class_mask_ || is_root_rule || initial_state_.sub_element_id != 0 ||
      initial_state_.element_id != grammar_->per_rule_fsms[init_rule_id_]->GetFsm().GetStart()) {
    return std::nullopt;
  }
  const auto& rule = grammar_->GetRule(init_rule_id_);
  const auto& body = grammar_->GetGrammarExpr(rule.body_expr_id);
  if (body.type != GrammarExprType::kChoices) {
    return std::nullopt;
  }
  int32_t character_class_expr_id = -1;
  bool is_character_class_plus = false;
  if (body.size() == 1) {
    if (rule.lookahead_assertion_id != -1) {
      return std::nullopt;
    }
    const auto& sequence = grammar_->GetGrammarExpr(body[0]);
    if (sequence.type != GrammarExprType::kSequence || sequence.size() != 1) {
      return std::nullopt;
    }
    character_class_expr_id = sequence[0];
  } else if (body.size() == 2) {
    int32_t base_character_class_id = -1;
    int32_t recursive_character_class_id = -1;
    for (int32_t sequence_id : body) {
      const auto& sequence = grammar_->GetGrammarExpr(sequence_id);
      if (sequence.type != GrammarExprType::kSequence) {
        return std::nullopt;
      }
      if (sequence.size() == 1 &&
          grammar_->GetGrammarExpr(sequence[0]).type == GrammarExprType::kCharacterClass) {
        base_character_class_id = sequence[0];
      } else if (sequence.size() == 2 &&
                 grammar_->GetGrammarExpr(sequence[0]).type == GrammarExprType::kCharacterClass) {
        const auto& recursive_reference = grammar_->GetGrammarExpr(sequence[1]);
        if (recursive_reference.type != GrammarExprType::kRuleRef ||
            recursive_reference[0] != init_rule_id_) {
          return std::nullopt;
        }
        recursive_character_class_id = sequence[0];
      } else {
        return std::nullopt;
      }
    }
    if (base_character_class_id == -1 || recursive_character_class_id == -1) {
      return std::nullopt;
    }
    const auto& base_character_class = grammar_->GetGrammarExpr(base_character_class_id);
    const auto& recursive_character_class = grammar_->GetGrammarExpr(recursive_character_class_id);
    if (base_character_class.size() != recursive_character_class.size() ||
        !std::equal(
            base_character_class.begin(),
            base_character_class.end(),
            recursive_character_class.begin()
        )) {
      return std::nullopt;
    }
    character_class_expr_id = base_character_class_id;
    is_character_class_plus = true;
  } else {
    return std::nullopt;
  }
  const auto& character_class = grammar_->GetGrammarExpr(character_class_expr_id);
  if (character_class.type != GrammarExprType::kCharacterClass) {
    return std::nullopt;
  }

  XGRAMMAR_DCHECK(character_class_token_summary_cache_ != nullptr);
  const auto& sorted_vocab = tokenizer_info_.GetSortedDecodedVocab();
  const auto summaries = character_class_token_summary_cache_->GetOrCreate(
      character_class,
      sorted_vocab,
      tokenizer_info_.ImplPtr()->GetSortedVocabLCPWithPrevious(),
      tokenizer_info_.ImplPtr()->GetAsciiStringSafeIndices(),
      tokenizer_info_.GetVocabSize()
  );
  if (is_character_class_plus) {
    return summaries->consumed_whole_token_count >= AdaptiveTokenMask::USE_BITSET_THRESHOLD
               ? AdaptiveTokenMask(
                     summaries->consumed_whole_token_bitset,
                     sorted_vocab,
                     /*additional_accepted_indices=*/{},
                     summaries->completed_prefix_unconsumed_indices
                 )
               : AdaptiveTokenMask(
                     tokenizer_info_.GetVocabSize(),
                     sorted_vocab,
                     summaries->small_consumed_whole_token_indices,
                     summaries->completed_prefix_unconsumed_indices
                 );
  }
  std::vector<int32_t> accepted_indices;
  std::vector<int32_t> uncertain_indices;
  accepted_indices.reserve(summaries->summaries.size() / 16);
  uncertain_indices.reserve(summaries->summaries.size());
  for (const auto& summary : summaries->summaries) {
    if (summary.consumed_whole_token && summary.locally_consumed_characters <= 1) {
      accepted_indices.push_back(summary.sorted_vocab_index);
    } else if (summary.has_completed_character_prefix) {
      uncertain_indices.push_back(summary.sorted_vocab_index);
    }
  }
  return AdaptiveTokenMask(
      tokenizer_info_.GetVocabSize(),
      sorted_vocab,
      std::move(accepted_indices),
      std::move(uncertain_indices)
  );
}

std::optional<AdaptiveTokenMask> GrammarMatcherForTokenMaskCache::GetLocalCompletionDirectMask(
    bool is_root_rule
) const {
  if (is_root_rule || initial_state_.sub_element_id != 0 || init_rule_id_ < 0 ||
      init_rule_id_ >= static_cast<int32_t>(local_completion_summaries_.size()) ||
      initial_state_.element_id != grammar_->per_rule_fsms[init_rule_id_]->GetFsm().GetStart()) {
    return std::nullopt;
  }
  const auto& summary = local_completion_summaries_[init_rule_id_];
  if (summary == nullptr) {
    return std::nullopt;
  }
  AdaptiveTokenMask result(
      summary->accepted_prefix_tokens,
      tokenizer_info_.GetSortedDecodedVocab(),
      /*additional_accepted_indices=*/{},
      summary->crossing_indices
  );
  result.local_completion_summary = summary;
  result.uncertain_token_bitset = summary->crossing_tokens;
  result.all_uncertain_tokens_are_json_string_crossing = summary->json_string_length_compatible;
  return result;
}

std::optional<AdaptiveTokenMask>
GrammarMatcherForTokenMaskCache::GetDelimitedRecursiveRunDirectMask(bool is_root_rule) const {
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  const auto& rule = grammar_->GetRule(init_rule_id_);
  if (is_root_rule || initial_state_.sub_element_id != 0 || rule.is_lazy ||
      rule.lookahead_assertion_id != -1 || rule.max_tokens != -1 || rule.max_chars != -1 ||
      rule.json_string_min_length != -1 || rule.json_string_max_length != -1 ||
      !rule.capture_name.empty() || !grammar_->per_rule_fsms[init_rule_id_].has_value() ||
      initial_state_.element_id != grammar_->per_rule_fsms[init_rule_id_]->GetFsm().GetStart()) {
    return std::nullopt;
  }
  const auto& body = grammar_->GetGrammarExpr(rule.body_expr_id);
  if (body.type != GrammarExprType::kChoices || body.size() != 2) {
    return std::nullopt;
  }
  int32_t recursive_sequence_id = -1;
  bool has_empty_choice = false;
  for (int32_t choice_id : body) {
    const auto& choice = grammar_->GetGrammarExpr(choice_id);
    if (choice.type == GrammarExprType::kEmptyStr) {
      has_empty_choice = true;
    } else if (choice.type == GrammarExprType::kSequence && choice.size() == 3) {
      recursive_sequence_id = choice_id;
    } else {
      return std::nullopt;
    }
  }
  if (!has_empty_choice || recursive_sequence_id == -1) {
    return std::nullopt;
  }
  const auto& recursive_sequence = grammar_->GetGrammarExpr(recursive_sequence_id);
  const auto& delimiter_expr = grammar_->GetGrammarExpr(recursive_sequence[0]);
  const auto& run_reference = grammar_->GetGrammarExpr(recursive_sequence[1]);
  const auto& self_reference = grammar_->GetGrammarExpr(recursive_sequence[2]);
  if (delimiter_expr.type != GrammarExprType::kByteString || delimiter_expr.size() != 1 ||
      delimiter_expr[0] < 0 || delimiter_expr[0] >= 128 ||
      run_reference.type != GrammarExprType::kRuleRef ||
      self_reference.type != GrammarExprType::kRuleRef || self_reference[0] != init_rule_id_) {
    return std::nullopt;
  }

  const int32_t run_rule_id = run_reference[0];
  const auto& run_rule = grammar_->GetRule(run_rule_id);
  const auto& run_body = grammar_->GetGrammarExpr(run_rule.body_expr_id);
  if (run_rule.is_lazy || run_rule.max_tokens != -1 || run_rule.max_chars != -1 ||
      run_rule.json_string_min_length != -1 || run_rule.json_string_max_length != -1 ||
      !run_rule.capture_name.empty() || run_body.type != GrammarExprType::kChoices ||
      run_body.size() != 2) {
    return std::nullopt;
  }
  int32_t recursive_character_class_id = -1;
  int32_t terminal_character_class_id = -1;
  for (int32_t choice_id : run_body) {
    const auto& choice = grammar_->GetGrammarExpr(choice_id);
    if (choice.type != GrammarExprType::kSequence) {
      return std::nullopt;
    }
    if (choice.size() == 1 &&
        grammar_->GetGrammarExpr(choice[0]).type == GrammarExprType::kCharacterClass) {
      terminal_character_class_id = choice[0];
    } else if (choice.size() == 2) {
      const auto& character_class = grammar_->GetGrammarExpr(choice[0]);
      const auto& recursive_reference = grammar_->GetGrammarExpr(choice[1]);
      if (character_class.type != GrammarExprType::kCharacterClass ||
          recursive_reference.type != GrammarExprType::kRuleRef ||
          recursive_reference[0] != run_rule_id) {
        return std::nullopt;
      }
      recursive_character_class_id = choice[0];
    } else {
      return std::nullopt;
    }
  }
  if (recursive_character_class_id == -1 || terminal_character_class_id == -1) {
    return std::nullopt;
  }
  const auto& character_class = grammar_->GetGrammarExpr(recursive_character_class_id);
  const auto& terminal_character_class = grammar_->GetGrammarExpr(terminal_character_class_id);
  if (character_class.size() != terminal_character_class.size() ||
      !std::equal(
          character_class.begin(), character_class.end(), terminal_character_class.begin()
      ) ||
      character_class.size() < 3 || character_class[0] != 0) {
    return std::nullopt;
  }
  std::array<uint8_t, 128> accepted_bytes{};
  for (int32_t index = 1; index < character_class.size(); index += 2) {
    if (index + 1 >= character_class.size() || character_class[index] < 0 ||
        character_class[index + 1] >= 128) {
      return std::nullopt;
    }
    for (int32_t byte = character_class[index]; byte <= character_class[index + 1]; ++byte) {
      accepted_bytes[byte] = true;
    }
  }
  const uint8_t delimiter = static_cast<uint8_t>(delimiter_expr[0]);
  if (accepted_bytes[delimiter]) {
    return std::nullopt;
  }

  const auto& sorted_vocab = tokenizer_info_.GetSortedDecodedVocab();
  const std::string delimiter_prefix(1, static_cast<char>(delimiter));
  auto first = std::lower_bound(
      sorted_vocab.begin(),
      sorted_vocab.end(),
      delimiter_prefix,
      [](const auto& token, const std::string& value) { return token.second < value; }
  );
  std::vector<int32_t> accepted_indices;
  std::vector<int32_t> uncertain_indices;
  for (auto it = first; it != sorted_vocab.end() && !it->second.empty() &&
                        static_cast<uint8_t>(it->second.front()) == delimiter;
       ++it) {
    const auto& token = it->second;
    bool consumed_run_character = false;
    bool crossing = false;
    for (int32_t byte_index = 1; byte_index < static_cast<int32_t>(token.size()); ++byte_index) {
      const uint8_t byte = static_cast<uint8_t>(token[byte_index]);
      if (byte < accepted_bytes.size() && accepted_bytes[byte]) {
        consumed_run_character = true;
        continue;
      }
      crossing = consumed_run_character;
      break;
    }
    const int32_t token_index = static_cast<int32_t>(it - sorted_vocab.begin());
    if (!crossing && (token.size() == 1 || consumed_run_character)) {
      accepted_indices.push_back(token_index);
    } else if (crossing) {
      uncertain_indices.push_back(token_index);
    }
  }
  return AdaptiveTokenMask(
      tokenizer_info_.GetVocabSize(), sorted_vocab, accepted_indices, uncertain_indices
  );
}

std::optional<AdaptiveTokenMask>
GrammarMatcherForTokenMaskCache::GetDelimitedRecursiveLabelDirectMask(bool is_root_rule) const {
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  const auto& rule = grammar_->GetRule(init_rule_id_);
  if (is_root_rule || initial_state_.sub_element_id != 0 || rule.is_lazy ||
      rule.lookahead_assertion_id != -1 || rule.max_tokens != -1 || rule.max_chars != -1 ||
      rule.json_string_min_length != -1 || rule.json_string_max_length != -1 ||
      !rule.capture_name.empty() || !grammar_->per_rule_fsms[init_rule_id_].has_value() ||
      initial_state_.element_id != grammar_->per_rule_fsms[init_rule_id_]->GetFsm().GetStart()) {
    return std::nullopt;
  }
  const auto& body = grammar_->GetGrammarExpr(rule.body_expr_id);
  if (body.type != GrammarExprType::kChoices || body.size() != 2) {
    return std::nullopt;
  }
  int32_t label_sequence_id = -1;
  bool has_empty_choice = false;
  for (int32_t choice_id : body) {
    const auto& choice = grammar_->GetGrammarExpr(choice_id);
    if (choice.type == GrammarExprType::kEmptyStr) {
      has_empty_choice = true;
    } else if (choice.type == GrammarExprType::kSequence && choice.size() == 5) {
      label_sequence_id = choice_id;
    } else {
      return std::nullopt;
    }
  }
  if (!has_empty_choice || label_sequence_id == -1) {
    return std::nullopt;
  }
  const auto& label_sequence = grammar_->GetGrammarExpr(label_sequence_id);
  const auto& delimiter_expr = grammar_->GetGrammarExpr(label_sequence[0]);
  const auto& first_character_class = grammar_->GetGrammarExpr(label_sequence[1]);
  const auto& middle_star = grammar_->GetGrammarExpr(label_sequence[2]);
  const auto& final_character_class = grammar_->GetGrammarExpr(label_sequence[3]);
  const auto& self_reference = grammar_->GetGrammarExpr(label_sequence[4]);
  if (delimiter_expr.type != GrammarExprType::kByteString || delimiter_expr.size() != 1 ||
      delimiter_expr[0] < 0 || delimiter_expr[0] >= 128 ||
      first_character_class.type != GrammarExprType::kCharacterClass ||
      middle_star.type != GrammarExprType::kCharacterClassStar ||
      final_character_class.type != GrammarExprType::kCharacterClass ||
      self_reference.type != GrammarExprType::kRuleRef || self_reference[0] != init_rule_id_) {
    return std::nullopt;
  }
  const auto build_ascii_byte_set = [](const auto& character_class
                                    ) -> std::optional<std::array<uint8_t, 128>> {
    if (character_class.size() < 3 || character_class[0] != 0) {
      return std::nullopt;
    }
    std::array<uint8_t, 128> result{};
    for (int32_t index = 1; index < character_class.size(); index += 2) {
      if (index + 1 >= character_class.size() || character_class[index] < 0 ||
          character_class[index + 1] >= 128) {
        return std::nullopt;
      }
      for (int32_t byte = character_class[index]; byte <= character_class[index + 1]; ++byte) {
        result[byte] = true;
      }
    }
    return result;
  };
  const auto first_bytes = build_ascii_byte_set(first_character_class);
  const auto middle_bytes = build_ascii_byte_set(middle_star);
  const auto final_bytes = build_ascii_byte_set(final_character_class);
  if (!first_bytes.has_value() || !middle_bytes.has_value() || !final_bytes.has_value()) {
    return std::nullopt;
  }
  for (int32_t byte = 0; byte < 128; ++byte) {
    if (((*first_bytes)[byte] || (*final_bytes)[byte]) && !(*middle_bytes)[byte]) {
      return std::nullopt;
    }
  }
  const uint8_t delimiter = static_cast<uint8_t>(delimiter_expr[0]);
  if ((*middle_bytes)[delimiter]) {
    return std::nullopt;
  }

  const auto& sorted_vocab = tokenizer_info_.GetSortedDecodedVocab();
  const std::string delimiter_prefix(1, static_cast<char>(delimiter));
  auto first = std::lower_bound(
      sorted_vocab.begin(),
      sorted_vocab.end(),
      delimiter_prefix,
      [](const auto& token, const std::string& value) { return token.second < value; }
  );
  std::vector<int32_t> accepted_indices;
  std::vector<int32_t> uncertain_indices;
  for (auto it = first; it != sorted_vocab.end() && !it->second.empty() &&
                        static_cast<uint8_t>(it->second.front()) == delimiter;
       ++it) {
    const auto& token = it->second;
    bool accepted_prefix = token.size() == 1;
    bool can_complete_label = false;
    bool crossing = false;
    if (token.size() > 1) {
      const uint8_t first_label_byte = static_cast<uint8_t>(token[1]);
      if (first_label_byte < first_bytes->size() && (*first_bytes)[first_label_byte]) {
        accepted_prefix = true;
        for (int32_t byte_index = 2; byte_index < static_cast<int32_t>(token.size());
             ++byte_index) {
          const uint8_t byte = static_cast<uint8_t>(token[byte_index]);
          if (byte < middle_bytes->size() && (*middle_bytes)[byte]) {
            can_complete_label = (*final_bytes)[byte];
            continue;
          }
          crossing = can_complete_label;
          accepted_prefix = false;
          break;
        }
      }
    }
    const int32_t token_index = static_cast<int32_t>(it - sorted_vocab.begin());
    if (accepted_prefix) {
      accepted_indices.push_back(token_index);
    } else if (crossing) {
      uncertain_indices.push_back(token_index);
    }
  }
  return AdaptiveTokenMask(
      tokenizer_info_.GetVocabSize(), sorted_vocab, accepted_indices, uncertain_indices
  );
}

std::optional<AdaptiveTokenMask> GrammarMatcherForTokenMaskCache::GetAsciiAlphanumericRunDirectMask(
    bool is_root_rule
) {
  const auto& rule = grammar_->GetRule(init_rule_id_);
  if (is_root_rule || initial_state_.sub_element_id != 0 || rule.is_lazy ||
      rule.lookahead_assertion_id == -1) {
    return std::nullopt;
  }
  const auto& rule_fsm = grammar_->per_rule_fsms[init_rule_id_]->GetFsm();
  const auto& fsm = rule_fsm.GetFsm();
  if (rule_fsm.IsEndState(initial_state_.element_id)) {
    return std::nullopt;
  }
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  const auto is_ascii_alphanumeric = [](int32_t byte) {
    return (byte >= '0' && byte <= '9') || (byte >= 'A' && byte <= 'Z') ||
           (byte >= 'a' && byte <= 'z');
  };
  const auto is_ascii_alphanumeric_range = [](int32_t min, int32_t max) {
    return (min >= '0' && max <= '9') || (min >= 'A' && max <= 'Z') || (min >= 'a' && max <= 'z');
  };
  const auto character_class_contains_all_alphanumeric = [&](const auto& character_class) {
    if (character_class.type != GrammarExprType::kCharacterClass &&
        character_class.type != GrammarExprType::kCharacterClassStar) {
      return false;
    }
    const bool is_negative = character_class[0];
    for (int32_t byte = 0; byte < 128; ++byte) {
      if (!is_ascii_alphanumeric(byte)) {
        continue;
      }
      bool in_ranges = false;
      for (int32_t index = 1; index < static_cast<int32_t>(character_class.size()); index += 2) {
        if (byte >= character_class[index] && byte <= character_class[index + 1]) {
          in_ranges = true;
          break;
        }
      }
      if (is_negative == in_ranges) {
        return false;
      }
    }
    return true;
  };
  const auto rule_accepts_any_alphanumeric_suffix = [&](int32_t rule_id) {
    const auto& rule = grammar_->GetRule(rule_id);
    const auto& body = grammar_->GetGrammarExpr(rule.body_expr_id);
    if (rule.is_lazy || body.type != GrammarExprType::kChoices) {
      return false;
    }
    bool accepts_empty = false;
    bool accepts_nonempty_run = false;
    for (int32_t choice_id : body) {
      const auto& choice = grammar_->GetGrammarExpr(choice_id);
      if (choice.type == GrammarExprType::kEmptyStr) {
        accepts_empty = true;
      } else if (choice.type == GrammarExprType::kSequence && choice.size() == 2) {
        const auto& prefix = grammar_->GetGrammarExpr(choice[0]);
        const auto& final = grammar_->GetGrammarExpr(choice[1]);
        if (prefix.type == GrammarExprType::kCharacterClassStar &&
            final.type == GrammarExprType::kCharacterClass &&
            character_class_contains_all_alphanumeric(prefix) &&
            character_class_contains_all_alphanumeric(final)) {
          accepts_nonempty_run = true;
        }
      }
    }
    return accepts_empty && accepts_nonempty_run;
  };

  std::array<int32_t, 256> transitions;
  transitions.fill(-1);
  for (const auto& edge : fsm.GetEdges(initial_state_.element_id)) {
    if (!edge.IsCharRange() || !is_ascii_alphanumeric_range(edge.min, edge.max)) {
      return std::nullopt;
    }
    for (int32_t byte = edge.min; byte <= edge.max; ++byte) {
      if (transitions[byte] != -1 && transitions[byte] != edge.target) {
        return std::nullopt;
      }
      transitions[byte] = edge.target;
    }
  }
  std::vector<int32_t> targets;
  for (int32_t byte = 0; byte < 256; ++byte) {
    if (is_ascii_alphanumeric(byte) != (transitions[byte] != -1)) {
      return std::nullopt;
    }
    if (transitions[byte] != -1 &&
        std::find(targets.begin(), targets.end(), transitions[byte]) == targets.end()) {
      targets.push_back(transitions[byte]);
    }
  }
  for (int32_t target : targets) {
    bool accepts_suffix = false;
    for (const auto& edge : fsm.GetEdges(target)) {
      if (edge.IsRuleRef() && rule_accepts_any_alphanumeric_suffix(edge.GetRefRuleId())) {
        accepts_suffix = true;
        break;
      }
    }
    if (!accepts_suffix) {
      return std::nullopt;
    }
  }
  const auto& sorted_vocab = tokenizer_info_.GetSortedDecodedVocab();
  tmp_accepted_indices_.reserve(sorted_vocab.size() / 4);
  tmp_uncertain_indices_.reserve(sorted_vocab.size() / 8);
  for (int32_t index = 0; index < static_cast<int32_t>(sorted_vocab.size()); ++index) {
    const auto& token = sorted_vocab[index].second;
    if (!token.empty() && std::all_of(token.begin(), token.end(), is_ascii_alphanumeric)) {
      tmp_accepted_indices_.push_back(index);
    } else if (!token.empty() && is_ascii_alphanumeric(static_cast<uint8_t>(token.front()))) {
      tmp_uncertain_indices_.push_back(index);
    }
  }
  return AdaptiveTokenMask(
      tokenizer_info_.GetVocabSize(), sorted_vocab, tmp_accepted_indices_, tmp_uncertain_indices_
  );
}

std::optional<AdaptiveTokenMask> GrammarMatcherForTokenMaskCache::GetFixedWidthJSONStringDirectMask(
    bool is_root_rule
) const {
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  const auto& rule = grammar_->GetRule(init_rule_id_);
  const auto& rule_body = grammar_->GetGrammarExpr(rule.body_expr_id);
  const auto& rule_fsm = grammar_->per_rule_fsms[init_rule_id_]->GetFsm();
  const auto& fsm = rule_fsm.GetFsm();
  if (is_root_rule || initial_state_.sub_element_id != 0 || rule.is_lazy ||
      rule_body.type != GrammarExprType::kRegex || !grammar_->GetRegexIsJSONString(rule_body) ||
      rule_fsm.IsEndState(initial_state_.element_id)) {
    return std::nullopt;
  }
  const auto get_ascii_wildcard_target = [&](int32_t state) -> int32_t {
    std::array<int32_t, 256> transitions;
    transitions.fill(-1);
    for (const auto& edge : fsm.GetEdges(state)) {
      if (!edge.IsCharRange()) {
        return -1;
      }
      for (int32_t byte = edge.min; byte <= edge.max; ++byte) {
        if (transitions[byte] != -1 && transitions[byte] != edge.target) {
          return -1;
        }
        transitions[byte] = edge.target;
      }
    }
    int32_t target = -1;
    for (int32_t byte = 0x20; byte < 0x7f; ++byte) {
      if (byte == '"' || byte == '\\') {
        continue;
      }
      if (transitions[byte] == -1) {
        return -1;
      }
      if (target == -1) {
        target = transitions[byte];
      } else if (target != transitions[byte]) {
        return -1;
      }
    }
    return target;
  };
  constexpr int32_t kMaxFixedCharacters = 8;
  std::vector<int32_t> scalar_boundary_states{initial_state_.element_id};
  while (static_cast<int32_t>(scalar_boundary_states.size()) <= kMaxFixedCharacters) {
    const int32_t target = get_ascii_wildcard_target(scalar_boundary_states.back());
    if (target == -1) {
      break;
    }
    scalar_boundary_states.push_back(target);
    if (rule_fsm.IsEndState(target)) {
      break;
    }
  }
  const int32_t fixed_characters = static_cast<int32_t>(scalar_boundary_states.size()) - 1;
  if (fixed_characters == 0 || fixed_characters > kMaxFixedCharacters ||
      (fixed_characters == kMaxFixedCharacters &&
       get_ascii_wildcard_target(scalar_boundary_states.back()) != -1)) {
    return std::nullopt;
  }
  for (int32_t index = 0; index < fixed_characters; ++index) {
    if (rule_fsm.IsEndState(scalar_boundary_states[index])) {
      return std::nullopt;
    }
  }

  struct LocalState {
    int32_t global_state;
    std::array<int32_t, 256> transitions;
  };
  constexpr size_t kMaxDirectStates = 64;
  std::vector<LocalState> local_states;
  local_states.reserve(kMaxDirectStates);
  const auto find_or_add_state = [&](int32_t global_state) -> int32_t {
    for (int32_t local_id = 0; local_id < static_cast<int32_t>(local_states.size()); ++local_id) {
      if (local_states[local_id].global_state == global_state) {
        return local_id;
      }
    }
    if (local_states.size() == kMaxDirectStates) {
      return -1;
    }
    LocalState state;
    state.global_state = global_state;
    state.transitions.fill(-2);
    local_states.push_back(std::move(state));
    return static_cast<int32_t>(local_states.size()) - 1;
  };
  const auto transition = [&](int32_t local_state, int32_t byte) -> int32_t {
    auto& cached = local_states[local_state].transitions[byte];
    if (cached != -2) {
      return cached;
    }
    int32_t global_target = -1;
    for (const auto& edge : fsm.GetEdges(local_states[local_state].global_state)) {
      if (!edge.IsCharRange()) {
        return -2;
      }
      if (byte >= edge.min && byte <= edge.max) {
        if (global_target != -1 && global_target != edge.target) {
          return -2;
        }
        global_target = edge.target;
      }
    }
    if (global_target == -1) {
      cached = -1;
      return cached;
    }
    const int32_t local_target = find_or_add_state(global_target);
    if (local_target == -1) {
      return -2;
    }
    cached = local_target;
    return cached;
  };

  const auto tokenizer_impl = tokenizer_info_.ImplPtr();
  const auto& sorted_vocab = tokenizer_info_.GetSortedDecodedVocab();
  const auto& ascii_safe_indices = tokenizer_impl->GetAsciiStringSafeIndices();
  DynamicBitset base_accepted = tokenizer_impl->GetAsciiStringSafeBitset();
  std::vector<int32_t> additional_accepted_indices;
  std::vector<int32_t> uncertain_indices;
  additional_accepted_indices.reserve(sorted_vocab.size() / 64);
  uncertain_indices.reserve(sorted_vocab.size() / 32);
  const int32_t boundary_local_state = find_or_add_state(scalar_boundary_states.back());
  for (int32_t index : ascii_safe_indices) {
    const auto& token = sorted_vocab[index].second;
    if (static_cast<int32_t>(token.size()) <= fixed_characters) {
      continue;
    }
    base_accepted.Reset(sorted_vocab[index].first);
    int32_t state = boundary_local_state;
    bool reached_end = rule_fsm.IsEndState(local_states[state].global_state);
    int32_t matched = fixed_characters;
    for (int32_t byte_index = fixed_characters; byte_index < static_cast<int32_t>(token.size());
         ++byte_index) {
      const int32_t next = transition(state, static_cast<uint8_t>(token[byte_index]));
      if (next == -2) {
        return std::nullopt;
      }
      if (next == -1) {
        break;
      }
      state = next;
      ++matched;
      reached_end = reached_end || rule_fsm.IsEndState(local_states[state].global_state);
    }
    if (matched == static_cast<int32_t>(token.size())) {
      additional_accepted_indices.push_back(index);
    } else if (reached_end) {
      uncertain_indices.push_back(index);
    }
  }
  for (int32_t index = 0; index < static_cast<int32_t>(sorted_vocab.size()); ++index) {
    if (!tokenizer_impl->GetAsciiStringSafeBitset()[sorted_vocab[index].first]) {
      uncertain_indices.push_back(index);
    }
  }
  std::sort(uncertain_indices.begin(), uncertain_indices.end());
  uncertain_indices.erase(
      std::unique(uncertain_indices.begin(), uncertain_indices.end()), uncertain_indices.end()
  );
  return AdaptiveTokenMask(
      base_accepted, sorted_vocab, additional_accepted_indices, uncertain_indices
  );
}

std::optional<AdaptiveTokenMask> GrammarMatcherForTokenMaskCache::GetAbsorbingEndStateDirectMask(
    bool is_root_rule
) const {
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  const auto& rule = grammar_->GetRule(init_rule_id_);
  const auto& rule_body = grammar_->GetGrammarExpr(rule.body_expr_id);
  const auto& rule_fsm = grammar_->per_rule_fsms[init_rule_id_]->GetFsm();
  const auto& fsm = rule_fsm.GetFsm();
  if (is_root_rule || initial_state_.sub_element_id != 0 || rule.is_lazy ||
      rule_body.type != GrammarExprType::kRegex || !grammar_->GetRegexIsJSONString(rule_body) ||
      !rule_fsm.IsEndState(initial_state_.element_id)) {
    return std::nullopt;
  }
  std::bitset<256> self_loop_bytes;
  for (const auto& edge : fsm.GetEdges(initial_state_.element_id)) {
    if (edge.IsCharRange() && edge.target == initial_state_.element_id) {
      for (int32_t byte = edge.min; byte <= edge.max; ++byte) {
        self_loop_bytes.set(byte);
      }
    }
  }
  for (int32_t byte = 0x20; byte < 0x7f; ++byte) {
    if (byte != '"' && byte != '\\' && !self_loop_bytes[byte]) {
      return std::nullopt;
    }
  }
  struct LocalState {
    int32_t global_state;
    std::array<int32_t, 256> transitions;
  };
  constexpr size_t kMaxDirectStates = 64;
  std::vector<LocalState> local_states;
  local_states.reserve(kMaxDirectStates);
  auto find_or_add_state = [&](int32_t global_state) -> int32_t {
    for (int32_t local_id = 0; local_id < static_cast<int32_t>(local_states.size()); ++local_id) {
      if (local_states[local_id].global_state == global_state) {
        return local_id;
      }
    }
    if (local_states.size() == kMaxDirectStates) {
      return -1;
    }
    LocalState state;
    state.global_state = global_state;
    state.transitions.fill(-1);
    local_states.push_back(std::move(state));
    return static_cast<int32_t>(local_states.size() - 1);
  };
  const int32_t initial_local_state = find_or_add_state(initial_state_.element_id);
  for (size_t local_id = 0; local_id < local_states.size(); ++local_id) {
    const int32_t global_state = local_states[local_id].global_state;
    std::array<int32_t, 256> global_transitions;
    global_transitions.fill(-1);
    for (const auto& edge : fsm.GetEdges(global_state)) {
      if (!edge.IsCharRange()) {
        return std::nullopt;
      }
      for (int32_t byte = edge.min; byte <= edge.max; ++byte) {
        if (global_transitions[byte] != -1 && global_transitions[byte] != edge.target) {
          return std::nullopt;
        }
        global_transitions[byte] = edge.target;
      }
    }
    for (int32_t byte = 0; byte < 256; ++byte) {
      if (global_transitions[byte] == -1) {
        continue;
      }
      const int32_t target_local_state = find_or_add_state(global_transitions[byte]);
      if (target_local_state == -1) {
        return std::nullopt;
      }
      local_states[local_id].transitions[byte] = target_local_state;
    }
  }
  const auto transition = [&](int32_t state, int32_t byte) {
    return local_states[state].transitions[byte];
  };
  const auto tokenizer_impl = tokenizer_info_.ImplPtr();
  const auto& sorted_vocab = tokenizer_info_.GetSortedDecodedVocab();
  const auto& base_accepted = tokenizer_impl->GetAsciiStringSafeBitset();
  std::vector<int32_t> additional_accepted_indices;
  std::vector<int32_t> uncertain_indices;
  additional_accepted_indices.reserve(sorted_vocab.size() / 32);
  uncertain_indices.reserve(sorted_vocab.size() / 128);
  for (int32_t index = 0; index < static_cast<int32_t>(sorted_vocab.size()); ++index) {
    const int32_t token_id = sorted_vocab[index].first;
    if (base_accepted[token_id]) {
      continue;
    }
    const auto& token = sorted_vocab[index].second;
    int32_t state = initial_local_state;
    bool reached_end = false;
    int32_t matched = 0;
    for (uint8_t byte : token) {
      const int32_t next = transition(state, byte);
      if (next == -1) {
        break;
      }
      state = next;
      ++matched;
      reached_end = reached_end || rule_fsm.IsEndState(local_states[state].global_state);
    }
    if (matched == static_cast<int32_t>(token.size())) {
      additional_accepted_indices.push_back(index);
    } else if (reached_end && matched > 0) {
      uncertain_indices.push_back(index);
    }
  }
  return AdaptiveTokenMask(
      base_accepted, sorted_vocab, additional_accepted_indices, uncertain_indices
  );
}

std::optional<AdaptiveTokenMask>
GrammarMatcherForTokenMaskCache::GetDeterministicByteLoopDirectMask(bool is_root_rule) const {
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  const auto& rule = grammar_->GetRule(init_rule_id_);
  const auto& rule_body = grammar_->GetGrammarExpr(rule.body_expr_id);
  if (!enable_direct_character_class_mask_ || is_root_rule || initial_state_.sub_element_id != 0 ||
      rule.is_lazy || rule.max_tokens != -1 || rule.max_chars != -1 ||
      rule.json_string_min_length != -1 || rule.json_string_max_length != -1 ||
      !rule.capture_name.empty() || rule_body.type != GrammarExprType::kRegex ||
      !grammar_->GetRegexIsByteMode(rule_body) ||
      !grammar_->per_rule_fsms[init_rule_id_].has_value()) {
    return std::nullopt;
  }

  const auto& rule_fsm = grammar_->per_rule_fsms[init_rule_id_]->GetFsm();
  const auto& fsm = rule_fsm.GetFsm();
  struct LocalState {
    int32_t global_state;
    std::array<int32_t, 256> transitions;
  };
  constexpr size_t kMaxDirectStates = 256;
  std::vector<LocalState> local_states;
  local_states.reserve(std::min<size_t>(kMaxDirectStates, grammar_->complete_fsm.NumStates()));
  std::unordered_map<int32_t, int32_t> global_to_local;
  global_to_local.reserve(local_states.capacity());
  const auto find_or_add_state = [&](int32_t global_state) -> int32_t {
    const auto existing = global_to_local.find(global_state);
    if (existing != global_to_local.end()) {
      return existing->second;
    }
    if (local_states.size() == kMaxDirectStates) {
      return -1;
    }
    LocalState state;
    state.global_state = global_state;
    state.transitions.fill(-1);
    const int32_t local_state = static_cast<int32_t>(local_states.size());
    local_states.push_back(std::move(state));
    global_to_local.emplace(global_state, local_state);
    return local_state;
  };

  const int32_t initial_local_state = find_or_add_state(initial_state_.element_id);
  for (size_t local_id = 0; local_id < local_states.size(); ++local_id) {
    std::array<int32_t, 256> global_transitions;
    global_transitions.fill(-1);
    for (const auto& edge : fsm.GetEdges(local_states[local_id].global_state)) {
      if (!edge.IsCharRange()) {
        return std::nullopt;
      }
      for (int32_t byte = edge.min; byte <= edge.max; ++byte) {
        if (global_transitions[byte] != -1 && global_transitions[byte] != edge.target) {
          return std::nullopt;
        }
        global_transitions[byte] = edge.target;
      }
    }
    for (int32_t byte = 0; byte < 256; ++byte) {
      if (global_transitions[byte] == -1) {
        continue;
      }
      const int32_t target_local_state = find_or_add_state(global_transitions[byte]);
      if (target_local_state == -1) {
        return std::nullopt;
      }
      local_states[local_id].transitions[byte] = target_local_state;
    }
  }

  // Find the largest byte class that enters one state and then stays there. Tokens made entirely
  // from this class are definitely accepted prefixes. This covers both an ordinary self-loop and
  // the non-accepting entry state produced for `+` without recognizing any particular regex.
  std::bitset<256> stable_loop_bytes;
  for (int32_t target = 0; target < static_cast<int32_t>(local_states.size()); ++target) {
    std::bitset<256> candidate;
    for (int32_t byte = 0; byte < 256; ++byte) {
      if (local_states[initial_local_state].transitions[byte] == target &&
          local_states[target].transitions[byte] == target) {
        candidate.set(byte);
      }
    }
    if (candidate.count() > stable_loop_bytes.count()) {
      stable_loop_bytes = candidate;
    }
  }
  if (stable_loop_bytes.none()) {
    return std::nullopt;
  }

  XGRAMMAR_DCHECK(character_class_token_summary_cache_ != nullptr);
  const auto& sorted_vocab = tokenizer_info_.GetSortedDecodedVocab();
  const auto loop_mask = character_class_token_summary_cache_->GetOrCreateASCIIByteLoopMask(
      stable_loop_bytes,
      sorted_vocab,
      tokenizer_info_.ImplPtr()->GetSortedVocabLCPWithPrevious(),
      tokenizer_info_.GetVocabSize()
  );
  std::optional<std::string> exact_literal_lookahead;
  if (rule.is_exact_lookahead && rule.lookahead_assertion_id != -1) {
    const auto& lookahead = grammar_->GetGrammarExpr(rule.lookahead_assertion_id);
    if (lookahead.type == GrammarExprType::kSequence && lookahead.size() == 1) {
      const auto& literal = grammar_->GetGrammarExpr(lookahead[0]);
      if (literal.type == GrammarExprType::kByteString && literal.size() != 0) {
        exact_literal_lookahead.emplace();
        exact_literal_lookahead->reserve(literal.size());
        for (int32_t byte : literal) {
          exact_literal_lookahead->push_back(static_cast<char>(byte));
        }
      }
    }
  }
  const auto crosses_literal_lookahead = [&](const std::string& token, size_t offset) {
    XGRAMMAR_DCHECK(exact_literal_lookahead.has_value());
    if (offset == token.size()) {
      return false;
    }
    const size_t compared = std::min(token.size() - offset, exact_literal_lookahead->size());
    return token.compare(offset, compared, *exact_literal_lookahead, 0, compared) == 0;
  };

  std::vector<int32_t> additional_accepted_indices;
  std::vector<int32_t> uncertain_indices;
  additional_accepted_indices.reserve(loop_mask->unaccepted_indices.size() / 16);
  uncertain_indices.reserve(loop_mask->unaccepted_indices.size() / 128);
  const auto& subtree_nodes_range = tokenizer_info_.GetTrieSubtreeNodesRange();
  const auto& lcp_with_previous = tokenizer_info_.ImplPtr()->GetSortedVocabLCPWithPrevious();
  std::vector<int32_t> state_history{initial_local_state};
  std::vector<uint8_t> reached_end_history{
      static_cast<uint8_t>(rule_fsm.IsEndState(local_states[initial_local_state].global_state))
  };
  int32_t previous_index = -1;
  for (size_t candidate = 0; candidate < loop_mask->unaccepted_indices.size(); ++candidate) {
    const int32_t index = loop_mask->unaccepted_indices[candidate];
    const auto& token = sorted_vocab[index].second;
    int32_t common_prefix = 0;
    if (previous_index != -1) {
      XGRAMMAR_DCHECK(previous_index < index);
      common_prefix = static_cast<int32_t>(token.size());
      for (int32_t lcp_index = previous_index + 1; lcp_index <= index; ++lcp_index) {
        common_prefix = std::min(common_prefix, lcp_with_previous[lcp_index]);
      }
      common_prefix = std::min(common_prefix, static_cast<int32_t>(state_history.size()) - 1);
    }
    state_history.resize(common_prefix + 1);
    reached_end_history.resize(common_prefix + 1);
    int32_t state = state_history.back();
    size_t matched = common_prefix;
    for (size_t offset = common_prefix; offset < token.size(); ++offset) {
      const int32_t next = local_states[state].transitions[static_cast<uint8_t>(token[offset])];
      if (next == -1) {
        break;
      }
      state = next;
      ++matched;
      state_history.push_back(state);
      reached_end_history.push_back(
          reached_end_history.back() || rule_fsm.IsEndState(local_states[state].global_state)
      );
    }
    previous_index = index;
    if (!token.empty() && matched == token.size()) {
      additional_accepted_indices.push_back(index);
    } else {
      bool can_cross_lookahead = false;
      if (exact_literal_lookahead.has_value()) {
        for (size_t offset = 0; offset <= matched; ++offset) {
          if (rule_fsm.IsEndState(local_states[state_history[offset]].global_state) &&
              crosses_literal_lookahead(token, offset)) {
            can_cross_lookahead = true;
            break;
          }
        }
      }
      if (exact_literal_lookahead.has_value() ? can_cross_lookahead : reached_end_history.back()) {
        uncertain_indices.push_back(index);
        continue;
      }
      const int32_t rejected_subtree_end = subtree_nodes_range[index];
      while (candidate + 1 < loop_mask->unaccepted_indices.size() &&
             loop_mask->unaccepted_indices[candidate + 1] < rejected_subtree_end) {
        ++candidate;
      }
    }
  }
  return AdaptiveTokenMask(
      loop_mask->accepted_bitset, sorted_vocab, additional_accepted_indices, uncertain_indices
  );
}

std::optional<AdaptiveTokenMask>
GrammarMatcherForTokenMaskCache::GetDeterministicBytePathDirectMask(bool is_root_rule) const {
  if (!enable_direct_character_class_mask_ || is_root_rule || initial_state_.sub_element_id != 0 ||
      grammar_->GetRule(init_rule_id_).is_lazy ||
      grammar_->GetRule(init_rule_id_).lookahead_assertion_id != -1) {
    return std::nullopt;
  }

  const auto& rule_fsm = grammar_->per_rule_fsms[init_rule_id_]->GetFsm();
  const auto& fsm = rule_fsm.GetFsm();
  struct PendingPath {
    int32_t state;
    std::string bytes;
  };
  std::vector<PendingPath> pending{{initial_state_.element_id, {}}};
  std::vector<std::string> accepted_prefixes;
  std::vector<std::string> boundary_prefixes;
  constexpr size_t kMaxDirectPaths = 32768;
  constexpr size_t kMaxDirectPathBytes = 256;
  constexpr size_t kMinBoundaryPrefixBytes = 2;
  size_t visited_paths = 0;
  while (!pending.empty()) {
    PendingPath path = std::move(pending.back());
    pending.pop_back();
    if (++visited_paths > kMaxDirectPaths) {
      return std::nullopt;
    }
    if (rule_fsm.IsEndState(path.state)) {
      // A token ending here is accepted locally. Longer tokens depend on either the outgoing path
      // or parent completion; keep only a selective boundary range for runtime classification.
      if (path.bytes.size() < kMinBoundaryPrefixBytes) {
        return std::nullopt;
      }
      boundary_prefixes.push_back(std::move(path.bytes));
      continue;
    }
    if (path.bytes.size() >= kMaxDirectPathBytes) {
      return std::nullopt;
    }

    const auto& edges = fsm.GetEdges(path.state);
    std::bitset<256> seen_bytes;
    bool deterministic_byte_edges = edges.size() != 0;
    for (const auto& edge : edges) {
      if (!edge.IsCharRange() || edge.min != edge.max || seen_bytes[edge.min]) {
        deterministic_byte_edges = false;
        break;
      }
      seen_bytes[edge.min] = true;
    }
    if (!deterministic_byte_edges) {
      // The rejected linear experiment admitted a one-byte boundary (commonly an opening quote)
      // before a broad string state, making a large part of the vocabulary uncertain. A two-byte
      // boundary remains selective while covering property-name tries followed by rule refs.
      if (path.bytes.size() < kMinBoundaryPrefixBytes) {
        return std::nullopt;
      }
      boundary_prefixes.push_back(std::move(path.bytes));
      continue;
    }
    for (const auto& edge : edges) {
      std::string next_bytes = path.bytes;
      next_bytes.push_back(static_cast<char>(edge.min));
      accepted_prefixes.push_back(next_bytes);
      pending.push_back(PendingPath{edge.target, std::move(next_bytes)});
    }
  }
  if (boundary_prefixes.empty() ||
      std::any_of(boundary_prefixes.begin(), boundary_prefixes.end(), [](const auto& prefix) {
        return prefix.empty();
      })) {
    return std::nullopt;
  }

  const auto& sorted_vocab = tokenizer_info_.GetSortedDecodedVocab();
  std::vector<int32_t> accepted_indices;
  std::vector<int32_t> uncertain_indices;

  // A token ending at any visited byte prefix is accepted locally. Equal decoded strings can
  // have multiple token ids, so collect the full equal range for every distinct prefix.
  std::sort(accepted_prefixes.begin(), accepted_prefixes.end());
  accepted_prefixes.erase(
      std::unique(accepted_prefixes.begin(), accepted_prefixes.end()), accepted_prefixes.end()
  );
  for (const auto& prefix : accepted_prefixes) {
    auto first = std::lower_bound(
        sorted_vocab.begin(),
        sorted_vocab.end(),
        prefix,
        [](const auto& token, const std::string& value) { return token.second < value; }
    );
    for (auto it = first; it != sorted_vocab.end() && it->second == prefix; ++it) {
      accepted_indices.push_back(static_cast<int32_t>(it - sorted_vocab.begin()));
    }
  }
  std::sort(accepted_indices.begin(), accepted_indices.end());
  accepted_indices.erase(
      std::unique(accepted_indices.begin(), accepted_indices.end()), accepted_indices.end()
  );

  // A token crossing a boundary depends on the remainder of this rule and, when the rule
  // completes, its parent parser states. Preserve that exact distinction by leaving only these
  // lexicographic prefix ranges for the runtime continuation check. Remove descendant boundaries
  // because their token ranges are already covered by the shortest ancestor.
  std::sort(boundary_prefixes.begin(), boundary_prefixes.end());
  boundary_prefixes.erase(
      std::unique(boundary_prefixes.begin(), boundary_prefixes.end()), boundary_prefixes.end()
  );
  std::vector<std::string> minimal_boundaries;
  for (auto& prefix : boundary_prefixes) {
    if (!minimal_boundaries.empty() && prefix.size() >= minimal_boundaries.back().size() &&
        prefix.compare(0, minimal_boundaries.back().size(), minimal_boundaries.back()) == 0) {
      continue;
    }
    minimal_boundaries.push_back(std::move(prefix));
  }
  for (const auto& prefix : minimal_boundaries) {
    auto first = std::lower_bound(
        sorted_vocab.begin(),
        sorted_vocab.end(),
        prefix,
        [](const auto& token, const std::string& value) { return token.second < value; }
    );
    for (auto it = first; it != sorted_vocab.end(); ++it) {
      const auto& token = it->second;
      if (token.size() < prefix.size() || token.compare(0, prefix.size(), prefix) != 0) {
        break;
      }
      if (token.size() > prefix.size()) {
        uncertain_indices.push_back(static_cast<int32_t>(it - sorted_vocab.begin()));
      }
    }
  }
  std::sort(uncertain_indices.begin(), uncertain_indices.end());
  uncertain_indices.erase(
      std::unique(uncertain_indices.begin(), uncertain_indices.end()), uncertain_indices.end()
  );

  return AdaptiveTokenMask(
      tokenizer_info_.GetVocabSize(), sorted_vocab, accepted_indices, uncertain_indices
  );
}

std::optional<AdaptiveTokenMask>
GrammarMatcherForTokenMaskCache::GetDeterministicBytePrefixDirectMask(bool is_root_rule) const {
  if (!enable_direct_character_class_mask_ || is_root_rule || initial_state_.sub_element_id != 0 ||
      grammar_->GetRule(init_rule_id_).is_lazy ||
      grammar_->GetRule(init_rule_id_).lookahead_assertion_id != -1) {
    return std::nullopt;
  }

  const auto& rule_fsm = grammar_->per_rule_fsms[init_rule_id_]->GetFsm();
  const auto& fsm = rule_fsm.GetFsm();
  int32_t state = initial_state_.element_id;
  std::string deterministic_bytes;
  constexpr size_t kMinDirectPrefixBytes = 2;
  constexpr size_t kMaxDirectPrefixBytes = 256;
  while (deterministic_bytes.size() < kMaxDirectPrefixBytes) {
    if (rule_fsm.IsEndState(state)) {
      // Completion can enter arbitrary parent states. The finite-path builder handles a terminal
      // end state exactly; accepting states that also continue retain the general builder.
      return std::nullopt;
    }
    const auto& edges = fsm.GetEdges(state);
    if (edges.size() != 1 || !edges[0].IsCharRange() || edges[0].min != edges[0].max) {
      break;
    }
    deterministic_bytes.push_back(static_cast<char>(edges[0].min));
    state = edges[0].target;
  }
  // The rejected linear experiment admitted one-byte boundaries such as a quote before a broad
  // string state, making a large vocabulary range uncertain. Two literal bytes are selective
  // enough for structural suffixes such as `\":` while still covering complete property names.
  if (deterministic_bytes.size() < kMinDirectPrefixBytes ||
      deterministic_bytes.size() == kMaxDirectPrefixBytes) {
    return std::nullopt;
  }

  const auto& sorted_vocab = tokenizer_info_.GetSortedDecodedVocab();
  std::vector<int32_t> accepted_indices;
  std::vector<int32_t> uncertain_indices;

  std::string prefix;
  prefix.reserve(deterministic_bytes.size());
  for (char byte : deterministic_bytes) {
    prefix.push_back(byte);
    auto first = std::lower_bound(
        sorted_vocab.begin(),
        sorted_vocab.end(),
        prefix,
        [](const auto& token, const std::string& value) { return token.second < value; }
    );
    for (auto it = first; it != sorted_vocab.end() && it->second == prefix; ++it) {
      accepted_indices.push_back(static_cast<int32_t>(it - sorted_vocab.begin()));
    }
  }

  auto first_crossing = std::lower_bound(
      sorted_vocab.begin(),
      sorted_vocab.end(),
      deterministic_bytes,
      [](const auto& token, const std::string& value) { return token.second < value; }
  );
  for (auto it = first_crossing; it != sorted_vocab.end(); ++it) {
    const auto& token = it->second;
    if (token.size() < deterministic_bytes.size() ||
        token.compare(0, deterministic_bytes.size(), deterministic_bytes) != 0) {
      break;
    }
    if (token.size() > deterministic_bytes.size()) {
      uncertain_indices.push_back(static_cast<int32_t>(it - sorted_vocab.begin()));
    }
  }

  return AdaptiveTokenMask(
      tokenizer_info_.GetVocabSize(), sorted_vocab, accepted_indices, uncertain_indices
  );
}

JSONStringSinkInfo GrammarMatcherForTokenMaskCache::GetJSONStringSinkInfo() const {
  JSONStringSinkInfo result;
  const auto& rule = grammar_->GetRule(init_rule_id_);
  if (has_char_budget_rules_ || rule.is_lazy || initial_state_.sub_element_id != 0) {
    return result;
  }

  const auto& rule_fsm = grammar_->per_rule_fsms[init_rule_id_]->GetFsm();
  const auto& fsm = rule_fsm.GetFsm();
  const auto& edges = fsm.GetEdges(initial_state_.element_id);
  auto accepts_all_ascii_string_bytes = [&](int32_t state) {
    std::bitset<256> accepted_bytes;
    for (const auto& edge : fsm.GetEdges(state)) {
      if (edge.IsCharRange()) {
        for (int32_t byte = edge.min; byte <= edge.max; ++byte) {
          accepted_bytes.set(byte);
        }
      }
    }
    for (int32_t byte = 0x20; byte < 0x7f; ++byte) {
      if (byte != '"' && byte != '\\' && !accepted_bytes[byte]) {
        return false;
      }
    }
    return true;
  };
  auto is_ascii_sink = [&](int32_t state) {
    if (rule_fsm.IsEndState(state)) {
      return false;
    }
    std::bitset<256> self_loop_bytes;
    for (const auto& edge : fsm.GetEdges(state)) {
      if (edge.IsCharRange() && edge.target == state) {
        for (int32_t byte = edge.min; byte <= edge.max; ++byte) {
          self_loop_bytes.set(byte);
        }
      }
    }
    for (int32_t byte = 0x20; byte < 0x7f; ++byte) {
      if (byte != '"' && byte != '\\' && !self_loop_bytes[byte]) {
        return false;
      }
    }
    return true;
  };

  std::vector<int32_t> checked_targets;
  std::vector<uint8_t> target_is_sink;
  for (const auto& edge : edges) {
    if (!edge.IsCharRange()) {
      continue;
    }
    auto target_it = std::find(checked_targets.begin(), checked_targets.end(), edge.target);
    bool is_sink;
    if (target_it != checked_targets.end()) {
      is_sink = target_is_sink[target_it - checked_targets.begin()];
    } else {
      is_sink = is_ascii_sink(edge.target);
      checked_targets.push_back(edge.target);
      target_is_sink.push_back(is_sink);
    }
    if (is_sink) {
      for (int32_t byte = edge.min; byte <= edge.max; ++byte) {
        result.ascii_transition_mask.set(byte);
      }
    }
  }

  const auto& rule_body = grammar_->GetGrammarExpr(rule.body_expr_id);
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  auto character_class_matches = [&](const Grammar::Impl::GrammarExpr& character_class,
                                     const std::vector<std::pair<int32_t, int32_t>>& expected) {
    if (character_class.type != GrammarExprType::kCharacterClass ||
        character_class.size() % 2 != 1) {
      return false;
    }

    // Compare the complete Unicode languages, rather than relying on one particular positive or
    // negative range spelling. Membership can change only at a range boundary, so one probe per
    // interval between all actual and expected boundaries proves equivalence over U+0000 through
    // U+10FFFF without enumerating every codepoint.
    constexpr int32_t kUnicodeEnd = 0x110000;
    std::vector<int32_t> boundaries{0, kUnicodeEnd};
    auto add_boundaries = [&](int32_t lower, int32_t upper) {
      if (lower >= 0 && lower < kUnicodeEnd) {
        boundaries.push_back(lower);
      }
      if (upper >= -1 && upper + 1 < kUnicodeEnd) {
        boundaries.push_back(upper + 1);
      }
    };
    for (int32_t index = 1; index + 1 < character_class.size(); index += 2) {
      add_boundaries(character_class[index], character_class[index + 1]);
    }
    for (const auto& [lower, upper] : expected) {
      add_boundaries(lower, upper);
    }
    std::sort(boundaries.begin(), boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());

    auto actual_contains = [&](int32_t codepoint) {
      bool in_stored_range = false;
      for (int32_t index = 1; index + 1 < character_class.size(); index += 2) {
        if (character_class[index] <= codepoint && codepoint <= character_class[index + 1]) {
          in_stored_range = true;
          break;
        }
      }
      return character_class[0] != 0 ? !in_stored_range : in_stored_range;
    };
    auto expected_contains = [&](int32_t codepoint) {
      return std::any_of(expected.begin(), expected.end(), [&](const auto& range) {
        return range.first <= codepoint && codepoint <= range.second;
      });
    };
    for (int32_t codepoint : boundaries) {
      if (codepoint < kUnicodeEnd && actual_contains(codepoint) != expected_contains(codepoint)) {
        return false;
      }
    }
    return true;
  };
  auto is_standard_json_escape_rule = [&](int32_t rule_id) {
    const auto& escape_rule = grammar_->GetRule(rule_id);
    if (escape_rule.is_lazy || escape_rule.max_tokens != -1 || escape_rule.max_chars != -1 ||
        escape_rule.json_string_min_length != -1 || escape_rule.json_string_max_length != -1) {
      return false;
    }
    // Lookahead propagation adds an exact reference to the following string-content rule onto
    // the standard escape helper. It does not narrow the surrounding `escape string_sub`
    // sequence, but no other lookahead is safe to ignore here.
    if (escape_rule.lookahead_assertion_id != -1) {
      const auto& lookahead = grammar_->GetGrammarExpr(escape_rule.lookahead_assertion_id);
      if (!escape_rule.is_exact_lookahead || lookahead.type != GrammarExprType::kSequence ||
          lookahead.size() != 1) {
        return false;
      }
      const auto& lookahead_reference = grammar_->GetGrammarExpr(lookahead[0]);
      if (lookahead_reference.type != GrammarExprType::kRuleRef ||
          lookahead_reference.size() != 1 || lookahead_reference[0] != init_rule_id_) {
        return false;
      }
    }
    const auto& escape_body = grammar_->GetGrammarExpr(escape_rule.body_expr_id);
    if (escape_body.type != GrammarExprType::kChoices || escape_body.size() != 2) {
      return false;
    }

    const std::vector<std::pair<int32_t, int32_t>> simple_escape_ranges{
        {'"', '"'},
        {'/', '/'},
        {'\\', '\\'},
        {'b', 'b'},
        {'f', 'f'},
        {'n', 'n'},
        {'r', 'r'},
        {'t', 't'}
    };
    const std::vector<std::pair<int32_t, int32_t>> hex_ranges{{'0', '9'}, {'A', 'F'}, {'a', 'f'}};
    bool has_simple_escape = false;
    bool has_unicode_escape = false;
    for (int32_t choice_id : escape_body) {
      const auto& choice = grammar_->GetGrammarExpr(choice_id);
      if (choice.type != GrammarExprType::kSequence) {
        return false;
      }
      if (choice.size() == 1 &&
          character_class_matches(grammar_->GetGrammarExpr(choice[0]), simple_escape_ranges)) {
        if (has_simple_escape) {
          return false;
        }
        has_simple_escape = true;
        continue;
      }
      if (choice.size() == 5) {
        const auto& unicode_prefix = grammar_->GetGrammarExpr(choice[0]);
        if (unicode_prefix.type != GrammarExprType::kByteString || unicode_prefix.size() != 1 ||
            unicode_prefix[0] != 'u') {
          return false;
        }
        for (int32_t index = 1; index < 5; ++index) {
          if (!character_class_matches(grammar_->GetGrammarExpr(choice[index]), hex_ranges)) {
            return false;
          }
        }
        if (has_unicode_escape) {
          return false;
        }
        has_unicode_escape = true;
        continue;
      }
      return false;
    }
    return has_simple_escape && has_unicode_escape;
  };
  if (!has_budget_rules_ && initial_state_.element_id == rule_fsm.GetStart() &&
      rule_body.type == Grammar::Impl::GrammarExprType::kChoices && rule_body.size() == 3) {
    bool has_closing_quote = false;
    bool has_normal_recursive_path = false;
    bool has_escape_recursive_path = false;
    for (int32_t sequence_id : rule_body) {
      const auto& sequence = grammar_->GetGrammarExpr(sequence_id);
      if (sequence.type == Grammar::Impl::GrammarExprType::kByteString) {
        if (sequence.size() != 1 || sequence[0] != '"') {
          return result;
        }
        has_closing_quote = true;
        continue;
      }
      if (sequence.type != Grammar::Impl::GrammarExprType::kSequence) {
        return result;
      }
      if (sequence.size() == 1) {
        const auto& closing_quote = grammar_->GetGrammarExpr(sequence[0]);
        if (closing_quote.type != Grammar::Impl::GrammarExprType::kByteString ||
            closing_quote.size() != 1 || closing_quote[0] != '"') {
          return result;
        }
        has_closing_quote = true;
      } else if (sequence.size() == 2) {
        const auto& character_class = grammar_->GetGrammarExpr(sequence[0]);
        const auto& recursive_reference = grammar_->GetGrammarExpr(sequence[1]);
        if (character_class.type != Grammar::Impl::GrammarExprType::kCharacterClass ||
            recursive_reference.type != Grammar::Impl::GrammarExprType::kRuleRef ||
            recursive_reference[0] != init_rule_id_) {
          return result;
        }
        const std::vector<std::pair<int32_t, int32_t>> normal_character_ranges{
            {0x20, 0x21}, {0x23, 0x5B}, {0x5D, 0x10FFFF}
        };
        if (!character_class_matches(character_class, normal_character_ranges)) {
          return result;
        }
        has_normal_recursive_path = true;
      } else if (sequence.size() == 3) {
        const auto& escape_literal = grammar_->GetGrammarExpr(sequence[0]);
        const auto& escape_reference = grammar_->GetGrammarExpr(sequence[1]);
        const auto& recursive_reference = grammar_->GetGrammarExpr(sequence[2]);
        if (escape_literal.type != Grammar::Impl::GrammarExprType::kByteString ||
            escape_literal.size() != 1 || escape_literal[0] != '\\' ||
            escape_reference.type != Grammar::Impl::GrammarExprType::kRuleRef ||
            !is_standard_json_escape_rule(escape_reference[0]) ||
            recursive_reference.type != Grammar::Impl::GrammarExprType::kRuleRef ||
            recursive_reference[0] != init_rule_id_) {
          return result;
        }
        has_escape_recursive_path = true;
      } else {
        return result;
      }
    }
    if (has_closing_quote && has_normal_recursive_path && has_escape_recursive_path) {
      result.current_state_accepts_ascii_safe_prefixes = true;
      return result;
    }
  }
  if (has_budget_rules_ || rule_body.type != Grammar::Impl::GrammarExprType::kRegex ||
      !grammar_->GetRegexHasJSONStringNormalSink(rule_body)) {
    return result;
  }
  // The marker is set only on the converter's additional-key exclusion regex. At every
  // codepoint boundary inside that regex, all ordinary JSON characters are valid prefixes: they
  // either continue an excluded key or make it distinct and enter the absorbing suffix. Requiring
  // all safe ASCII outgoing bytes mechanically excludes the opening-quote, escape, UTF-8
  // continuation, and end states.
  if (accepts_all_ascii_string_bytes(initial_state_.element_id)) {
    result.current_state_accepts_content_prefixes = true;
  }
  result.content_prefix_transition_mask = result.ascii_transition_mask;
  return result;
}

AdaptiveTokenMask GrammarMatcherForTokenMaskCache::BuildAdaptiveTokenMask(
    bool rejected_filled,
    const std::vector<int32_t>& accepted_indices,
    const std::vector<int32_t>& rejected_indices,
    const std::vector<int32_t>& uncertain_indices
) const {
  const auto& sorted_vocab = tokenizer_info_.GetSortedDecodedVocab();
  if (speculative_character_class_summary_) {
    return AdaptiveTokenMask(
        speculative_character_class_summary_->consumed_whole_token_bitset,
        sorted_vocab,
        accepted_indices,
        uncertain_indices
    );
  }
  if (tmp_base_accepted_bitset_.has_value()) {
    return AdaptiveTokenMask(
        *tmp_base_accepted_bitset_, sorted_vocab, accepted_indices, uncertain_indices
    );
  }
  if (rejected_filled) {
    return AdaptiveTokenMask(
        tokenizer_info_.GetVocabSize(),
        sorted_vocab,
        accepted_indices,
        rejected_indices,
        uncertain_indices
    );
  }
  return AdaptiveTokenMask(
      tokenizer_info_.GetVocabSize(), sorted_vocab, accepted_indices, uncertain_indices
  );
}

AdaptiveTokenMask GrammarMatcherForTokenMaskCache::GetAdaptiveTokenMask(bool is_root_rule) {
  tmp_accepted_indices_.clear();
  tmp_rejected_indices_.clear();
  tmp_uncertain_indices_.clear();
  tmp_rejected_by_lookahead_indices_.clear();
  tmp_accepted_by_lookahead_indices_.clear();
  tmp_can_reach_end_prefix_or_stack_.clear();
  tmp_can_reach_end_stack_.clear();
  tmp_base_accepted_bitset_.reset();
  speculative_character_class_summary_.reset();
  // For every character in the current token, stores whether it is possible to reach the end of
  // the rule when matching until this character. Store it in a stack for later rollback.
  tmp_can_reach_end_stack_.push_back(false);
  tmp_can_reach_end_prefix_or_stack_.push_back(false);

  auto direct_local_completion_mask = GetLocalCompletionDirectMask(is_root_rule);
  if (direct_local_completion_mask.has_value()) {
    return std::move(*direct_local_completion_mask);
  }

  // Try to get the crossing cache.
  bool rule_level_cache_is_available = !has_char_budget_rules_ && rule_level_cache_.has_value() &&
                                       grammar_->per_rule_fsm_hashes[init_rule_id_].has_value();
  std::optional<uint64_t> fsm_hash = std::nullopt;
  int32_t new_state_id = -1;
  std::optional<AdaptiveTokenMask> crossing_cache = std::nullopt;
  int lookahead_id = grammar_->GetRule(initial_state_.rule_id).lookahead_assertion_id;
  bool is_exact_lookahead = grammar_->GetRule(initial_state_.rule_id).is_exact_lookahead;
  std::optional<uint64_t> lookahead_hash = std::nullopt;
  if (rule_level_cache_is_available) {
    lookahead_hash = GrammarFSMHasher::HashSequence(grammar_, lookahead_id);
    const auto& original_to_new_id = grammar_->per_rule_fsm_new_state_ids[init_rule_id_];
    fsm_hash = grammar_->per_rule_fsm_hashes[init_rule_id_].value();
    for (const auto& original_new_pair : original_to_new_id) {
      if (original_new_pair.first == initial_state_.element_id) {
        new_state_id = original_new_pair.second;
        break;
      }
    }
    XGRAMMAR_DCHECK(new_state_id != -1);
    const auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
    if (lookahead_hash.has_value()) {
      crossing_cache = rule_level_cache_->GetCache(
          HashCombine(fsm_hash.value(), lookahead_hash.value(), is_exact_lookahead),
          new_state_id,
          fsm.GetNodeNum(),
          fsm.GetEdgeNum()
      );
      if (crossing_cache.has_value()) {
        // A perfect match.
        return crossing_cache.value();
      }
    }
    crossing_cache = rule_level_cache_->GetCache(
        fsm_hash.value(), new_state_id, fsm.GetNodeNum(), fsm.GetEdgeNum()
    );
    // If the rule doesn't have a lookahead, then it is exactly the same fsm.
    if (crossing_cache.has_value()) {
      // The standard JSON string-content rule leaves every token that crosses its closing quote
      // uncertain. The runtime parser must consult the parent state for those tokens anyway, so
      // replaying each one here merely to pre-apply the rule's lexical lookahead duplicates that
      // work. Keeping the broader base-rule uncertainty is conservative and avoids a vocabulary-
      // wide lookahead walk for quote-heavy tokenizers. For ordinary tokenizers the crossing set
      // is small, and resolving it once here is far cheaper than re-walking it on every runtime
      // mask fill of a length-constrained string, so the deferral is gated on the set size.
      if (!is_root_rule &&
          tokenizer_info_.ImplPtr()->GetJSONStringCrossingIndices().size() >=
              kJSONStringDeferCrossingTokensThreshold &&
          GetJSONStringSinkInfo().current_state_accepts_ascii_safe_prefixes) {
        return crossing_cache.value();
      }
      AdaptCacheWithLookahead(&crossing_cache.value(), is_root_rule);
      return std::move(crossing_cache.value());
    }
  }

  auto direct_character_class_mask = GetSingleCharacterClassDirectMask(is_root_rule);
  if (direct_character_class_mask.has_value()) {
    if (rule_level_cache_is_available) {
      const auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
      rule_level_cache_->AddCache(
          fsm_hash.value(),
          new_state_id,
          fsm.GetNodeNum(),
          fsm.GetEdgeNum(),
          *direct_character_class_mask
      );
    }
    return std::move(*direct_character_class_mask);
  }

  auto direct_delimited_recursive_run_mask = GetDelimitedRecursiveRunDirectMask(is_root_rule);
  if (direct_delimited_recursive_run_mask.has_value()) {
    return std::move(*direct_delimited_recursive_run_mask);
  }

  auto direct_delimited_recursive_label_mask = GetDelimitedRecursiveLabelDirectMask(is_root_rule);
  if (direct_delimited_recursive_label_mask.has_value()) {
    return std::move(*direct_delimited_recursive_label_mask);
  }

  auto direct_ascii_alphanumeric_mask = GetAsciiAlphanumericRunDirectMask(is_root_rule);
  if (direct_ascii_alphanumeric_mask.has_value()) {
    return std::move(*direct_ascii_alphanumeric_mask);
  }

  auto direct_fixed_width_json_string_mask = GetFixedWidthJSONStringDirectMask(is_root_rule);
  if (direct_fixed_width_json_string_mask.has_value()) {
    return std::move(*direct_fixed_width_json_string_mask);
  }

  auto direct_absorbing_end_mask = GetAbsorbingEndStateDirectMask(is_root_rule);
  if (direct_absorbing_end_mask.has_value()) {
    return std::move(*direct_absorbing_end_mask);
  }

  auto direct_deterministic_byte_loop_mask = GetDeterministicByteLoopDirectMask(is_root_rule);
  if (direct_deterministic_byte_loop_mask.has_value()) {
    if (rule_level_cache_is_available) {
      const auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
      rule_level_cache_->AddCache(
          fsm_hash.value(),
          new_state_id,
          fsm.GetNodeNum(),
          fsm.GetEdgeNum(),
          *direct_deterministic_byte_loop_mask
      );
    }
    return std::move(*direct_deterministic_byte_loop_mask);
  }

  auto direct_byte_path_mask = GetDeterministicBytePathDirectMask(is_root_rule);
  if (direct_byte_path_mask.has_value()) {
    if (rule_level_cache_is_available) {
      const auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
      rule_level_cache_->AddCache(
          fsm_hash.value(), new_state_id, fsm.GetNodeNum(), fsm.GetEdgeNum(), *direct_byte_path_mask
      );
    }
    return std::move(*direct_byte_path_mask);
  }

  auto direct_byte_prefix_mask = GetDeterministicBytePrefixDirectMask(is_root_rule);
  if (direct_byte_prefix_mask.has_value()) {
    if (rule_level_cache_is_available) {
      const auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
      rule_level_cache_->AddCache(
          fsm_hash.value(),
          new_state_id,
          fsm.GetNodeNum(),
          fsm.GetEdgeNum(),
          *direct_byte_prefix_mask
      );
    }
    return std::move(*direct_byte_prefix_mask);
  }

  std::bitset<256> first_character_mask;
  GetFirstCharacterMask(first_character_mask);

  // Token edge accepted indices (for byte path skip + merge).
  const auto& token_edge_accepted = GetTokenEdgeAcceptedIndices();

  // Byte path: skip tokens already accepted by token edges.
  bool rejected_filled;
  if (first_character_mask.none()) {
    rejected_filled = false;
  } else {
    rejected_filled = GetTokenMaskWithFirstCharacterCheck(
        first_character_mask, is_root_rule, token_edge_accepted
    );
  }

  // Token edges are rechecked at runtime when a character budget is present because accepting
  // one can enter a budgeted rule within the same token.
  if (!token_edge_accepted.empty()) {
    if (has_char_budget_rules_ || has_json_string_length_rules_) {
      IntsetUnion(&tmp_uncertain_indices_, token_edge_accepted);
    } else {
      IntsetUnion(&tmp_accepted_indices_, token_edge_accepted);
      IntsetDifference(&tmp_uncertain_indices_, token_edge_accepted);
    }
    IntsetDifference(&tmp_rejected_indices_, token_edge_accepted);
  }
  if (rejected_filled) {
    auto return_value = BuildAdaptiveTokenMask(
        true, tmp_accepted_indices_, tmp_rejected_indices_, tmp_uncertain_indices_
    );
    if (rule_level_cache_is_available) {
      if (lookahead_id == -1 && !is_root_rule) {
        // If the rule doesn't have a lookahead, then it is exactly the same fsm.
        auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
        rule_level_cache_->AddCache(
            fsm_hash.value(), new_state_id, fsm.GetNodeNum(), fsm.GetEdgeNum(), return_value
        );
        return return_value;
      }

      // We can add a cache for basic fsm, and a better one for lookahead.
      // All the tokens rejected by lookahead should be uncertain.
      IntsetUnion(&tmp_uncertain_indices_, tmp_rejected_by_lookahead_indices_);
      IntsetUnion(&tmp_uncertain_indices_, tmp_accepted_by_lookahead_indices_);
      std::vector<int32_t> rejected_indices_without_lookahead;
      std::vector<int32_t> accepted_indices_without_lookahead;
      rejected_indices_without_lookahead.reserve(
          tmp_rejected_indices_.size() - tmp_rejected_by_lookahead_indices_.size()
      );
      accepted_indices_without_lookahead.reserve(
          tmp_accepted_indices_.size() - tmp_accepted_by_lookahead_indices_.size()
      );
      std::set_difference(
          tmp_rejected_indices_.begin(),
          tmp_rejected_indices_.end(),
          tmp_rejected_by_lookahead_indices_.begin(),
          tmp_rejected_by_lookahead_indices_.end(),
          std::back_inserter(rejected_indices_without_lookahead)
      );
      std::set_difference(
          tmp_accepted_indices_.begin(),
          tmp_accepted_indices_.end(),
          tmp_accepted_by_lookahead_indices_.begin(),
          tmp_accepted_by_lookahead_indices_.end(),
          std::back_inserter(accepted_indices_without_lookahead)
      );
      auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
      rule_level_cache_->AddCache(
          fsm_hash.value(),
          new_state_id,
          fsm.GetNodeNum(),
          fsm.GetEdgeNum(),
          BuildAdaptiveTokenMask(
              true,
              accepted_indices_without_lookahead,
              rejected_indices_without_lookahead,
              tmp_uncertain_indices_
          )
      );
      if (lookahead_hash.has_value()) {
        auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
        rule_level_cache_->AddCache(
            HashCombine(fsm_hash.value(), lookahead_hash.value(), is_exact_lookahead),
            new_state_id,
            fsm.GetNodeNum(),
            fsm.GetEdgeNum(),
            return_value
        );
      }
    }
    return return_value;
  } else {
    auto return_value =
        BuildAdaptiveTokenMask(false, tmp_accepted_indices_, {}, tmp_uncertain_indices_);

    if (rule_level_cache_is_available) {
      // Prepare for cache.
      auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
      if (lookahead_id == -1 && !is_root_rule) {
        // If the rule doesn't have a lookahead, then it is exactly the same fsm.
        rule_level_cache_->AddCache(
            fsm_hash.value(), new_state_id, fsm.GetNodeNum(), fsm.GetEdgeNum(), return_value
        );
        return return_value;
      }

      // Add 2 caches.
      IntsetUnion(&tmp_uncertain_indices_, tmp_rejected_by_lookahead_indices_);
      IntsetUnion(&tmp_uncertain_indices_, tmp_accepted_by_lookahead_indices_);
      std::vector<int32_t> accepted_indices_without_lookahead;
      accepted_indices_without_lookahead.reserve(
          tmp_accepted_indices_.size() - tmp_accepted_by_lookahead_indices_.size()
      );
      std::set_difference(
          tmp_accepted_indices_.begin(),
          tmp_accepted_indices_.end(),
          tmp_accepted_by_lookahead_indices_.begin(),
          tmp_accepted_by_lookahead_indices_.end(),
          std::back_inserter(accepted_indices_without_lookahead)
      );
      rule_level_cache_->AddCache(
          fsm_hash.value(),
          new_state_id,
          fsm.GetNodeNum(),
          fsm.GetEdgeNum(),
          BuildAdaptiveTokenMask(
              false, accepted_indices_without_lookahead, {}, tmp_uncertain_indices_
          )
      );

      if (lookahead_hash.has_value()) {
        rule_level_cache_->AddCache(
            HashCombine(fsm_hash.value(), lookahead_hash.value(), is_exact_lookahead),
            new_state_id,
            fsm.GetNodeNum(),
            fsm.GetEdgeNum(),
            return_value
        );
      }
    }
    return return_value;
  }
}

void CompiledGrammar::Impl::EnsureRuleLevelMetadata() {
  if (!rule_level_metadata_enabled) {
    return;
  }
  std::call_once(rule_level_metadata_once, [this]() {
    std::lock_guard<std::mutex> lock(rule_level_metadata_mutex);
    GrammarFSMHasher().Apply(&grammar);
    rule_level_cacheable = GetRuleLevelCacheableRules(grammar);

    if (builtin_rule_level_cache_seed_source == nullptr) {
      return;
    }
    for (int32_t rule_id = 0; rule_id < static_cast<int32_t>(grammar->NumRules()); ++rule_id) {
      const auto& hash = grammar->per_rule_fsm_hashes[rule_id];
      if (!hash.has_value() ||
          std::find(builtin_rule_fsm_hashes.begin(), builtin_rule_fsm_hashes.end(), *hash) ==
              builtin_rule_fsm_hashes.end()) {
        continue;
      }
      const auto& fsm = grammar->per_rule_fsms[rule_id].value();
      for (const auto& [original_state_id, normalized_state_id] :
           grammar->per_rule_fsm_new_state_ids[rule_id]) {
        if (normalized_state_id != 0) {
          continue;
        }
        if (auto cached = builtin_rule_level_cache_seed_source->GetCache(
                *hash, normalized_state_id, fsm.GetNodeNum(), fsm.GetEdgeNum()
            )) {
          rule_level_cache->AddCache(
              *hash, normalized_state_id, fsm.GetNodeNum(), fsm.GetEdgeNum(), std::move(*cached)
          );
        }
        break;
      }
      break;
    }
  });
}

const AdaptiveTokenMask& CompiledGrammar::Impl::GetAdaptiveTokenMask(
    const ParserState& state, bool is_root_rule
) {
  if (!enable_dynamic_compilation) {
    const auto it = adaptive_token_mask_cache.find(state);
    XGRAMMAR_CHECK(it != adaptive_token_mask_cache.end())
        << "The token mask cache is incomplete while dynamic compilation is disabled: " << state;
    return it->second;
  }

  EnsureRuleLevelMetadata();
  std::lock_guard<std::mutex> lock(adaptive_token_mask_cache_mutex);
  const auto existing = adaptive_token_mask_cache.find(state);
  if (existing != adaptive_token_mask_cache.end()) {
    return existing->second;
  }

  const ParserState cache_state(
      state.rule_id,
      state.sequence_id,
      state.element_id,
      ParserState::kNoPrevInputPos,
      -1,
      state.sub_element_id
  );
  const bool is_context_independent =
      state.rule_id >= 0 && state.rule_id < static_cast<int32_t>(rule_level_cacheable.size()) &&
      rule_level_cacheable[state.rule_id];
  std::optional<RuleLevelCache> retained_rule_level_cache;
  if (rule_level_cache != nullptr && is_context_independent) {
    retained_rule_level_cache = *rule_level_cache;
  }
  AdaptiveTokenMask mask = GrammarMatcherForTokenMaskCache(
                               grammar,
                               cache_state,
                               tag_dispatch_rule_id_to_second_slicing_bitset,
                               tokenizer_info,
                               retained_rule_level_cache,
                               character_class_token_summary_cache,
                               local_completion_summaries,
                               is_context_independent,
                               earley_parser_grammar_features
  )
                               .GetAdaptiveTokenMask(is_root_rule);
  mask.RecomputeJSONStringMetadata(tokenizer_info);
  return adaptive_token_mask_cache.emplace(cache_state, std::move(mask)).first->second;
}

const CharacterClassRepeatTokenMask& CompiledGrammar::Impl::GetCharacterClassRepeatTokenMask(
    int32_t character_class_expr_id, int32_t max_characters
) {
  std::lock_guard<std::mutex> lock(character_class_repeat_token_masks_mutex);
  const uint64_t cache_key = (static_cast<uint64_t>(character_class_expr_id) << 32) |
                             static_cast<uint32_t>(max_characters + 1);
  const auto existing = character_class_repeat_token_masks.find(cache_key);
  if (existing != character_class_repeat_token_masks.end()) {
    return *existing->second;
  }

  const auto& sorted_vocab = tokenizer_info.GetSortedDecodedVocab();
  XGRAMMAR_DCHECK(character_class_token_summary_cache != nullptr);
  auto repeat_mask = character_class_token_summary_cache->GetOrCreateRepeatMask(
      grammar->GetGrammarExpr(character_class_expr_id),
      sorted_vocab,
      tokenizer_info.ImplPtr()->GetSortedVocabLCPWithPrevious(),
      tokenizer_info.ImplPtr()->GetAsciiStringSafeIndices(),
      tokenizer_info.ImplPtr()->GetJSONStringCrossingFlags(),
      tokenizer_info.GetVocabSize(),
      max_characters
  );
  return *character_class_repeat_token_masks.emplace(cache_key, std::move(repeat_mask))
              .first->second;
}

void CompiledGrammar::Impl::MaterializeAdaptiveTokenMaskCache() {
  if (tokenizer_info.GetVocabSize() == 0) {
    return;
  }
  const int32_t root_rule_id = grammar->GetRootRuleId();
  for (int32_t rule_id = 0; rule_id < static_cast<int32_t>(grammar->NumRules()); ++rule_id) {
    const auto& rule_fsm = grammar->per_rule_fsms[rule_id];
    XGRAMMAR_DCHECK(rule_fsm.has_value());
    ParserState state(
        rule_id, grammar->GetRule(rule_id).body_expr_id, 0, ParserState::kNoPrevInputPos, -1, 0
    );
    std::unordered_set<int> reachable_states;
    rule_fsm->GetFsm().GetReachableStates(&reachable_states);
    for (int32_t element_id : reachable_states) {
      if (!rule_fsm->GetFsm().IsScanableState(element_id)) {
        continue;
      }
      state.element_id = element_id;
      GetAdaptiveTokenMask(state, rule_id == root_rule_id);
    }
  }
}

/******************* GrammarCompilerNoCache *******************/

/*!
 * \brief The base class for the grammar compiler. Handles the compilation logic without cache.
 */
class GrammarCompilerSub {
 public:
  GrammarCompilerSub(
      const TokenizerInfo& tokenizer_info,
      int max_threads,
      std::optional<RuleLevelCache> rule_level_cache,
      bool enable_dynamic_compilation
  )
      : tokenizer_info_(tokenizer_info),
        max_threads_(max_threads),
        rule_level_cache_(rule_level_cache),
        builtin_rule_level_cache_(
            enable_dynamic_compilation && !rule_level_cache.has_value()
                ? std::make_shared<RuleLevelCache>()
                : nullptr
        ),
        enable_dynamic_compilation_(enable_dynamic_compilation) {
    if (builtin_rule_level_cache_ != nullptr && tokenizer_info_.GetVocabSize() != 0) {
      MultiThreadCompileGrammar(
          Grammar::BuiltinJSONGrammar(), /*regex_fsm_cache=*/nullptr, /*materialize_masks=*/true
      );
    }
  }

  CompiledGrammar CompileBuiltinJSONGrammar();

  CompiledGrammar CompileJSONSchema(
      const std::string& schema,
      bool any_whitespace,
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool strict_mode,
      std::optional<int> max_whitespace_cnt,
      bool any_order
  );

  CompiledGrammar CompileRegex(const std::string& regex);

  CompiledGrammar CompileStructuralTag(const std::string& structural_tag_json);

  CompiledGrammar CompileGrammar(const Grammar& grammar);

  CompiledGrammar CompileGrammar(const std::string& ebnf_str, std::string root_rule_name);

  void ClearCharacterClassTokenSummaryCache() { character_class_token_summary_cache_->Clear(); }

  void ClearTagDispatchSlicingCache() {
    std::lock_guard<std::mutex> lock(tag_dispatch_slicing_cache_mutex_);
    tag_dispatch_slicing_cache_.clear();
  }

 private:
  /*! \brief The main logic. Compile the grammar with multi-threading. */
  CompiledGrammar MultiThreadCompileGrammar(
      Grammar grammar, RegexFSMCache* regex_fsm_cache = nullptr, bool materialize_masks = false
  );
  /*! \brief Optimization for TagDispatch.
   *  \param compiled_grammar_impl the compiled_grammar to be optimized.
   *  \param tag_dispatch_rule_id_to_second_slicing_bitset Return value. Mapping from the rule_id to
   * the definite accepted token mask.
   */
  void TagDispatchOptimization(
      std::shared_ptr<CompiledGrammar::Impl> compiled_grammar_impl,
      std::unordered_map<int32_t, DynamicBitset>* tag_dispatch_rule_id_to_second_slicing_bitset
  );

  std::shared_ptr<const DynamicBitset> GetTagDispatchSecondSlicingBitset(
      std::vector<std::string> patterns
  );

  /*! \brief The vocabulary associated with this storage class. */
  const TokenizerInfo tokenizer_info_;
  /*! \brief The maximum number of threads to use. */
  const int max_threads_;

  /*! \brief The manager of the rule level cache.*/
  std::optional<RuleLevelCache> rule_level_cache_;

  std::shared_ptr<RuleLevelCache> builtin_rule_level_cache_;

  std::vector<uint64_t> builtin_rule_fsm_hashes_;

  std::shared_ptr<CharacterClassTokenSummaryCache> character_class_token_summary_cache_ =
      std::make_shared<CharacterClassTokenSummaryCache>();
  /*! \brief Whether token masks are generated on first use. */
  const bool enable_dynamic_compilation_;
  struct StringVectorHash {
    size_t operator()(const std::vector<std::string>& strings) const {
      uint64_t result = 0;
      for (const auto& string : strings) {
        HashCombineBinary(result, std::hash<std::string>{}(string));
      }
      return result;
    }
  };
  std::mutex tag_dispatch_slicing_cache_mutex_;
  std::unordered_map<
      std::vector<std::string>,
      std::shared_ptr<const DynamicBitset>,
      StringVectorHash>
      tag_dispatch_slicing_cache_;
};

CompiledGrammar GrammarCompilerSub::MultiThreadCompileGrammar(
    Grammar grammar_unoptimized, RegexFSMCache* regex_fsm_cache, bool materialize_masks
) {
  auto compiled_grammar_impl = std::make_shared<CompiledGrammar::Impl>();
  compiled_grammar_impl->grammar = GrammarOptimizer::Apply(
      grammar_unoptimized,
      !enable_dynamic_compilation_,
      regex_fsm_cache,
      !enable_dynamic_compilation_
  );
  compiled_grammar_impl->tokenizer_info = tokenizer_info_;
  compiled_grammar_impl->enable_dynamic_compilation = enable_dynamic_compilation_;
  compiled_grammar_impl->earley_parser_grammar_features =
      std::make_shared<const EarleyParserGrammarFeatures>(compiled_grammar_impl->grammar);
  if (tokenizer_info_.GetVocabSize() == 0) {
    return CompiledGrammar(compiled_grammar_impl);
  }
  compiled_grammar_impl->local_completion_summaries = BuildLocalCompletionTokenSummaries(
      compiled_grammar_impl->grammar, compiled_grammar_impl->tokenizer_info
  );
  std::unordered_map<int32_t, DynamicBitset> tag_dispatch_rule_id_to_second_slicing_bitset;
  TagDispatchOptimization(compiled_grammar_impl, &tag_dispatch_rule_id_to_second_slicing_bitset);

  // Dynamic compilation also uses rule hashes to reuse masks between structurally identical
  // context-independent rules in one compiled grammar. The public compiler cache only controls
  // whether that reuse extends across separately compiled grammars.
  compiled_grammar_impl->rule_level_metadata_enabled =
      rule_level_cache_.has_value() || enable_dynamic_compilation_;
  if (enable_dynamic_compilation_) {
    compiled_grammar_impl->character_class_token_summary_cache =
        character_class_token_summary_cache_;
    // Context-independent rule metadata also drives matcher-local character-class repeat
    // masks. Keep it available when the public compiler cache is disabled; otherwise bounded
    // repeats fall back to materializing one full-vocabulary mask for every repeat count.
    compiled_grammar_impl->rule_level_cache =
        materialize_masks ? builtin_rule_level_cache_
                          : (rule_level_cache_.has_value()
                                 ? std::make_shared<RuleLevelCache>(rule_level_cache_.value())
                                 : std::make_shared<RuleLevelCache>());
    if (!materialize_masks && !rule_level_cache_.has_value()) {
      // Defer copying builtin JSON string masks together with the hashes that identify the
      // matching rules. The source cache and hash list are immutable after compiler construction.
      compiled_grammar_impl->builtin_rule_level_cache_seed_source = builtin_rule_level_cache_;
      compiled_grammar_impl->builtin_rule_fsm_hashes = builtin_rule_fsm_hashes_;
    }
    if (materialize_masks) {
      compiled_grammar_impl->EnsureRuleLevelMetadata();
    }
    if (materialize_masks) {
      builtin_rule_fsm_hashes_.clear();
      for (int32_t rule_id = 0;
           rule_id < static_cast<int32_t>(compiled_grammar_impl->grammar->NumRules());
           ++rule_id) {
        if (compiled_grammar_impl->grammar->GetRule(rule_id).name != "characters_item") {
          continue;
        }
        const auto& hash = compiled_grammar_impl->grammar->per_rule_fsm_hashes[rule_id];
        if (hash.has_value()) {
          builtin_rule_fsm_hashes_.push_back(*hash);
        }
      }
    }
    compiled_grammar_impl->tag_dispatch_rule_id_to_second_slicing_bitset =
        std::move(tag_dispatch_rule_id_to_second_slicing_bitset);
    if (materialize_masks) {
      for (int32_t rule_id = 0;
           rule_id < static_cast<int32_t>(compiled_grammar_impl->grammar->NumRules());
           ++rule_id) {
        if (compiled_grammar_impl->grammar->GetRule(rule_id).name != "characters_item") {
          continue;
        }
        const auto& rule_fsm = compiled_grammar_impl->grammar->per_rule_fsms[rule_id].value();
        ParserState state(
            rule_id,
            compiled_grammar_impl->grammar->GetRule(rule_id).body_expr_id,
            0,
            ParserState::kNoPrevInputPos,
            -1,
            0
        );
        for (const auto& [element_id, normalized_state_id] :
             compiled_grammar_impl->grammar->per_rule_fsm_new_state_ids[rule_id]) {
          if (normalized_state_id != 0 || !rule_fsm.GetFsm().IsScanableState(element_id)) {
            continue;
          }
          state.element_id = element_id;
          compiled_grammar_impl->GetAdaptiveTokenMask(state, /*is_root_rule=*/false);
          break;
        }
      }
    }
    return CompiledGrammar(compiled_grammar_impl);
  }

  compiled_grammar_impl->EnsureRuleLevelMetadata();

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
    std::optional<RuleLevelCache> retained_rule_level_cache;
    const bool is_context_independent =
        state.rule_id >= 0 &&
        state.rule_id < static_cast<int32_t>(compiled_grammar_impl->rule_level_cacheable.size()) &&
        compiled_grammar_impl->rule_level_cacheable[state.rule_id];
    if (rule_level_cache_.has_value() && is_context_independent) {
      retained_rule_level_cache = *rule_level_cache_;
    }
    auto grammar_matcher = GrammarMatcherForTokenMaskCache(
        compiled_grammar_impl->grammar,
        state,
        tag_dispatch_rule_id_to_second_slicing_bitset,
        tokenizer_info_,
        retained_rule_level_cache,
        character_class_token_summary_cache_,
        compiled_grammar_impl->local_completion_summaries,
        false,
        compiled_grammar_impl->earley_parser_grammar_features
    );
    auto cur_adaptive_token_mask_cache = grammar_matcher.GetAdaptiveTokenMask(is_root_rule);
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

  auto root_rule_id = compiled_grammar_impl->grammar->GetRootRuleId();

  for (int32_t rule_id = 0; rule_id < static_cast<int>(compiled_grammar_impl->grammar->NumRules());
       ++rule_id) {
    auto rule = compiled_grammar_impl->grammar->GetRule(rule_id);
    const auto& rule_fsm = compiled_grammar_impl->grammar->per_rule_fsms[rule_id];
    XGRAMMAR_DCHECK(rule_fsm.has_value());
    auto cur_stack_element =
        ParserState(rule_id, rule.body_expr_id, 0, ParserState::kNoPrevInputPos, 0);
    std::unordered_set<int> reachable_states;
    rule_fsm->GetFsm().GetReachableStates(&reachable_states);
    for (int i : reachable_states) {
      cur_stack_element.element_id = i;
      if (!rule_fsm->GetFsm().IsScanableState(i)) {
        continue;
      }
      add_task_adaptive_token_mask(cur_stack_element, rule_id == root_rule_id);
    }
  }

  if (max_threads_ > 1) {
    thread_pool->Join();
  }

  return CompiledGrammar(compiled_grammar_impl);
}

CompiledGrammar GrammarCompilerSub::CompileBuiltinJSONGrammar() {
  return MultiThreadCompileGrammar(Grammar::BuiltinJSONGrammar());
}

CompiledGrammar GrammarCompilerSub::CompileJSONSchema(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    std::optional<int> max_whitespace_cnt,
    bool any_order
) {
  RegexFSMCache regex_fsm_cache;
  Grammar grammar = GrammarNormalizer::Apply(JSONSchemaToGrammar(
      schema,
      any_whitespace,
      indent,
      separators,
      strict_mode,
      max_whitespace_cnt,
      any_order,
      JSONFormat::kJSON,
      &regex_fsm_cache
  ));
  return MultiThreadCompileGrammar(std::move(grammar), &regex_fsm_cache);
}

CompiledGrammar GrammarCompilerSub::CompileStructuralTag(const std::string& structural_tag_json) {
  auto result = StructuralTagToGrammar(
                    structural_tag_json,
                    tokenizer_info_,
                    /*normalize=*/true,
                    /*normalize_json_schema_subgrammars=*/false
  )
                    .ToVariant();
  XGRAMMAR_CHECK(std::holds_alternative<Grammar>(result))
      << GetMessageFromVariantError(std::get<1>(result));
  return MultiThreadCompileGrammar(std::get<0>(result));
}

CompiledGrammar GrammarCompilerSub::CompileRegex(const std::string& regex) {
  return MultiThreadCompileGrammar(Grammar::FromRegex(regex));
}

CompiledGrammar GrammarCompilerSub::CompileGrammar(const Grammar& grammar) {
  return MultiThreadCompileGrammar(grammar);
}

CompiledGrammar GrammarCompilerSub::CompileGrammar(
    const std::string& ebnf_str, std::string root_rule_name
) {
  return MultiThreadCompileGrammar(Grammar::FromEBNF(ebnf_str, root_rule_name));
}

void GrammarCompilerSub::TagDispatchOptimization(
    std::shared_ptr<CompiledGrammar::Impl> compiled_grammar_impl,
    std::unordered_map<int32_t, DynamicBitset>* tag_dispatch_rule_id_to_second_slicing_bitset
) {
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  tag_dispatch_rule_id_to_second_slicing_bitset->clear();

  // Optimization for TagDispatch: Precompute the definitely accepted tokens.
  for (int i = 0; i < compiled_grammar_impl->grammar->NumRules(); i++) {
    const auto& rule = compiled_grammar_impl->grammar->GetRule(i);
    const auto& rule_body = compiled_grammar_impl->grammar->GetGrammarExpr(rule.body_expr_id);
    if (rule_body.type != GrammarExprType::kTagDispatch) {
      continue;
    }
    XGRAMMAR_DCHECK(rule_body.type == GrammarExprType::kTagDispatch);
    Grammar::Impl::TagDispatch tag_dispatch =
        compiled_grammar_impl->GetGrammar()->GetTagDispatch(rule.body_expr_id);
    std::vector<std::string> patterns;
    patterns.reserve(tag_dispatch.tag_rule_pairs.size() + tag_dispatch.excludes.size());
    for (const auto& [trigger, rule_id] : tag_dispatch.tag_rule_pairs) {
      patterns.push_back(trigger);
    }
    patterns.insert(patterns.end(), tag_dispatch.excludes.begin(), tag_dispatch.excludes.end());
    std::sort(patterns.begin(), patterns.end());
    patterns.erase(std::unique(patterns.begin(), patterns.end()), patterns.end());
    (*tag_dispatch_rule_id_to_second_slicing_bitset)[i] =
        *GetTagDispatchSecondSlicingBitset(std::move(patterns));
  }
}

std::shared_ptr<const DynamicBitset> GrammarCompilerSub::GetTagDispatchSecondSlicingBitset(
    std::vector<std::string> patterns
) {
  {
    std::lock_guard<std::mutex> lock(tag_dispatch_slicing_cache_mutex_);
    auto it = tag_dispatch_slicing_cache_.find(patterns);
    if (it != tag_dispatch_slicing_cache_.end()) {
      return it->second;
    }
  }

  const auto& sorted_decoded_vocab = tokenizer_info_.GetSortedDecodedVocab();
  auto computed = std::make_shared<DynamicBitset>(sorted_decoded_vocab.size());
  for (int32_t index = 0; index < static_cast<int32_t>(sorted_decoded_vocab.size()); ++index) {
    const auto& token = sorted_decoded_vocab[index].second;
    bool definitely_accepted = token.empty();
    if (!definitely_accepted) {
      definitely_accepted =
          std::none_of(patterns.begin(), patterns.end(), [&](const std::string& pattern) {
            return token.find(pattern, 1) != std::string::npos;
          });
    }
    if (definitely_accepted) {
      computed->Set(index);
    }
  }

  constexpr size_t kMaxTagDispatchSlicingCacheEntries = 64;
  std::lock_guard<std::mutex> lock(tag_dispatch_slicing_cache_mutex_);
  auto it = tag_dispatch_slicing_cache_.find(patterns);
  if (it != tag_dispatch_slicing_cache_.end()) {
    return it->second;
  }
  if (tag_dispatch_slicing_cache_.size() < kMaxTagDispatchSlicingCacheEntries) {
    tag_dispatch_slicing_cache_.emplace(std::move(patterns), computed);
  }
  return computed;
}

/******************* GrammarCompiler::Impl *******************/

/*!
 * \brief The keys for the cache. This is defined here instead of inside the GrammarCompiler::Impl
 * class due C++ template specialization and hash specialization rules.
 */
class GrammarCompilerCacheKeys {
 public:
  struct SchemaKey {
    std::string schema;
    bool any_whitespace;
    std::optional<int> indent;
    std::optional<std::pair<std::string, std::string>> separators;
    bool strict_mode;
    std::optional<int> max_whitespace_cnt;
    bool any_order;

    XGRAMMAR_EQUAL_BY_MEMBERS(
        SchemaKey,
        &SchemaKey::schema,
        &SchemaKey::any_whitespace,
        &SchemaKey::indent,
        &SchemaKey::separators,
        &SchemaKey::strict_mode,
        &SchemaKey::max_whitespace_cnt,
        &SchemaKey::any_order
    );
  };

  struct StructuralTagKey {
    std::string structural_tag_json;

    XGRAMMAR_EQUAL_BY_MEMBERS(StructuralTagKey, &StructuralTagKey::structural_tag_json);
  };

  struct GrammarKey {
    std::string ebnf_str;
    std::string root_rule_name;

    XGRAMMAR_EQUAL_BY_MEMBERS(GrammarKey, &GrammarKey::ebnf_str, &GrammarKey::root_rule_name);
  };

  struct RegexKey {
    std::string regex;

    XGRAMMAR_EQUAL_BY_MEMBERS(RegexKey, &RegexKey::regex);
  };

  struct BuiltinJSONGrammarKey {
    XGRAMMAR_EQUAL_BY_MEMBERS_EMPTY(BuiltinJSONGrammarKey);
  };

  using UnionKey =
      std::variant<SchemaKey, StructuralTagKey, GrammarKey, RegexKey, BuiltinJSONGrammarKey>;
};

}  // namespace xgrammar

XGRAMMAR_HASH_BY_MEMBERS(
    xgrammar::GrammarCompilerCacheKeys::SchemaKey,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::schema,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::any_whitespace,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::indent,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::separators,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::strict_mode,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::max_whitespace_cnt,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::any_order
);

XGRAMMAR_HASH_BY_MEMBERS(
    xgrammar::GrammarCompilerCacheKeys::StructuralTagKey,
    &xgrammar::GrammarCompilerCacheKeys::StructuralTagKey::structural_tag_json
);

XGRAMMAR_HASH_BY_MEMBERS(
    xgrammar::GrammarCompilerCacheKeys::GrammarKey,
    &xgrammar::GrammarCompilerCacheKeys::GrammarKey::ebnf_str,
    &xgrammar::GrammarCompilerCacheKeys::GrammarKey::root_rule_name
);

XGRAMMAR_HASH_BY_MEMBERS(
    xgrammar::GrammarCompilerCacheKeys::RegexKey,
    &xgrammar::GrammarCompilerCacheKeys::RegexKey::regex
);

XGRAMMAR_HASH_BY_MEMBERS_EMPTY(xgrammar::GrammarCompilerCacheKeys::BuiltinJSONGrammarKey);

namespace xgrammar {

/*!
 * \brief The implementation of the grammar compiler with cache. It calls the no cache compiler
 * to compile the grammar, and implements the cache logic upon it.
 */
class GrammarCompiler::Impl {
 public:
  Impl(
      const TokenizerInfo& tokenizer_info,
      int max_threads,
      bool cache_enabled,
      int64_t max_memory_bytes,
      bool enable_dynamic_compilation
  )
      : cache_enabled_(cache_enabled && (!enable_dynamic_compilation || max_memory_bytes == -1)),
        rule_level_cache_(
            cache_enabled
                ? std::optional<RuleLevelCache>(
                      max_memory_bytes == -1
                          ? static_cast<std::size_t>(-1)
                          : static_cast<std::size_t>(max_memory_bytes - max_memory_bytes / 3 * 2)
                  )
                : std::nullopt
        ),
        no_cache_compiler_(
            tokenizer_info, max_threads, rule_level_cache_, enable_dynamic_compilation
        ),
        grammar_level_cache_(
            max_memory_bytes == -1 ? static_cast<std::size_t>(-1)
                                   : static_cast<std::size_t>(max_memory_bytes / 3 * 2),
            Computer(*this)
        ) {
    if (max_memory_bytes < -1) {
      XGRAMMAR_LOG(FATAL) << "Invalid max_memory_bytes: " << max_memory_bytes << ". "
                          << "It should be -1 (unlimited) or a non-negative integer.";
    }
  }

  CompiledGrammar CompileBuiltinJSONGrammar();

  CompiledGrammar CompileJSONSchema(
      const std::string& schema,
      bool any_whitespace,
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool strict_mode,
      std::optional<int> max_whitespace_cnt,
      bool any_order
  );

  CompiledGrammar CompileStructuralTag(const std::string& structural_tag_json);

  CompiledGrammar CompileRegex(const std::string& regex);

  CompiledGrammar CompileGrammar(const Grammar& grammar);

  CompiledGrammar CompileGrammar(const std::string& ebnf_str, std::string root_rule_name);

  void ClearCache();

  int64_t GetCacheSizeBytes() const;

  int64_t CacheLimitBytes() const;

 private:
  using SchemaKey = GrammarCompilerCacheKeys::SchemaKey;
  using StructuralTagKey = GrammarCompilerCacheKeys::StructuralTagKey;
  using GrammarKey = GrammarCompilerCacheKeys::GrammarKey;
  using RegexKey = GrammarCompilerCacheKeys::RegexKey;
  using BuiltinJSONGrammarKey = GrammarCompilerCacheKeys::BuiltinJSONGrammarKey;
  using UnionKey = GrammarCompilerCacheKeys::UnionKey;

  CompiledGrammar Compute(const UnionKey& key);

  struct Computer {
    Computer(Impl& compiler) : compiler(compiler) {}
    // Forward the key to GrammarCompiler::Impl::Compute(key)
    CompiledGrammar operator()(const UnionKey& key) const { return compiler.Compute(key); }
    GrammarCompiler::Impl& compiler;
  };

  struct SizeEstimator {
    std::size_t operator()(const CompiledGrammar& value) const { return value.MemorySizeBytes(); }
  };

  /*! \brief Whether the cache is enabled. */
  const bool cache_enabled_;

  /*! \brief The crossing cache manager for compiled grammars. */
  std::optional<RuleLevelCache> rule_level_cache_ = std::nullopt;

  /*! \brief The no cache compiler. */
  GrammarCompilerSub no_cache_compiler_;

  /*! \brief The cache for compiled grammars. */
  ThreadSafeLRUCache<UnionKey, CompiledGrammar, Computer, SizeEstimator> grammar_level_cache_;
};

CompiledGrammar GrammarCompiler::Impl::Compute(const UnionKey& key) {
  return std::visit(
      [this](const auto& key) -> CompiledGrammar {
        using KeyType = std::decay_t<decltype(key)>;
        if constexpr (std::is_same_v<KeyType, GrammarKey>) {
          const auto& [ebnf_str, root_rule_name] = key;
          return this->no_cache_compiler_.CompileGrammar(ebnf_str, root_rule_name);
        } else if constexpr (std::is_same_v<KeyType, SchemaKey>) {
          const auto& [schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, any_order] =
              key;
          return this->no_cache_compiler_.CompileJSONSchema(
              schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, any_order
          );
        } else if constexpr (std::is_same_v<KeyType, StructuralTagKey>) {
          const auto& [structural_tag_json] = key;
          return this->no_cache_compiler_.CompileStructuralTag(structural_tag_json);
        } else if constexpr (std::is_same_v<KeyType, RegexKey>) {
          const auto& [regex] = key;
          return this->no_cache_compiler_.CompileRegex(regex);
        } else if constexpr (std::is_same_v<KeyType, BuiltinJSONGrammarKey>) {
          return this->no_cache_compiler_.CompileBuiltinJSONGrammar();
        } else {
          XGRAMMAR_UNREACHABLE();
        }
      },
      key
  );
}

CompiledGrammar GrammarCompiler::Impl::CompileBuiltinJSONGrammar() {
  if (!cache_enabled_) {
    return no_cache_compiler_.CompileBuiltinJSONGrammar();
  }
  return grammar_level_cache_.Get(BuiltinJSONGrammarKey{});
}

CompiledGrammar GrammarCompiler::Impl::CompileJSONSchema(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    std::optional<int> max_whitespace_cnt,
    bool any_order
) {
  if (!cache_enabled_) {
    return no_cache_compiler_.CompileJSONSchema(
        schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, any_order
    );
  }
  return grammar_level_cache_.Get(SchemaKey{
      schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, any_order
  });
}

CompiledGrammar GrammarCompiler::Impl::CompileStructuralTag(const std::string& structural_tag_json
) {
  if (!cache_enabled_) {
    return no_cache_compiler_.CompileStructuralTag(structural_tag_json);
  }
  return grammar_level_cache_.Get(StructuralTagKey{structural_tag_json});
}

CompiledGrammar GrammarCompiler::Impl::CompileRegex(const std::string& regex) {
  if (!cache_enabled_) {
    return no_cache_compiler_.CompileRegex(regex);
  }
  return grammar_level_cache_.Get(RegexKey{regex});
}

CompiledGrammar GrammarCompiler::Impl::CompileGrammar(const Grammar& grammar) {
  if (!cache_enabled_) {
    return no_cache_compiler_.CompileGrammar(grammar);
  }
  return grammar_level_cache_.Get(GrammarKey{grammar.ToString(), grammar->GetRootRule().name});
}

CompiledGrammar GrammarCompiler::Impl::CompileGrammar(
    const std::string& ebnf_str, std::string root_rule_name
) {
  if (!cache_enabled_) {
    return no_cache_compiler_.CompileGrammar(ebnf_str, root_rule_name);
  }
  return grammar_level_cache_.Get(GrammarKey{ebnf_str, root_rule_name});
}

void GrammarCompiler::Impl::ClearCache() {
  grammar_level_cache_.Clear();
  no_cache_compiler_.ClearCharacterClassTokenSummaryCache();
  no_cache_compiler_.ClearTagDispatchSlicingCache();
  if (rule_level_cache_.has_value()) {
    rule_level_cache_->ClearCache();
  }
}

int64_t GrammarCompiler::Impl::GetCacheSizeBytes() const {
  return static_cast<int64_t>(grammar_level_cache_.MemorySize()) +
         static_cast<int64_t>(MemorySize(rule_level_cache_));
}

int64_t GrammarCompiler::Impl::CacheLimitBytes() const {
  const auto size = grammar_level_cache_.MaxMemorySize();
  if (size == grammar_level_cache_.kUnlimitedSize) return -1;
  return static_cast<int64_t>(size) + (rule_level_cache_.has_value()
                                           ? static_cast<int64_t>(rule_level_cache_->GetMaxSize())
                                           : 0);
}

/******************* GrammarCompiler *******************/

GrammarCompiler::GrammarCompiler(
    const TokenizerInfo& tokenizer_info,
    int max_threads,
    bool cache_enabled,
    int64_t max_memory_bytes
)
    : GrammarCompiler(tokenizer_info, max_threads, cache_enabled, max_memory_bytes, false) {}

GrammarCompiler::GrammarCompiler(
    const TokenizerInfo& tokenizer_info,
    int max_threads,
    bool cache_enabled,
    int64_t max_memory_bytes,
    bool enable_dynamic_compilation
)
    : pimpl_(std::make_shared<Impl>(
          tokenizer_info, max_threads, cache_enabled, max_memory_bytes, enable_dynamic_compilation
      )) {}

CompiledGrammar GrammarCompiler::CompileJSONSchema(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    std::optional<int> max_whitespace_cnt,
    bool any_order
) {
  return pimpl_->CompileJSONSchema(
      schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, any_order
  );
}

CompiledGrammar GrammarCompiler::CompileBuiltinJSONGrammar() {
  return pimpl_->CompileBuiltinJSONGrammar();
}

CompiledGrammar GrammarCompiler::CompileStructuralTag(const std::string& structural_tag_json) {
  return pimpl_->CompileStructuralTag(structural_tag_json);
}

CompiledGrammar GrammarCompiler::CompileRegex(const std::string& regex) {
  return pimpl_->CompileRegex(regex);
}

CompiledGrammar GrammarCompiler::CompileGrammar(const Grammar& grammar) {
  return pimpl_->CompileGrammar(grammar);
}

CompiledGrammar GrammarCompiler::CompileGrammar(
    const std::string& ebnf_str, const std::string& root_rule_name
) {
  return pimpl_->CompileGrammar(ebnf_str, root_rule_name);
}

void GrammarCompiler::ClearCache() { pimpl_->ClearCache(); }

int64_t GrammarCompiler::GetCacheSizeBytes() const { return pimpl_->GetCacheSizeBytes(); }

int64_t GrammarCompiler::CacheLimitBytes() const { return pimpl_->CacheLimitBytes(); }

}  // namespace xgrammar
