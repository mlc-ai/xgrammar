/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/local_completion_token_summary.h
 * \brief Vocabulary classification for repeatable rules terminated by a single delimiter byte.
 */

#ifndef XGRAMMAR_LOCAL_COMPLETION_TOKEN_SUMMARY_H_
#define XGRAMMAR_LOCAL_COMPLETION_TOKEN_SUMMARY_H_

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "compiled_grammar_impl.h"
#include "fsm.h"
#include "grammar_impl.h"
#include "support/dynamic_bitset.h"
#include "tokenizer_info_impl.h"

namespace xgrammar {

// Vocabulary classification at the start of a repeatable, single-delimiter rule.
// Tokens ending exactly at the delimiter are accepted, matching main's mask semantics.
// Tokens with a suffix after it remain uncertain and use the existing runtime parser.
struct LocalCompletionTokenSummary {
  AdaptiveTokenMask mask;
};

namespace local_completion {
struct RuleFSMConfiguration {
  int32_t rule_id;
  int32_t state;
  std::vector<std::pair<int32_t, int32_t>> returns;

  bool operator==(const RuleFSMConfiguration& other) const {
    return rule_id == other.rule_id && state == other.state && returns == other.returns;
  }
};

struct RuleFSMConfigurationHash {
  size_t operator()(const RuleFSMConfiguration& configuration) const {
    size_t result = std::hash<int32_t>{}(configuration.rule_id);
    result ^=
        std::hash<int32_t>{}(configuration.state) + 0x9e3779b9 + (result << 6) + (result >> 2);
    for (const auto& [rule_id, state] : configuration.returns) {
      result ^= std::hash<int32_t>{}(rule_id) + 0x9e3779b9 + (result << 6) + (result >> 2);
      result ^= std::hash<int32_t>{}(state) + 0x9e3779b9 + (result << 6) + (result >> 2);
    }
    return result;
  }
};

inline Result<FSMWithStartEnd> FlattenLocalCompletionFSM(
    const Grammar& grammar, int32_t root_rule_id, int32_t max_num_states
) {
  FSM result_fsm;
  std::vector<int32_t> end_states;
  std::vector<RuleFSMConfiguration> configurations;
  std::unordered_map<RuleFSMConfiguration, int32_t, RuleFSMConfigurationHash> state_by_config;
  std::queue<int32_t> queue;

  auto add_configuration = [&](RuleFSMConfiguration configuration) {
    auto [it, inserted] = state_by_config.emplace(configuration, -1);
    if (!inserted) {
      return it->second;
    }
    if (static_cast<int32_t>(configurations.size()) >= max_num_states) {
      state_by_config.erase(it);
      return FSM::kNoNextState;
    }
    int32_t state = result_fsm.AddState();
    it->second = state;
    configurations.push_back(std::move(configuration));
    queue.push(state);
    return state;
  };

  const auto& root_fsm = grammar->per_rule_fsms.at(root_rule_id)->GetFsm();
  int32_t start_state = add_configuration({root_rule_id, root_fsm.GetStart(), {}});
  XGRAMMAR_DCHECK(start_state == 0);

  while (!queue.empty()) {
    int32_t result_state = queue.front();
    queue.pop();
    RuleFSMConfiguration configuration = configurations[result_state];
    const auto& rule_fsm = grammar->per_rule_fsms.at(configuration.rule_id)->GetFsm();

    if (rule_fsm.IsEndState(configuration.state)) {
      if (configuration.returns.empty()) {
        end_states.push_back(result_state);
      } else {
        auto return_configuration = configuration;
        auto [return_rule_id, return_state] = return_configuration.returns.back();
        return_configuration.returns.pop_back();
        return_configuration.rule_id = return_rule_id;
        return_configuration.state = return_state;
        int32_t target = add_configuration(std::move(return_configuration));
        if (target == FSM::kNoNextState) {
          return ResultErr(
              "The number of states needed to flatten a regular expression exceeds the limit "
              "of " +
              std::to_string(max_num_states)
          );
        }
        result_fsm.AddEpsilonEdge(result_state, target);
      }
    }

    for (const auto& edge : rule_fsm.GetFsm().GetEdges(configuration.state)) {
      RuleFSMConfiguration next = configuration;
      if (edge.IsRuleRef()) {
        int32_t referenced_rule_id = edge.GetRefRuleId();
        if (referenced_rule_id < 0 || referenced_rule_id >= grammar->NumRules() ||
            !grammar->per_rule_fsms.at(referenced_rule_id).has_value()) {
          return ResultErr("invalid rule reference while flattening a regular expression");
        }
        bool is_tail_call =
            rule_fsm.IsEndState(edge.target) && rule_fsm.GetFsm().GetEdges(edge.target).size() == 0;
        if (!is_tail_call) {
          next.returns.push_back({configuration.rule_id, edge.target});
        }
        next.rule_id = referenced_rule_id;
        next.state = grammar->per_rule_fsms.at(referenced_rule_id)->GetFsm().GetStart();
      } else if (edge.IsEpsilon()) {
        next.state = edge.target;
      } else if (edge.IsCharRange()) {
        next.state = edge.target;
      } else {
        return ResultErr("unsupported transition while flattening a regular expression");
      }

      int32_t target = add_configuration(std::move(next));
      if (target == FSM::kNoNextState) {
        return ResultErr(
            "The number of states needed to flatten a regular expression exceeds the limit of " +
            std::to_string(max_num_states)
        );
      }
      if (edge.IsCharRange()) {
        result_fsm.AddEdge(result_state, target, edge.min, edge.max);
      } else {
        result_fsm.AddEpsilonEdge(result_state, target);
      }
    }
  }

  FSMWithStartEnd result(result_fsm, start_state, std::move(end_states));
  return ResultOk(result.SimplifyEpsilon().MergeEquivalentStates());
}
inline std::vector<uint8_t> GetRedundantLookaheadRules(const Grammar& grammar) {
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
inline std::vector<uint8_t> GetPureFSMSemanticsRules(const Grammar& grammar) {
  const auto redundant_lookahead = GetRedundantLookaheadRules(grammar);
  std::vector<uint8_t> result(grammar->NumRules(), true);
  std::vector<std::vector<int32_t>> referencing_rules(grammar->NumRules());
  std::vector<int32_t> impure_rules;
  for (int32_t rule_id = 0; rule_id < grammar->NumRules(); ++rule_id) {
    const auto& rule = grammar->GetRule(rule_id);
    result[rule_id] = rule.max_tokens == -1 && rule.max_chars == -1 && !rule.is_lazy &&
                      !rule.temperature.has_value() && rule.capture_name.empty() &&
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
  // enforced by the existing lookahead adapter when its summary is used. A
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
inline std::shared_ptr<const LocalCompletionTokenSummary> BuildLocalCompletionTokenSummary(
    const Grammar& grammar,
    int32_t rule_id,
    const TokenizerInfo& tokenizer_info,
    bool has_pure_fsm_semantics,
    const std::vector<int32_t>& lcp_with_previous
) {
  if (rule_id < 0 || rule_id >= grammar->NumRules() || tokenizer_info.GetVocabSize() == 0 ||
      !grammar->per_rule_fsms[rule_id].has_value()) {
    return nullptr;
  }
  if (!has_pure_fsm_semantics) {
    return nullptr;
  }

  auto flattened_result = FlattenLocalCompletionFSM(grammar, rule_id, 256);
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
  auto summary = std::make_shared<LocalCompletionTokenSummary>();
  DynamicBitset accepted_prefix_tokens(tokenizer_info.GetVocabSize());
  std::vector<int32_t> crossing_indices;
  crossing_indices.reserve(sorted_vocab.size() / 16);

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
        if (previous_decision_prefix == static_cast<int32_t>(token.size())) {
          accepted_prefix_tokens.Set(sorted_vocab[token_index].first);
        } else {
          crossing_indices.push_back(token_index);
        }
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
        if (previous_decision_prefix == static_cast<int32_t>(token.size())) {
          accepted_prefix_tokens.Set(sorted_vocab[token_index].first);
        } else {
          crossing_indices.push_back(token_index);
        }
        break;
      }
    }
    if (previous_decision == PreviousDecision::kNone && can_reach_end[state]) {
      accepted_prefix_tokens.Set(sorted_vocab[token_index].first);
    }
  }
  if (crossing_indices.empty()) {
    return nullptr;
  }

  // Preserve main's sparse/dense storage choice, including output padding behavior.
  std::vector<int32_t> accepted, rejected;
  size_t crossing = 0;
  for (int32_t i = 0; i < static_cast<int32_t>(sorted_vocab.size()); ++i) {
    if (crossing < crossing_indices.size() && crossing_indices[crossing] == i) {
      ++crossing;
    } else if (accepted_prefix_tokens[sorted_vocab[i].first]) {
      accepted.push_back(i);
    } else {
      rejected.push_back(i);
    }
  }
  summary->mask = AdaptiveTokenMask(
      tokenizer_info.GetVocabSize(), sorted_vocab, accepted, rejected, crossing_indices
  );
  return summary;
}

}  // namespace local_completion

/*!
 * \brief Per-grammar local-completion summaries, built on first use.
 *
 * Most rules of a grammar are never reached by a given generation, and building a summary walks
 * the whole vocabulary, so nothing is computed at compile time. The grammar-wide eligibility
 * analysis runs on the first request and each rule's summary on the first request for that rule.
 * Requests are serialized by a mutex: JIT mask generation calls this from several matchers.
 */
class LocalCompletionSummaryTable {
 public:
  /*! \brief The summary of rule `rule_id`, or nullptr when the rule is not eligible. */
  std::shared_ptr<const LocalCompletionTokenSummary> Get(
      const Grammar& grammar, const TokenizerInfo& tokenizer_info, int32_t rule_id
  ) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!analyzed_) {
      analyzed_ = true;
      const int32_t num_rules = grammar->NumRules();
      summaries_.resize(num_rules);
      built_.assign(num_rules, false);
      // Character budgets are context-dependent and retain the generic mask path.
      enabled_ = true;
      for (int32_t i = 0; i < num_rules; ++i) {
        enabled_ = enabled_ && grammar->GetRule(i).max_chars == -1;
      }
      if (enabled_) {
        pure_ = local_completion::GetPureFSMSemanticsRules(grammar);
      }
    }
    if (!enabled_ || rule_id < 0 || rule_id >= static_cast<int32_t>(summaries_.size()) ||
        rule_id == grammar->GetRootRuleId()) {
      return nullptr;
    }
    if (!built_[rule_id]) {
      built_[rule_id] = true;
      // Reuse the tokenizer-scoped common-prefix table shared with the other direct builders.
      summaries_[rule_id] = local_completion::BuildLocalCompletionTokenSummary(
          grammar,
          rule_id,
          tokenizer_info,
          pure_[rule_id],
          tokenizer_info.ImplPtr()->GetSortedVocabLCPWithPrevious()
      );
    }
    return summaries_[rule_id];
  }

 private:
  std::mutex mutex_;
  bool analyzed_ = false;
  bool enabled_ = false;
  std::vector<uint8_t> pure_;
  std::vector<uint8_t> built_;
  std::vector<std::shared_ptr<const LocalCompletionTokenSummary>> summaries_;
};
}  // namespace xgrammar
#endif  // XGRAMMAR_LOCAL_COMPLETION_TOKEN_SUMMARY_H_
