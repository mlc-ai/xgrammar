/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/earley_parser.cc
 */

#include "earley_parser.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <utility>
#include <vector>

#include "fsm.h"
#include "grammar_impl.h"
#include "support/encoding.h"
#include "support/logging.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

using GrammarExprType = Grammar::Impl::GrammarExprType;

using GrammarExpr = Grammar::Impl::GrammarExpr;

const EarleyParser::UniqueKeySemanticState& EarleyParser::GetUniqueKeySemanticState(int32_t state_id
) const {
  static const UniqueKeySemanticState kEmptyState;
  if (state_id == 0) {
    return kEmptyState;
  }
  XGRAMMAR_DCHECK(unique_key_context_storage_ != nullptr);
  XGRAMMAR_DCHECK(
      state_id > 0 &&
      state_id < static_cast<int32_t>(unique_key_context_storage_->semantic_states.size())
  );
  return unique_key_context_storage_->semantic_states[state_id];
}

const EarleyParser::UniqueKeyContextNode& EarleyParser::GetUniqueKeyContext(int32_t context_id
) const {
  static const UniqueKeyContextNode kRootContext;
  if (context_id == 0) {
    return kRootContext;
  }
  XGRAMMAR_DCHECK(unique_key_context_storage_ != nullptr);
  XGRAMMAR_DCHECK(
      context_id > 0 &&
      context_id < static_cast<int32_t>(unique_key_context_storage_->contexts.size())
  );
  return unique_key_context_storage_->contexts[context_id];
}

void EarleyParser::EnsureUniqueKeyContextStorage() {
  if (unique_key_context_storage_ == nullptr) {
    unique_key_context_storage_ = std::make_shared<UniqueKeyContextStorage>();
  } else if (unique_key_context_storage_.use_count() > 1) {
    unique_key_context_storage_ =
        std::make_shared<UniqueKeyContextStorage>(*unique_key_context_storage_);
  }
}

int32_t EarleyParser::InternUniqueKeySemanticState(
    int32_t current_context_id, int32_t entry_context_id
) {
  if (current_context_id == 0 && entry_context_id == 0) {
    return 0;
  }
  EnsureUniqueKeyContextStorage();
  const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(current_context_id)) << 32) |
                       static_cast<uint32_t>(entry_context_id);
  if (auto it = unique_key_context_storage_->semantic_state_ids.find(key);
      it != unique_key_context_storage_->semantic_state_ids.end()) {
    return it->second;
  }
  XGRAMMAR_CHECK(
      unique_key_context_storage_->semantic_states.size() <
      static_cast<size_t>(std::numeric_limits<int32_t>::max())
  ) << "Too many branch-local DynamicTag uniqueness states";
  const int32_t result = static_cast<int32_t>(unique_key_context_storage_->semantic_states.size());
  unique_key_context_storage_->semantic_states.push_back({current_context_id, entry_context_id});
  unique_key_context_storage_->semantic_state_ids.emplace(key, result);
  return result;
}

int32_t EarleyParser::InternUniqueKeyContext(UniqueKeyContextNode node) {
  node.signature_hash = HashCombine(
      static_cast<uint8_t>(node.kind),
      node.parent_id,
      node.scope_rule_id,
      node.scope_start_pos,
      node.name_start_byte,
      node.byte_length,
      node.nearest_scope_context_id,
      node.name_hash
  );
  EnsureUniqueKeyContextStorage();
  const auto range =
      unique_key_context_storage_->context_ids_by_hash.equal_range(node.signature_hash);
  for (auto it = range.first; it != range.second; ++it) {
    const auto& existing = unique_key_context_storage_->contexts[it->second];
    if (existing.kind == node.kind && existing.parent_id == node.parent_id &&
        existing.scope_rule_id == node.scope_rule_id &&
        existing.scope_start_pos == node.scope_start_pos &&
        existing.name_start_byte == node.name_start_byte &&
        existing.byte_length == node.byte_length &&
        existing.nearest_scope_context_id == node.nearest_scope_context_id &&
        existing.name_hash == node.name_hash) {
      return it->second;
    }
  }
  XGRAMMAR_CHECK(
      unique_key_context_storage_->contexts.size() <
      static_cast<size_t>(std::numeric_limits<int32_t>::max())
  ) << "Too many branch-local DynamicTag uniqueness contexts";
  const int32_t result = static_cast<int32_t>(unique_key_context_storage_->contexts.size());
  unique_key_context_storage_->contexts.push_back(std::move(node));
  unique_key_context_storage_->context_ids_by_hash.emplace(
      unique_key_context_storage_->contexts.back().signature_hash, result
  );
  if (unique_key_context_storage_->contexts.back().kind == UniqueKeyContextNode::Kind::kKey) {
    const auto& context = unique_key_context_storage_->contexts.back();
    unique_key_context_storage_->key_context_ids_by_scope_and_name_hash.emplace(
        HashCombine(context.nearest_scope_context_id, context.name_hash), result
    );
  }
  return result;
}

int32_t EarleyParser::MakeChildUniqueKeyState(
    int32_t parent_state_id, int32_t child_rule_id, int32_t child_start_pos
) {
  if (!track_unique_key_contexts_) {
    return parent_state_id;
  }
  const auto parent = GetUniqueKeySemanticState(parent_state_id);
  int32_t child_context_id = parent.current_context_id;
  if (IsUniqueKeyScopeRule(child_rule_id)) {
    UniqueKeyContextNode scope;
    scope.kind = UniqueKeyContextNode::Kind::kScope;
    scope.parent_id = child_context_id;
    scope.scope_rule_id = child_rule_id;
    scope.scope_start_pos = child_start_pos;
    child_context_id = InternUniqueKeyContext(std::move(scope));
  } else if (parent.current_context_id == parent.entry_context_id) {
    return parent_state_id;
  }
  return InternUniqueKeySemanticState(child_context_id, parent.current_context_id);
}

std::optional<int32_t> EarleyParser::CompleteUniqueKeyState(
    int32_t parent_state_id, const ParserState& completed_state
) {
  if (!track_unique_key_contexts_) {
    return parent_state_id;
  }
  const auto parent = GetUniqueKeySemanticState(parent_state_id);
  const auto child = GetUniqueKeySemanticState(completed_state.unique_key_state_id);
  if (parent.current_context_id != child.entry_context_id) {
    return std::nullopt;
  }
  int32_t completed_context_id = child.current_context_id;
  if (IsUniqueKeyScopeRule(completed_state.rule_id)) {
    const auto& current = GetUniqueKeyContext(completed_context_id);
    if (current.kind == UniqueKeyContextNode::Kind::kKey) {
      completed_context_id = current.nearest_scope_context_id;
    }
    const auto& scope = GetUniqueKeyContext(completed_context_id);
    if (scope.kind != UniqueKeyContextNode::Kind::kScope ||
        scope.scope_rule_id != completed_state.rule_id ||
        scope.scope_start_pos != completed_state.rule_start_pos) {
      return std::nullopt;
    }
    completed_context_id = scope.parent_id;
  }
  if (completed_context_id == parent.current_context_id) {
    return parent_state_id;
  }
  if (completed_context_id == child.current_context_id &&
      parent.entry_context_id == child.entry_context_id) {
    return completed_state.unique_key_state_id;
  }
  return InternUniqueKeySemanticState(completed_context_id, parent.entry_context_id);
}

int32_t EarleyParser::AppendUniqueKey(
    int32_t state_id, int32_t name_start_byte, int32_t byte_length, uint64_t name_hash
) {
  XGRAMMAR_DCHECK(track_unique_key_contexts_);
  const auto semantic_state = GetUniqueKeySemanticState(state_id);
  const auto& previous = GetUniqueKeyContext(semantic_state.current_context_id);
  XGRAMMAR_DCHECK(
      previous.kind == UniqueKeyContextNode::Kind::kScope ||
      previous.kind == UniqueKeyContextNode::Kind::kKey
  );
  UniqueKeyContextNode key;
  key.kind = UniqueKeyContextNode::Kind::kKey;
  key.parent_id = semantic_state.current_context_id;
  key.name_start_byte = name_start_byte;
  key.byte_length = byte_length;
  key.nearest_scope_context_id = previous.kind == UniqueKeyContextNode::Kind::kScope
                                     ? semantic_state.current_context_id
                                     : previous.nearest_scope_context_id;
  XGRAMMAR_DCHECK(key.nearest_scope_context_id > 0);
  key.scope_key_depth =
      previous.kind == UniqueKeyContextNode::Kind::kScope ? 1 : previous.scope_key_depth + 1;
  key.name_hash = name_hash;
  const uint64_t bloom_bit_1 = uint64_t{1} << (name_hash & 63);
  const uint64_t bloom_bit_2 = uint64_t{1} << ((name_hash >> 32) & 63);
  key.name_bloom = previous.name_bloom | bloom_bit_1 | bloom_bit_2;
  const int32_t key_context_id = InternUniqueKeyContext(std::move(key));
  return InternUniqueKeySemanticState(key_context_id, semantic_state.entry_context_id);
}

int32_t EarleyParser::RemapUniqueKeyStateRows(
    int32_t state_id, int32_t from_row, int32_t to_row, UniqueKeyRowRemapCache* remap_cache
) {
  if (state_id == 0 || from_row == to_row) {
    return state_id;
  }
  XGRAMMAR_DCHECK(remap_cache != nullptr);
  auto& remapped_context_ids = remap_cache->context_ids;
  auto& remapped_semantic_state_ids = remap_cache->semantic_state_ids;
  auto& context_path = remap_cache->context_path;
  if (auto it = remapped_semantic_state_ids.find(state_id);
      it != remapped_semantic_state_ids.end()) {
    return it->second;
  }
  if (remapped_context_ids.empty()) {
    remapped_context_ids.emplace(0, 0);
  }

  const auto semantic_state = GetUniqueKeySemanticState(state_id);
  auto remap_context = [&](int32_t context_id) {
    context_path.clear();
    auto remapped_it = remapped_context_ids.find(context_id);
    while (remapped_it == remapped_context_ids.end()) {
      context_path.push_back(context_id);
      context_id = GetUniqueKeyContext(context_id).parent_id;
      remapped_it = remapped_context_ids.find(context_id);
    }

    int32_t remapped_id = remapped_it->second;
    while (!context_path.empty()) {
      const int32_t original_id = context_path.back();
      context_path.pop_back();
      auto context = GetUniqueKeyContext(original_id);
      auto parent_it = remapped_context_ids.find(context.parent_id);
      XGRAMMAR_DCHECK(parent_it != remapped_context_ids.end());
      context.parent_id = parent_it->second;
      if (context.kind == UniqueKeyContextNode::Kind::kKey) {
        auto scope_it = remapped_context_ids.find(context.nearest_scope_context_id);
        XGRAMMAR_DCHECK(scope_it != remapped_context_ids.end());
        context.nearest_scope_context_id = scope_it->second;
      }
      if (context.kind == UniqueKeyContextNode::Kind::kScope &&
          context.scope_start_pos == from_row) {
        context.scope_start_pos = to_row;
      }
      remapped_id = InternUniqueKeyContext(std::move(context));
      remapped_context_ids.emplace(original_id, remapped_id);
    }
    return remapped_id;
  };

  const int32_t result = InternUniqueKeySemanticState(
      remap_context(semantic_state.current_context_id),
      remap_context(semantic_state.entry_context_id)
  );
  remapped_semantic_state_ids.emplace(state_id, result);
  return result;
}

EarleyParser::UniqueKeyContextSnapshot EarleyParser::SnapshotUniqueKeyContexts() const {
  if (unique_key_context_storage_ == nullptr) {
    return {};
  }
  return {
      unique_key_context_storage_->contexts.size(),
      unique_key_context_storage_->semantic_states.size()
  };
}

void EarleyParser::RestoreUniqueKeyContexts(const UniqueKeyContextSnapshot& snapshot) {
  if (unique_key_context_storage_ == nullptr) {
    return;
  }
  XGRAMMAR_DCHECK(snapshot.num_contexts >= 1 && snapshot.num_semantic_states >= 1);
  XGRAMMAR_DCHECK(snapshot.num_contexts <= unique_key_context_storage_->contexts.size());
  XGRAMMAR_DCHECK(
      snapshot.num_semantic_states <= unique_key_context_storage_->semantic_states.size()
  );
  if (snapshot.num_contexts == unique_key_context_storage_->contexts.size() &&
      snapshot.num_semantic_states == unique_key_context_storage_->semantic_states.size()) {
    return;
  }
  EnsureUniqueKeyContextStorage();
  for (size_t index = unique_key_context_storage_->semantic_states.size();
       index > snapshot.num_semantic_states;
       --index) {
    const auto& state = unique_key_context_storage_->semantic_states[index - 1];
    const uint64_t key =
        (static_cast<uint64_t>(static_cast<uint32_t>(state.current_context_id)) << 32) |
        static_cast<uint32_t>(state.entry_context_id);
    unique_key_context_storage_->semantic_state_ids.erase(key);
  }
  unique_key_context_storage_->semantic_states.resize(snapshot.num_semantic_states);
  for (size_t index = unique_key_context_storage_->contexts.size(); index > snapshot.num_contexts;
       --index) {
    const auto& context = unique_key_context_storage_->contexts[index - 1];
    if (context.kind == UniqueKeyContextNode::Kind::kKey) {
      const auto name_range =
          unique_key_context_storage_->key_context_ids_by_scope_and_name_hash.equal_range(
              HashCombine(context.nearest_scope_context_id, context.name_hash)
          );
      bool found = false;
      for (auto it = name_range.first; it != name_range.second; ++it) {
        if (it->second == static_cast<int32_t>(index - 1)) {
          unique_key_context_storage_->key_context_ids_by_scope_and_name_hash.erase(it);
          found = true;
          break;
        }
      }
      XGRAMMAR_CHECK(found) << "DynamicTag unique-key index is inconsistent";
    }
    const auto range =
        unique_key_context_storage_->context_ids_by_hash.equal_range(context.signature_hash);
    for (auto it = range.first; it != range.second; ++it) {
      if (it->second == static_cast<int32_t>(index - 1)) {
        unique_key_context_storage_->context_ids_by_hash.erase(it);
        break;
      }
    }
  }
  unique_key_context_storage_->contexts.resize(snapshot.num_contexts);
}

bool EarleyParser::IsCompleted() const { return is_completed_.back(); }

bool EarleyParser::CompletionConsumedMarker(const ParserState& state) const {
  const auto& body = grammar_->GetGrammarExpr(grammar_->GetRule(state.rule_id).body_expr_id);
  if (body.type != GrammarExprType::kTagDispatch) {
    return true;
  }
  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[state.rule_id].has_value());
  const auto& fsm = grammar_->per_rule_fsms[state.rule_id]->GetFsm().GetFsm();
  return fsm.GetEdges(state.element_id).size() == 0;
}

std::vector<CaptureOccurrence> EarleyParser::CollectStopCaptureTargets(const ParserState& state
) const {
  // Follow only the parent links of this concrete rule occurrence. A byte-overlap test at
  // materialization time cannot distinguish an actual captured ancestor from an unrelated
  // Earley branch that happens to cover the same input.
  std::vector<CaptureOccurrence> targets;
  std::vector<CaptureOccurrence> pending{{state.rule_id, state.rule_start_pos}};
  std::unordered_set<int64_t> visited;
  while (!pending.empty()) {
    CaptureOccurrence occurrence = pending.back();
    pending.pop_back();
    int64_t occurrence_key = (static_cast<int64_t>(occurrence.rule_id) << 32) |
                             static_cast<uint32_t>(occurrence.start_pos);
    if (!visited.insert(occurrence_key).second) {
      continue;
    }
    if (RuleHasCapture(occurrence.rule_id)) {
      targets.push_back(occurrence);
    }
    if (occurrence.start_pos == ParserState::kNoPrevInputPos) {
      continue;
    }
    const auto& parent_states = rule_id_to_completable_states_[occurrence.start_pos];
    for (const auto& [ref_rule_id, parent_state] : parent_states) {
      if (ref_rule_id != occurrence.rule_id || parent_state.rule_id < 0) {
        continue;
      }
      pending.push_back({parent_state.rule_id, parent_state.rule_start_pos});
    }
  }
  return targets;
}

void EarleyParser::RecordCaptureEvent(const ParserState& state, bool marker_present) {
  const auto* suffix_stop_info = grammar_->GetSuffixStopInfo(state.rule_id);
  bool marker_consumed =
      marker_present && suffix_stop_info != nullptr &&
      (suffix_stop_info->hidden_suffix_bytes > 0 || suffix_stop_info->hidden_stop_bytes > 0) &&
      CompletionConsumedMarker(state);
  const int32_t hidden_suffix_bytes = marker_consumed ? suffix_stop_info->hidden_suffix_bytes : 0;
  const int32_t hidden_stop_bytes = marker_consumed ? suffix_stop_info->hidden_stop_bytes : 0;

  int32_t event_start_pos = state.rule_start_pos;
  if (marker_consumed && suffix_stop_info->body_rule_id == state.rule_id) {
    // A self-referencing body helper marks the zero-width event inserted immediately after a
    // dynamic string trigger. Its capture span is the fixed-length marker that precedes it.
    XGRAMMAR_DCHECK(event_start_pos != ParserState::kNoPrevInputPos);
    event_start_pos -= std::max(hidden_suffix_bytes, hidden_stop_bytes);
    XGRAMMAR_DCHECK(event_start_pos >= 0);
  }

  std::vector<CaptureOccurrence> stop_capture_targets =
      hidden_stop_bytes > 0 ? CollectStopCaptureTargets(state) : std::vector<CaptureOccurrence>{};

  capture_event_history_.PushBackInLatestRow(
      {state.rule_id,
       event_start_pos,
       state.rule_start_pos,
       hidden_suffix_bytes,
       hidden_stop_bytes,
       std::move(stop_capture_targets)}
  );
}

int32_t EarleyParser::ResolveActiveTemperatureRule(int32_t rule_id, int32_t inherited_rule_id)
    const {
  return grammar_->GetRule(rule_id).temperature.has_value() ? rule_id : inherited_rule_id;
}

void EarleyParser::PopLastStates(int32_t cnt) {
  stop_token_is_accepted_ = false;
  if (cnt >= static_cast<int32_t>(rule_id_to_completable_states_.size())) {
    XGRAMMAR_LOG(FATAL) << "The number of states to be popped is larger than the size of states.";
  }
  rule_id_to_completable_states_.PopBack(cnt);
  is_completed_.erase(is_completed_.end() - cnt, is_completed_.end());
  scanable_state_history_.PopBack(cnt);
  if (capture_tracking_) {
    capture_event_history_.PopBack(cnt);
  }
  if (has_char_budget_rules_) {
    char_count_history_.erase(char_count_history_.end() - cnt, char_count_history_.end());
    char_budget_entry_history_.erase(
        char_budget_entry_history_.end() - cnt, char_budget_entry_history_.end()
    );
  }
}

void EarleyParser::Complete(const ParserState& state, bool debug_print, bool marker_present) {
  // Record capture and hidden-span events. This is only enabled during definitive advances;
  // speculative completions (mask computation, lookahead) never record events.
  if (capture_recording_ && RuleNeedsCaptureEvent(state.rule_id)) {
    RecordCaptureEvent(state, marker_present);
  }
  if (state.rule_id != -1 && grammar_->GetRule(state.rule_id).is_lazy) {
    tmp_completed_lazy_occurrences_.emplace_back(state.rule_id, state.rule_start_pos);
  }
  // Check if a rule is completed.
  if (state.rule_start_pos == ParserState::kNoPrevInputPos) {
    // assert: if a root rule can achieve here, then it must be completed.
    if (debug_print) {
      XGRAMMAR_LOG(INFO) << "The root rule is completed.";
    }
    tmp_accept_stop_token_ = true;
    return;
  }
  if (debug_print) {
    XGRAMMAR_LOG(INFO) << "The rule " << state.rule_id << ": "
                       << grammar_->GetRule(state.rule_id).name
                       << " is completed, trying to complete its parent states.";
  }

  // Check all the possible parent states.
  const auto& parent_states_map = rule_id_to_completable_states_[state.rule_start_pos];
  for (const auto& [ref_id, parent_state] : parent_states_map) {
    if (ref_id != state.rule_id) {
      continue;
    }
    auto completed_parent = parent_state;
    if (track_unique_key_contexts_) {
      auto completed_unique_key_state =
          CompleteUniqueKeyState(parent_state.unique_key_state_id, state);
      if (!completed_unique_key_state.has_value()) {
        continue;
      }
      completed_parent.unique_key_state_id = *completed_unique_key_state;
    }
    XGRAMMAR_DCHECK(
        completed_parent.rule_id == -1 ||
        grammar_->per_rule_fsms[completed_parent.rule_id].has_value()
    );
    if (completed_parent.rule_id == -1) {
      const auto& parent_expr = grammar_->GetGrammarExpr(completed_parent.sequence_id);
      const auto& element_expr = grammar_->GetGrammarExpr(parent_expr[completed_parent.element_id]);
      // The new rule is not referenced by a fsm.
      XGRAMMAR_DCHECK(
          element_expr.type == GrammarExprType::kRuleRef ||
          element_expr.type == GrammarExprType::kRepeat
      );
      if (element_expr.type == GrammarExprType::kRuleRef) {
        completed_parent.element_id++;
        completed_parent.repeat_count = 0;
        Enqueue(std::move(completed_parent));
        continue;
      }
      XGRAMMAR_DCHECK(element_expr.type == GrammarExprType::kRepeat);
      // The parent state is a repeat, we need to increase the repeat count.
      auto new_state = completed_parent;
      const int32_t& min_repeat_count = element_expr[1];
      const int32_t& max_repeat_count = element_expr[2];
      new_state.repeat_count++;
      // The repeat rule can be completed, and we advance the state. Don't forget to
      // reset the repeat count.
      if (new_state.repeat_count >= min_repeat_count) {
        auto completed_repeat = completed_parent;
        completed_repeat.element_id++;
        completed_repeat.repeat_count = 0;
        Enqueue(std::move(completed_repeat));
      }
      // If the repeat count is less than the max repeat count, we can continue to
      // visit the repeat state for another round.
      if (new_state.repeat_count < max_repeat_count) {
        Enqueue(new_state);
      }
      continue;
    }
    // If the rule is referenced by a fsm, we need to advance the fsm.
    XGRAMMAR_DCHECK(grammar_->per_rule_fsms[completed_parent.rule_id].has_value());

    // Check if the parent_state sits on a kRepeatRef edge
    bool handled_as_repeat = false;
    const auto& parent_fsm = grammar_->per_rule_fsms[completed_parent.rule_id].value();
    for (const auto& edge : parent_fsm.GetFsm().GetFsm().GetEdges(completed_parent.element_id)) {
      // Because of invariance, a state with a kRepeatRef edge has exactly one outgoing edge.
      if (!edge.IsRepeatRef()) continue;
      auto info = grammar_->complete_fsm.GetRepeatEdgeInfo(edge.GetAuxIndex());
      if (info.RuleId() != ref_id) continue;
      handled_as_repeat = true;
      int32_t new_count = completed_parent.repeat_count + 1;
      if (new_count >= info.Lower()) {
        auto completed_repeat = completed_parent;
        completed_repeat.element_id = edge.target;
        completed_repeat.repeat_count = 0;
        Enqueue(std::move(completed_repeat));
      }
      if (new_count < info.Upper()) {
        auto continued_repeat = completed_parent;
        continued_repeat.repeat_count = new_count;
        Enqueue(std::move(continued_repeat));
      }
      break;
    }
    if (!handled_as_repeat) {
      Enqueue(std::move(completed_parent));
    }
  }
}

std::pair</* scanable */ bool, /* completable */ bool> EarleyParser::Predict(
    const ParserState& state, bool debug_print
) {
  // Check if the rule has a corresponding FSM.
  if (state.rule_id != -1) {
    XGRAMMAR_DCHECK(grammar_->per_rule_fsms[state.rule_id].has_value());
    const uint8_t flags = GetFsmStateFlags(state.rule_id, state.element_id);
    if (flags & kFsmStateNonTerminal) {
      ExpandNextRuleRefElementOnFSM(state, debug_print);
    }
    return std::make_pair(flags & kFsmStateScanable, flags & kFsmStateEnd);
  }
  const GrammarExpr& grammar_expr = grammar_->GetGrammarExpr(state.sequence_id);
  XGRAMMAR_DCHECK(
      grammar_expr.type == GrammarExprType::kSequence ||
      grammar_expr.type == GrammarExprType::kEmptyStr
  );
  if (state.element_id == grammar_expr.size()) {
    // The rule is completed.
    return std::make_pair(false, true);
  }
  const auto& element_expr = grammar_->GetGrammarExpr(grammar_expr[state.element_id]);
  switch (element_expr.type) {
    case GrammarExprType::kRuleRef: {
      ExpandNextRuleRefElement(state, grammar_expr, &element_expr, debug_print);
      return std::make_pair(false, false);
    }
    case GrammarExprType::kCharacterClassStar: {
      if (state.sub_element_id == 0) {
        auto completed_star = state;
        completed_star.element_id++;
        completed_star.sub_element_id = 0;
        completed_star.repeat_count = 0;
        completed_star.partial_codepoint = 0;
        Enqueue(std::move(completed_star));
      }
      return std::make_pair(true, false);
    }
    case GrammarExprType::kRepeat: {
      const int32_t& min_repeat_count = element_expr[1];
      const int32_t& max_repeat_count = element_expr[2];
      // If the current repeat count is less than the max repeat count,
      // we can expand the next rule reference element.
      XGRAMMAR_DCHECK(state.repeat_count <= max_repeat_count);
      ExpandNextRuleRefElement(state, grammar_expr, &element_expr, debug_print);
      if (state.repeat_count >= min_repeat_count) {
        auto completed_repeat = state;
        completed_repeat.element_id++;
        completed_repeat.sub_element_id = 0;
        completed_repeat.repeat_count = 0;
        completed_repeat.partial_codepoint = 0;
        Enqueue(std::move(completed_repeat));
      }
      return std::make_pair(false, false);
    }
    case GrammarExprType::kByteString:
    case GrammarExprType::kCharacterClass: {
      return std::make_pair(true, false);  // The element is scanable, but not completable.
    }
    case GrammarExprType::kToken:
    case GrammarExprType::kExcludeToken: {
      return std::make_pair(false, false);
    }
    default: {
      XGRAMMAR_LOG(FATAL) << "The element type is not supported! The type is: "
                          << int(element_expr.type);
      XGRAMMAR_UNREACHABLE();
    }
  }
}

void EarleyParser::Scan(const ParserState& state, const uint8_t ch) {
  XGRAMMAR_DCHECK(state.rule_id == -1 || grammar_->per_rule_fsms[state.rule_id].has_value());
  if (state.rule_id == -1) {
    const auto& cur_rule = grammar_->GetGrammarExpr(state.sequence_id);
    const auto& element_expr = grammar_->GetGrammarExpr(cur_rule[state.element_id]);
    // The element is a rule reference, we do not need to scan it.
    switch (element_expr.type) {
      case (GrammarExprType::kByteString): {
        AdvanceByteString(state, ch, element_expr);
        break;
      }
      case (GrammarExprType::kCharacterClass): {
        AdvanceCharacterClass(state, ch, element_expr);
        break;
      }
      case (GrammarExprType::kCharacterClassStar): {
        AdvanceCharacterClassStar(state, ch, element_expr);
        break;
      }
      default: {
        XGRAMMAR_LOG(FATAL) << "The element type is not supported! The type is: "
                            << int(element_expr.type);
        XGRAMMAR_UNREACHABLE();
      }
    }
  } else {
    AdvanceFsm(state, ch);
  }
}

/*!
  \note The workflow of Advance is as follows:
  1. Scan all the states in the latest states. Add all the possible states
  to the next states.
  2. If the next states are empty, then the character is not accepted.
  3. If the next states are not empty, then the character is accepted. Moreover,
  we need to complete and predict the next states.

  \note Thus, when initializing the Earley parser, we need to add the initial state
  to the history_states[0], and perform prediction and completion on the initial state.
*/
bool EarleyParser::Advance(const uint8_t ch, bool debug_print) {
  // Initialize the containers.
  XGRAMMAR_DCHECK(tmp_process_state_queue_.empty())
      << "The tmp_process_state_queue_ should be empty before the scan.";
  tmp_states_visited_in_queue_.Clear();
  tmp_states_to_be_added_.clear();
  tmp_accept_stop_token_ = false;
  tmp_completed_lazy_occurrences_.clear();
  if (has_char_budget_rules_) {
    tmp_char_budget_entered_ = char_budget_entry_history_.back();
    char_count_history_.push_back(GetCurrentCharIndex() + StartsUTF8Codepoint(ch));
  }
  const auto& latest_states = scanable_state_history_[scanable_state_history_.size() - 1];
  // Scan all the scanable states.
  for (const auto& state : latest_states) {
    if (skip_expired_states_ && IsExpiredState(state)) {
      continue;
    }
    Scan(state, ch);
  }

  // Check if the character is accepted.
  if (tmp_process_state_queue_.empty() && tmp_states_to_be_added_.empty()) {
    if (has_char_budget_rules_) {
      char_count_history_.pop_back();
    }
    return false;
  }

  // execute Predict and Complete for all states in the queue until empty.
  rule_id_to_completable_states_.PushBack(std::vector<std::pair<int32_t, ParserState>>());
  if (capture_tracking_) {
    capture_event_history_.PushBack(std::vector<CaptureEvent>());
  }
  while (!tmp_process_state_queue_.empty()) {
    const auto state = std::move(tmp_process_state_queue_.front());
    tmp_process_state_queue_.pop();
    auto [scanable, completable] = Predict(state, debug_print);
    if (completable) {
      Complete(state, debug_print);
    }
    if (scanable) {
      tmp_states_to_be_added_.push_back(state);
    }
  }

  // Check if the grammar is completed, and add the scannable states to the history.
  if (!tmp_completed_lazy_occurrences_.empty()) {
    RemoveCommittedLazyStates();
  }
  if (!unique_key_scope_rules_.empty() && tmp_states_to_be_added_.empty() &&
      !tmp_accept_stop_token_) {
    rule_id_to_completable_states_.PopBack(1);
    if (capture_tracking_) {
      capture_event_history_.PopBack(1);
    }
    if (has_char_budget_rules_) {
      char_count_history_.pop_back();
    }
    return false;
  }
  is_completed_.push_back(tmp_accept_stop_token_);
  scanable_state_history_.PushBack(tmp_states_to_be_added_);
  if (has_char_budget_rules_) {
    char_budget_entry_history_.push_back(tmp_char_budget_entered_);
  }
  return true;
}

void EarleyParser::RemoveCommittedLazyStates() {
  auto is_committed = [&](const ParserState& state) {
    for (const auto& [rule_id, rule_start_pos] : tmp_completed_lazy_occurrences_) {
      if (state.rule_id == rule_id && state.rule_start_pos == rule_start_pos) {
        return true;
      }
    }
    return false;
  };
  tmp_states_to_be_added_.erase(
      std::remove_if(tmp_states_to_be_added_.begin(), tmp_states_to_be_added_.end(), is_committed),
      tmp_states_to_be_added_.end()
  );
}

EarleyParser::EarleyParser(
    const Grammar& grammar, std::optional<ParserState> initial_state, bool track_unique_key_contexts
)
    : grammar_(grammar),
      fsm_state_flags_cache_(grammar->NumRules()),
      rule_is_nullable_(grammar->NumRules(), 0),
      track_unique_key_contexts_(track_unique_key_contexts) {
  if (!grammar->optimized) {
    XGRAMMAR_LOG(FATAL) << "The grammar is not optimized. Please optimize the grammar before using "
                           "the Earley parser.";
  }
  for (int32_t i = 0; i < grammar_->NumRules(); ++i) {
    const auto& rule = grammar_->GetRule(i);
    has_budget_rules_ = has_budget_rules_ || rule.max_tokens >= 0;
    has_char_budget_rules_ = has_char_budget_rules_ || rule.max_chars >= 0;
    const auto* suffix_stop_info = grammar_->GetSuffixStopInfo(i);
    capture_tracking_ =
        capture_tracking_ || !rule.capture_name.empty() ||
        (suffix_stop_info != nullptr && !suffix_stop_info->stop_capture_name.empty());
    has_hidden_capture_rules_ =
        has_hidden_capture_rules_ ||
        (suffix_stop_info != nullptr &&
         (suffix_stop_info->hidden_suffix_bytes > 0 || suffix_stop_info->hidden_stop_bytes > 0));
    if (IsDynamicTagRule(i)) {
      has_dynamic_tag_rules_ = true;
      const auto body = grammar_->GetGrammarExpr(rule.body_expr_id);
      int32_t scope_rule_id = grammar_->GetDynamicTagUniqueKeyScopeRuleId(body);
      if (scope_rule_id >= 0) {
        XGRAMMAR_DCHECK(scope_rule_id < grammar_->NumRules());
        if (unique_key_scope_rules_.empty()) {
          unique_key_scope_rules_.assign(grammar_->NumRules(), 0);
          dynamic_tag_unique_key_scope_rule_ids_.assign(grammar_->NumRules(), -1);
        }
        unique_key_scope_rules_[scope_rule_id] = 1;
        dynamic_tag_unique_key_scope_rule_ids_[i] = scope_rule_id;
      }
    }
  }
  track_unique_key_contexts_ = track_unique_key_contexts_ && !unique_key_scope_rules_.empty();
  for (int32_t rule_id : grammar_->allow_empty_rule_ids) {
    rule_is_nullable_[rule_id] = true;
  }
  PushStateAndExpand(initial_state.has_value() ? *initial_state : RootInitialState());
}

uint8_t EarleyParser::InitializeFsmStateFlags(int32_t rule_id, int32_t state_id) {
  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[rule_id].has_value());
  const auto& fsm = grammar_->per_rule_fsms[rule_id]->GetFsm();
  auto& flags_cache = fsm_state_flags_cache_[rule_id];
  if (flags_cache.empty()) {
    flags_cache.resize(fsm.NumStates());
  }
  XGRAMMAR_DCHECK(state_id >= 0 && state_id < static_cast<int32_t>(flags_cache.size()));
  uint8_t& flags = flags_cache[state_id];
  if (flags & kFsmStateInitialized) {
    return flags;
  }

  flags = kFsmStateInitialized;
  if (fsm.IsEndState(state_id)) {
    flags |= kFsmStateEnd;
  }
  const auto& edges = fsm.GetFsm().GetEdges(state_id);
  if (edges.size() != 0) {
    flags |= kFsmStateHasEdges;
  }
  if (!has_dynamic_tag_rules_) {
    for (const auto& edge : edges) {
      if (edge.IsCharRange() || edge.IsToken() || edge.IsExcludeToken()) {
        flags |= kFsmStateScanable;
      } else if (edge.IsRuleRef() || edge.IsEpsilon() || edge.IsRepeatRef()) {
        flags |= kFsmStateNonTerminal;
      }
    }
    return flags;
  }
  for (const auto& edge : edges) {
    if (edge.IsCharRange() || edge.IsToken() || edge.IsExcludeToken() || edge.IsBackReference()) {
      flags |= kFsmStateScanable;
    } else if (edge.IsRuleRef() || edge.IsCaptureStart() || edge.IsCaptureEnd() ||
               edge.IsEpsilon() || edge.IsRepeatRef()) {
      flags |= kFsmStateNonTerminal;
    }
  }
  return flags;
}

ParserState EarleyParser::RootInitialState() {
  const auto root_rule_id = grammar_->GetRootRuleId();
  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[root_rule_id].has_value());
  ParserState result(
      root_rule_id,
      grammar_->GetRule(root_rule_id).body_expr_id,
      grammar_->per_rule_fsms[root_rule_id]->GetFsm().GetStart(),
      ParserState::kNoPrevInputPos,
      DeadlineForRule(root_rule_id, -1),
      0,
      0,
      0,
      ResolveActiveTemperatureRule(root_rule_id, -1),
      CharDeadlineForRule(root_rule_id, -1)
  );
  if (track_unique_key_contexts_) {
    result.unique_key_state_id = MakeChildUniqueKeyState(
        /*parent_state_id=*/0, root_rule_id, ParserState::kNoPrevInputPos
    );
  }
  return result;
}

void EarleyParser::PushStateAndExpand(const ParserState& state) {
  tmp_states_visited_in_queue_.Clear();
  tmp_accept_stop_token_ = false;
  tmp_states_to_be_added_.clear();
  tmp_completed_lazy_occurrences_.clear();
  Enqueue(state);
  rule_id_to_completable_states_.PushBack(std::vector<std::pair<int32_t, ParserState>>());
  if (capture_tracking_) {
    capture_event_history_.PushBack(std::vector<CaptureEvent>());
  }
  while (!tmp_process_state_queue_.empty()) {
    const auto state = tmp_process_state_queue_.front();
    tmp_process_state_queue_.pop();
    auto [scanable, completable] = Predict(state);
    if (completable) {
      Complete(state);
    }
    if (scanable) {
      tmp_states_to_be_added_.push_back(state);
    }
  }
  if (!tmp_completed_lazy_occurrences_.empty()) {
    RemoveCommittedLazyStates();
  }
  is_completed_.push_back(tmp_accept_stop_token_);
  scanable_state_history_.PushBack(tmp_states_to_be_added_);
  if (has_char_budget_rules_) {
    char_count_history_.push_back(GetCurrentCharIndex());
    char_budget_entry_history_.push_back(tmp_char_budget_entered_);
  }
}

void EarleyParser::Reset() {
  rule_id_to_completable_states_.PopBack(rule_id_to_completable_states_.size());
  scanable_state_history_.PopBack(scanable_state_history_.size());
  is_completed_.clear();
  stop_token_is_accepted_ = false;
  if (capture_tracking_) {
    capture_event_history_.PopBack(capture_event_history_.size());
  }
  char_count_history_.clear();
  char_budget_entry_history_.clear();
  tmp_char_budget_entered_ = false;
  capture_recording_ = false;
  RestoreUniqueKeyContexts({});
  XGRAMMAR_DCHECK(tmp_process_state_queue_.empty());
  PushStateAndExpand(RootInitialState());
}

void EarleyParser::ExpandNextRuleRefElement(
    const ParserState& state,
    const GrammarExpr& grammar_expr,
    const GrammarExpr* sub_grammar_expr,
    bool debug_print
) {
  // Path A. The rule has a corresponding FSM.
  XGRAMMAR_DCHECK(!(state.rule_id != -1 && grammar_->per_rule_fsms[state.rule_id].has_value()));
  XGRAMMAR_DCHECK(grammar_expr.type == GrammarExprType::kSequence);
  XGRAMMAR_DCHECK(
      sub_grammar_expr->type == GrammarExprType::kRuleRef ||
      sub_grammar_expr->type == GrammarExprType::kRepeat
  );
  auto ref_rule_id = (*sub_grammar_expr)[0];

  if (debug_print) {
    XGRAMMAR_LOG(INFO) << "The rule " << state.rule_id << ": "
                       << grammar_->GetRule(state.rule_id).name << " predict the new rule "
                       << ref_rule_id << ": " << grammar_->GetRule(ref_rule_id).name << ".";
  }

  bool right_recursion_to_root = false;
  // The right-recursion optimization elides the completion of the parent rule (and, in the
  // to-root case, corrupts the start position of the child rule), so it must be disabled when
  // either rule produces capture-history events or the parent closes a unique-key scope.
  if (state.element_id != grammar_expr.size() - 1 ||
      sub_grammar_expr->type == GrammarExprType::kRepeat ||
      (state.rule_start_pos == rule_id_to_completable_states_.size() - 1) ||
      RuleNeedsCaptureEvent(state.rule_id) || RuleNeedsCaptureEvent(ref_rule_id) ||
      (!unique_key_scope_rules_.empty() && IsUniqueKeyScopeRule(state.rule_id))) {
    // It's not the right recursion, or it's the root rule.
    rule_id_to_completable_states_.PushBackInLatestRow(std::make_pair(ref_rule_id, state));
  } else {
    if (state.rule_start_pos == ParserState::kNoPrevInputPos) {
      right_recursion_to_root = true;
    } else {
      // If it's the right recursion, we need to add the ancestors of the parent state.
      const auto in_vec = [&](const ParserState& state_) {
        return std::find_if(
                   rule_id_to_completable_states_.Back().begin(),
                   rule_id_to_completable_states_.Back().end(),
                   [&](const auto& s) {
                     return StateEqualForParsing()(s.second, state_) && s.first == ref_rule_id;
                   }
               ) != rule_id_to_completable_states_.Back().end();
      };
      const auto& parent_states_map = rule_id_to_completable_states_[state.rule_start_pos];
      std::vector<std::pair<int32_t, ParserState>> to_added_states;
      for (const auto& parent_state_iter : parent_states_map) {
        if (parent_state_iter.first != state.rule_id) continue;
        auto parent_state = parent_state_iter.second;
        if (track_unique_key_contexts_) {
          // Preserve the branch-local mutations that the elided completion would have carried
          // into this ancestor before the recursive child eventually completes it.
          auto propagated_state = CompleteUniqueKeyState(parent_state.unique_key_state_id, state);
          if (!propagated_state.has_value()) {
            continue;
          }
          parent_state.unique_key_state_id = *propagated_state;
        }
        if (!in_vec(parent_state)) {
          to_added_states.push_back({ref_rule_id, std::move(parent_state)});
        }
      }
      for (const auto& to_add_state : to_added_states) {
        rule_id_to_completable_states_.PushBackInLatestRow(to_add_state);
      }
    }
  }

  if (IsRuleNullable(ref_rule_id)) {
    XGRAMMAR_DCHECK(grammar_expr.type == GrammarExprType::kSequence);
    auto skipped_nullable = state;
    skipped_nullable.element_id++;
    skipped_nullable.sub_element_id = 0;
    skipped_nullable.repeat_count = 0;
    skipped_nullable.partial_codepoint = 0;
    Enqueue(std::move(skipped_nullable));
  }

  // If the reference rule is not visited, we need to add it to the queue.
  const auto& ref_rule = grammar_->GetRule(ref_rule_id);
  if (ref_rule.max_chars >= 0) {
    tmp_char_budget_entered_ = true;
  }
  const auto& ref_grammar_expr_id = ref_rule.body_expr_id;

  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[ref_rule_id].has_value());
  const auto& ref_fsm = grammar_->per_rule_fsms[ref_rule_id].value();
  const int32_t child_start_pos = right_recursion_to_root
                                      ? ParserState::kNoPrevInputPos
                                      : int32_t(rule_id_to_completable_states_.size() - 1);
  Enqueue(ParserState{
      ref_rule_id,
      ref_grammar_expr_id,
      ref_fsm.GetFsm().GetStart(),
      child_start_pos,
      DeadlineForRule(ref_rule_id, state.budget_deadline),
      0,
      0,
      0,
      ResolveActiveTemperatureRule(ref_rule_id, state.active_temperature_rule_id),
      CharDeadlineForRule(ref_rule_id, state.char_budget_deadline),
      track_unique_key_contexts_
          ? MakeChildUniqueKeyState(state.unique_key_state_id, ref_rule_id, child_start_pos)
          : state.unique_key_state_id
  });
}

void EarleyParser::ExpandNextRuleRefElementOnFSM(const ParserState& state, bool debug_print) {
  XGRAMMAR_DCHECK(state.rule_id != -1 && grammar_->per_rule_fsms[state.rule_id].has_value());
  const auto& fsm = grammar_->per_rule_fsms[state.rule_id].value();

  // Add the rule reference pairs, and enqueue the epsilon edges.
  for (const auto& edge : fsm.GetFsm().GetFsm().GetEdges(state.element_id)) {
    if (edge.IsEpsilon()) {
      Enqueue(ParserState{
          state.rule_id,
          state.sequence_id,
          edge.target,
          state.rule_start_pos,
          state.budget_deadline,
          state.sub_element_id,
          0,
          state.partial_codepoint,
          state.active_temperature_rule_id,
          state.char_budget_deadline,
          state.unique_key_state_id
      });
      continue;
    }
    if (edge.IsCaptureStart()) {
      XGRAMMAR_DCHECK(IsDynamicTagRule(state.rule_id));
      auto capture_started = state;
      capture_started.element_id = edge.target;
      capture_started.sub_element_id =
          static_cast<int32_t>(rule_id_to_completable_states_.size()) - 1;
      Enqueue(std::move(capture_started));
      continue;
    }
    if (edge.IsCaptureEnd()) {
      XGRAMMAR_DCHECK(IsDynamicTagRule(state.rule_id));
      auto capture_ended = state;
      capture_ended.element_id = edge.target;
      capture_ended.partial_codepoint =
          static_cast<int32_t>(rule_id_to_completable_states_.size()) - 1;
      Enqueue(std::move(capture_ended));
      continue;
    }

    ParserState transition_state = state;
    int target;
    int ref_rule_id;
    bool is_repeat = false;
    RepeatEdgeRef repeat_info{nullptr};

    if (edge.IsRuleRef()) {
      if (IsUniqueKeyDynamicTagRule(state.rule_id) &&
          !PrepareDynamicTagContent(&transition_state)) {
        continue;
      }
      target = edge.target;
      ref_rule_id = edge.GetRefRuleId();
    } else if (edge.IsRepeatRef()) {
      is_repeat = true;
      repeat_info = grammar_->complete_fsm.GetRepeatEdgeInfo(edge.GetAuxIndex());
      target = edge.target;
      ref_rule_id = repeat_info.RuleId();

      if (transition_state.repeat_count >= repeat_info.Lower()) {
        auto completed_repeat = transition_state;
        completed_repeat.element_id = target;
        completed_repeat.repeat_count = 0;
        Enqueue(std::move(completed_repeat));
      }
      if (transition_state.repeat_count >= repeat_info.Upper()) {
        continue;
      }
    } else {
      continue;
    }
    bool right_recursion_to_root = false;
    if (debug_print) {
      XGRAMMAR_LOG(INFO) << "The rule " << state.rule_id << ": "
                         << grammar_->GetRule(state.rule_id).name << " predict the new rule "
                         << ref_rule_id << ": " << grammar_->GetRule(ref_rule_id).name << ".";
    }
    const uint8_t target_flags = GetFsmStateFlags(state.rule_id, target);
    if (!is_repeat && !(target_flags & kFsmStateHasEdges) && (target_flags & kFsmStateEnd) &&
        state.rule_start_pos != static_cast<int32_t>(rule_id_to_completable_states_.size() - 1) &&
        !RuleNeedsCaptureEvent(state.rule_id) && !RuleNeedsCaptureEvent(ref_rule_id) &&
        (unique_key_scope_rules_.empty() || !IsUniqueKeyScopeRule(state.rule_id))) {
      // It's a right recursion. We can optimize it. The optimization elides the completion of
      // the parent rule, so it is disabled for capture events and unique-key scope boundaries.
      // If it's the right recursion, we need to add the ancestors of the parent state.
      if (state.rule_start_pos == ParserState::kNoPrevInputPos) {
        // In this case, we can mark the new state as the root state to speed up.
        right_recursion_to_root = true;
      } else {
        const auto in_vec = [&](const ParserState& state_) {
          return std::find_if(
                     rule_id_to_completable_states_.Back().begin(),
                     rule_id_to_completable_states_.Back().end(),
                     [&](const auto& s) {
                       return StateEqualForParsing()(s.second, state_) && s.first == ref_rule_id;
                     }
                 ) != rule_id_to_completable_states_.Back().end();
        };
        const auto& parent_states_map = rule_id_to_completable_states_[state.rule_start_pos];
        std::vector<std::pair<int32_t, ParserState>> to_added_states;
        for (const auto& parent_state_iter : parent_states_map) {
          if (parent_state_iter.first != state.rule_id) continue;
          auto parent_state = parent_state_iter.second;
          if (track_unique_key_contexts_) {
            // Preserve the branch-local mutations that the elided completion would have carried
            // into this ancestor before the recursive child eventually completes it.
            auto propagated_state =
                CompleteUniqueKeyState(parent_state.unique_key_state_id, transition_state);
            if (!propagated_state.has_value()) {
              continue;
            }
            parent_state.unique_key_state_id = *propagated_state;
          }
          if (!in_vec(parent_state)) {
            to_added_states.push_back({ref_rule_id, std::move(parent_state)});
          }
        }
        for (const auto& to_add_state : to_added_states) {
          rule_id_to_completable_states_.PushBackInLatestRow(to_add_state);
        }
      }
    } else {
      if (is_repeat) {
        // For kRepeatRef: store element_id = source state, preserve repeat_count
        rule_id_to_completable_states_.PushBackInLatestRow(
            {ref_rule_id,
             ParserState{
                 state.rule_id,
                 state.sequence_id,
                 state.element_id,
                 state.rule_start_pos,
                 state.budget_deadline,
                 state.sub_element_id,
                 state.repeat_count,
                 state.partial_codepoint,
                 state.active_temperature_rule_id,
                 state.char_budget_deadline,
                 transition_state.unique_key_state_id
             }}
        );
      } else {
        // For kRuleRef: store element_id = target (post-transition state)
        rule_id_to_completable_states_.PushBackInLatestRow(
            {ref_rule_id,
             ParserState{
                 state.rule_id,
                 state.sequence_id,
                 target,
                 state.rule_start_pos,
                 state.budget_deadline,
                 state.sub_element_id,
                 0,
                 state.partial_codepoint,
                 state.active_temperature_rule_id,
                 state.char_budget_deadline,
                 transition_state.unique_key_state_id
             }}
        );
      }
    }

    // Check if the reference rule can be empty.
    if (!is_repeat && IsRuleNullable(ref_rule_id)) {
      auto skipped_nullable = transition_state;
      skipped_nullable.element_id = target;
      skipped_nullable.repeat_count = 0;
      Enqueue(std::move(skipped_nullable));
    }

    // If the reference rule is not visited, we need to add it to the queue.
    const auto& ref_rule = grammar_->GetRule(ref_rule_id);
    if (ref_rule.max_chars >= 0) {
      tmp_char_budget_entered_ = true;
    }
    const auto& ref_grammar_expr_id = ref_rule.body_expr_id;

    XGRAMMAR_DCHECK(grammar_->per_rule_fsms[ref_rule_id].has_value());
    const auto& ref_fsm = grammar_->per_rule_fsms[ref_rule_id].value();
    const int32_t child_start_pos = right_recursion_to_root
                                        ? ParserState::kNoPrevInputPos
                                        : int32_t(rule_id_to_completable_states_.size() - 1);
    Enqueue(ParserState{
        ref_rule_id,
        ref_grammar_expr_id,
        ref_fsm.GetFsm().GetStart(),
        child_start_pos,
        DeadlineForRule(ref_rule_id, transition_state.budget_deadline),
        0,
        0,
        0,
        ResolveActiveTemperatureRule(ref_rule_id, transition_state.active_temperature_rule_id),
        CharDeadlineForRule(ref_rule_id, transition_state.char_budget_deadline),
        track_unique_key_contexts_
            ? MakeChildUniqueKeyState(
                  transition_state.unique_key_state_id, ref_rule_id, child_start_pos
              )
            : transition_state.unique_key_state_id
    });
  }
}

void EarleyParser::AdvanceByteString(
    const ParserState& state, const uint8_t ch, const GrammarExpr& sub_rule
) {
  XGRAMMAR_DCHECK(sub_rule.type == GrammarExprType::kByteString);
  XGRAMMAR_DCHECK(sub_rule.size() > state.sub_element_id);
  if (static_cast<uint8_t>(sub_rule[state.sub_element_id]) == ch) {
    auto new_state = state;
    new_state.sub_element_id++;
    if (new_state.sub_element_id == sub_rule.size()) {
      new_state.element_id++;
      new_state.sub_element_id = 0;
      Enqueue(new_state);
      // Assert: In a sequence, the bytestring can't be skipped. So the state can't be repeated.
    } else {
      tmp_states_to_be_added_.push_back(new_state);
    }
  }
  return;
}

void EarleyParser::AdvanceCharacterClass(
    const ParserState& state, const uint8_t ch, const GrammarExpr& sub_sequence
) {
  XGRAMMAR_DCHECK(sub_sequence.type == GrammarExprType::kCharacterClass)
      << "The element type is not supported!";

  bool is_negative = static_cast<bool>(sub_sequence[0]);

  // The state is matching a UTF8 character (continuation bytes).
  if (state.sub_element_id > 0) {
    if ((ch & 0xC0) == 0x80) {
      auto new_state = state;
      new_state.sub_element_id--;
      // Accumulate the codepoint from continuation byte
      new_state.partial_codepoint = (new_state.partial_codepoint << 6) | (ch & 0x3F);

      // Check if the UTF8 character is completed.
      if (new_state.sub_element_id == 0) {
        if (is_negative) {
          // For negative classes, accept if codepoint is NOT in any range
          bool matches_range = false;
          for (int i = 1; i < sub_sequence.size(); i += 2) {
            if (new_state.partial_codepoint >= sub_sequence[i] &&
                new_state.partial_codepoint <= sub_sequence[i + 1]) {
              matches_range = true;
              break;
            }
          }
          if (!matches_range) {
            new_state.element_id++;
            new_state.partial_codepoint = 0;
            Enqueue(new_state);
          }
        } else {
          // For positive classes, accept if codepoint IS in a range
          bool matches_range = false;
          for (int i = 1; i < sub_sequence.size(); i += 2) {
            if (new_state.partial_codepoint >= sub_sequence[i] &&
                new_state.partial_codepoint <= sub_sequence[i + 1]) {
              matches_range = true;
              break;
            }
          }
          if (matches_range) {
            new_state.element_id++;
            new_state.partial_codepoint = 0;
            Enqueue(new_state);
          }
        }
      } else {
        // Check if partial codepoint could still potentially match any range
        int32_t remaining_bytes = new_state.sub_element_id;
        int32_t min_codepoint = new_state.partial_codepoint << (6 * remaining_bytes);
        int32_t max_codepoint = min_codepoint | ((1 << (6 * remaining_bytes)) - 1);

        bool could_match = false;
        for (int i = 1; i < sub_sequence.size(); i += 2) {
          int32_t lower = sub_sequence[i];
          int32_t upper = sub_sequence[i + 1];
          if (max_codepoint >= lower && min_codepoint <= upper) {
            could_match = true;
            break;
          }
        }

        // For negative classes: always continue (will verify on final byte)
        // For positive classes: only continue if some range could match
        bool should_continue = is_negative ? true : could_match;
        if (should_continue) {
          tmp_states_to_be_added_.push_back(new_state);
        }
      }
    }
    return;
  }

  // Handle non-ASCII first bytes
  if (!isascii(ch)) {
    auto [accepted, num_bytes, partial] = HandleUTF8FirstByte(ch);
    if (!accepted) {
      return;
    }

    XGRAMMAR_DCHECK(num_bytes > 1);

    // Compute possible codepoint range for this first byte
    int32_t min_codepoint = partial << (6 * (num_bytes - 1));
    int32_t max_codepoint = min_codepoint | ((1 << (6 * (num_bytes - 1))) - 1);

    // Check if any stored range could potentially match
    bool could_match = false;
    for (int i = 1; i < sub_sequence.size(); i += 2) {
      int32_t lower = sub_sequence[i];
      int32_t upper = sub_sequence[i + 1];
      // Check for overlap between [min_codepoint, max_codepoint] and [lower, upper]
      if (max_codepoint >= lower && min_codepoint <= upper) {
        could_match = true;
        break;
      }
    }

    // For negative classes: accept if no range could match (will verify on final byte)
    // For positive classes: accept if some range could match (will verify on final byte)
    bool should_continue = is_negative ? true : could_match;

    if (should_continue) {
      auto new_state = state;
      new_state.sub_element_id = num_bytes - 1;
      new_state.partial_codepoint = partial;
      tmp_states_to_be_added_.push_back(new_state);
    }
    return;
  }

  // ASCII handling (unchanged)
  for (int i = 1; i < sub_sequence.size(); i += 2) {
    if (static_cast<uint8_t>(sub_sequence[i]) <= ch &&
        ch <= static_cast<uint8_t>(sub_sequence[i + 1])) {
      if (!is_negative) {
        auto new_state = state;
        new_state.element_id++;
        new_state.sub_element_id = 0;
        Enqueue(new_state);
      }
      return;
    }
  }
  if (is_negative) {
    auto new_state = state;
    new_state.element_id++;
    new_state.sub_element_id = 0;
    Enqueue(new_state);
  }
}

void EarleyParser::AdvanceCharacterClassStar(
    const ParserState& state, const uint8_t ch, const GrammarExpr& sub_sequence
) {
  XGRAMMAR_DCHECK(sub_sequence.type == GrammarExprType::kCharacterClassStar)
      << "The element type is not supported!";

  bool is_negative = static_cast<bool>(sub_sequence[0]);

  // The state is matching a UTF8 character (continuation bytes).
  if (state.sub_element_id > 0) {
    if ((ch & 0xC0) == 0x80) {
      auto new_state = state;
      new_state.sub_element_id--;
      // Accumulate the codepoint from continuation byte
      new_state.partial_codepoint = (new_state.partial_codepoint << 6) | (ch & 0x3F);

      // Check if the UTF8 character is completed.
      if (new_state.sub_element_id == 0) {
        if (is_negative) {
          // For negative classes, accept if codepoint is NOT in any range
          bool matches_range = false;
          for (int i = 1; i < sub_sequence.size(); i += 2) {
            if (new_state.partial_codepoint >= sub_sequence[i] &&
                new_state.partial_codepoint <= sub_sequence[i + 1]) {
              matches_range = true;
              break;
            }
          }
          if (!matches_range) {
            new_state.partial_codepoint = 0;
            Enqueue(new_state);
          }
        } else {
          // For positive classes, accept if codepoint IS in a range
          bool matches_range = false;
          for (int i = 1; i < sub_sequence.size(); i += 2) {
            if (new_state.partial_codepoint >= sub_sequence[i] &&
                new_state.partial_codepoint <= sub_sequence[i + 1]) {
              matches_range = true;
              break;
            }
          }
          if (matches_range) {
            new_state.partial_codepoint = 0;
            Enqueue(new_state);
          }
        }
      } else {
        // Check if partial codepoint could still potentially match any range
        int32_t remaining_bytes = new_state.sub_element_id;
        int32_t min_codepoint = new_state.partial_codepoint << (6 * remaining_bytes);
        int32_t max_codepoint = min_codepoint | ((1 << (6 * remaining_bytes)) - 1);

        bool could_match = false;
        for (int i = 1; i < sub_sequence.size(); i += 2) {
          int32_t lower = sub_sequence[i];
          int32_t upper = sub_sequence[i + 1];
          if (max_codepoint >= lower && min_codepoint <= upper) {
            could_match = true;
            break;
          }
        }

        // For negative classes: always continue (will verify on final byte)
        // For positive classes: only continue if some range could match
        bool should_continue = is_negative ? true : could_match;
        if (should_continue) {
          tmp_states_to_be_added_.push_back(new_state);
        }
      }
    }
    return;
  }

  // Handle non-ASCII first bytes
  if (!isascii(ch)) {
    auto [accepted, num_bytes, partial] = HandleUTF8FirstByte(ch);
    if (!accepted) {
      return;
    }

    XGRAMMAR_DCHECK(num_bytes > 1);

    // Compute possible codepoint range for this first byte
    int32_t min_codepoint = partial << (6 * (num_bytes - 1));
    int32_t max_codepoint = min_codepoint | ((1 << (6 * (num_bytes - 1))) - 1);

    // Check if any stored range could potentially match
    bool could_match = false;
    for (int i = 1; i < sub_sequence.size(); i += 2) {
      int32_t lower = sub_sequence[i];
      int32_t upper = sub_sequence[i + 1];
      // Check for overlap between [min_codepoint, max_codepoint] and [lower, upper]
      if (max_codepoint >= lower && min_codepoint <= upper) {
        could_match = true;
        break;
      }
    }

    // For negative classes: accept if no range could match (will verify on final byte)
    // For positive classes: accept if some range could match (will verify on final byte)
    bool should_continue = is_negative ? true : could_match;

    if (should_continue) {
      auto new_state = state;
      new_state.sub_element_id = num_bytes - 1;
      new_state.partial_codepoint = partial;
      tmp_states_to_be_added_.push_back(new_state);
    }
    return;
  }

  // ASCII handling (unchanged)
  for (int i = 1; i < sub_sequence.size(); i += 2) {
    if (static_cast<uint8_t>(sub_sequence[i]) <= ch &&
        ch <= static_cast<uint8_t>(sub_sequence[i + 1])) {
      if (!is_negative) {
        Enqueue(state);
      }
      return;
    }
  }
  if (is_negative) {
    Enqueue(state);
  }
}

void EarleyParser::AdvanceFsm(const ParserState& state, const uint8_t ch) {
  XGRAMMAR_DCHECK(state.rule_id != -1 && grammar_->per_rule_fsms[state.rule_id].has_value());
  const auto& current_fsm = grammar_->per_rule_fsms[state.rule_id].value();
  const auto& edges = current_fsm.GetFsm().GetFsm().GetEdges(state.element_id);
  if (!has_dynamic_tag_rules_) {
    for (const auto& edge : edges) {
      if ((!edge.IsCharRange()) || ch < edge.min || ch > edge.max) {
        continue;
      }
      auto new_state = state;
      new_state.element_id = edge.target;
      const uint8_t flags = GetFsmStateFlags(state.rule_id, edge.target);
      if (!(flags & kFsmStateNonTerminal) && !(flags & kFsmStateEnd) &&
          (flags & kFsmStateScanable)) {
        EnqueueWithoutProcessing(std::move(new_state));
      } else {
        Enqueue(std::move(new_state));
      }
    }
    return;
  }
  for (const auto& edge : edges) {
    if (edge.IsBackReference()) {
      auto progress = GetBackReferenceProgress(state);
      if (!progress.has_value()) {
        if (AllowWildcardBackReference()) {
          EnqueueWithoutProcessing(state);
          auto completed = state;
          completed.element_id = edge.target;
          completed.sub_element_id = 0;
          completed.repeat_count = 0;
          completed.partial_codepoint = 0;
          Enqueue(std::move(completed));
        }
        continue;
      }
      if (progress->next_byte != ch) {
        continue;
      }
      auto new_state = state;
      if (progress->is_last_byte) {
        new_state.element_id = edge.target;
        new_state.sub_element_id = 0;
        new_state.repeat_count = 0;
        new_state.partial_codepoint = 0;
        Enqueue(std::move(new_state));
      } else {
        ++new_state.repeat_count;
        EnqueueWithoutProcessing(std::move(new_state));
      }
      continue;
    }
    if ((!edge.IsCharRange()) || ch < edge.min || ch > edge.max) {
      continue;
    }
    auto new_state = state;
    new_state.element_id = edge.target;
    const uint8_t flags = GetFsmStateFlags(state.rule_id, edge.target);
    if (!(flags & kFsmStateNonTerminal) && !(flags & kFsmStateEnd) && (flags & kFsmStateScanable)) {
      EnqueueWithoutProcessing(std::move(new_state));
    } else {
      Enqueue(std::move(new_state));
    }
  }
}

void EarleyParser::ScanAtomicToken(const ParserState& state, int32_t token_id) {
  if (state.rule_id == -1) return;
  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[state.rule_id].has_value());
  const auto& current_fsm = grammar_->per_rule_fsms[state.rule_id].value();
  for (const auto& edge : current_fsm.GetFsm().GetFsm().GetEdges(state.element_id)) {
    bool matched = false;
    if (edge.IsToken()) {
      auto info = current_fsm.GetFsm().GetFsm().GetTokenEdgeInfo(edge.GetAuxIndex());
      matched = info.Contains(token_id);
    } else if (edge.IsExcludeToken()) {
      auto info = current_fsm.GetFsm().GetFsm().GetExcludeTokenEdgeInfo(edge.GetAuxIndex());
      matched = info.Accepts(token_id);
    }
    if (!matched) continue;
    auto new_state = state;
    new_state.element_id = edge.target;
    const uint8_t flags = GetFsmStateFlags(state.rule_id, edge.target);
    if (!(flags & kFsmStateNonTerminal) && !(flags & kFsmStateEnd) && (flags & kFsmStateScanable)) {
      EnqueueWithoutProcessing(std::move(new_state));
    } else {
      Enqueue(std::move(new_state));
    }
  }
}

bool EarleyParser::AdvanceAtomicToken(
    int32_t token_id, bool debug_print, int32_t token_char_count
) {
  XGRAMMAR_DCHECK(tmp_process_state_queue_.empty())
      << "The tmp_process_state_queue_ should be empty before AdvanceAtomicToken.";
  tmp_states_visited_in_queue_.Clear();
  tmp_states_to_be_added_.clear();
  tmp_accept_stop_token_ = false;
  tmp_completed_lazy_occurrences_.clear();
  if (has_char_budget_rules_) {
    tmp_char_budget_entered_ = char_budget_entry_history_.back();
    char_count_history_.push_back(GetCurrentCharIndex() + token_char_count);
  }
  const auto& latest_states = scanable_state_history_[scanable_state_history_.size() - 1];
  for (const auto& state : latest_states) {
    if (skip_expired_states_ && IsExpiredState(state)) {
      continue;
    }
    ScanAtomicToken(state, token_id);
  }
  if (tmp_process_state_queue_.empty() && tmp_states_to_be_added_.empty()) {
    if (has_char_budget_rules_) {
      char_count_history_.pop_back();
    }
    return false;
  }
  rule_id_to_completable_states_.PushBack(std::vector<std::pair<int32_t, ParserState>>());
  if (capture_tracking_) {
    capture_event_history_.PushBack(std::vector<CaptureEvent>());
  }
  while (!tmp_process_state_queue_.empty()) {
    const auto state = std::move(tmp_process_state_queue_.front());
    tmp_process_state_queue_.pop();
    auto [scanable, completable] = Predict(state, debug_print);
    if (completable) {
      Complete(state, debug_print);
    }
    if (scanable) {
      tmp_states_to_be_added_.push_back(state);
    }
  }
  if (!tmp_completed_lazy_occurrences_.empty()) {
    RemoveCommittedLazyStates();
  }
  if (!unique_key_scope_rules_.empty() && tmp_states_to_be_added_.empty() &&
      !tmp_accept_stop_token_) {
    rule_id_to_completable_states_.PopBack(1);
    if (capture_tracking_) {
      capture_event_history_.PopBack(1);
    }
    if (has_char_budget_rules_) {
      char_count_history_.pop_back();
    }
    return false;
  }
  is_completed_.push_back(tmp_accept_stop_token_);
  scanable_state_history_.PushBack(tmp_states_to_be_added_);
  if (has_char_budget_rules_) {
    char_budget_entry_history_.push_back(tmp_char_budget_entered_);
  }
  return true;
}

bool RepeatDetector::IsVisited(const ParserState& state) const {
  // If the size is larger than the threshold, then we use the set to check.
  if (size_ > transition_threshold_) {
    return visited_set_.find(state) != visited_set_.end();
  }
  return std::find_if(
             visited_vector_.begin(),
             visited_vector_.begin() + size_,
             [&state](const ParserState& s) { return StateEqualForParsing()(state, s); }
         ) != visited_vector_.begin() + size_;
}

void RepeatDetector::Insert(const ParserState& state) {
  if (size_ == transition_threshold_) {
    for (const auto& s : visited_vector_) {
      visited_set_.insert(s);
    }
  }
  size_++;
  if (size_ > transition_threshold_) {
    visited_set_.insert(state);
  } else {
    visited_vector_[size_ - 1] = state;
  }
}

void RepeatDetector::Clear() {
  if (size_ > transition_threshold_) {
    visited_set_.clear();
  }
  size_ = 0;
}

}  // namespace xgrammar
