/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/grammar_optimizer.cc
 */

#include "grammar_optimizer.h"

#include <bitset>
#include <queue>
#include <set>

#include "fsm_builder.h"
#include "grammar_functor.h"
#include "support/logging.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

using GrammarExpr = Grammar::Impl::GrammarExpr;
using ExprType = Grammar::Impl::GrammarExprType;
class GrammarOptimizerImpl : public GrammarMutator {
 public:
  /*! \brief Optimize the grammar. */
  Grammar Apply(const Grammar& grammar) final {
    XGRAMMAR_ICHECK(!grammar->optimized);
    Grammar new_grammar_base = Grammar(std::make_shared<Grammar::Impl>(*grammar.ImplPtr()));
    new_grammar_base = RuleInliner::Apply(new_grammar_base);
    new_grammar_base = DeadCodeEliminator::Apply(new_grammar_base);
    new_grammar_base = LookaheadAssertionAnalyzer::Apply(new_grammar_base);
    new_grammar_base->allow_empty_rule_ids = AllowEmptyRuleAnalyzer::Apply(new_grammar_base);
    RepetitionNormalizer::Apply(&new_grammar_base);
    GrammarFSMBuilder::Apply(&new_grammar_base);
    new_grammar_base->optimized = true;
    XGRAMMAR_ICHECK(!grammar->optimized);
    return new_grammar_base;
  }
};

class GrammarFSMBuilderImpl {
 public:
  void Apply(Grammar* grammar) {
    FSM complete_fsm;
    std::vector<std::optional<FSMWithStartEnd>> per_rule_fsms((*grammar)->NumRules());
    std::unordered_map<int, int> state_mapping;

    for (int i = 0; i < (*grammar)->NumRules(); ++i) {
      auto rule = (*grammar)->GetRule(i);
      auto grammar_expr = (*grammar)->GetGrammarExpr(rule.body_expr_id);
      if (grammar_expr.type == ExprType::kTagDispatch) {
        auto rule_fsm = TagDispatch((*grammar)->GetTagDispatch(grammar_expr));
        XGRAMMAR_CHECK(rule_fsm.has_value()) << "Failed to build tag dispatch fsm for rule " << i;
        per_rule_fsms[i] = rule_fsm->AddToCompleteFSM(&complete_fsm, &state_mapping);
      } else {
        XGRAMMAR_DCHECK(grammar_expr.type == ExprType::kChoices);
        auto rule_fsm = Choices(grammar_expr, *grammar);
        if (rule_fsm.has_value()) {
          per_rule_fsms[i] = rule_fsm->AddToCompleteFSM(&complete_fsm, &state_mapping);
        }
      }
    }

    // Compress to compact fsm
    CompactFSM compact_complete_fsm = complete_fsm.ToCompact();
    std::vector<std::optional<CompactFSMWithStartEnd>> compact_per_rule_fsms((*grammar)->NumRules()
    );
    for (int i = 0; i < (*grammar)->NumRules(); ++i) {
      if (per_rule_fsms[i]) {
        compact_per_rule_fsms[i] = CompactFSMWithStartEnd(
            compact_complete_fsm, per_rule_fsms[i]->GetStart(), per_rule_fsms[i]->GetEnds()
        );
      }
    }

    (*grammar)->complete_fsm = std::move(compact_complete_fsm);
    (*grammar)->per_rule_fsms = std::move(compact_per_rule_fsms);
  }

  /* Basic Building functions.*/
  static std::optional<FSMWithStartEnd> RuleRef(const GrammarExpr& expr);
  static std::optional<FSMWithStartEnd> CharacterClass(const GrammarExpr& expr);
  static std::optional<FSMWithStartEnd> ByteString(const GrammarExpr& expr);
  static std::optional<FSMWithStartEnd> Sequence(const GrammarExpr& expr, const Grammar& grammar);
  static std::optional<FSMWithStartEnd> Choices(const GrammarExpr& expr, const Grammar& grammar);
  static std::optional<FSMWithStartEnd> TagDispatch(const Grammar::Impl::TagDispatch& tag_dispatch);

  /* Building tool funtions.*/
  static std::optional<FSMWithStartEnd> BuildTagDispatchWithEOSStop(
      const std::vector<std::pair<std::string, int>>& tag_dispatch_rules, bool loop_after_dispatch
  );
  static std::optional<FSMWithStartEnd> BuildTagDispatchWithStopString(
      const std::vector<std::pair<std::string, int>>& tag_dispatch_rules,
      const std::vector<std::string>& stop_strings,
      bool loop_after_dispatch
  );
  static std::optional<FSMWithStartEnd> BuildNegativeCharacterClass(const GrammarExpr& expr);
};

const static uint32_t kMax1ByteUnicode = 0x7F;
const static uint32_t kMin2BytesUnicode = 0xC080;
const static uint32_t kMax2BytesUnicode = 0xDFBF;
const static uint32_t kMin3BytesUnicode = 0xE08080;
const static uint32_t kMax3BytesUnicode = 0xEFBFBF;
const static uint32_t kMin4BytesUnicode = 0xF0808080;
const static uint32_t kMax4BytesUnicode = 0xF7BFBFBF;

// This function will add a range [min, max] of characters to the FSM, and the length
// of the characters are the same.
void AddSameLengthCharacterRange(
    FSMWithStartEnd& fsm, int from, int to, uint32_t min, uint32_t max
) {
  uint8_t byte_min[4] = {
      static_cast<uint8_t>(min & 0xFF),
      static_cast<uint8_t>(min >> 8),
      static_cast<uint8_t>(min >> 16),
      static_cast<uint8_t>(min >> 24)
  };
  uint8_t byte_max[4] = {
      static_cast<uint8_t>(max & 0xFF),
      static_cast<uint8_t>(max >> 8),
      static_cast<uint8_t>(max >> 16),
      static_cast<uint8_t>(max >> 24)
  };

  // ASCII.
  if (byte_max[1] == 0) {
    fsm.GetFsm().AddEdge(from, to, byte_min[0], byte_max[0]);
    return;
  }

  if (byte_max[3] != 0) {
    // 4-byte unicode.
    if (byte_max[3] == byte_min[3]) {
      int tmp_state = fsm.AddState();
      fsm.GetFsm().AddEdge(from, tmp_state, byte_min[3], byte_max[3]);
      min = (min & 0x00FFFFFF);
      max = (max & 0x00FFFFFF);
      AddSameLengthCharacterRange(fsm, tmp_state, to, min, max);
      return;
    }
    if ((min & 0x00FFFFFF) != 0x808080) {
      int tmp_state_min = fsm.AddState();
      fsm.GetFsm().AddEdge(from, tmp_state_min, byte_min[3], byte_min[3]);
      AddSameLengthCharacterRange(fsm, tmp_state_min, to, (min & 0x00FFFFFF), 0x00BFBFBF);
    } else {
      byte_min[3]--;
    }
    if ((max & 0x00FFFFFF) != 0xBFBFBF) {
      int tmp_state_max = fsm.AddState();
      fsm.GetFsm().AddEdge(from, tmp_state_max, byte_max[3], byte_max[3]);
      AddSameLengthCharacterRange(fsm, tmp_state_max, to, 0x00808080, (max & 0x00FFFFFF));
    } else {
      byte_max[3]++;
    }
    if (byte_max[3] - byte_min[3] > 1) {
      int tmp_state_mid = fsm.AddState();
      // First byte.
      fsm.GetFsm().AddEdge(from, tmp_state_mid, byte_min[3] + 1, byte_max[3] - 1);
      int tmp_state_mid2 = fsm.AddState();
      // Second byte.
      fsm.GetFsm().AddEdge(tmp_state_mid, tmp_state_mid2, 0x80, 0xBF);
      int tmp_state_mid3 = fsm.AddState();
      // Third byte.
      fsm.GetFsm().AddEdge(tmp_state_mid2, tmp_state_mid3, 0x80, 0xBF);
      // Last byte.
      fsm.GetFsm().AddEdge(tmp_state_mid3, to, 0x80, 0xBF);
    }
    return;
  }
  if (byte_max[2] != 0) {
    // 3 byte unicode.
    if (byte_max[2] == byte_min[2]) {
      int tmp_state = fsm.AddState();
      fsm.GetFsm().AddEdge(from, tmp_state, byte_min[2], byte_max[2]);
      min = (min & 0x00FFFF);
      max = (max & 0x00FFFF);
      AddSameLengthCharacterRange(fsm, tmp_state, to, min, max);
      return;
    }
    if ((min & 0x00FFFF) != 0x8080) {
      int tmp_state_min = fsm.AddState();
      fsm.GetFsm().AddEdge(from, tmp_state_min, byte_min[2], byte_min[2]);
      AddSameLengthCharacterRange(fsm, tmp_state_min, to, (min & 0x00FFFF), 0x00BFBF);
    } else {
      byte_min[2]--;
    }
    if ((max & 0x00FFFF) != 0xBFBF) {
      int tmp_state_max = fsm.AddState();
      fsm.GetFsm().AddEdge(from, tmp_state_max, byte_max[2], byte_max[2]);
      AddSameLengthCharacterRange(fsm, tmp_state_max, to, 0x0080, (max & 0x00FFFF));
    } else {
      byte_max[2]++;
    }
    if (byte_max[2] - byte_min[2] > 1) {
      int tmp_state_mid = fsm.AddState();
      // First byte.
      fsm.GetFsm().AddEdge(from, tmp_state_mid, byte_min[2] + 1, byte_max[2] - 1);
      int tmp_state_mid2 = fsm.AddState();
      // Second byte.
      fsm.GetFsm().AddEdge(tmp_state_mid, tmp_state_mid2, 0x80, 0xBF);
      // Last byte.
      fsm.GetFsm().AddEdge(tmp_state_mid2, to, 0x80, 0xBF);
    }
    return;
  }

  // 2 byte unicode.
  if (byte_max[1] == byte_min[1]) {
    int tmp_state = fsm.AddState();
    fsm.GetFsm().AddEdge(from, tmp_state, byte_min[1], byte_max[1]);
    min = (min & 0x00FF);
    max = (max & 0x00FF);
    AddSameLengthCharacterRange(fsm, tmp_state, to, min, max);
    return;
  }
  if ((min & 0x00FF) != 0x80) {
    int tmp_state_min = fsm.AddState();
    fsm.GetFsm().AddEdge(from, tmp_state_min, byte_min[1], byte_min[1]);
    AddSameLengthCharacterRange(fsm, tmp_state_min, to, (min & 0x00FF), 0x00BF);
  } else {
    byte_min[1]--;
  }
  if ((max & 0x00FF) != 0xBF) {
    int tmp_state_max = fsm.AddState();
    fsm.GetFsm().AddEdge(from, tmp_state_max, byte_max[1], byte_max[1]);
    AddSameLengthCharacterRange(fsm, tmp_state_max, to, 0x0080, (max & 0x00FF));
  } else {
    byte_max[1]++;
  }
  if (byte_max[1] - byte_min[1] > 1) {
    int tmp_state_mid = fsm.AddState();
    // First byte.
    fsm.GetFsm().AddEdge(from, tmp_state_mid, byte_min[1] + 1, byte_max[1] - 1);
    fsm.GetFsm().AddEdge(tmp_state_mid, to, 0x80, 0xBF);
  }
  return;
}

// This function will add a range [min, max] of unicode characters to the FSM.
void AddCharacterRange(FSMWithStartEnd& fsm, int from, int to, uint32_t min, uint32_t max) {
  XGRAMMAR_CHECK(min <= max) << "Invalid character range: min (" << min << ") > max (" << max
                             << ")";
  // Ensure max and min are valid unicode value.
  if (max > kMax4BytesUnicode) {
    max = kMax4BytesUnicode;
  } else if (max > kMax3BytesUnicode) {
    if (max < kMin4BytesUnicode) {
      max = kMax3BytesUnicode;
    }
  } else if (max > kMax2BytesUnicode) {
    if (max < kMin3BytesUnicode) {
      max = kMax2BytesUnicode;
    }
  } else if (max < kMin2BytesUnicode && (max > kMax1ByteUnicode)) {
    max = kMax1ByteUnicode;
  }

  if (min > kMax4BytesUnicode) {
    min = kMax4BytesUnicode;
  } else if (min > kMax3BytesUnicode) {
    if (min < kMin4BytesUnicode) {
      min = kMin4BytesUnicode;
    }
  } else if (min > kMax2BytesUnicode) {
    if (min < kMin3BytesUnicode) {
      min = kMin3BytesUnicode;
    }
  } else if (min < kMin2BytesUnicode && (min > kMax1ByteUnicode)) {
    min = kMin2BytesUnicode;
  }

  // Step2. Divide the range into several ranges, which contain characters with different lengths.
  if (max <= kMax1ByteUnicode) {
    AddSameLengthCharacterRange(fsm, from, to, min, max);
    return;
  }
  if (max <= kMax2BytesUnicode) {
    if (min >= kMin2BytesUnicode) {
      AddSameLengthCharacterRange(fsm, from, to, min, max);
    } else {
      AddSameLengthCharacterRange(fsm, from, to, min, kMax1ByteUnicode);
      AddSameLengthCharacterRange(fsm, from, to, kMin2BytesUnicode, max);
    }
    return;
  }
  if (max <= kMax3BytesUnicode) {
    if (min >= kMin3BytesUnicode) {
      AddSameLengthCharacterRange(fsm, from, to, min, max);
    } else if (min >= kMin2BytesUnicode) {
      AddSameLengthCharacterRange(fsm, from, to, min, kMax2BytesUnicode);
      AddSameLengthCharacterRange(fsm, from, to, kMin3BytesUnicode, max);
    } else {
      AddSameLengthCharacterRange(fsm, from, to, min, kMax1ByteUnicode);
      AddSameLengthCharacterRange(fsm, from, to, kMin2BytesUnicode, kMax2BytesUnicode);
      AddSameLengthCharacterRange(fsm, from, to, kMin3BytesUnicode, max);
    }
    return;
  }
  XGRAMMAR_CHECK(max <= kMax4BytesUnicode);
  if (min >= kMin4BytesUnicode) {
    AddSameLengthCharacterRange(fsm, from, to, min, max);
  } else if (min >= kMin3BytesUnicode) {
    AddSameLengthCharacterRange(fsm, from, to, min, kMax3BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin4BytesUnicode, max);
  } else if (min >= kMin2BytesUnicode) {
    AddSameLengthCharacterRange(fsm, from, to, min, kMax2BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin3BytesUnicode, kMax3BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin4BytesUnicode, max);
  } else {
    AddSameLengthCharacterRange(fsm, from, to, min, kMax1ByteUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin2BytesUnicode, kMax2BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin3BytesUnicode, kMax3BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin4BytesUnicode, max);
  }
  return;
}

std::optional<FSMWithStartEnd> GrammarFSMBuilderImpl::BuildNegativeCharacterClass(
    const GrammarExpr& expr
) {
  XGRAMMAR_DCHECK(
      expr.type == ExprType::kCharacterClass || expr.type == ExprType::kCharacterClassStar
  );
  XGRAMMAR_DCHECK(expr[0]);  // Negative character class should be true.
  std::bitset<128> char_set;
  for (int i = 1; i < static_cast<int>(expr.size()); i += 2) {
    uint8_t byte_min = static_cast<uint8_t>(expr[i]);
    uint8_t byte_max = static_cast<uint8_t>(expr[i + 1]);
    if (byte_max > 128) {
      XGRAMMAR_LOG(WARNING) << "Negative Character class contains byte greater than 127, "
                            << "clamping to 127.";
      byte_max = 127;
    }
    for (uint8_t j = byte_min; j <= byte_max; ++j) {
      char_set.set(j);
    }
  }

  // Construct the basic FSM.
  FSMWithStartEnd result_fsm;
  int start_state = result_fsm.AddState();
  bool is_star = expr.type == ExprType::kCharacterClassStar;
  result_fsm.SetStartState(start_state);
  int end_state = -1;
  if (is_star) {
    end_state = start_state;
  } else {
    end_state = result_fsm.AddState();
  }
  result_fsm.AddEndState(end_state);
  int left_bound = -1;
  for (int i = 0; i < 128; ++i) {
    if (!char_set[i]) {
      left_bound = i;
      int right_bound = i + 1;
      while (right_bound < 128 && !char_set[right_bound]) {
        right_bound++;
      }
      result_fsm.GetFsm().AddEdge(
          start_state,
          end_state,
          static_cast<uint8_t>(left_bound),
          static_cast<uint8_t>(right_bound - 1)
      );
      i = right_bound;
    }
  }
  AddCharacterRange(result_fsm, start_state, end_state, kMin2BytesUnicode, kMax4BytesUnicode);
  return result_fsm;
}

std::optional<FSMWithStartEnd> GrammarFSMBuilderImpl::CharacterClass(const GrammarExpr& expr) {
  bool is_negative = expr[0];
  FSMWithStartEnd result_fsm;
  if (is_negative) {
    auto optional_fsm = BuildNegativeCharacterClass(expr);
    if (!optional_fsm.has_value()) {
      return std::nullopt;
    }
    return result_fsm = std::move(optional_fsm.value());
  }
  int start_state = result_fsm.AddState();
  result_fsm.SetStartState(start_state);
  bool is_star = expr.type == ExprType::kCharacterClassStar;
  int end_state = -1;
  if (is_star) {
    end_state = start_state;
  } else {
    end_state = result_fsm.AddState();
  }
  result_fsm.AddEndState(end_state);
  for (int i = 1; i < static_cast<int>(expr.size()); i += 2) {
    uint8_t byte_min = static_cast<uint8_t>(expr[i]);
    uint8_t byte_max = static_cast<uint8_t>(expr[i + 1]);
    result_fsm.GetFsm().AddEdge(start_state, end_state, byte_min, byte_max);
  }
  return result_fsm;
}

std::optional<FSMWithStartEnd> GrammarFSMBuilderImpl::Sequence(
    const GrammarExpr& expr, const Grammar& grammar
) {
  std::vector<FSMWithStartEnd> fsm_lists;

  // Build the fsm of sub-expressions.
  for (const auto& sequence_id : expr) {
    const auto& sequence_expr = grammar->GetGrammarExpr(sequence_id);
    switch (sequence_expr.type) {
      case (ExprType::kByteString): {
        auto fsm = ByteString(sequence_expr);
        if (!fsm.has_value()) {
          return std::nullopt;
        }
        fsm_lists.push_back(std::move(fsm.value()));
        break;
      }
      case (ExprType::kRuleRef): {
        auto fsm = RuleRef(sequence_expr);
        if (!fsm.has_value()) {
          return std::nullopt;
        }
        fsm_lists.push_back(std::move(fsm.value()));
        break;
      }
      case (ExprType::kCharacterClass):
      case (ExprType::kCharacterClassStar): {
        auto fsm = CharacterClass(sequence_expr);
        if (!fsm.has_value()) {
          return std::nullopt;
        }
        fsm_lists.push_back(std::move(fsm.value()));
        break;
      }
      default: {
        return std::nullopt;
      }
    }
  }

  // Check if the sequence is empty.
  if (fsm_lists.empty()) {
    FSMWithStartEnd empty_fsm;
    empty_fsm.AddState();
    empty_fsm.SetStartState(0);
    empty_fsm.AddEndState(0);
    return empty_fsm;
  }

  return FSMWithStartEnd::Concat(fsm_lists);
}

std::optional<FSMWithStartEnd> GrammarFSMBuilderImpl::RuleRef(const GrammarExpr& expr) {
  FSMWithStartEnd result_fsm;
  result_fsm.AddState();
  result_fsm.AddState();
  result_fsm.SetStartState(0);
  result_fsm.AddEndState(1);
  result_fsm.GetFsm().AddRuleEdge(0, 1, expr[0]);
  return result_fsm;
}

std::optional<FSMWithStartEnd> GrammarFSMBuilderImpl::ByteString(const GrammarExpr& expr) {
  XGRAMMAR_DCHECK(expr.type == ExprType::kByteString);
  FSMWithStartEnd result_fsm;
  int current_state = result_fsm.AddState();
  result_fsm.SetStartState(current_state);
  for (const auto& byte : expr) {
    int next_state = result_fsm.AddState();
    result_fsm.GetFsm().AddEdge(
        current_state, next_state, static_cast<uint8_t>(byte), static_cast<uint8_t>(byte)
    );
    current_state = next_state;
  }
  result_fsm.AddEndState(current_state);
  return result_fsm;
}

std::optional<FSMWithStartEnd> GrammarFSMBuilderImpl::Choices(
    const GrammarExpr& expr, const Grammar& grammar
) {
  XGRAMMAR_DCHECK(expr.type == ExprType::kChoices);
  std::vector<FSMWithStartEnd> fsm_list;
  bool nullable = false;
  for (const auto& choice_id : expr) {
    const auto& choice_expr = grammar->GetGrammarExpr(choice_id);
    // The choice expression should be either a sequence or an empty string.
    if (choice_expr.type == ExprType::kEmptyStr) {
      nullable = true;
      continue;
    }
    XGRAMMAR_DCHECK(choice_expr.type == ExprType::kSequence);
    auto fsm_result = Sequence(choice_expr, grammar);
    if (!fsm_result.has_value()) {
      return std::nullopt;
    }
    fsm_list.push_back(std::move(fsm_result.value()));
  }

  if (fsm_list.empty()) {
    // It's an empty rule.
    FSMWithStartEnd empty_fsm;
    empty_fsm.AddState();
    empty_fsm.SetStartState(0);
    empty_fsm.AddEndState(0);
    return empty_fsm;
  }
  if (nullable) {
    FSMWithStartEnd null_fsm;
    null_fsm.AddState();
    null_fsm.SetStartState(0);
    null_fsm.AddEndState(0);
    fsm_list.push_back(std::move(null_fsm));
  }

  auto result = FSMWithStartEnd::Union(fsm_list);
  result = result.SimplifyEpsilon();
  result = result.MergeEquivalentSuccessors();
  auto result_raw = result.MinimizeDFA();
  if (result_raw.IsOk()) {
    result = std::move(result_raw).Unwrap();
  }
  return result;
}

std::optional<FSMWithStartEnd> GrammarFSMBuilderImpl::BuildTagDispatchWithStopString(
    const std::vector<std::pair<std::string, int>>& tag_dispatch_rules,
    const std::vector<std::string>& stop_strings,
    bool loop_after_dispatch
) {
  XGRAMMAR_DCHECK(stop_strings.size() > 0);
  std::vector<std::string> tag_names;
  tag_names.reserve(tag_dispatch_rules.size());
  for (const auto& [tag_name, tag_id] : tag_dispatch_rules) {
    tag_names.push_back(tag_name);
  }
  for (const auto& stop_string : stop_strings) {
    tag_names.push_back(stop_string);
  }
  std::vector<int> trie_end_states;
  auto trie_result = TrieFSMBuilder::Build(tag_names, &trie_end_states, false, true);
  if (!trie_result.has_value()) {
    return std::nullopt;
  }
  auto trie_fsm = trie_result->GetFsm();
  auto start = trie_result->GetStart();
  std::unordered_set<int> old_ends;
  for (int end = 0; end < trie_result->NumStates(); end++) {
    if (trie_result->IsEndState(end)) {
      old_ends.insert(end);
    }
  }
  std::unordered_set<int> ends;

  // The final end states are the end of each stop string.
  for (int i = static_cast<int>(tag_dispatch_rules.size());
       i < static_cast<int>(trie_end_states.size());
       i++) {
    ends.insert(trie_end_states[i]);
  }

  if (loop_after_dispatch) {
    for (int i = 0; i < static_cast<int>(tag_dispatch_rules.size()); i++) {
      trie_fsm.AddRuleEdge(trie_end_states[i], start, tag_dispatch_rules[i].second);
    }
  } else {
    // We should first build a new FSM that only contains the stop strings.
    tag_names.clear();
    for (const auto& stop_string : stop_strings) {
      tag_names.push_back(stop_string);
    }
    std::vector<int> stop_end_states;
    auto stop_trie_result = TrieFSMBuilder::Build(tag_names, nullptr, false, false);
    XGRAMMAR_DCHECK(stop_trie_result.has_value());
    auto stop_trie_fsm = stop_trie_result->GetFsm();
    auto stop_trie_start = stop_trie_result->GetStart();
    std::unordered_set<int> stop_trie_ends;
    for (int end = 0; end < stop_trie_result->NumStates(); end++) {
      if (stop_trie_result->IsEndState(end)) {
        stop_trie_ends.insert(end);
      }
    }

    std::unordered_map<int, int> stop_trie_to_trie_map;
    trie_fsm.AddFSM(stop_trie_fsm, &stop_trie_to_trie_map);
    int start_of_stop_trie = stop_trie_to_trie_map[stop_trie_start];
    for (auto state : stop_trie_ends) {
      ends.insert(stop_trie_to_trie_map[state]);
    }

    for (int i = 0; i < static_cast<int>(tag_dispatch_rules.size()); i++) {
      trie_fsm.AddRuleEdge(trie_end_states[i], start_of_stop_trie, tag_dispatch_rules[i].second);
    }
  }

  return FSMWithStartEnd(trie_fsm, start, ends);
}

std::optional<FSMWithStartEnd> GrammarFSMBuilderImpl::BuildTagDispatchWithEOSStop(
    const std::vector<std::pair<std::string, int>>& tag_dispatch_rules, bool loop_after_dispatch
) {
  std::vector<std::string> tag_names;
  tag_names.reserve(tag_dispatch_rules.size());
  for (const auto& [tag_name, tag_id] : tag_dispatch_rules) {
    tag_names.push_back(tag_name);
  }
  std::vector<int> end_states;
  auto trie_result = TrieFSMBuilder::Build(tag_names, &end_states, false, true);
  if (!trie_result.has_value()) {
    return std::nullopt;
  }
  auto trie_fsm = trie_result->GetFsm();
  auto start = trie_result->GetStart();
  std::unordered_set<int> old_ends;
  std::unordered_set<int> ends;
  for (int end = 0; end < trie_result->NumStates(); end++) {
    if (trie_result->IsEndState(end)) {
      old_ends.insert(end);
    }
  }

  // The final end states are all but old_ends.
  for (int i = 0; i < trie_fsm.NumStates(); i++) {
    if (old_ends.count(i) == 0) {
      ends.insert(i);
    }
  }

  // Add rule ref edges
  for (int i = 0; i < static_cast<int>(tag_dispatch_rules.size()); i++) {
    int next_state;
    if (loop_after_dispatch) {
      next_state = start;
    } else {
      next_state = trie_fsm.AddState();
      ends.insert(next_state);
    }
    trie_fsm.AddRuleEdge(end_states[i], next_state, tag_dispatch_rules[i].second);
  }

  return FSMWithStartEnd(trie_fsm, start, ends);
}

std::optional<FSMWithStartEnd> GrammarFSMBuilderImpl::TagDispatch(
    const Grammar::Impl::TagDispatch& tag_dispatch
) {
  if (tag_dispatch.stop_eos) {
    return BuildTagDispatchWithEOSStop(
        tag_dispatch.tag_rule_pairs, tag_dispatch.loop_after_dispatch
    );
  } else {
    return BuildTagDispatchWithStopString(
        tag_dispatch.tag_rule_pairs, tag_dispatch.stop_str, tag_dispatch.loop_after_dispatch
    );
  }
}

class RepetitionNormalizerImpl {
 public:
  void Apply(Grammar* grammar) {
    for (int i = 0; i < (*grammar)->NumGrammarExprs(); ++i) {
      auto expr = (*grammar)->GetGrammarExpr(i);
      if (expr.type != Grammar::Impl::GrammarExprType::kRepeat) {
        continue;
      }
      int repeat_rule_id = expr[0];
      grammar->ImplPtr()->GetRule(repeat_rule_id).is_exact_lookahead = true;
      if (std::binary_search(
              (*grammar)->allow_empty_rule_ids.begin(),
              (*grammar)->allow_empty_rule_ids.end(),
              repeat_rule_id
          )) {
        // The repeated rule can be empty, so we need to normalize it.
        expr.SetData(1, 0);  // Set min repeat to 0
      }
    }
  }
};
/*!
 * \brief Analyze all referenced rules or the main rule. Return a list of all referenced rule ids.
 * This is useful for dead code elimination.
 */
class UsedRulesAnalyzer : public GrammarVisitor<std::vector<int32_t>> {
 public:
  UsedRulesAnalyzer() = default;

  std::vector<int32_t> Apply(const Grammar& grammar) final {
    InitGrammar(grammar);

    std::set<int32_t> visited;

    std::queue<int32_t>().swap(visit_queue_);

    visit_queue_.push(base_grammar_->GetRootRuleId());
    while (!visit_queue_.empty()) {
      auto rule_id = visit_queue_.front();
      visit_queue_.pop();
      if (visited.count(rule_id)) {
        continue;
      }
      visited.insert(rule_id);
      auto rule = base_grammar_->GetRule(rule_id);
      VisitExpr(rule.body_expr_id);
    }

    return std::vector<int32_t>(visited.begin(), visited.end());
  }

  void VisitTagDispatch(const GrammarExpr& grammar_expr) {
    for (int i = 0; i < grammar_expr.size() - 3; i += 2) {
      visit_queue_.push(grammar_expr[i + 1]);
    }
  }

  void VisitRuleRef(const GrammarExpr& grammar_expr) { visit_queue_.push(grammar_expr[0]); }

  void VisitRepeat(const GrammarExpr& grammar_expr) { visit_queue_.push(grammar_expr[0]); }

 private:
  std::queue<int32_t> visit_queue_;
};

class DeadCodeEliminatorImpl : public GrammarMutator {
 public:
  using GrammarMutator::Apply;
  using GrammarMutator::GrammarMutator;

  Grammar Apply(const Grammar& grammar) final {
    InitGrammar(grammar);
    InitBuilder();
    auto used_rules = UsedRulesAnalyzer().Apply(grammar);
    rule_id_map_.clear();
    for (auto rule_id : used_rules) {
      rule_id_map_[rule_id] = builder_->AddEmptyRule(grammar->GetRule(rule_id).name);
    }
    for (auto rule_id : used_rules) {
      auto rule = grammar->GetRule(rule_id);
      auto new_body_expr_id = VisitExpr(rule.body_expr_id);
      builder_->UpdateRuleBody(rule_id_map_[rule_id], new_body_expr_id);
      builder_->UpdateLookaheadAssertion(
          rule_id_map_[rule_id], VisitLookaheadAssertion(rule.lookahead_assertion_id)
      );
    }
    XGRAMMAR_CHECK(rule_id_map_.count(grammar->GetRootRuleId()) > 0);
    return builder_->Get(rule_id_map_[grammar->GetRootRuleId()]);
  }

  int32_t VisitTagDispatch(const GrammarExpr& grammar_expr) final {
    Grammar::Impl::TagDispatch tag_dispatch = base_grammar_->GetTagDispatch(grammar_expr);
    for (auto& [tag, rule_id] : tag_dispatch.tag_rule_pairs) {
      XGRAMMAR_DCHECK(rule_id_map_.count(rule_id) > 0);
      rule_id = rule_id_map_[rule_id];
    }

    return builder_->AddTagDispatch(tag_dispatch);
  }

  int32_t VisitRuleRef(const GrammarExpr& grammar_expr) final {
    XGRAMMAR_DCHECK(rule_id_map_.count(grammar_expr[0]) > 0);
    auto new_rule_id = rule_id_map_[grammar_expr[0]];
    return builder_->AddRuleRef(new_rule_id);
  }

  int32_t VisitRepeat(const GrammarExpr& grammar_expr) final {
    XGRAMMAR_DCHECK(rule_id_map_.count(grammar_expr[0]) > 0);
    auto new_rule_id = rule_id_map_[grammar_expr[0]];
    return builder_->AddRepeat(new_rule_id, grammar_expr[1], grammar_expr[2]);
  }

 private:
  std::unordered_map<int32_t, int32_t> rule_id_map_;
};

/*!
 * \brief Inline rules that can be inlined.
 *
 * Now we only inline rule references that:
 * 1. at the beginning of a sequence
 * 2. The rule should be a sequence of choices, cannot be empty, cannot refer to other rules
 */
class RuleInlinerImpl : public GrammarMutator {
 public:
  using GrammarMutator::Apply;
  using GrammarMutator::GrammarMutator;

 private:
  int32_t VisitChoices(const GrammarExpr& grammar_expr) final {
    std::vector<int32_t> new_choice_ids;
    for (int i : grammar_expr) {
      auto choice_expr = base_grammar_->GetGrammarExpr(i);
      if (choice_expr.type == ExprType::kEmptyStr) {
        new_choice_ids.push_back(VisitExpr(i));
        continue;
      }
      XGRAMMAR_ICHECK(choice_expr.type == ExprType::kSequence);
      auto first_element = base_grammar_->GetGrammarExpr(choice_expr[0]);
      if (first_element.type != ExprType::kRuleRef) {
        new_choice_ids.push_back(VisitExpr(choice_expr));
        continue;
      }
      auto rule_ref_id = first_element[0];
      if (can_rule_be_inlined_.count(rule_ref_id) == 0) {
        can_rule_be_inlined_[rule_ref_id] = CheckIfRuleCanBeInlined(rule_ref_id);
      }
      if (!can_rule_be_inlined_[rule_ref_id]) {
        new_choice_ids.push_back(VisitExpr(choice_expr));
        continue;
      }

      // Do inlining
      std::vector<int32_t> other_elements;
      for (int i = 1; i < choice_expr.size(); ++i) {
        other_elements.push_back(VisitExpr(choice_expr[i]));
      }

      auto ref_rule = base_grammar_->GetRule(rule_ref_id);
      auto ref_grammar_expr = base_grammar_->GetGrammarExpr(ref_rule.body_expr_id);

      for (auto ref_choice_id : ref_grammar_expr) {
        auto ref_choice_expr = base_grammar_->GetGrammarExpr(ref_choice_id);
        XGRAMMAR_ICHECK(ref_choice_expr.type == ExprType::kSequence);
        std::vector<int32_t> choice_to_add;
        for (auto ref_element_id : ref_choice_expr) {
          choice_to_add.push_back(VisitExpr(ref_element_id));
        }
        choice_to_add.insert(choice_to_add.end(), other_elements.begin(), other_elements.end());
        new_choice_ids.push_back(builder_->AddSequence(choice_to_add));
      }
    }
    return builder_->AddChoices(new_choice_ids);
  }

  /**
   * The rule should be: a sequence of choices, cannot be empty, cannot refer to other rules
   */
  bool CheckIfRuleCanBeInlined(int32_t rule_id) {
    auto rule = base_grammar_->GetRule(rule_id);
    auto grammar_expr = base_grammar_->GetGrammarExpr(rule.body_expr_id);
    if (grammar_expr.type != ExprType::kChoices) {
      return false;
    }
    if (grammar_expr.size() == 0) {
      return false;
    }
    for (auto choice_id : grammar_expr) {
      auto choice_expr = base_grammar_->GetGrammarExpr(choice_id);
      if (choice_expr.type == ExprType::kEmptyStr) {
        return false;
      }
      XGRAMMAR_ICHECK(choice_expr.type == ExprType::kSequence);
      for (auto element_id : choice_expr) {
        auto element_expr = base_grammar_->GetGrammarExpr(element_id);
        if (element_expr.type == ExprType::kRuleRef) {
          return false;
        }
      }
    }
    return true;
  }

  std::unordered_map<int32_t, bool> can_rule_be_inlined_;
};

/*!
 * \brief Finds the rule reference graph of a grammar.
 *
 * The rule reference graph shows which rules reference which other rules.
 * The returned graph is inverted: it points from referee to referer.
 */
class RuleRefGraphFinder : public GrammarVisitor<std::vector<std::vector<int32_t>>> {
 public:
  RuleRefGraphFinder() = default;

  std::vector<std::vector<int32_t>> Apply(const Grammar& grammar) {
    InitGrammar(grammar);
    rule_visit_graph_ = std::vector<std::vector<int32_t>>(base_grammar_->NumRules());
    for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
      auto rule = base_grammar_->GetRule(i);
      auto grammar_expr = base_grammar_->GetGrammarExpr(rule.body_expr_id);
      cur_rule_id_ = i;
      VisitExpr(grammar_expr);
    }
    for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
      std::sort(rule_visit_graph_[i].begin(), rule_visit_graph_[i].end());
      auto end_it = std::unique(rule_visit_graph_[i].begin(), rule_visit_graph_[i].end());
      rule_visit_graph_[i].erase(end_it, rule_visit_graph_[i].end());
    }
    return std::move(rule_visit_graph_);
  }

 private:
  void VisitRuleRef(const GrammarExpr& grammar_expr) {
    rule_visit_graph_[grammar_expr[0]].push_back(cur_rule_id_);
  }

  void VisitRepeat(const GrammarExpr& grammar_expr) {
    rule_visit_graph_[grammar_expr[0]].push_back(cur_rule_id_);
  }

  void VisitTagDispatch(const GrammarExpr& grammar_expr) {
    for (int i = 1; i < grammar_expr.size() - 3; i += 2) {
      rule_visit_graph_[grammar_expr[i]].push_back(cur_rule_id_);
    }
  }

  // Inversed reference graph: pointing from referee to referer
  std::vector<std::vector<int32_t>> rule_visit_graph_;
  int32_t cur_rule_id_;
};

/*!
 * \brief Analyzes which rules in a grammar can match the empty string.
 */
class AllowEmptyRuleAnalyzerImpl : public GrammarVisitor<std::vector<int32_t>> {
 public:
  AllowEmptyRuleAnalyzerImpl() = default;

  std::vector<int32_t> Apply(const Grammar& grammar) final {
    InitGrammar(grammar);

    // Step 1: Find rules that explicitly allow empty string
    std::unordered_set<int32_t> empty_rule_id_set;
    FindExplicitEmptyRules(&empty_rule_id_set);

    // Step 2: Find rules that indirectly allow empty string. Using the Bellman-Ford algorithm
    // on the rule reference graph.
    std::vector<std::vector<int32_t>> rule_ref_graph = RuleRefGraphFinder().Apply(grammar);
    FindIndirectEmptyRules(&empty_rule_id_set, rule_ref_graph);

    auto result = std::vector<int32_t>(empty_rule_id_set.begin(), empty_rule_id_set.end());
    std::sort(result.begin(), result.end());
    return result;
  }

  void FindExplicitEmptyRules(std::unordered_set<int32_t>* empty_rule_id_set) {
    for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
      auto rule = base_grammar_->GetRule(i);
      auto grammar_expr = base_grammar_->GetGrammarExpr(rule.body_expr_id);
      if (grammar_expr.type == ExprType::kTagDispatch) {
        continue;
      }

      XGRAMMAR_DCHECK(grammar_expr.type == ExprType::kChoices);
      if (base_grammar_->GetGrammarExpr(grammar_expr[0]).type == ExprType::kEmptyStr) {
        empty_rule_id_set->insert(i);
        continue;
      }

      for (auto seq_id : grammar_expr) {
        auto seq_expr = base_grammar_->GetGrammarExpr(seq_id);
        if (std::all_of(seq_expr.begin(), seq_expr.end(), [&](int32_t i) {
              return base_grammar_->GetGrammarExpr(i).type == ExprType::kCharacterClassStar;
            })) {
          empty_rule_id_set->insert(i);
          break;
        }
      }
    }
  }

  bool SeqExprIsEpsilon(
      const GrammarExpr& seq_expr, const std::unordered_set<int32_t>& empty_rule_id_set
  ) {
    if (seq_expr.type == ExprType::kEmptyStr) {
      return true;
    }
    XGRAMMAR_DCHECK(seq_expr.type == ExprType::kSequence);

    return std::all_of(seq_expr.begin(), seq_expr.end(), [&](int32_t i) {
      auto element_expr = base_grammar_->GetGrammarExpr(i);
      return (element_expr.type == ExprType::kRuleRef && empty_rule_id_set.count(element_expr[0])
             ) ||
             element_expr.type == ExprType::kCharacterClassStar ||
             (element_expr.type == ExprType::kRepeat &&
              (empty_rule_id_set.count(element_expr[0]) || element_expr[1] == 0));
    });
  }

  void FindIndirectEmptyRules(
      std::unordered_set<int32_t>* empty_rule_id_set,
      const std::vector<std::vector<int32_t>>& rule_ref_graph
  ) {
    std::queue<int32_t> queue;
    for (auto i : *empty_rule_id_set) {
      queue.push(i);
    }

    while (!queue.empty()) {
      auto rule_id = queue.front();
      queue.pop();
      XGRAMMAR_DCHECK(rule_id >= 0 && rule_id < static_cast<int>(rule_ref_graph.size()));
      for (auto referer_rule_id : rule_ref_graph[rule_id]) {
        if (empty_rule_id_set->count(referer_rule_id)) {
          continue;
        }
        auto rule = base_grammar_->GetRule(referer_rule_id);
        auto grammar_expr = base_grammar_->GetGrammarExpr(rule.body_expr_id);

        XGRAMMAR_DCHECK(grammar_expr.type != ExprType::kTagDispatch)
            << "TagDispatch rules should already exist in empty_rule_id_set";

        bool is_epsilon = std::any_of(grammar_expr.begin(), grammar_expr.end(), [&](int32_t i) {
          auto seq_expr = base_grammar_->GetGrammarExpr(i);
          return SeqExprIsEpsilon(seq_expr, *empty_rule_id_set);
        });

        if (is_epsilon) {
          empty_rule_id_set->insert(referer_rule_id);
          queue.push(referer_rule_id);
        }
      }
    }
  }
};

class LookaheadAssertionAnalyzerImpl : public GrammarMutator {
 public:
  using GrammarMutator::GrammarMutator;

  Grammar Apply(const Grammar& grammar) final {
    InitGrammar(grammar);
    InitBuilder(grammar);
    auto root_rule = grammar->GetRootRule();
    for (int i = 0; i < static_cast<int>(grammar->NumRules()); ++i) {
      auto rule = grammar->GetRule(i);
      if (i == grammar->GetRootRuleId()) {
        continue;
      }
      if (rule.lookahead_assertion_id != -1) {
        builder_->UpdateLookaheadExact(i, IsExactLookaheadAssertion(i));
        continue;
      }
      auto look_head_assertion_id = DetectLookaheadAssertion(i);
      if (look_head_assertion_id != -1) {
        builder_->UpdateLookaheadAssertion(i, look_head_assertion_id);
        builder_->UpdateLookaheadExact(i);
      }
    }
    return builder_->Get(grammar->GetRootRuleId());
  }

  bool IsExactLookaheadAssertion(int32_t rule_id) {
    XGRAMMAR_DCHECK(base_grammar_->GetRule(rule_id).lookahead_assertion_id != -1);
    bool found = false;
    for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
      auto rule = base_grammar_->GetRule(i);
      auto grammar_expr = base_grammar_->GetGrammarExpr(rule.body_expr_id);
      if (grammar_expr.type == GrammarExprType::kTagDispatch) {
        for (int j = 1; j < grammar_expr.size() - 3; j += 2) {
          if (grammar_expr[j] == rule_id) {
            return false;
          }
        }
        continue;
      }
      XGRAMMAR_DCHECK(grammar_expr.type == GrammarExprType::kChoices);
      for (auto sequence_id : grammar_expr) {
        auto sequence_expr = base_grammar_->GetGrammarExpr(sequence_id);
        if (sequence_expr.type != GrammarExprType::kSequence) {
          continue;
        }
        auto last_element = base_grammar_->GetGrammarExpr(sequence_expr.end()[-1]);
        if (last_element.type == GrammarExprType::kRuleRef && last_element[0] == rule_id &&
            i != rule_id) {
          return false;
        }

        for (int j = 0; j < sequence_expr.size() - 1; ++j) {
          auto element_expr = base_grammar_->GetGrammarExpr(sequence_expr[j]);
          if (element_expr.type != GrammarExprType::kRuleRef || element_expr[0] != rule_id) {
            continue;
          }
          if (found) {
            return false;
          }
          found = true;
        }
      }
    }
    return found;
  }

  int32_t DetectLookaheadAssertion(int32_t rule_id) {
    std::vector<int32_t> found_sequence;  // Element ids
    bool found = false;
    for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
      auto rule = base_grammar_->GetRule(i);
      auto grammar_expr = base_grammar_->GetGrammarExpr(rule.body_expr_id);
      if (grammar_expr.type == GrammarExprType::kTagDispatch) {
        for (int j = 1; j < grammar_expr.size() - 3; j += 2) {
          if (grammar_expr[j] == rule_id) {
            return -1;
          }
        }
        continue;
      }
      XGRAMMAR_DCHECK(grammar_expr.type == GrammarExprType::kChoices);
      for (auto sequence_id : grammar_expr) {
        auto sequence_expr = base_grammar_->GetGrammarExpr(sequence_id);
        if (sequence_expr.type != GrammarExprType::kSequence) {
          continue;
        }
        auto last_element = base_grammar_->GetGrammarExpr(sequence_expr.end()[-1]);
        if (last_element.type == GrammarExprType::kRuleRef && last_element[0] == rule_id &&
            i != rule_id) {
          return -1;
        }

        for (int j = 0; j < sequence_expr.size() - 1; ++j) {
          auto element_expr = base_grammar_->GetGrammarExpr(sequence_expr[j]);
          if (element_expr.type != GrammarExprType::kRuleRef || element_expr[0] != rule_id) {
            continue;
          }
          if (found) {
            return -1;
          }
          found = true;
          for (int k = j + 1; k < sequence_expr.size(); ++k) {
            found_sequence.push_back(sequence_expr[k]);
          }
        }
      }
    }

    if (!found) {
      return -1;
    }
    return builder_->AddSequence(found_sequence);
  }
};

/*************************** Forward grammar optimizers to their impl ***************************/

Grammar GrammarOptimizer::Optimize(const Grammar& grammar) {
  return GrammarOptimizerImpl().Apply(grammar);
}

void GrammarFSMBuilder::Apply(Grammar* grammar) { GrammarFSMBuilderImpl().Apply(grammar); }

void RepetitionNormalizer::Apply(Grammar* grammar) { RepetitionNormalizerImpl().Apply(grammar); }

std::optional<FSMWithStartEnd> GrammarFSMBuilder::RuleRef(const GrammarExpr& expr) {
  return GrammarFSMBuilderImpl::RuleRef(expr);
}

std::optional<FSMWithStartEnd> GrammarFSMBuilder::CharacterClass(const GrammarExpr& expr) {
  return GrammarFSMBuilderImpl::CharacterClass(expr);
}

std::optional<FSMWithStartEnd> GrammarFSMBuilder::ByteString(const GrammarExpr& expr) {
  return GrammarFSMBuilderImpl::ByteString(expr);
}

std::optional<FSMWithStartEnd> GrammarFSMBuilder::Sequence(
    const GrammarExpr& expr, const Grammar& grammar
) {
  return GrammarFSMBuilderImpl::Sequence(expr, grammar);
}

std::optional<FSMWithStartEnd> GrammarFSMBuilder::Choices(
    const GrammarExpr& expr, const Grammar& grammar
) {
  return GrammarFSMBuilderImpl::Choices(expr, grammar);
}

std::optional<FSMWithStartEnd> GrammarFSMBuilder::TagDispatch(
    const Grammar::Impl::TagDispatch& tag_dispatch
) {
  return GrammarFSMBuilderImpl::TagDispatch(tag_dispatch);
}

Grammar DeadCodeEliminator::Apply(const Grammar& grammar) {
  return DeadCodeEliminatorImpl().Apply(grammar);
}

Grammar RuleInliner::Apply(const Grammar& grammar) { return RuleInlinerImpl().Apply(grammar); }

std::vector<int32_t> AllowEmptyRuleAnalyzer::Apply(const Grammar& grammar) {
  return AllowEmptyRuleAnalyzerImpl().Apply(grammar);
}

Grammar LookaheadAssertionAnalyzer::Apply(const Grammar& grammar) {
  return LookaheadAssertionAnalyzerImpl().Apply(grammar);
}

}  // namespace xgrammar
