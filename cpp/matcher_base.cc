/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/regex_converter.cc
 */

#include "matcher_base.h"

#include <algorithm>
#include <vector>

#include "grammar_data_structure.h"
#include "matcher_data_structure.h"
#include "support/encoding.h"

namespace xgrammar {

/*! \brief Check the codepoint is contained in the character class. */
bool GrammarMatcherBase::CheckIfAccepted(const RulePosition& rule_position, uint8_t char_value)
    const {
  auto current_sequence = grammar_->GetRuleExpr(rule_position.sequence_id);
  auto current_element = grammar_->GetRuleExpr(current_sequence[rule_position.element_id]);
  if (current_element.type == RuleExprType::kCharacterClass ||
      current_element.type == RuleExprType::kCharacterClassStar) {
    if (rule_position.left_utf8_bytes > 0) {
      return (char_value & 0xC0) == 0x80;
    }
    auto [accepted, num_bytes, codepoint] = HandleUTF8FirstByte(char_value);
    if (!accepted) {
      return false;
    }
    bool is_negative = static_cast<bool>(current_element[0]);
    if (num_bytes > 1) {
      return is_negative;
    }
    for (int i = 1; i < current_element.size(); i += 2) {
      if (current_element[i] <= char_value && char_value <= current_element[i + 1]) {
        return !is_negative;
      }
    }
    return is_negative;
  } else if (current_element.type == RuleExprType::kByteString) {
    return current_element[rule_position.element_in_string] == char_value;
  } else {
    XGRAMMAR_LOG(FATAL) << "Unexpected RuleExprType in CheckIfAccepted: "
                        << static_cast<int>(current_element.type);
  }
}

RulePosition GrammarMatcherBase::UpdatePositionWithChar(
    const RulePosition& rule_position, uint8_t char_value
) const {
  auto current_sequence = grammar_->GetRuleExpr(rule_position.sequence_id);
  auto current_element = grammar_->GetRuleExpr(current_sequence[rule_position.element_id]);
  RulePosition new_rule_position = rule_position;
  switch (current_element.type) {
    case RuleExprType::kCharacterClass: {
      if (rule_position.left_utf8_bytes > 1) {
        new_rule_position.left_utf8_bytes -= 1;
        return new_rule_position;
      } else if (rule_position.left_utf8_bytes == 1) {
        return GetNextPositionInSequence(rule_position, true).second;
      }
      // If no left utf8 bytes, check the first byte to find the left bytes needed.
      XGRAMMAR_DCHECK(rule_position.left_utf8_bytes == 0);
      auto [accepted, num_bytes, codepoint] = HandleUTF8FirstByte(char_value);
      XGRAMMAR_DCHECK(accepted);
      if (num_bytes > 1) {
        new_rule_position.left_utf8_bytes = num_bytes - 1;
        return new_rule_position;
      }
      return GetNextPositionInSequence(rule_position, true).second;
    }
    case RuleExprType::kCharacterClassStar: {
      if (rule_position.left_utf8_bytes >= 1) {
        new_rule_position.left_utf8_bytes -= 1;
      } else {
        XGRAMMAR_DCHECK(rule_position.left_utf8_bytes == 0);
        auto [accepted, num_bytes, codepoint] = HandleUTF8FirstByte(char_value);
        XGRAMMAR_DCHECK(accepted);
        new_rule_position.left_utf8_bytes = num_bytes - 1;
      }
      return new_rule_position;
    }
    case RuleExprType::kByteString: {
      if (rule_position.element_in_string + 1 < current_element.size()) {
        new_rule_position.element_in_string += 1;
        return new_rule_position;
      }
      return GetNextPositionInSequence(rule_position, true).second;
    }
    default:
      XGRAMMAR_LOG(FATAL) << "Unexpected RuleExprType in UpdatePositionWithChar: "
                          << static_cast<int>(current_element.type);
  }
}

bool GrammarMatcherBase::AcceptChar(uint8_t char_value, bool debug_print) {
  if (debug_print) {
    XGRAMMAR_LOG(INFO) << "Matching char: " << static_cast<int>(char_value) << " \""
                       << PrintAsEscapedUTF8(char_value) << "\"";
    XGRAMMAR_LOG(INFO) << "Previous stack: " << PrintStackState();
  }
  const auto& prev_stack_tops = stack_tops_history_.GetLatest();

  tmp_new_stack_tops_.clear();
  for (auto prev_top : prev_stack_tops) {
    auto cur_rule_position = persistent_stack_[prev_top];
    auto current_sequence = grammar_->GetRuleExpr(cur_rule_position.sequence_id);
    if (cur_rule_position.parent_id == RulePosition::kNoParent &&
        cur_rule_position.element_id == current_sequence.size()) {
      // This RulePosition means previous elements has matched the complete rule.
      // But we are still need to accept a new character, so this stack will become invalid.
      continue;
    }

    auto accepted = CheckIfAccepted(cur_rule_position, char_value);
    if (!accepted) {
      continue;
    }

    auto new_rule_position = UpdatePositionWithChar(cur_rule_position, char_value);

    if (new_rule_position == cur_rule_position) {
      ExpandRulePosition(new_rule_position, &tmp_new_stack_tops_, true, prev_top);
    } else {
      ExpandRulePosition(new_rule_position, &tmp_new_stack_tops_, true);
    }
  }
  if (tmp_new_stack_tops_.empty()) {
    if (debug_print) {
      XGRAMMAR_LOG(INFO) << "Character " << static_cast<int>(char_value) << " \""
                         << PrintAsEscapedUTF8(char_value) << "\" Rejected";
    }
    return false;
  }
  stack_tops_history_.PushHistory(tmp_new_stack_tops_);
  if (debug_print) {
    XGRAMMAR_LOG(INFO) << "Character: " << static_cast<int>(char_value) << " \""
                       << PrintAsEscapedUTF8(char_value) << "\" Accepted";
    XGRAMMAR_LOG(INFO) << "New stack after acceptance: " << PrintStackState();
  }

  constexpr bool DEBUG_CHECK_WELL_FORMED = false;
  if (DEBUG_CHECK_WELL_FORMED) {
    stack_tops_history_.CheckWellFormed();
  }
  return true;
}

bool GrammarMatcherBase::CanReachEnd() const {
  const auto& last_stack_tops = stack_tops_history_.GetLatest();
  return std::any_of(last_stack_tops.begin(), last_stack_tops.end(), [&](int32_t id) {
    return persistent_stack_.IsEndPosition(persistent_stack_[id]);
  });
}

void GrammarMatcherBase::RollbackChars(int rollback_cnt) {
  stack_tops_history_.Rollback(rollback_cnt);
}

void GrammarMatcherBase::DiscardEarliestChars(int discard_cnt) {
  stack_tops_history_.DiscardEarliest(discard_cnt);
}

std::string GrammarMatcherBase::PrintStackState(int steps_behind_latest) const {
  return stack_tops_history_.PrintHistory(steps_behind_latest);
}

void GrammarMatcherBase::PushInitialState(
    RulePosition init_rule_position, bool expand_init_rule_position
) {
  if (init_rule_position == kInvalidRulePosition) {
    // Initialize the stack with the root rule.
    auto root_rule = grammar_->GetRootRule();
    auto root_rule_body = grammar_->GetRuleExpr(root_rule.body_expr_id);
    tmp_new_stack_tops_.clear();
    for (auto i : root_rule_body) {
      auto init_rule_position = RulePosition(0, i, 0, RulePosition::kNoParent);
      if (expand_init_rule_position) {
        ExpandRulePosition(init_rule_position, &tmp_new_stack_tops_, true);
      } else {
        tmp_new_stack_tops_.push_back(persistent_stack_.NewNode(init_rule_position));
      }
    }
    stack_tops_history_.PushHistory(tmp_new_stack_tops_);
  } else {
    if (expand_init_rule_position) {
      tmp_new_stack_tops_.clear();
      ExpandRulePosition(init_rule_position, &tmp_new_stack_tops_, true);
      stack_tops_history_.PushHistory(tmp_new_stack_tops_);
    } else {
      stack_tops_history_.PushHistory({persistent_stack_.NewNode(init_rule_position)});
    }
  }
}

std::pair<bool, RulePosition> GrammarMatcherBase::GetNextPositionInSequence(
    const RulePosition& rule_position, bool consider_parent
) const {
  auto sequence = grammar_->GetRuleExpr(rule_position.sequence_id);

  auto next_position = rule_position;
  next_position.element_id += 1;
  next_position.element_in_string = 0;
  next_position.left_utf8_bytes = 0;

  XGRAMMAR_DCHECK(next_position.element_id <= sequence.size());

  if (next_position.element_id < sequence.size()) {
    return {true, next_position};
  }

  if (!consider_parent) {
    return {false, kInvalidRulePosition};
  }

  // Find the next position in the parent rule
  while (next_position.parent_id != RulePosition::kNoParent) {
    next_position = persistent_stack_[next_position.parent_id];
    next_position.element_id += 1;
    XGRAMMAR_DCHECK(next_position.element_in_string == 0);
    XGRAMMAR_DCHECK(next_position.left_utf8_bytes == 0);

    sequence = grammar_->GetRuleExpr(next_position.sequence_id);
    XGRAMMAR_DCHECK(next_position.element_id <= sequence.size());

    if (next_position.element_id < sequence.size()) {
      break;
    }
  }

  return {true, next_position};
}

bool GrammarMatcherBase::ExpandRulePosition(
    RulePosition cur_rule_position,
    std::vector<int32_t>* new_stack_tops,
    bool consider_parent,
    int32_t first_id_if_inserted
) {
  bool is_first = false;
  bool is_iteration_successful = true;

  for (; is_iteration_successful;
       std::tie(is_iteration_successful, cur_rule_position) =
           GetNextPositionInSequence(cur_rule_position, consider_parent)) {
    // Insert the node to the tree, if not inserted before.
    int32_t new_node_id;
    if (is_first && first_id_if_inserted != -1) {
      new_node_id = first_id_if_inserted;
    } else {
      new_node_id = persistent_stack_.NewNode(cur_rule_position);
    }
    is_first = false;

    // Case 1. The current position points to the end of the grammar.
    if (consider_parent) {
      if (persistent_stack_.IsEndPosition(cur_rule_position)) {
        new_stack_tops->push_back(new_node_id);
        return true;
      }
    } else {
      XGRAMMAR_DCHECK(!persistent_stack_.IsEndPosition(cur_rule_position));
    }

    auto sequence = grammar_->GetRuleExpr(cur_rule_position.sequence_id);
    auto element = grammar_->GetRuleExpr(sequence[cur_rule_position.element_id]);
    bool can_be_empty = false;

    if (element.type == RuleExprType::kRuleRef) {
      // Case 2. The current position refers to another rule.
      auto ref_rule = grammar_->GetRule(element[0]);
      auto ref_rule_body = grammar_->GetRuleExpr(ref_rule.body_expr_id);
      XGRAMMAR_DCHECK(ref_rule_body.type == RuleExprType::kChoices);

      for (auto sequence_id : ref_rule_body) {
        auto ref_rule_sequence = grammar_->GetRuleExpr(sequence_id);
        if (ref_rule_sequence.type == RuleExprType::kEmptyStr) {
          can_be_empty = true;
          continue;
        }
        auto ref_rule_position = RulePosition(element[0], sequence_id, 0, new_node_id);
        // Find the positions in every choice of the referred rule
        can_be_empty |= ExpandRulePosition(ref_rule_position, new_stack_tops, false);
      }
    } else if (element.type == RuleExprType::kCharacterClass ||
               element.type == RuleExprType::kByteString) {
      // Case 3. Character class or byte string. cannot be empty.
      new_stack_tops->push_back(new_node_id);
      can_be_empty = false;
    } else {
      XGRAMMAR_DCHECK(element.type == RuleExprType::kCharacterClassStar);
      // Case 4. Character class star. Might be empty.
      new_stack_tops->push_back(new_node_id);
      can_be_empty = cur_rule_position.left_utf8_bytes == 0;
    }

    if (!can_be_empty) {
      return false;
    }
  }
  return true;
}
}  // namespace xgrammar
