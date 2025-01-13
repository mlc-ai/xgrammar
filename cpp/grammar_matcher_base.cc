/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_matcher_base.cc
 * \brief This source file implements the basic matching algorithm from strings to grammar.
 * matcher.cc will handle the logic related to LLM tokens, like accepting tokens, leveraging the
 * token mask cache to generate the mask, etc.
 */

#include "grammar_matcher_base.h"

#include <algorithm>
#include <vector>

#include "grammar_data_structure.h"
#include "persistent_stack.h"
#include "support/encoding.h"

namespace xgrammar {

/*! \brief Check the codepoint is contained in the character class. */
bool GrammarMatcherBase::CheckIfAccepted(const StackElement& stack_element, uint8_t char_value)
    const {
  auto current_sequence = grammar_->GetRuleExpr(stack_element.sequence_id);
  auto current_element = grammar_->GetRuleExpr(current_sequence[stack_element.element_id]);
  if (current_element.type == RuleExprType::kCharacterClass ||
      current_element.type == RuleExprType::kCharacterClassStar) {
    if (stack_element.left_utf8_bytes > 0) {
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
    return current_element[stack_element.element_in_string] == char_value;
  } else {
    XGRAMMAR_LOG(FATAL) << "Unexpected RuleExprType in CheckIfAccepted: "
                        << static_cast<int>(current_element.type);
  }
}

StackElement GrammarMatcherBase::MoveToNextPosition(StackElement stack_element) {
  stack_element.element_id += 1;
  stack_element.element_in_string = 0;
  stack_element.left_utf8_bytes = 0;

  XGRAMMAR_DCHECK(
      stack_element.element_id <= grammar_->GetRuleExpr(stack_element.sequence_id).size()
  );
  return stack_element;
}

StackElement GrammarMatcherBase::AdvanceStackElementWithChar(
    const StackElement& stack_element, uint8_t char_value
) {
  auto current_sequence = grammar_->GetRuleExpr(stack_element.sequence_id);
  auto current_element = grammar_->GetRuleExpr(current_sequence[stack_element.element_id]);
  StackElement new_stack_element = stack_element;
  switch (current_element.type) {
    case RuleExprType::kCharacterClass: {
      if (stack_element.left_utf8_bytes > 1) {
        new_stack_element.left_utf8_bytes -= 1;
        return new_stack_element;
      } else if (stack_element.left_utf8_bytes == 1) {
        return MoveToNextPosition(stack_element);
      }
      // If no left utf8 bytes, check the first byte to find the left bytes needed.
      XGRAMMAR_DCHECK(stack_element.left_utf8_bytes == 0);
      auto [accepted, num_bytes, codepoint] = HandleUTF8FirstByte(char_value);
      XGRAMMAR_DCHECK(accepted);
      if (num_bytes > 1) {
        new_stack_element.left_utf8_bytes = num_bytes - 1;
        return new_stack_element;
      }
      return MoveToNextPosition(stack_element);
    }
    case RuleExprType::kCharacterClassStar: {
      if (stack_element.left_utf8_bytes >= 1) {
        new_stack_element.left_utf8_bytes -= 1;
      } else {
        XGRAMMAR_DCHECK(stack_element.left_utf8_bytes == 0);
        auto [accepted, num_bytes, codepoint] = HandleUTF8FirstByte(char_value);
        XGRAMMAR_DCHECK(accepted);
        new_stack_element.left_utf8_bytes = num_bytes - 1;
      }
      return new_stack_element;
    }
    case RuleExprType::kByteString: {
      if (stack_element.element_in_string + 1 < current_element.size()) {
        new_stack_element.element_in_string += 1;
        return new_stack_element;
      }
      return MoveToNextPosition(stack_element);
    }
    default:
      XGRAMMAR_LOG(FATAL) << "Unexpected RuleExprType in AdvanceStackElementWithChar: "
                          << static_cast<int>(current_element.type);
  }
}

void GrammarMatcherBase::ExpandEquivalentStackElements(
    StackElement cur_stack_element,
    std::vector<int32_t>* new_stack_tops,
    int32_t cur_stack_element_id,
    bool consider_parent
) {
  auto f_add_stack_element = [&](const StackElement& stack_element) {
    if (cur_stack_element_id != -1) {
      return cur_stack_element_id;
    } else {
      return persistent_stack_.NewNode(stack_element);
    }
  };

  auto cur_sequence = grammar_->GetRuleExpr(cur_stack_element.sequence_id);

  // Case 1. The stack element points to the end of a rule.
  if (cur_sequence.size() == cur_stack_element.element_id) {
    if (cur_stack_element.parent_id == StackElement::kNoParent) {
      // Case 1.1. The stack element points to the end of the grammar (meaning the matching
      // succeeded). Insert it and add as a stack top.
      new_stack_tops->push_back(f_add_stack_element(cur_stack_element));
    } else if (consider_parent) {
      // Case 1.2. When consider_parent is true, we should recurse to the parent rule.
      auto new_stack_element = persistent_stack_[cur_stack_element.parent_id];
      new_stack_element.element_id += 1;
      XGRAMMAR_DCHECK(new_stack_element.element_in_string == 0);
      XGRAMMAR_DCHECK(new_stack_element.left_utf8_bytes == 0);
      ExpandEquivalentStackElements(new_stack_element, new_stack_tops, -1, consider_parent);
    }
    // Case 1.3. When consider_parent is false, we do nothing and return.
    return;
  }

  auto current_element = grammar_->GetRuleExpr(cur_sequence[cur_stack_element.element_id]);
  auto stack_element_id = f_add_stack_element(cur_stack_element);

  if (current_element.type == RuleExprType::kCharacterClass ||
      current_element.type == RuleExprType::kByteString ||
      current_element.type == RuleExprType::kCharacterClassStar) {
    new_stack_tops->push_back(stack_element_id);
  } else {
    XGRAMMAR_DCHECK(current_element.type == RuleExprType::kRuleRef);
    auto ref_rule_body = grammar_->GetRuleExpr(grammar_->GetRule(current_element[0]).body_expr_id);
    XGRAMMAR_DCHECK(ref_rule_body.type == RuleExprType::kChoices);
    for (auto sequence_id : ref_rule_body) {
      auto ref_rule_sequence = grammar_->GetRuleExpr(sequence_id);
      if (ref_rule_sequence.type == RuleExprType::kEmptyStr) {
        continue;
      }
      auto new_stack_element = StackElement(current_element[0], sequence_id, 0, stack_element_id);
      ExpandEquivalentStackElements(new_stack_element, new_stack_tops, -1, false);
    }
  }

  if ((current_element.type == RuleExprType::kCharacterClassStar &&
       cur_stack_element.left_utf8_bytes == 0) ||
      (current_element.type == RuleExprType::kRuleRef && std::find(
                                                             grammar_->allow_empty_rule_ids.begin(),
                                                             grammar_->allow_empty_rule_ids.end(),
                                                             current_element[0]
                                                         ) != grammar_->allow_empty_rule_ids.end()
      )) {
    auto next_stack_element = MoveToNextPosition(cur_stack_element);
    ExpandEquivalentStackElements(next_stack_element, new_stack_tops, -1, consider_parent);
  }
}

// while (next_position.parent_id != StackElement::kNoParent) {
//   next_position = persistent_stack_[next_position.parent_id];
//   next_position.element_id += 1;
//   XGRAMMAR_DCHECK(next_position.element_in_string == 0);
//   XGRAMMAR_DCHECK(next_position.left_utf8_bytes == 0);

//   sequence = grammar_->GetRuleExpr(next_position.sequence_id);
//   XGRAMMAR_DCHECK(next_position.element_id <= sequence.size());

//   if (next_position.element_id < sequence.size()) {
//     break;
//   }
// }

//   bool is_first = false;
//   bool is_iteration_successful = true;

//   for (; is_iteration_successful;
//        std::tie(is_iteration_successful, cur_stack_element) =
//            MoveToNextPosition(cur_stack_element, !is_inner_recursion)) {
//     // Insert the node to the tree, if not inserted before.
//     int32_t new_node_id;
//     if (is_first && first_id_if_inserted != -1) {
//       new_node_id = first_id_if_inserted;
//     } else {
//       new_node_id = persistent_stack_.NewNode(cur_stack_element);
//     }
//     is_first = false;

//     // Case 1. The current position points to the end of the grammar.
//     if (!is_inner_recursion) {
//       if (persistent_stack_.IsEndOfGrammar(cur_stack_element)) {
//         new_stack_tops->push_back(new_node_id);
//         return true;
//       }
//     } else {
//       XGRAMMAR_DCHECK(!persistent_stack_.IsEndOfGrammar(cur_stack_element));
//     }

//     auto sequence = grammar_->GetRuleExpr(cur_stack_element.sequence_id);
//     auto element = grammar_->GetRuleExpr(sequence[cur_stack_element.element_id]);
//     bool can_be_empty = false;

//     if (element.type == RuleExprType::kRuleRef) {
//       // Case 2. The current position refers to another rule.
//       auto ref_rule = grammar_->GetRule(element[0]);
//       auto ref_rule_body = grammar_->GetRuleExpr(ref_rule.body_expr_id);
//       XGRAMMAR_DCHECK(ref_rule_body.type == RuleExprType::kChoices);

//       for (auto sequence_id : ref_rule_body) {
//         auto ref_rule_sequence = grammar_->GetRuleExpr(sequence_id);
//         if (ref_rule_sequence.type == RuleExprType::kEmptyStr) {
//           can_be_empty = true;
//           continue;
//         }
//         auto ref_stack_element = StackElement(element[0], sequence_id, 0, new_node_id);
//         // Find the positions in every choice of the referred rule
//         can_be_empty |=
//             ExpandEquivalentStackElements(ref_stack_element, new_stack_tops, -1, false);
//       }
//     } else if (element.type == RuleExprType::kCharacterClass ||
//                element.type == RuleExprType::kByteString) {
//       // Case 3. Character class or byte string. cannot be empty.
//       new_stack_tops->push_back(new_node_id);
//       can_be_empty = false;
//     } else {
//       XGRAMMAR_DCHECK(element.type == RuleExprType::kCharacterClassStar);
//       // Case 4. Character class star. Might be empty.
//       new_stack_tops->push_back(new_node_id);
//       can_be_empty = cur_stack_element.left_utf8_bytes == 0;
//     }

//     if (!can_be_empty) {
//       return false;
//     }
//   }
//   return true;
// }

bool GrammarMatcherBase::AcceptChar(uint8_t char_value, bool debug_print) {
  if (debug_print) {
    XGRAMMAR_LOG(INFO) << "Matching char: " << static_cast<int>(char_value) << " \""
                       << PrintAsEscapedUTF8(char_value) << "\"";
    XGRAMMAR_LOG(INFO) << "Previous stack: " << PrintStackState();
  }
  const auto& prev_stack_tops = stack_tops_history_.GetLatest();

  tmp_new_stack_tops_.clear();
  for (auto prev_top : prev_stack_tops) {
    auto cur_stack_element = persistent_stack_[prev_top];
    if (persistent_stack_.IsEndOfGrammar(cur_stack_element)) {
      // This StackElement means previous elements has matched the complete rule.
      // But we are still need to accept a new character, so this stack will become invalid.
      continue;
    }

    auto accepted = CheckIfAccepted(cur_stack_element, char_value);
    if (!accepted) {
      continue;
    }

    auto new_stack_element = AdvanceStackElementWithChar(cur_stack_element, char_value);

    if (new_stack_element == cur_stack_element) {
      ExpandEquivalentStackElements(new_stack_element, &tmp_new_stack_tops_, prev_top);
    } else {
      ExpandEquivalentStackElements(new_stack_element, &tmp_new_stack_tops_);
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
    return persistent_stack_.IsEndOfGrammar(persistent_stack_[id]);
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
    StackElement init_stack_element, bool expand_init_stack_element
) {
  if (init_stack_element == kInvalidStackElement) {
    // Initialize the stack with the root rule.
    auto root_rule = grammar_->GetRootRule();
    auto root_rule_body = grammar_->GetRuleExpr(root_rule.body_expr_id);
    tmp_new_stack_tops_.clear();
    for (auto i : root_rule_body) {
      auto init_stack_element = StackElement(0, i, 0, StackElement::kNoParent);
      if (expand_init_stack_element) {
        ExpandEquivalentStackElements(init_stack_element, &tmp_new_stack_tops_);
      } else {
        tmp_new_stack_tops_.push_back(persistent_stack_.NewNode(init_stack_element));
      }
    }
    stack_tops_history_.PushHistory(tmp_new_stack_tops_);
  } else {
    if (expand_init_stack_element) {
      tmp_new_stack_tops_.clear();
      ExpandEquivalentStackElements(init_stack_element, &tmp_new_stack_tops_);
      stack_tops_history_.PushHistory(tmp_new_stack_tops_);
    } else {
      stack_tops_history_.PushHistory({persistent_stack_.NewNode(init_stack_element)});
    }
  }
}

}  // namespace xgrammar
