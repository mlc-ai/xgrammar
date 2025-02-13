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
#include "support/utils.h"

namespace xgrammar {

constexpr int32_t kUnexpandedRuleStartSequenceId = 128000;

constexpr int32_t kDispatchedTagDispatchElementId = -1;

/*! \brief Check the codepoint is contained in the character class. */
bool GrammarMatcherBase::CheckIfAccepted(const StackElement& stack_element, uint8_t char_value)
    const {
  auto current_sequence = grammar_->GetRuleExpr(stack_element.sequence_id);
  if (current_sequence.type == Grammar::Impl::RuleExprType::kTagDispatch) {
    XGRAMMAR_DCHECK(stack_element.element_id != -1);
    return true;
  }

  if (stack_element.parent_id == StackElement::kNoParent &&
      current_sequence.size() == stack_element.element_id) {
    // This StackElement means previous elements has matched the complete rule.
    // But we are still need to accept a new character, so this stack will become invalid.
    return false;
  }

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

StackElement GrammarMatcherBase::MoveToNextPosition(const StackElement& stack_element) {
  StackElement new_stack_element = stack_element;
  new_stack_element.element_id += 1;
  new_stack_element.element_in_string = 0;
  new_stack_element.left_utf8_bytes = 0;

  XGRAMMAR_DCHECK(
      new_stack_element.element_id <= grammar_->GetRuleExpr(stack_element.sequence_id).size()
  );
  return new_stack_element;
}

StackElement GrammarMatcherBase::AdvanceStackElementWithChar(
    const StackElement& stack_element, uint8_t char_value
) {
  auto current_sequence = grammar_->GetRuleExpr(stack_element.sequence_id);
  if (current_sequence.type == Grammar::Impl::RuleExprType::kTagDispatch) {
    auto root_tag_dispatch_fsm = grammar_->root_tag_dispatch_fsm;
    if (!root_tag_dispatch_fsm) {
      XGRAMMAR_LOG(FATAL) << "The grammar does not have a root tag dispatch rule; it is not built.";
      XGRAMMAR_UNREACHABLE();
    }
    auto start_node = root_tag_dispatch_fsm->StartNode();
    auto next_node = root_tag_dispatch_fsm->Transition(stack_element.element_id, char_value);
    auto new_stack_element = stack_element;
    if (next_node == CompactFSM::NO_TRANSITION) {
      // Case 1. The new char cannot continue to be accepted by the tag dispatch fsm.
      // We try to accept the new char from the start node. If accepted, we go to the target node.
      // If it still cannot be accepted, we stay at the start node.
      auto new_next_node = root_tag_dispatch_fsm->Transition(start_node, char_value);
      new_stack_element.element_id =
          new_next_node == CompactFSM::NO_TRANSITION ? start_node : new_next_node;
    } else if (!root_tag_dispatch_fsm->IsEndNode(next_node)) {
      // Case 2. The new char can continue to be accepted by the tag dispatch fsm.
      // We need to update the element id to the next node.
      new_stack_element.element_id = next_node;
    } else {
      // Case 3. The new char can continue to be accepted by the tag dispatch fsm.
      // We need to dispatch the tag dispatch fsm to the end node.
      // We need to create a new stack element to represent the dispatched tag dispatch.
      new_stack_element.element_id = kDispatchedTagDispatchElementId;
      auto new_stack_element_id = persistent_stack_.NewNode(new_stack_element);
      XGRAMMAR_DCHECK(grammar_->tag_dispatch_end_node_to_rule_id.count(next_node))
          << "The end node of the tag dispatch fsm does not correspond to any rule id";
      auto refered_rule_id = grammar_->tag_dispatch_end_node_to_rule_id.at(next_node);
      new_stack_element =
          StackElement(refered_rule_id, kUnexpandedRuleStartSequenceId, 0, new_stack_element_id);
    }
    return new_stack_element;
  }

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
    const StackElement& cur_stack_element,
    std::vector<int32_t>* new_stack_tops,
    int32_t cur_stack_element_id,
    bool consider_parent
) {
  auto f_add_current_stack_element = [&]() {
    if (cur_stack_element_id != -1) {
      return cur_stack_element_id;
    } else {
      return persistent_stack_.NewNode(cur_stack_element);
    }
  };

  // Step 1. Handle unexpanded rules.
  if (cur_stack_element.sequence_id == kUnexpandedRuleStartSequenceId) {
    auto cur_rule_id = cur_stack_element.rule_id;
    auto cur_rule_body_id = grammar_->GetRule(cur_rule_id).body_expr_id;
    auto cur_rule_body = grammar_->GetRuleExpr(cur_rule_body_id);

    if (cur_rule_body.type == RuleExprType::kTagDispatch) {
      auto new_stack_element = StackElement(
          cur_rule_id,
          cur_rule_body_id,
          grammar_->root_tag_dispatch_fsm->StartNode(),
          cur_stack_element.parent_id
      );
      new_stack_tops->push_back(persistent_stack_.NewNode(new_stack_element));
      return;
    } else {
      XGRAMMAR_DCHECK(cur_rule_body.type == RuleExprType::kChoices);
      for (auto sequence_id : cur_rule_body) {
        auto ref_rule_sequence = grammar_->GetRuleExpr(sequence_id);
        if (ref_rule_sequence.type == RuleExprType::kEmptyStr &&
            cur_stack_element.parent_id != StackElement::kNoParent) {
          // If the empty string is in a root rule, it indicates the end of the grammar and we
          // just add it as a stack top to indicate the matching ends.
          continue;
        }
        auto new_stack_element =
            StackElement(cur_rule_id, sequence_id, 0, cur_stack_element.parent_id);
        ExpandEquivalentStackElements(new_stack_element, new_stack_tops, -1, false);
      }
      return;
    }
  }

  auto cur_sequence = grammar_->GetRuleExpr(cur_stack_element.sequence_id);

  // If the current sequence is a tag dispatch, it do not have any other equivalent stack elements.
  if (cur_sequence.type == RuleExprType::kTagDispatch) {
    new_stack_tops->push_back(f_add_current_stack_element());
    return;
  }

  // Step 2. The stack element points to the end of a rule.
  if (cur_sequence.size() == cur_stack_element.element_id) {
    if (cur_stack_element.parent_id == StackElement::kNoParent) {
      // Case 2.1. The stack element points to the end of the grammar (meaning the matching
      // succeeded). Insert it and add as a stack top.
      new_stack_tops->push_back(f_add_current_stack_element());
    } else if (consider_parent) {
      // Case 2.2. When consider_parent is true, we should recurse to the parent rule.
      auto new_stack_element = persistent_stack_[cur_stack_element.parent_id];
      auto parent_sequence = grammar_->GetRuleExpr(new_stack_element.sequence_id);
      if (parent_sequence.type == RuleExprType::kTagDispatch) {
        new_stack_element.element_id = grammar_->root_tag_dispatch_fsm->StartNode();
      } else {
        new_stack_element.element_id += 1;
      }
      XGRAMMAR_DCHECK(new_stack_element.element_in_string == 0);
      XGRAMMAR_DCHECK(new_stack_element.left_utf8_bytes == 0);
      ExpandEquivalentStackElements(new_stack_element, new_stack_tops, -1, consider_parent);
    }
    // Case 2.3. When consider_parent is false, we do nothing and return.
    return;
  }

  auto current_element = grammar_->GetRuleExpr(cur_sequence[cur_stack_element.element_id]);
  auto stack_element_id = f_add_current_stack_element();

  // Step 3. Iterate into sub rules
  if (current_element.type == RuleExprType::kRuleRef) {
    ExpandEquivalentStackElements(
        StackElement(current_element[0], kUnexpandedRuleStartSequenceId, 0, stack_element_id),
        new_stack_tops,
        -1,
        false
    );
  } else {
    XGRAMMAR_DCHECK(
        current_element.type == RuleExprType::kCharacterClass ||
        current_element.type == RuleExprType::kByteString ||
        current_element.type == RuleExprType::kCharacterClassStar
    );
    new_stack_tops->push_back(stack_element_id);
  }

  // Step 4. Check the next element in the same rule
  auto exist_in_vector = [](const std::vector<int32_t>& vec, int32_t value) {
    return std::find(vec.begin(), vec.end(), value) != vec.end();
  };
  if ((current_element.type == RuleExprType::kCharacterClassStar &&
       cur_stack_element.left_utf8_bytes == 0) ||
      (current_element.type == RuleExprType::kRuleRef &&
       exist_in_vector(grammar_->allow_empty_rule_ids, current_element[0]))) {
    auto next_stack_element = MoveToNextPosition(cur_stack_element);
    ExpandEquivalentStackElements(next_stack_element, new_stack_tops, -1, consider_parent);
  }
}

bool GrammarMatcherBase::AcceptChar(uint8_t char_value, bool debug_print) {
  if (debug_print) {
    XGRAMMAR_LOG(INFO) << "Trying to accept char: " << static_cast<int>(char_value) << " \""
                       << PrintAsEscapedUTF8(char_value) << "\"";
  }
  const auto& prev_stack_tops = stack_tops_history_.GetLatest();

  tmp_new_stack_tops_.clear();
  for (auto prev_top : prev_stack_tops) {
    auto cur_stack_element = persistent_stack_[prev_top];

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

std::string GrammarMatcherBase::PrintStackState(int steps_before_latest) const {
  return stack_tops_history_.PrintHistory(steps_before_latest);
}

void GrammarMatcherBase::PushInitialState(
    const StackElement& init_stack_element, bool expand_init_stack_element
) {
  if (init_stack_element == kInvalidStackElement) {
    // Initialize the stack with the root rule.
    auto init_stack_element = StackElement(
        grammar_->GetRootRuleId(), kUnexpandedRuleStartSequenceId, 0, StackElement::kNoParent
    );
    tmp_new_stack_tops_.clear();
    ExpandEquivalentStackElements(init_stack_element, &tmp_new_stack_tops_);
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
