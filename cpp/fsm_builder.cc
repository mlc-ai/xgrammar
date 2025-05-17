/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm_builder.cc
 */

#include "fsm_builder.h"

#include <cstdint>
#include <memory>
#include <stack>
#include <variant>

#include "fsm.h"
#include "support/logging.h"
#include "support/utils.h"

namespace xgrammar {

Result<RegexIR> FSMBuilder::BuildRegexIR(const std::string& regex) {
  // initialization.
  grammar_ = regex;
  current_parsing_index_ = 0;
  while (!stack_.empty()) {
    stack_.pop();
  }
  CheckStartEndOfRegex();

  while (current_parsing_index_ < grammar_.size()) {
    if (Peek() == '[') {
      bool successful = HandleCharacterClass();
      if (!successful) {
        return Result<RegexIR>::Err(std::make_shared<Error>("Unmatched '['"));
      }
      continue;
    }

    if (Peek() == '(' || Peek() == '|') {
      stack_.push(Peek());
      current_parsing_index_++;
      continue;
    }

    if (Peek() == '+' || Peek() == '?' || Peek() == '*') {
      HandleSymbol();
      continue;
    }

    if (Peek() == '{') {
      int8_t repeat_status_code = TryHandleRepeat();
      if (repeat_status_code == kIsRepeat) {
        continue;
      }
      if (repeat_status_code == kParsingFailure) {
        return Result<RegexIR>::Err(std::make_shared<Error>(
            "Parsing failure: there's no elements before a repeat quantifier!"
        ));
      }
      XGRAMMAR_DCHECK(repeat_status_code == kIsNotRepeat);
      // It's just a character.
    }

    if (Peek() == ')') {
      bool successful = HandleBracket();
      if (!successful) {
        return Result<RegexIR>::Err(std::make_shared<Error>("Unmatched ')'"));
      }
      continue;
    }
    // It's matching characters at present.
    HandleString();
  }

  return BuildRegexIRFromStack();
}

void FSMBuilder::CheckStartEndOfRegex() {
  // The first character is ignored.
  if (!grammar_.empty() && grammar_[0] == '^') {
    current_parsing_index_ = 1;
  }
  if (!grammar_.empty() && grammar_[grammar_.size() - 1] == '$') {
    grammar_ = grammar_.substr(0, grammar_.size() - 1);
  }
}

void FSMBuilder::ConsumeWhiteSpace() {
  while (current_parsing_index_ < (grammar_.size() - 1) &&
         (grammar_[current_parsing_index_] == ' ' || grammar_[current_parsing_index_] == '\n' ||
          grammar_[current_parsing_index_] == '\t')) {
    current_parsing_index_++;
  }
}

void FSMBuilder::HandleString() {
  XGRAMMAR_DCHECK(current_parsing_index_ < grammar_.size());
  RegexIR::Leaf character_node;
  character_node.regex = grammar_[current_parsing_index_];
  stack_.push(character_node);
  current_parsing_index_++;
}

bool FSMBuilder::HandleCharacterClass() {
  XGRAMMAR_DCHECK(current_parsing_index_ < grammar_.size());
  XGRAMMAR_DCHECK(grammar_[current_parsing_index_] == '[');
  int left_bracket_index = current_parsing_index_;
  while (current_parsing_index_ < grammar_.size() && grammar_[current_parsing_index_] != ']') {
    if (grammar_[current_parsing_index_] == '\\') {
      current_parsing_index_++;
    }
    current_parsing_index_++;
  }
  if (current_parsing_index_ >= grammar_.size()) {
    return false;  // Error: unmatched '['
  }
  RegexIR::Leaf character_class_node;
  character_class_node.regex =
      grammar_.substr(left_bracket_index, current_parsing_index_ - left_bracket_index + 1);
  stack_.push(character_class_node);
  current_parsing_index_++;
  return true;
}

bool FSMBuilder::HandleSymbol() {
  XGRAMMAR_DCHECK(current_parsing_index_ < grammar_.size());
  if (stack_.empty()) {
    return false;
  }
  RegexIR::Symbol symbol_node;
  auto last_element = std::move(stack_.top());
  if (std::holds_alternative<char>(last_element)) {
    return false;  // Error: no node before operator
  }
  stack_.pop();
  symbol_node.node = std::make_shared<RegexIR::Node>(std::get<RegexIR::Node>(last_element));
  switch (grammar_[current_parsing_index_]) {
    case '+':
      symbol_node.symbol = RegexIR::RegexSymbol::plus;
      break;
    case '?':
      symbol_node.symbol = RegexIR::RegexSymbol::optional;
      break;
    case '*':
      symbol_node.symbol = RegexIR::RegexSymbol::star;
      break;
    default:
      XGRAMMAR_LOG(FATAL) << "Invalid grammar: invalid operator!";
  }
  stack_.push(symbol_node);
  current_parsing_index_++;
  return true;
}

int8_t FSMBuilder::TryHandleRepeat() {
  XGRAMMAR_DCHECK(current_parsing_index_ < grammar_.size());
  XGRAMMAR_DCHECK(Peek() == '{');
  int start_index = current_parsing_index_;
  current_parsing_index_++;

  ConsumeWhiteSpace();
  int lower_bound = ParsingPositveInteger();
  if (lower_bound == kNotNumber) {
    // It's not a repeat. It's just a character.
    current_parsing_index_ = start_index;
    return kIsNotRepeat;
  }
  ConsumeWhiteSpace();

  if (Peek() != ',' && Peek() != '}') {
    current_parsing_index_ = start_index;
    return kIsNotRepeat;
  }

  // Handle the case of {n}.
  if (Peek() == '}') {
    if (stack_.empty()) {
      return kParsingFailure;
    }
    auto last_element = std::move(stack_.top());
    if (std::holds_alternative<char>(last_element)) {
      current_parsing_index_ = start_index;
      return kParsingFailure;
    }
    stack_.pop();
    RegexIR::Repeat repeat_node;
    repeat_node.lower_bound = lower_bound;
    repeat_node.upper_bound = lower_bound;
    repeat_node.node = std::make_shared<RegexIR::Node>(std::get<RegexIR::Node>(last_element));
    stack_.push(repeat_node);
    current_parsing_index_++;
    return kIsRepeat;
  }

  // Handling the cases of {n,} and {n,m}.
  XGRAMMAR_DCHECK(Peek() == ',');
  current_parsing_index_++;
  ConsumeWhiteSpace();
  int upper_bound = ParsingPositveInteger();
  ConsumeWhiteSpace();

  if (Peek() != '}') {
    current_parsing_index_ = start_index;
    return kIsNotRepeat;
  }

  if (stack_.empty()) {
    return kParsingFailure;
  }
  auto last_element = std::move(stack_.top());
  if (std::holds_alternative<char>(last_element)) {
    current_parsing_index_ = start_index;
    return kParsingFailure;
  }
  stack_.pop();
  RegexIR::Repeat repeat_node;
  repeat_node.lower_bound = lower_bound;
  if (upper_bound == kNotNumber) {
    repeat_node.upper_bound = RegexIR::KRepeatNoUpperBound;
  } else {
    repeat_node.upper_bound = upper_bound;
  }
  repeat_node.node = std::make_shared<RegexIR::Node>(std::get<RegexIR::Node>(last_element));
  stack_.push(repeat_node);
  current_parsing_index_++;
  return kIsRepeat;
}

bool FSMBuilder::HandleBracket() {
  XGRAMMAR_DCHECK(current_parsing_index_ < grammar_.size());
  XGRAMMAR_DCHECK(grammar_[current_parsing_index_] == ')');
  std::stack<IRNode> element_in_bracket;
  bool paired = false;
  bool unioned = false;

  while ((!stack_.empty()) && (!paired)) {
    auto node = stack_.top();
    stack_.pop();
    if (std::holds_alternative<char>(node)) {
      char c = std::get<char>(node);
      if (c == '(') {
        // The bracket is paired. Stop popping.
        paired = true;
        break;
      }
      if (c == '|') {
        // It's a bracket like (a|b|c).
        unioned = true;
      }
      element_in_bracket.push(node);
    } else {
      // It's a node, i.e. some regex.
      element_in_bracket.push(node);
    }
  }

  if (!paired) {
    return false;
  }

  // It's a empty bracket, just ignore it.
  if (element_in_bracket.empty()) {
    current_parsing_index_++;
    return true;
  }

  // In the bracket, there's no union operator.
  if (!unioned) {
    RegexIR::Bracket bracket;
    while (!element_in_bracket.empty()) {
      auto node = element_in_bracket.top();
      element_in_bracket.pop();
      auto child = std::get<RegexIR::Node>(node);
      bracket.nodes.push_back(child);
    }
    stack_.push(bracket);
    current_parsing_index_++;
    return true;
  }

  RegexIR::Union union_node;
  RegexIR::Bracket bracket;
  while (!element_in_bracket.empty()) {
    auto node = element_in_bracket.top();
    element_in_bracket.pop();
    if (std::holds_alternative<char>(node)) {
      char c = std::get<char>(node);
      XGRAMMAR_DCHECK(c == '|');
      union_node.nodes.push_back(bracket);
      bracket.nodes.clear();
      continue;
    }
    XGRAMMAR_DCHECK(std::holds_alternative<RegexIR::Node>(node));
    auto child = std::get<RegexIR::Node>(node);
    bracket.nodes.push_back(child);
  }

  // Add the last bracket.
  union_node.nodes.push_back(bracket);
  stack_.push(union_node);
  current_parsing_index_++;
  return true;
}

Result<RegexIR> FSMBuilder::BuildRegexIRFromStack() {
  // The post-processing of the stack.
  RegexIR ir;
  std::vector<RegexIR::Node> res_nodes;
  std::vector<decltype(res_nodes)> union_node_list;
  bool unioned = false;
  while (!stack_.empty()) {
    if (std::holds_alternative<char>(stack_.top())) {
      char c = std::get<char>(stack_.top());
      XGRAMMAR_DCHECK(c == '|');
      union_node_list.push_back(res_nodes);
      res_nodes.clear();
      unioned = true;
      stack_.pop();
      continue;
    }
    auto node = stack_.top();
    stack_.pop();
    auto child = std::get<RegexIR::Node>(node);
    res_nodes.push_back(std::move(child));
  }
  if (!unioned) {
    for (auto it = res_nodes.rbegin(); it != res_nodes.rend(); ++it) {
      ir.nodes.push_back(std::move(*it));
    }
  } else {
    union_node_list.push_back(res_nodes);
    RegexIR::Union union_node;
    for (auto it = union_node_list.begin(); it != union_node_list.end(); ++it) {
      RegexIR::Bracket bracket;
      for (auto node = it->rbegin(); node != it->rend(); ++node) {
        bracket.nodes.push_back(std::move(*node));
      }
      union_node.nodes.push_back(std::move(bracket));
    }
    ir.nodes.push_back(std::move(union_node));
  }
  return Result<RegexIR>::Ok(ir);
}

}  // namespace xgrammar
