/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm_builder.cc
 */

#include "fsm_builder.h"

#include <cstdint>
#include <memory>
#include <variant>

#include "fsm.h"
#include "support/logging.h"
#include "support/utils.h"

namespace xgrammar {

Result<RegexIR> FSMBuilder::BuildRegexIR(const std::string& regex) {
  // initialization.
  RegexIR result_ir;
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
    }

    if (Peek() == ')') {
      HandleBracket();
      continue;
    }
    // It's matching characters at present.
    HandleString();
  }

  // TODO(Linzhang): Build the RegexIR.
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

}  // namespace xgrammar
