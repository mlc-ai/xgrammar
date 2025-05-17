/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm_builder.cc
 */

#include "fsm_builder.h"

#include <cstddef>

#include "fsm.h"
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
      HandleCharacterClass();
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
      bool is_repeated = TryHandleRepeat();
      if (is_repeated) {
        continue;
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

}  // namespace xgrammar
