/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm_builder.h
 */
#ifndef XGRAMMAR_FSM_BUILDER_H_
#define XGRAMMAR_FSM_BUILDER_H_

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <stack>
#include <string>

#include "fsm.h"
#include "support/logging.h"
#include "support/utils.h"

namespace xgrammar {

class FSMBuilder {
 private:
  using IRNode = std::variant<RegexIR::Node, char>;

  /*! \brief The grammar string. */
  std::string grammar_;

  /*! \brief The parsing stack. */
  std::stack<IRNode> stack_;

  size_t current_parsing_index_ = 0;

  /************* Parsing Helper functions *************/

  void ConsumeWhiteSpace();
  bool HandleCharacterClass();
  bool HandleBracket();
  bool HandleSymbol();
  void HandleString();
  void CheckStartEndOfRegex();
  const char& Peek() {
    XGRAMMAR_DCHECK(current_parsing_index_ < grammar_.size());
    return grammar_[current_parsing_index_];
  }

  static const int8_t kIsRepeat = 0;

  static const int8_t kIsNotRepeat = 1;

  static const int8_t kParsingFailure = 2;

  static const int32_t kNotNumber = -1;

  /*! \brief Try to handle {n (,(m)?)?}.
      \return 0 if parsing successfully, 1 if it's not a repeat, 2 if parsing failure.*/
  int8_t TryHandleRepeat();
  int32_t ParsingPositveInteger() {
    int32_t result = 0;
    bool is_number = false;
    while (current_parsing_index_ < grammar_.size() && grammar_[current_parsing_index_] >= '0' &&
           grammar_[current_parsing_index_] <= '9') {
      is_number = true;
      result = result * 10 + (grammar_[current_parsing_index_] - '0');
      current_parsing_index_++;
    }
    if (!is_number) {
      return kNotNumber;
    }
    return result;
  }

  Result<RegexIR> BuildRegexIRFromStack();

 public:
  FSMBuilder() = default;

  /*! \brief Build a finite state machine with a given expression.
      \param rule_expr the expression of the rule.
      \return the corresponding fsm if successful, err otherwise. */
  Result<FSMWithStartEnd> BuildFSMFromRule(const std::string& rule_expr);

  /*! \brief Build a finite state machine with a given regex.
      \param regex the expression of the regex.
      \return the corresponding fsm if successful, err otherwise. */
  Result<FSMWithStartEnd> BuildFSMFromRegex(const std::string& regex) {
    const auto& ir_result = BuildRegexIR(regex);
    if (ir_result.IsErr()) {
      return Result<FSMWithStartEnd>::Err(ir_result.UnwrapErr());
    }
    XGRAMMAR_DCHECK(ir_result.IsOk());
    return ir_result.Unwrap().Build();
  }

  /*! \brief build the regex IR from the given regex. */
  Result<RegexIR> BuildRegexIR(const std::string& regex);
};

}  // namespace xgrammar

#endif
