/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm_builder.h
 */
#ifndef XGRAMMAR_FSM_BUILDER_H_
#define XGRAMMAR_FSM_BUILDER_H_

#include <stack>
#include <string>

#include "fsm.h"
#include "support/utils.h"

namespace xgrammar {

class FSMBuilder {
 private:
  using IRNode = std::variant<RegexIR::Node, char>;

  /*! \brief The grammar string. */
  std::string grammar_;

  /*! \brief The parsing stack. */
  std::stack<IRNode> stack_;

  /*! \brief build the regex IR from the given regex. */
  Result<RegexIR> BuildRegexIR(const std::string& regex);

  int current_parsing_index_ = 0;

  /************* Parsing Helper functions *************/

  void ConsumeWhitespace();
  void HandleCharacterClass();
  void HandleBracket();

 public:
  FSMBuilder() = default;

  /*! \brief Build a finite state machine with a given expression.
      \param rule_expr the expression of the rule.
      \return the corresponding fsm if successful, err otherwise. */
  Result<FSMWithStartEnd> BuildFSMFromRule(const std::string& rule_expr);

  /*! \brief Build a finite state machine with a given regex.
      \param regex the expression of the regex.
      \return the corresponding fsm if successful, err otherwise. */
  Result<FSMWithStartEnd> BuildFSMFromRegex(const std::string& regex);
};

}  // namespace xgrammar

#endif
