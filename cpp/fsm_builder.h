/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm_builder.h
 */
#ifndef XGRAMMAR_FSM_BUILDER_H_
#define XGRAMMAR_FSM_BUILDER_H_

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <optional>
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
  bool HandleBracket(std::optional<Result<RegexIR>>& lookahead_fsm);
  bool HandleSymbol();
  void HandleStringInRegex();
  void CheckStartEndOfRegex();
  bool HandleLookAhead();
  bool HandleRuleRef(const std::unordered_map<std::string, int>& rule_name_to_id);
  bool HandleString();
  bool HandleRegex();

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

  int32_t ParsingPositveInteger();

  Result<RegexIR> BuildRegexIRFromStack();

 public:
  FSMBuilder() = default;

  /*! \brief Build a finite state machine with a given expression.
      \param rule_expr the expression of the rule.
      \param rule_name_to_id the mapping from the rule name to the rule id.
      \return the corresponding fsm if successful, err otherwise. */
  Result<FSMWithStartEnd> BuildFSMFromRule(
      const std::string& rule_expr,
      const std::unordered_map<std::string, int>& rule_name_to_id,
      std::optional<Result<RegexIR>>& lookahead_fsm
  );

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

/*! \brief The function is used to get FSMs from a given grammar.
    \param grammar The given grammar.
    The grammar should be in the such format:
    rule1 ::= (rule2 | (rule3)+)? /[0-9]abc/ "abc"
    i.e. the lhs is the name of the rule, '::=' means 'is defined as'.
    Between the '/', is a regex; In other cases, they are composed of
    rules and strings. If some characters are Between the '"', then it's a string.
    Moreover, to denote a '/' in regex, please use '\/' in the grammar.
    \param root_rule The root grammar.
    \return If everthing is OK, then a FSMGroups will be returned. Otherwise, it will return an
   error.
      */
Result<FSMGroup> GrammarToFSMs(const std::string& grammar, std::string root_rule);

}  // namespace xgrammar

#endif
