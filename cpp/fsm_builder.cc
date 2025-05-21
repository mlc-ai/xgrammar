/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm_builder.cc
 */

#include "fsm_builder.h"

#include <sys/types.h>

#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stack>
#include <utility>
#include <variant>
#include <vector>

#include "fsm.h"
#include "support/encoding.h"
#include "support/logging.h"
#include "support/utils.h"

namespace xgrammar {

Result<RegexIR> FSMBuilder::BuildRegexIR(const std::string& regex) {
  // initialization.
  std::optional<Result<RegexIR>> lookahead_fsm = std::nullopt;
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
      bool successful = HandleSymbol();
      if (!successful) {
        return Result<RegexIR>::Err(
            std::make_shared<Error>("Parsing failure: there's no elements before a quantifier!")
        );
      }
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
      bool successful = HandleBracket(lookahead_fsm);
      if (!successful) {
        return Result<RegexIR>::Err(std::make_shared<Error>("Unmatched ')'"));
      }
      continue;
    }
    // It's matching characters at present.
    HandleStringInRegex();
  }
  if (lookahead_fsm.has_value()) {
    return Result<RegexIR>::Err(
        std::make_shared<Error>("There should be no lookahead fsm in the regex!")
    );
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

void FSMBuilder::HandleStringInRegex() {
  XGRAMMAR_DCHECK(current_parsing_index_ < grammar_.size());
  RegexIR::Leaf character_node;
  if (Peek() == '\\') {
    character_node.regex = Peek();
    current_parsing_index_++;
    character_node.regex += Peek();
    current_parsing_index_++;
  } else {
    character_node.regex = grammar_[current_parsing_index_];
    current_parsing_index_++;
  }
  stack_.push(character_node);
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

bool FSMBuilder::HandleBracket(std::optional<Result<RegexIR>>& lookahead_fsm) {
  XGRAMMAR_DCHECK(current_parsing_index_ < grammar_.size());
  XGRAMMAR_DCHECK(grammar_[current_parsing_index_] == ')');
  std::stack<IRNode> element_in_bracket;
  bool paired = false;
  bool unioned = false;
  bool is_lookahead = false;

  while ((!stack_.empty()) && (!paired)) {
    auto node = stack_.top();
    stack_.pop();
    if (std::holds_alternative<char>(node)) {
      char c = std::get<char>(node);
      if (c == '(' || c == '=') {
        // The bracket is paired. Stop popping.
        paired = true;
        if (c == '=') {
          is_lookahead = true;
        }
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
    if (is_lookahead) {
      if (lookahead_fsm.has_value()) {
        return false;  // Error: there's a lookahead fsm before.
      }
      RegexIR lookahead_ir;
      lookahead_ir.nodes.push_back(bracket);
      lookahead_fsm = Result<RegexIR>::Ok(lookahead_ir);
    } else {
      stack_.push(bracket);
    }
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
  if (is_lookahead) {
    if (lookahead_fsm.has_value()) {
      return false;  // Error: there's a lookahead fsm before.
    }
    RegexIR lookahead_ir;
    lookahead_ir.nodes.push_back(bracket);
    lookahead_fsm = Result<RegexIR>::Ok(lookahead_ir);
  } else {
    union_node.nodes.push_back(bracket);
    stack_.push(union_node);
  }
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

void ConsumeWhiteSpaces(std::string& str) {
  size_t start = 0;
  while (start < str.size() && (str[start] == ' ' || str[start] == '\t' || str[start] == '\n')) {
    start++;
  }
  size_t end = str.size() - 1;
  while (end > start && (str[end] == ' ' || str[end] == '\t' || str[end] == '\n')) {
    end--;
  }
  str = str.substr(start, end - start + 1);
}

void SplitRules(
    const std::string& grammar,
    std::vector<std::string>& lhs_group,
    std::vector<std::string>& rhs_group
) {
  // The two variables should be indexes of new lines.
  size_t last_end_of_the_line = 0;
  last_end_of_the_line = grammar.find('\n');
  if (last_end_of_the_line == std::string::npos) {
    // The grammar is a single line.
    size_t define_symbol = grammar.find("::=");
    if (define_symbol == std::string::npos) {
      XGRAMMAR_LOG(WARNING) << "There's something surprising in the grammar: " << grammar;
      return;
    }
    std::string lhs = grammar.substr(0, define_symbol - last_end_of_the_line - 1);
    std::string rhs = grammar.substr(define_symbol + 3, grammar.size() - define_symbol - 3);
    ConsumeWhiteSpaces(lhs);
    ConsumeWhiteSpaces(rhs);
    lhs_group.push_back(lhs);
    rhs_group.push_back(rhs);
    return;
  }
  size_t end_of_the_line = grammar.find('\n', last_end_of_the_line + 1);
  while (end_of_the_line != std::string::npos) {
    // Check if the line has a defination.
    size_t define_symbol = grammar.find("::=", last_end_of_the_line);
    if (define_symbol == std::string::npos || define_symbol > end_of_the_line) {
      // The line should be empty.
      if (end_of_the_line - last_end_of_the_line != 1) {
        // This line contains something surprising.
        XGRAMMAR_LOG(WARNING
        ) << "There's something surprising in the grammar: "
          << grammar.substr(last_end_of_the_line + 1, end_of_the_line - last_end_of_the_line);
      }
    } else {
      // This line is splitted successfully.
      std::string lhs =
          grammar.substr(last_end_of_the_line + 1, define_symbol - last_end_of_the_line - 1);
      std::string rhs = grammar.substr(define_symbol + 3, end_of_the_line - define_symbol - 3);
      ConsumeWhiteSpaces(lhs);
      ConsumeWhiteSpaces(rhs);
      lhs_group.push_back(lhs);
      rhs_group.push_back(rhs);
    }
    // Search for the next '\n'.
    last_end_of_the_line = end_of_the_line;
    end_of_the_line = grammar.find('\n', end_of_the_line + 1);
  }
  // Handle the last rule.
  size_t define_symbol = grammar.find("::=", last_end_of_the_line);
  if (define_symbol == std::string::npos || define_symbol > end_of_the_line) {
    // The line should be empty.
    if (end_of_the_line - last_end_of_the_line != 1) {
      // This line contains something surprising.
      size_t no_white_space =
          grammar.substr(last_end_of_the_line + 1, end_of_the_line - last_end_of_the_line)
              .find_first_not_of(' ');
      if (no_white_space != std::string::npos) {
        XGRAMMAR_LOG(WARNING
        ) << "There's something surprising in the grammar: "
          << grammar.substr(last_end_of_the_line + 1, end_of_the_line - last_end_of_the_line);
      }
    }
  } else {
    // This line is splitted successfully.
    std::string lhs =
        grammar.substr(last_end_of_the_line + 1, define_symbol - last_end_of_the_line - 1);
    std::string rhs = grammar.substr(define_symbol + 3, end_of_the_line - define_symbol - 3);
    ConsumeWhiteSpaces(lhs);
    ConsumeWhiteSpaces(rhs);
    lhs_group.push_back(lhs);
    rhs_group.push_back(rhs);
  }
}

bool FSMGroup::BuildNameIdMap(
    const std::vector<std::string>& rule_names, const std::string& root_rule
) {
  // The mapping should be built before the fsms are built.
  XGRAMMAR_DCHECK(fsms_.size() == 0);
  XGRAMMAR_DCHECK(rule_names_.empty());
  XGRAMMAR_DCHECK(rule_name_to_id_.empty());
  for (const auto& rule_name : rule_names) {
    if (rule_name_to_id_.find(rule_name) == rule_name_to_id_.end()) {
      rule_name_to_id_[rule_name] = rule_names_.size();
      rule_names_.push_back(rule_name);
    }
  }
  return rule_name_to_id_.find(root_rule) != rule_name_to_id_.end();
}

Result<FSMGroup> GrammarToFSMs(const std::string& grammar, std::string root_rule) {
  FSMGroup fsm_group;
  std::vector<std::string> lhs;
  std::vector<std::string> rhs;
  SplitRules(grammar, lhs, rhs);
  ConsumeWhiteSpaces(root_rule);
  bool has_root = fsm_group.BuildNameIdMap(lhs, root_rule);
  if (!has_root) {
    return Result<FSMGroup>::Err(std::make_shared<Error>("Root rule isn't found in the grammar!"));
  }
  fsm_group.root_rule_id_ = fsm_group.rule_name_to_id_[root_rule];
  XGRAMMAR_DCHECK(lhs.size() == rhs.size());
  fsm_group.fsms_.reserve(rhs.size());
  for (size_t i = 0; i < rhs.size(); ++i) {
    const auto& rule_expr = rhs[i];
    const auto& rule_name = lhs[i];
    // Build the fsm for the rule.
    FSMBuilder builder;
    std::optional<Result<RegexIR>> lookahead_fsm = std::nullopt;
    auto fsm = builder.BuildFSMFromRule(rule_expr, fsm_group.rule_name_to_id_, lookahead_fsm);
    if (fsm.IsErr()) {
      XGRAMMAR_LOG(INFO) << "Error building fsm for rule " << rule_name << ": "
                         << fsm.UnwrapErr()->what();
      return Result<FSMGroup>::Err(fsm.UnwrapErr());
    }
    if (lookahead_fsm.has_value()) {
      const auto& ir_result = lookahead_fsm.value();
      if (ir_result.IsErr()) {
        return Result<FSMGroup>::Err(ir_result.UnwrapErr());
      }
      XGRAMMAR_DCHECK(ir_result.IsOk());
      const auto& ir = ir_result.Unwrap();
      auto lookahead_fsm_result = ir.Build();
      if (lookahead_fsm_result.IsErr()) {
        return Result<FSMGroup>::Err(lookahead_fsm_result.UnwrapErr());
      }
      XGRAMMAR_DCHECK(lookahead_fsm_result.IsOk());
      const auto& lookahead_fsm = lookahead_fsm_result.Unwrap();
      fsm_group.lookahead_fsms_[i] = lookahead_fsm;
    }
    // Since we number the rules in order, if the id is larger than the size of the fsms, we need
    // to add it. Otherwise, it has been added, then we just need to union it.
    int id = fsm_group.rule_name_to_id_[rule_name];
    if (fsm_group.fsms_.size() <= static_cast<size_t>(id)) {
      fsm_group.fsms_.push_back(fsm.Unwrap());
    } else {
      fsm_group.fsms_[id] = FSMWithStartEnd::Union({fsm_group.fsms_[id], fsm.Unwrap()});
    }
  }
  return Result<FSMGroup>::Ok(fsm_group);
}

Result<FSMWithStartEnd> FSMBuilder::BuildFSMFromRule(
    const std::string& rule_expr,
    const std::unordered_map<std::string, int>& rule_name_to_id,
    std::optional<Result<RegexIR>>& lookahead_fsm
) {
  // Initialization.
  grammar_ = rule_expr;
  current_parsing_index_ = 0;
  while (!stack_.empty()) {
    stack_.pop();
  }

  while (current_parsing_index_ < grammar_.size()) {
    ConsumeWhiteSpace();
    if (current_parsing_index_ >= grammar_.size()) {
      break;
    }
    if (lookahead_fsm.has_value()) {
      XGRAMMAR_LOG(WARNING) << "The lookahead fsm will be recognized as the last element!";
    }
    if (Peek() == '[') {
      bool successful = HandleCharacterClass();
      if (!successful) {
        return Result<FSMWithStartEnd>::Err(std::make_shared<Error>("Unmatched '['"));
      }
      continue;
    }

    if (Peek() == '(' || Peek() == '|') {
      stack_.push(Peek());
      current_parsing_index_++;
      continue;
    }

    if (Peek() == '=') {
      bool successful = HandleLookAhead();
      if (!successful) {
        return Result<FSMWithStartEnd>::Err(
            std::make_shared<Error>("Invalid grammar: lookahead '=' is invalid!")
        );
      }
      continue;
    }

    if (Peek() == '+' || Peek() == '?' || Peek() == '*') {
      bool successful = HandleSymbol();
      if (!successful) {
        return Result<FSMWithStartEnd>::Err(
            std::make_shared<Error>("Invalid grammar: no node before operator!")
        );
      }
      continue;
    }

    if (Peek() == '{') {
      int8_t repeat_status_code = TryHandleRepeat();
      if (repeat_status_code == kIsRepeat) {
        continue;
      }
      if (repeat_status_code == kParsingFailure) {
        return Result<FSMWithStartEnd>::Err(
            std::make_shared<Error>("Invalid grammar: no node before repeat quantifier!")
        );
      }
      XGRAMMAR_DCHECK(repeat_status_code == kIsNotRepeat);
    }

    if (Peek() == ')') {
      bool successful = HandleBracket(lookahead_fsm);
      if (!successful) {
        return Result<FSMWithStartEnd>::Err(std::make_shared<Error>("Unmatched ')'"));
      }
      continue;
    }

    if (Peek() == '"') {
      bool successful = HandleString();
      if (!successful) {
        return Result<FSMWithStartEnd>::Err(std::make_shared<Error>("Unmatched '\"'"));
      }
      continue;
    }

    if (Peek() == '/') {
      bool successful = HandleRegex();
      if (!successful) {
        return Result<FSMWithStartEnd>::Err(
            std::make_shared<Error>("Unmatched '/' or invalid regex!")
        );
      }
      continue;
    }

    bool successful = HandleRuleRef(rule_name_to_id);
    if (!successful) {
      return Result<FSMWithStartEnd>::Err(
          std::make_shared<Error>("Invalid grammar: the rule doesn't exist!")
      );
    }
  }
  const auto& ir_result = BuildRegexIRFromStack();
  if (ir_result.IsErr()) {
    return Result<FSMWithStartEnd>::Err(ir_result.UnwrapErr());
  }
  XGRAMMAR_DCHECK(ir_result.IsOk());
  const auto& ir = ir_result.Unwrap();
  return ir.Build();
}

bool FSMBuilder::HandleRuleRef(const std::unordered_map<std::string, int>& rule_name_to_id) {
  XGRAMMAR_DCHECK(current_parsing_index_ < grammar_.size());
  int start_index = current_parsing_index_;
  while (current_parsing_index_ < grammar_.size() &&
         (Peek() == '_' || (Peek() >= 'a' && Peek() <= 'z') || (Peek() >= 'A' && Peek() <= 'Z') ||
          (Peek() >= '0' && Peek() <= '9'))) {
    current_parsing_index_++;
  }
  std::string rule_name = grammar_.substr(start_index, current_parsing_index_ - start_index);
  if (rule_name_to_id.find(rule_name) == rule_name_to_id.end()) {
    // The rule doesn't exist.
    return false;
  }
  RegexIR::RuleRef rule_ref;
  rule_ref.rule_id = rule_name_to_id.at(rule_name);
  stack_.push(rule_ref);
  return true;
}

bool FSMBuilder::HandleString() {
  XGRAMMAR_DCHECK(current_parsing_index_ < grammar_.size());
  XGRAMMAR_DCHECK(grammar_[current_parsing_index_] == '"');
  int start_index = current_parsing_index_;
  current_parsing_index_++;
  while (current_parsing_index_ < grammar_.size() && grammar_[current_parsing_index_] != '"') {
    if (grammar_[current_parsing_index_] == '\\') {
      current_parsing_index_++;
    }
    current_parsing_index_++;
  }
  if (current_parsing_index_ >= grammar_.size()) {
    return false;  // Error: unmatched '"'
  }
  RegexIR::Leaf string_node;
  string_node.regex = grammar_.substr(start_index + 1, current_parsing_index_ - start_index - 1);
  string_node.is_literal = true;
  stack_.push(string_node);
  current_parsing_index_++;
  return true;
}

bool FSMBuilder::HandleRegex() {
  XGRAMMAR_DCHECK(current_parsing_index_ < grammar_.size());
  XGRAMMAR_DCHECK(grammar_[current_parsing_index_] == '/');
  int start_index = current_parsing_index_;
  current_parsing_index_++;
  while (current_parsing_index_ < grammar_.size() && grammar_[current_parsing_index_] != '/') {
    if (grammar_[current_parsing_index_] == '\\') {
      current_parsing_index_++;
    }
    current_parsing_index_++;
  }
  if (current_parsing_index_ >= grammar_.size()) {
    return false;  // Error: unmatched '/'
  }
  std::string regex = grammar_.substr(start_index + 1, current_parsing_index_ - start_index - 1);
  FSMBuilder regex_builder;
  auto fsm_result = regex_builder.BuildRegexIR(regex);
  if (fsm_result.IsErr()) {
    std::cout << regex << std::endl;
    return false;  // Error: invalid regex
  }
  XGRAMMAR_DCHECK(fsm_result.IsOk());
  for (const auto& node : fsm_result.Unwrap().nodes) {
    stack_.push(node);
  }
  current_parsing_index_++;
  return true;
}

int FSMBuilder::ParsingPositveInteger() {
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

bool FSMBuilder::HandleLookAhead() {
  if (stack_.empty()) {
    return false;
  }

  const auto& node = stack_.top();
  if (!std::holds_alternative<char>(node)) {
    return false;
  }

  char c = std::get<char>(node);
  if (c != '(') {
    return false;
  }
  // It'a valid lookahead.
  stack_.pop();
  stack_.push('=');  // Push the lookahead operator.
  current_parsing_index_++;
  return true;
}

const uint32_t kParseFailure = -1;

std::pair<uint32_t, uint32_t> NextUnicode(const std::string& str, size_t& index) {
  const auto [accept, num_byte, _] = HandleUTF8FirstByte(str[index]);
  if (!accept) {
    return std::make_pair(kParseFailure, 0);
  }
  uint32_t unicode = 0;
  for (int i = 0; i < num_byte; i++) {
    unicode <<= 8;
    unicode += static_cast<uint8_t>(str[index + i]);
  }
  return std::make_pair(num_byte, unicode);
}

FSMWithStartEnd::FSMWithStartEnd(const std::string& regex) {
  is_dfa = true;
  start = 0;
  auto& edges = fsm.edges;
  // Handle the regex string.
  if (!(regex[0] == '[' && regex[regex.size() - 1] == ']')) {
    edges.push_back(std::vector<FSMEdge>());
    for (size_t i = 0; i < regex.size(); i++) {
      if (regex[i] != '\\') {
        if (regex[i] == '.') {
          // Accept all unicode characters will add 6 new states.
          edges.back().emplace_back(0, 0x7f, edges.size() + 6);
          AcceptAllUnicodeCharacters(edges.size() - 1, edges.size() + 6);
        } else {
          edges.back().emplace_back(
              (unsigned char)(regex[i]), (unsigned char)(regex[i]), edges.size()
          );
        }
        edges.push_back(std::vector<FSMEdge>());
        continue;
      }
      auto [length, escape_vector] = HandleEscapes(regex, i);
      for (const auto& escape : escape_vector) {
        edges.back().emplace_back(
            (unsigned char)(escape.first), (unsigned char)(escape.second), edges.size()
        );
      }
      edges.push_back(std::vector<FSMEdge>());
      if (length > 3) {
        XGRAMMAR_LOG(WARNING) << "Such a long escape sequence is not supported: "
                              << regex.substr(i, length);
      }
      i = i + length - 1;
    }
    ends.insert(edges.size() - 1);
    return;
  }

  // Handle the character class.
  XGRAMMAR_DCHECK((regex[0] == '[' && regex[regex.size() - 1] == ']'));
  std::vector<std::pair<uint32_t, uint32_t>> char_class_edges;
  edges.resize(2);
  AddEndNode(1);
  bool negative = regex[1] == '^';
  uint32_t last_character = kParseFailure;
  bool is_possible_range = false;
  size_t index = negative ? 2 : 1;
  while (index < regex.size() - 1) {
    if (regex[index] == '-') {
      index++;
      if (is_possible_range) {
        XGRAMMAR_LOG(FATAL) << "Invalid regex: " << regex;
      }
      if (last_character == kParseFailure) {
        // It's a '-' at the beginning of the character class.
        char_class_edges.emplace_back('-', '-');
        continue;
      }
      is_possible_range = true;
      continue;
    }

    if (regex[index] == '\\') {
      const auto& [length, escape_vector] = HandleEscapes(regex, index);
      index += length;
      if (escape_vector.size() != 1 || escape_vector[0].first != escape_vector[0].second) {
        // It's a multi-match escape sequence.
        char_class_edges.insert(char_class_edges.end(), escape_vector.begin(), escape_vector.end());
        if (is_possible_range) {
          char_class_edges.emplace_back('-', '-');
          is_possible_range = false;
        }
        if (last_character != kParseFailure) {
          char_class_edges.emplace_back(last_character, last_character);
        }
        last_character = kParseFailure;
        continue;
      }

      if (is_possible_range) {
        if (escape_vector[0].first < last_character) {
          XGRAMMAR_LOG(FATAL) << "Invalid regex: " << regex;
        }
        char_class_edges.emplace_back(last_character, escape_vector[0].first);
        last_character = kParseFailure;
        is_possible_range = false;
        continue;
      }

      if (last_character != kParseFailure) {
        char_class_edges.emplace_back(last_character, last_character);
      }
      last_character = escape_vector[0].first;
      continue;
    }

    auto [length, unicode] = NextUnicode(regex, index);
    if (length == kParseFailure) {
      XGRAMMAR_LOG(FATAL) << "Invalid regex: " << regex;
    }
    if (is_possible_range) {
      if (unicode < last_character) {
        XGRAMMAR_LOG(FATAL) << "Invalid regex: " << regex;
      }
      char_class_edges.emplace_back(last_character, unicode);
      last_character = kParseFailure;
      is_possible_range = false;
      index += length;
      continue;
    }

    if (last_character != kParseFailure) {
      char_class_edges.emplace_back(last_character, last_character);
    }
    last_character = unicode;
    index += length;
  }
  // Handle the last character.
  if (last_character != kParseFailure) {
    char_class_edges.emplace_back(last_character, last_character);
  }
  // The last character is '-'.
  if (is_possible_range) {
    char_class_edges.emplace_back('-', '-');
  }

  for (const auto& edge : char_class_edges) {
    if (edge.second > 0x7f) {
      if (negative) {
        XGRAMMAR_LOG(FATAL) << "In a negative character class, the range should be in [0, 0x7f].";
      }
    }
  }
  BuildCharacterClass(char_class_edges, negative, 0, 1);
}

void FSMWithStartEnd::BuildCharacterClass(
    std::vector<std::pair<uint32_t, uint32_t>>& char_class_edges,
    bool is_negative,
    int start_node,
    int end_node
) {
  auto& edges = fsm.edges[start_node];
  if (is_negative) {
    std::vector<bool> is_accepted(0x80, true);
    for (const auto& edge : char_class_edges) {
      for (int i = edge.first; i <= int(edge.second); i++) {
        is_accepted[i] = false;
      }
    }
    for (int i = 0; i < 0x80; i++) {
      if (is_accepted[i]) {
        bool has_end = false;
        for (int j = i + 1; j < 0x80; j++) {
          if (!is_accepted[j]) {
            has_end = true;
            edges.emplace_back(i, j - 1, end_node);
            i = j;
            break;
          }
        }
        if (!has_end) {
          edges.emplace_back(i, 0x7f, end_node);
          break;
        }
      }
    }
    AcceptAllUnicodeCharacters(start_node, end_node);
    return;
  }
  for (const auto& edge : char_class_edges) {
    // TODO(Linzhang): support the unicode range here.
    if (edge.second < 0x80) {
      edges.emplace_back(edge.first, edge.second, end_node);
    } else {
      AddUnicodeEdge(edge.first, edge.second, start_node, end_node);
    }
  }
}

void FSMWithStartEnd::AddUnicodeEdge(
    uint32_t min_ch, uint32_t max_ch, int start_node, int end_node
) {
  static const uint8_t& utf8_successor_character_min = 0x80;
  static const uint8_t& utf8_successor_character_max = 0xbF;
  static const uint32_t& min_2_byte_character = 0xc280;
  static const uint32_t& max_2_byte_character = 0xdfbf;
  static const uint32_t& min_3_byte_character = 0xe08080;
  static const uint32_t& max_3_byte_character = 0xefbfbf;
  static const uint32_t& min_4_byte_character = 0xf0808080;
  uint8_t min_char[4] = {0};
  uint8_t max_char[4] = {0};
  int min_length = 0;
  uint32_t tmp_min_ch = min_ch;
  uint32_t tmp_max_ch = max_ch;
  while (tmp_min_ch > 0) {
    min_char[min_length++] = tmp_min_ch & 0xff;
    tmp_min_ch >>= 8;
  }
  int max_length = 0;
  while (tmp_max_ch > 0) {
    max_char[max_length++] = tmp_max_ch & 0xff;
    tmp_max_ch >>= 8;
  }
  XGRAMMAR_DCHECK(max_length > 1);
  if (min_length != max_length) {
    switch (min_length) {
      case (1): {
        AddUnicodeEdge(min_ch, 0x7f, start_node, end_node);
        AddUnicodeEdge(min_2_byte_character, max_ch, start_node, end_node);
        break;
      }
      case (2): {
        AddUnicodeEdge(min_ch, max_2_byte_character, start_node, end_node);
        AddUnicodeEdge(min_3_byte_character, max_ch, start_node, end_node);
        break;
      }
      case (3): {
        AddUnicodeEdge(min_ch, max_3_byte_character, start_node, end_node);
        AddUnicodeEdge(min_4_byte_character, max_ch, start_node, end_node);
        break;
      }
      default: {
        XGRAMMAR_LOG(FATAL) << "Invalid unicode character: " << min_ch;
      }
    }
    return;
  }
  XGRAMMAR_DCHECK(min_length == max_length);
  size_t current_node_index = start_node;
  switch (min_length) {
    case (1): {
      fsm.edges[current_node_index].emplace_back(min_char[0], max_char[0], end_node);
      break;
    }
    case (2): {
      if (min_char[1] == max_char[1]) {
        fsm.edges.emplace_back();
        fsm.edges[current_node_index].emplace_back(min_char[1], max_char[1], fsm.edges.size() - 1);
        current_node_index = fsm.edges.size() - 1;
        fsm.edges[current_node_index].emplace_back(min_char[0], max_char[0], end_node);
        return;
      }
      XGRAMMAR_DCHECK(min_char[1] < max_char[1]);
      // Consider 3 situations:
      // 1. min_char[1].
      fsm.edges.emplace_back();
      fsm.edges[current_node_index].emplace_back(min_char[1], min_char[1], fsm.edges.size() - 1);
      current_node_index = fsm.edges.size() - 1;
      fsm.edges[current_node_index].emplace_back(
          min_char[0], utf8_successor_character_max, end_node
      );
      current_node_index = start_node;

      // 2. max_char[1].
      fsm.edges.emplace_back();
      fsm.edges[current_node_index].emplace_back(max_char[1], max_char[1], fsm.edges.size() - 1);
      current_node_index = fsm.edges.size() - 1;
      fsm.edges[current_node_index].emplace_back(
          utf8_successor_character_min, max_char[0], end_node
      );
      current_node_index = start_node;
      // 3. (min_char[1], max_char[1]).
      if (min_char[1] + 1 < max_char[1]) {
        fsm.edges.emplace_back();
        fsm.edges[current_node_index].emplace_back(
            min_char[1] + 1, max_char[1] - 1, fsm.edges.size() - 1
        );
        current_node_index = fsm.edges.size() - 1;
        fsm.edges[current_node_index].emplace_back(
            utf8_successor_character_min, utf8_successor_character_max, end_node
        );
      }
      break;
    }
    case (3): {
      if (min_char[2] == max_char[2]) {
        fsm.edges.emplace_back();
        fsm.edges[current_node_index].emplace_back(min_char[2], max_char[2], fsm.edges.size() - 1);
        current_node_index = fsm.edges.size() - 1;
        min_ch &= 0x0000ffff;
        max_ch &= 0x0000ffff;
        AddUnicodeEdge(min_ch, max_ch, current_node_index, end_node);
        return;
      }
      XGRAMMAR_DCHECK(min_char[2] < max_char[2]);
      // Consider 3 situations:
      // 1. min_char[2].
      fsm.edges.emplace_back();
      fsm.edges[current_node_index].emplace_back(min_char[2], min_char[2], fsm.edges.size() - 1);
      current_node_index = fsm.edges.size() - 1;
      uint32_t largest_left_2_byte = utf8_successor_character_max;
      largest_left_2_byte <<= 8;
      largest_left_2_byte += utf8_successor_character_max;
      AddUnicodeEdge(min_ch & 0x0000ffff, largest_left_2_byte, current_node_index, end_node);
      current_node_index = start_node;

      // 2. max_char[2].
      fsm.edges.emplace_back();
      fsm.edges[current_node_index].emplace_back(max_char[2], max_char[2], fsm.edges.size() - 1);
      current_node_index = fsm.edges.size() - 1;
      uint32_t smallest_right_2_byte = utf8_successor_character_min;
      smallest_right_2_byte <<= 8;
      smallest_right_2_byte += utf8_successor_character_min;
      AddUnicodeEdge(smallest_right_2_byte, max_ch & 0x0000ffff, current_node_index, end_node);
      current_node_index = start_node;

      // 3. (min_char[2], max_char[2]).
      if (min_char[2] + 1 < max_char[2]) {
        fsm.edges.emplace_back();
        fsm.edges[current_node_index].emplace_back(
            min_char[2] + 1, max_char[2] - 1, fsm.edges.size() - 1
        );
        current_node_index = fsm.edges.size() - 1;
        fsm.edges.emplace_back();
        fsm.edges[current_node_index].emplace_back(
            utf8_successor_character_min, utf8_successor_character_max, fsm.edges.size() - 1
        );
        current_node_index = fsm.edges.size() - 1;
        fsm.edges[current_node_index].emplace_back(
            utf8_successor_character_min, utf8_successor_character_max, end_node
        );
      }
      break;
    }
    case (4): {
      if (min_char[3] == max_char[3]) {
        fsm.edges.emplace_back();
        fsm.edges[current_node_index].emplace_back(min_char[3], max_char[3], fsm.edges.size() - 1);
        current_node_index = fsm.edges.size() - 1;
        min_ch &= 0x00ffffff;
        max_ch &= 0x00ffffff;
        AddUnicodeEdge(min_ch, max_ch, current_node_index, end_node);
        return;
      }
      XGRAMMAR_DCHECK(min_char[3] < max_char[3]);
      // Consider 3 situations:
      // 1. min_char[3].
      fsm.edges.emplace_back();
      fsm.edges[current_node_index].emplace_back(min_char[3], min_char[3], fsm.edges.size() - 1);
      current_node_index = fsm.edges.size() - 1;
      uint32_t largest_left_3_byte = utf8_successor_character_max;
      largest_left_3_byte <<= 8;
      largest_left_3_byte += utf8_successor_character_max;
      largest_left_3_byte <<= 8;
      largest_left_3_byte += utf8_successor_character_max;
      AddUnicodeEdge(min_ch & 0x00ffffff, largest_left_3_byte, current_node_index, end_node);
      current_node_index = start_node;

      // 2. max_char[3].
      fsm.edges.emplace_back();
      fsm.edges[current_node_index].emplace_back(max_char[3], max_char[3], fsm.edges.size() - 1);
      current_node_index = fsm.edges.size() - 1;
      uint32_t smallest_right_3_byte = utf8_successor_character_min;
      smallest_right_3_byte <<= 8;
      smallest_right_3_byte += utf8_successor_character_min;
      smallest_right_3_byte <<= 8;
      smallest_right_3_byte += utf8_successor_character_min;
      AddUnicodeEdge(smallest_right_3_byte, max_ch & 0x00ffffff, current_node_index, end_node);
      current_node_index = start_node;

      // 3. (min_char[3], max_char[3]).
      if (min_char[3] + 1 < max_char[3]) {
        fsm.edges.emplace_back();
        fsm.edges[current_node_index].emplace_back(
            min_char[3] + 1, max_char[3] - 1, fsm.edges.size() - 1
        );
        current_node_index = fsm.edges.size() - 1;
        fsm.edges.emplace_back();
        fsm.edges[current_node_index].emplace_back(
            utf8_successor_character_min, utf8_successor_character_max, fsm.edges.size() - 1
        );
        current_node_index = fsm.edges.size() - 1;
        fsm.edges.emplace_back();
        fsm.edges[current_node_index].emplace_back(
            utf8_successor_character_min, utf8_successor_character_max, fsm.edges.size() - 1
        );
        current_node_index = fsm.edges.size() - 1;
        fsm.edges[current_node_index].emplace_back(
            utf8_successor_character_min, utf8_successor_character_max, end_node
        );
      }
      break;
    }
    default: {
      XGRAMMAR_LOG(FATAL) << "Invalid unicode character: " << min_ch;
    }
  }
}

void FSMWithStartEnd::AcceptAllUnicodeCharacters(const int& from_node, const int& to_node) {
  static const uint8_t& utf8_successor_character_min = 0x80;
  static const uint8_t& utf8_successor_character_max = 0xbF;
  static const uint8_t& min_start_character_of_2_byte = 0xc2;
  static const uint8_t& max_start_character_of_2_byte = 0xdf;
  static const uint8_t& min_start_character_of_3_byte = 0xe0;
  static const uint8_t& max_start_character_of_3_byte = 0xef;
  static const uint8_t& min_start_character_of_4_byte = 0xf0;
  static const uint8_t& max_start_character_of_4_byte = 0xf4;
  // First, Handle the 2-byte characters.
  fsm.edges.emplace_back();
  fsm.edges[from_node].push_back(FSMEdge{
      min_start_character_of_2_byte,
      max_start_character_of_2_byte,
      static_cast<int>(fsm.edges.size() - 1)
  });
  fsm.edges.back().push_back(
      FSMEdge{utf8_successor_character_min, utf8_successor_character_max, to_node}
  );

  // Handle the 3-byte characters.
  fsm.edges.emplace_back();
  fsm.edges.emplace_back();
  fsm.edges[from_node].push_back(FSMEdge{
      min_start_character_of_3_byte,
      max_start_character_of_3_byte,
      static_cast<int>(fsm.edges.size() - 2)
  });
  fsm.edges[fsm.edges.size() - 2].push_back(FSMEdge{
      utf8_successor_character_min,
      utf8_successor_character_max,
      static_cast<int>(fsm.edges.size() - 1)
  });
  fsm.edges[fsm.edges.size() - 1].push_back(
      FSMEdge{utf8_successor_character_min, utf8_successor_character_max, to_node}
  );

  // Handle the 4-byte characters.
  fsm.edges.emplace_back();
  fsm.edges.emplace_back();
  fsm.edges.emplace_back();
  fsm.edges[from_node].push_back(FSMEdge{
      min_start_character_of_4_byte,
      max_start_character_of_4_byte,
      static_cast<int>(fsm.edges.size() - 3)
  });
  fsm.edges[fsm.edges.size() - 3].push_back(FSMEdge{
      utf8_successor_character_min,
      utf8_successor_character_max,
      static_cast<int>(fsm.edges.size() - 2)
  });
  fsm.edges[fsm.edges.size() - 2].push_back(FSMEdge{
      utf8_successor_character_min,
      utf8_successor_character_max,
      static_cast<int>(fsm.edges.size() - 1)
  });
  fsm.edges[fsm.edges.size() - 1].push_back(
      FSMEdge{utf8_successor_character_min, utf8_successor_character_max, to_node}
  );
}

}  // namespace xgrammar
