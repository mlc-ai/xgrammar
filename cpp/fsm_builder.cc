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
      bool successful = HandleBracket();
      if (!successful) {
        return Result<RegexIR>::Err(std::make_shared<Error>("Unmatched ')'"));
      }
      continue;
    }
    // It's matching characters at present.
    HandleStringInRegex();
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
  while (last_end_of_the_line < grammar.size() && grammar[last_end_of_the_line] == '\n') {
    last_end_of_the_line++;
  }
  size_t end_of_the_line = grammar.find('\n', last_end_of_the_line);
  last_end_of_the_line--;
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
    auto fsm = builder.BuildFSMFromRule(rule_expr, fsm_group.rule_name_to_id_);
    if (fsm.IsErr()) {
      XGRAMMAR_LOG(INFO) << "Error building fsm for rule " << rule_name << ": "
                         << fsm.UnwrapErr()->what();
      return Result<FSMGroup>::Err(fsm.UnwrapErr());
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
    const std::string& rule_expr, const std::unordered_map<std::string, int>& rule_name_to_id
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
      bool successful = HandleBracket();
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

}  // namespace xgrammar
