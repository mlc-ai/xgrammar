/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm_builder.cc
 */
#include "fsm_builder.h"

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <set>
#include <stack>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "fsm.h"
#include "support/logging.h"
#include "support/utils.h"

namespace xgrammar {

class RegexIR {
 public:
  struct Leaf;

  struct Symbol;

  struct Union;

  struct Bracket;

  struct Repeat;

  static constexpr int kRepeatNoUpperBound = -1;

  using State = std::variant<Leaf, Symbol, Union, Bracket, Repeat>;

  // This struct is used to store the string in regex, or
  // the character class in regex.
  struct Leaf {
    std::string regex;
  };

  // This struct is used to store the symbol in regex, i.e.
  // +, *, ?
  enum class RegexSymbol {
    star,
    plus,
    optional,
  };

  struct Bracket {
    std::vector<State> states;
  };

  struct Symbol {
    RegexSymbol symbol;
    std::vector<State> state;
  };

  // This struct is used to represent a union symbol.
  struct Union {
    std::vector<State> states;
  };

  struct Repeat {
    std::vector<State> states;
    int lower_bound = 0;
    int upper_bound = 0;
  };

  struct LookAhead {
    bool is_positive;
    std::vector<State> states;
  };

  // This struct is used to represent a bracket in regex.
  std::vector<State> states;

  /*!
    \brief Constructs a NFA from the regex IR.
  */
  Result<FSMWithStartEnd> Build() const;

  /*!
    \brief the visit function for the variant.
  */
  Result<FSMWithStartEnd> visit(const Leaf& state) const;

  Result<FSMWithStartEnd> visit(const Symbol& state) const;

  Result<FSMWithStartEnd> visit(const Union& state) const;

  Result<FSMWithStartEnd> visit(const Bracket& state) const;

  Result<FSMWithStartEnd> visit(const Repeat& state) const;

  Result<FSMWithStartEnd> visit(const LookAhead& state) const;

 private:
  /*!
   * \brief Construct a FSM from a regex string.
   * \details The regex string should only be the format like "abx" or [a-c0-9].
   * \details Any symbols like "a|b" or "a*b" are not supported.
   * \param regex The regex string.
   * \return The FSM with start and end states.
   */
  static FSMWithStartEnd BuildLeafFSMFromRegex(const std::string& regex);

  /*!
   * \brief Handle escape characters.
   * \param regex the corresponding string.
   * \param start the pos escape characters start.
   */
  static std::vector<std::pair<int, int>> HandleEscapes(const std::string& regex, int start);

  /*!
   * \brief Check repeat in regex. i.e {...} and {...,...}
   * \param regex The regex string.
   * \param start The start position of the repeat. i.e. regex[start] == '{'.
   * After the function, start will be the position of '}'.
   * \return The repeat range.
   */
  static Result<std::pair<int, int>> CheckRepeat(const std::string& regex, int& start);

  friend class RegexFSMBuilder;
};

Result<std::pair<int, int>> RegexIR::CheckRepeat(const std::string& regex, int& start) {
  if (regex[start] != '{') {
    return ResultErr("Invalid repeat format1");
  }
  int lower_bound = 0;
  int upper_bound = RegexIR::kRepeatNoUpperBound;
  std::string num_str;
  XGRAMMAR_DCHECK(regex[start] == '{');
  start++;
  while (static_cast<size_t>(start) < regex.size() && regex[start] == ' ') {
    start++;
  }
  while (static_cast<size_t>(start) < regex.size() && std::isdigit(regex[start])) {
    num_str += regex[start];
    start++;
  }
  if (num_str.empty()) {
    return ResultErr("Invalid repeat format2");
  }
  lower_bound = std::stoi(num_str);
  while (static_cast<size_t>(start) < regex.size() && regex[start] == ' ') {
    start++;
  }
  // The format is {n}
  if (regex[start] == '}') {
    upper_bound = lower_bound;
    return ResultOk(std::make_pair(lower_bound, upper_bound));
  }
  if (regex[start] != ',') {
    return ResultErr("Invalid repeat format3");
  }
  XGRAMMAR_DCHECK(regex[start] == ',');
  start++;
  while (static_cast<size_t>(start) < regex.size() && regex[start] == ' ') {
    start++;
  }
  // The format is {n,}
  if (regex[start] == '}') {
    return ResultOk(std::make_pair(lower_bound, upper_bound));
  }
  num_str.clear();
  while (static_cast<size_t>(start) < regex.size() && std::isdigit(regex[start])) {
    num_str += regex[start];
    start++;
  }
  if (num_str.empty()) {
    return ResultErr("Invalid repeat format4");
  }
  upper_bound = std::stoi(num_str);
  while (static_cast<size_t>(start) < regex.size() && regex[start] == ' ') {
    start++;
  }
  if (regex[start] != '}') {
    return ResultErr("Invalid repeat format5");
  }
  XGRAMMAR_DCHECK(regex[start] == '}');
  return ResultOk(std::make_pair(lower_bound, upper_bound));
}

Result<FSMWithStartEnd> RegexIR::Build() const {
  if (states.empty()) {
    FSM empty_fsm(1);
    FSMWithStartEnd result(empty_fsm, 0, {true}, false);
    return ResultOk(std::move(result));
  }
  std::vector<FSMWithStartEnd> fsm_list;
  for (const auto& state : states) {
    auto visited = std::visit([&](auto&& arg) { return visit(arg); }, state);
    if (visited.IsErr()) {
      return visited;
    }
    fsm_list.push_back(std::move(visited).Unwrap());
  }
  if (fsm_list.size() > 1) {
    return ResultOk(FSMWithStartEnd::Concat(fsm_list));
  } else {
    // If there is only one FSM, return it directly.
    return ResultOk(std::move(fsm_list[0]));
  }
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Leaf& state) const {
  FSMWithStartEnd result = BuildLeafFSMFromRegex(state.regex);
  return ResultOk(std::move(result));
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Union& state) const {
  std::vector<FSMWithStartEnd> fsm_list;
  for (const auto& child : state.states) {
    auto visited = std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, child);
    if (visited.IsErr()) {
      return visited;
    }
    fsm_list.push_back(std::move(visited).Unwrap());
  }
  if (fsm_list.size() <= 1) {
    return ResultErr("Invalid union");
  }
  return ResultOk(FSMWithStartEnd::Union(fsm_list));
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Symbol& state) const {
  if (state.state.size() != 1) {
    return ResultErr("Invalid symbol");
  }
  Result<FSMWithStartEnd> child_result =
      std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, state.state[0]);
  if (child_result.IsErr()) {
    return child_result;
  }
  auto child = std::move(child_result).Unwrap();

  switch (state.symbol) {
    case RegexIR::RegexSymbol::plus: {
      return ResultOk(child.Plus());
    }
    case RegexIR::RegexSymbol::star: {
      return ResultOk(child.Star());
    }
    case RegexIR::RegexSymbol::optional: {
      return ResultOk(child.Optional());
    }
    default: {
      XGRAMMAR_LOG(FATAL) << "Unknown regex symbol: " << static_cast<int>(state.symbol);
    }
  }
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Bracket& state) const {
  std::vector<FSMWithStartEnd> fsm_list;
  for (const auto& child : state.states) {
    auto visited = std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, child);
    if (visited.IsErr()) {
      return visited;
    }
    fsm_list.push_back(std::move(visited).Unwrap());
  }
  if (fsm_list.empty()) {
    return ResultErr("Invalid bracket");
  }
  return ResultOk(FSMWithStartEnd::Concat(fsm_list));
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Repeat& state) const {
  if (state.states.size() != 1) {
    return ResultErr("Invalid repeat");
  }
  Result<FSMWithStartEnd> child_result =
      std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, state.states[0]);
  if (child_result.IsErr()) {
    return child_result;
  }
  FSMWithStartEnd child = std::move(child_result).Unwrap();
  FSMWithStartEnd result = child.Copy();
  std::unordered_set<int> new_ends;

  if (state.lower_bound == 1) {
    // Insert the first end state.
    for (int end = 0; end < result.NumStates(); ++end) {
      if (result.IsEndState(end)) {
        new_ends.insert(end);
      }
    }
  }

  // Handling {n,}
  if (state.upper_bound == RegexIR::kRepeatNoUpperBound) {
    for (int i = 2; i < state.lower_bound; i++) {
      result = FSMWithStartEnd::Concat(std::vector<FSMWithStartEnd>{result, child});
    }
    int end_state_of_lower_bound_fsm = -1;
    for (int end = 0; end < result.NumStates(); ++end) {
      if (result.IsEndState(end)) {
        end_state_of_lower_bound_fsm = end;
        break;
      }
    }
    XGRAMMAR_DCHECK(end_state_of_lower_bound_fsm != -1)
        << "No end state found in the lower bound FSM.";
    result = FSMWithStartEnd::Concat(std::vector<FSMWithStartEnd>{result, child});
    for (int end = 0; end < result.NumStates(); ++end) {
      if (result.IsEndState(end)) {
        result.GetFsm().AddEpsilonEdge(end, end_state_of_lower_bound_fsm);
      }
    }
    return ResultOk(std::move(result));
  }
  // Handling {n, m} or {n}
  for (int i = 2; i <= state.upper_bound; i++) {
    result = FSMWithStartEnd::Concat(std::vector<FSMWithStartEnd>{result, child});
    if (i >= state.lower_bound) {
      for (int end = 0; end < result.NumStates(); ++end) {
        if (result.IsEndState(end)) {
          new_ends.insert(end);
        }
      }
    }
  }
  for (const auto& end : new_ends) {
    result.AddEndState(end);
  }
  return ResultOk(std::move(result));
}

FSMWithStartEnd RegexIR::BuildLeafFSMFromRegex(const std::string& regex) {
  FSM empty_fsm(0);
  FSMWithStartEnd result(empty_fsm, 0, {}, true);
  // Handle the regex string.
  if (!(regex[0] == '[' && regex[regex.size() - 1] == ']')) {
    result.AddState();
    for (size_t i = 0; i < regex.size(); i++) {
      if (regex[i] != '\\') {
        if (regex[i] == '.') {
          result.GetFsm().AddEdge(result.NumStates() - 1, result.NumStates(), 0, 0xFF);
        } else {
          result.GetFsm().AddEdge(
              result.NumStates() - 1,
              result.NumStates(),
              static_cast<uint8_t>(regex[i]),
              static_cast<uint8_t>(regex[i])
          );
        }
        result.AddState();
        continue;
      }
      std::vector<std::pair<int, int>> escape_vector = HandleEscapes(regex, i);
      for (const auto& escape : escape_vector) {
        result.GetFsm().AddEdge(
            result.NumStates() - 1,
            result.NumStates(),
            static_cast<uint8_t>(escape.first),
            static_cast<uint8_t>(escape.second)
        );
      }
      result.AddState();
      i++;
    }
    result.AddEndState(result.NumStates() - 1);
  } else if (regex[0] == '[' && regex[regex.size() - 1] == ']') {
    // Handle the character class.
    result.AddState();
    result.AddState();
    result.AddEndState(1);
    bool reverse = regex[1] == '^';
    for (size_t i = reverse ? 2 : 1; i < regex.size() - 1; i++) {
      if (regex[i] != '\\') {
        if (!(((i + 2) < regex.size() - 1) && regex[i + 1] == '-')) {
          // A single char.
          result.GetFsm().AddEdge(
              0, 1, static_cast<uint8_t>(regex[i]), static_cast<uint8_t>(regex[i])
          );
          continue;
        }
        // Handle the char range.
        if (regex[i + 2] != '\\') {
          result.GetFsm().AddEdge(
              0, 1, static_cast<uint8_t>(regex[i]), static_cast<uint8_t>(regex[i + 2])
          );
          i = i + 2;
          continue;
        }
        auto escaped_edges = HandleEscapes(regex, i + 2);
        // Means it's not a range.
        if (escaped_edges.size() != 1 || escaped_edges[0].first != escaped_edges[0].second) {
          result.GetFsm().AddEdge(
              0, 1, static_cast<uint8_t>(regex[i]), static_cast<uint8_t>(regex[i])
          );
          continue;
        }
        result.GetFsm().AddEdge(
            0, 1, static_cast<uint8_t>(regex[0]), static_cast<uint8_t>(escaped_edges[0].first)
        );
        i = i + 3;
        continue;
      }
      auto escaped_edges = HandleEscapes(regex, i);
      i = i + 1;
      if (escaped_edges.size() != 1 || escaped_edges[0].first != escaped_edges[0].second) {
        // It's a multi-match escape char.
        for (const auto& edge : escaped_edges) {
          result.GetFsm().AddEdge(
              0, 1, static_cast<uint8_t>(edge.first), static_cast<uint8_t>(edge.second)
          );
        }
        continue;
      }
      if (!(((i + 2) < regex.size() - 1) && regex[i + 1] == '-')) {
        result.GetFsm().AddEdge(
            0,
            1,
            static_cast<uint8_t>(escaped_edges[0].first),
            static_cast<uint8_t>(escaped_edges[0].second)
        );
        continue;
      }
      if (regex[i + 2] != '\\') {
        result.GetFsm().AddEdge(
            0, 1, static_cast<uint8_t>(escaped_edges[0].first), static_cast<uint8_t>(regex[i + 2])
        );
        i = i + 2;
        continue;
      }
      auto rhs_escaped_edges = HandleEscapes(regex, i + 2);
      if (rhs_escaped_edges.size() != 1 ||
          rhs_escaped_edges[0].first != rhs_escaped_edges[0].second) {
        result.GetFsm().AddEdge(
            0,
            1,
            static_cast<uint8_t>(escaped_edges[0].first),
            static_cast<uint8_t>(escaped_edges[0].second)
        );
        continue;
      }
      result.GetFsm().AddEdge(
          0,
          1,
          static_cast<uint8_t>(escaped_edges[0].first),
          static_cast<uint8_t>(rhs_escaped_edges[0].first)
      );
      i = i + 3;
      continue;
    }
    bool has_edge[0x100];
    memset(has_edge, 0, sizeof(has_edge));
    FSM new_fsm(2);
    for (const auto& edge : result.GetFsm().GetEdges(0)) {
      for (int i = edge.min; i <= edge.max; i++) {
        has_edge[i] = true;
      }
    }
    // Simplify the edges. e.g [abc] -> [a-c]
    int last = -1;
    if (reverse) {
      for (int i = 0; i < 0x100; i++) {
        if (!has_edge[i]) {
          if (last == -1) {
            last = i;
          }
          continue;
        }
        if (last != -1) {
          new_fsm.AddEdge(0, 1, last, i - 1);
          last = -1;
        }
      }
      if (last != -1) {
        new_fsm.AddEdge(0, 1, last, 0xFF);
      }
    } else {
      for (int i = 0; i < 0x100; i++) {
        if (has_edge[i]) {
          if (last == -1) {
            last = i;
          }
          continue;
        }
        if (last != -1) {
          new_fsm.AddEdge(0, 1, last, i - 1);
          last = -1;
        }
      }
      if (last != -1) {
        new_fsm.AddEdge(0, 1, last, 0xFF);
      }
    }
    std::vector<bool> ends(new_fsm.NumStates(), false);
    ends[1] = true;
    result = FSMWithStartEnd(new_fsm, 0, ends, false);
  } else {
    // TODO: The support for rules.
    XGRAMMAR_LOG(WARNING) << "rule is not supported yet.";
  }
  return result;
}

std::vector<std::pair<int, int>> RegexIR::HandleEscapes(const std::string& regex, int start) {
  std::vector<std::pair<int, int>> result;
  switch (regex[start + 1]) {
    case 'n': {
      return std::vector<std::pair<int, int>>(1, std::make_pair('\n', '\n'));
    }
    case 't': {
      return std::vector<std::pair<int, int>>(1, std::make_pair('\t', '\t'));
    }
    case 'r': {
      return std::vector<std::pair<int, int>>(1, std::make_pair('\r', '\r'));
    }

    case '0': {
      return std::vector<std::pair<int, int>>(1, std::make_pair('\0', '\0'));
    }
    case 's': {
      return std::vector<std::pair<int, int>>(1, std::make_pair(0, ' '));
    }
    case 'S': {
      return std::vector<std::pair<int, int>>(1, std::make_pair(' ' + 1, 0x00FF));
    }
    case 'd': {
      return std::vector<std::pair<int, int>>(1, std::make_pair('0', '9'));
    }
    case 'D': {
      std::vector<std::pair<int, int>> result;
      result.emplace_back(0, '0' - 1);
      result.emplace_back('9' + 1, 0x00FF);
      return result;
    }
    case 'w': {
      std::vector<std::pair<int, int>> result;
      result.emplace_back('0', '9');
      result.emplace_back('a', 'z');
      result.emplace_back('A', 'Z');
      result.emplace_back('_', '_');
      return result;
    }
    case 'W': {
      std::vector<std::pair<int, int>> result;
      result.emplace_back(0, '0' - 1);
      result.emplace_back('9' + 1, 'A' - 1);
      result.emplace_back('Z' + 1, '_' - 1);
      result.emplace_back('_' + 1, 'a' - 1);
      result.emplace_back('z' + 1, 0x00FF);
      return result;
    }
    default: {
      return std::vector<std::pair<int, int>>(
          1, std::make_pair(regex[start + 1], regex[start + 1])
      );
    }
  }
}

Result<FSMWithStartEnd> RegexFSMBuilder::Build(const std::string& regex) {
  RegexIR ir;
  using IRState = std::variant<RegexIR::State, char>;
  // We use a stack to store the states.
  std::stack<IRState> stack;
  int left_middle_bracket = -1;
  for (int i = 0; i < static_cast<int>(regex.size()); i++) {
    if (i == 0 && regex[i] == '^') {
      continue;
    }
    if (i == static_cast<int>(regex.size()) - 1 && regex[i] == '$') {
      continue;
    }
    // Handle The class.
    if (regex[i] == '[') {
      if (left_middle_bracket != -1) {
        return ResultErr("Nested middle bracket!");
      }
      left_middle_bracket = i;
      continue;
    }
    if (regex[i] == ']') {
      if (left_middle_bracket == -1) {
        return ResultErr("Invalid middle bracket!");
      }
      RegexIR::Leaf leaf;
      leaf.regex = regex.substr(left_middle_bracket, i - left_middle_bracket + 1);
      stack.push(leaf);
      left_middle_bracket = -1;
      continue;
    }
    if (left_middle_bracket != -1) {
      if (regex[i] == '\\') {
        i++;
      }
      continue;
    }
    if (regex[i] == '+' || regex[i] == '*' || regex[i] == '?') {
      if (stack.empty()) {
        return ResultErr("Invalid regex: no state before operator!");
      }
      auto state = stack.top();
      if (std::holds_alternative<char>(state)) {
        return ResultErr("Invalid regex: no state before operator!");
      }
      stack.pop();
      auto child = std::get<RegexIR::State>(state);
      RegexIR::Symbol symbol;
      symbol.state.push_back(child);
      switch (regex[i]) {
        case '+': {
          symbol.symbol = RegexIR::RegexSymbol::plus;
          break;
        }
        case '*': {
          symbol.symbol = RegexIR::RegexSymbol::star;
          break;
        }
        case '?': {
          symbol.symbol = RegexIR::RegexSymbol::optional;
          break;
        }
      }
      stack.push(symbol);
      continue;
    }
    if (regex[i] == '(' || regex[i] == '|') {
      stack.push(regex[i]);
      if (i < static_cast<int>(regex.size()) - 2 && regex[i] == '(' && regex[i + 1] == '?' &&
          regex[i + 2] == ':') {
        i += 2;
        continue;
      }
      if (i < static_cast<int>(regex.size()) - 2 && regex[i] == '(' && regex[i + 1] == '?' &&
          (regex[i + 2] == '!' || regex[i + 2] == '=')) {
        i += 2;
        // TODO(Linzhang Li): Handling the lookahead.
        continue;
      }
      continue;
    }
    if (regex[i] == ')') {
      std::stack<IRState> states;
      bool paired = false;
      bool unioned = false;
      while ((!stack.empty()) && (!paired)) {
        auto state = stack.top();
        stack.pop();
        if (std::holds_alternative<char>(state)) {
          char c = std::get<char>(state);
          if (c == '(') {
            paired = true;
            break;
          }
          if (c == '|') {
            unioned = true;
          }
          states.push(state);
        } else {
          states.push(state);
        }
      }
      if (!paired) {
        return ResultErr("Invalid regex: no paired bracket!" + std::to_string(__LINE__));
      }
      if (states.empty()) {
        continue;
      }
      if (!unioned) {
        RegexIR::Bracket bracket;
        while (!states.empty()) {
          auto state = states.top();
          states.pop();
          auto child = std::get<RegexIR::State>(state);
          bracket.states.push_back(child);
        }
        stack.push(bracket);
      } else {
        RegexIR::Union union_state;
        RegexIR::Bracket bracket;
        while (!states.empty()) {
          auto state = states.top();
          states.pop();
          if (std::holds_alternative<char>(state)) {
            char c = std::get<char>(state);
            if (c == '|') {
              union_state.states.push_back(bracket);
              bracket.states.clear();
              continue;
            }
            return ResultErr("Invalid regex: no paired bracket!" + std::to_string(__LINE__));
          }
          if (std::holds_alternative<RegexIR::State>(state)) {
            auto child = std::get<RegexIR::State>(state);
            bracket.states.push_back(child);
            continue;
          }
          return ResultErr("Invalid regex: no paired bracket!" + std::to_string(__LINE__));
        }
        union_state.states.push_back(bracket);
        stack.push(union_state);
      }
      continue;
    }
    if (regex[i] == '{') {
      if (stack.empty()) {
        return ResultErr("Invalid regex: no state before repeat!");
      }
      auto state = stack.top();
      if (std::holds_alternative<char>(state)) {
        return ResultErr("Invalid regex: no state before repeat!");
      }
      stack.pop();
      auto bounds_result = RegexIR::CheckRepeat(regex, i);
      if (bounds_result.IsErr()) {
        return ResultErr(std::move(bounds_result).UnwrapErr());
      }
      auto bounds = std::move(bounds_result).Unwrap();
      auto child = std::get<RegexIR::State>(state);
      RegexIR::Repeat repeat;
      repeat.lower_bound = bounds.first;
      repeat.upper_bound = bounds.second;
      repeat.states.push_back(child);
      stack.push(repeat);
      continue;
    }
    RegexIR::Leaf leaf;
    if (regex[i] != '\\') {
      leaf.regex = regex[i];
    } else {
      leaf.regex = regex.substr(i, 2);
      i++;
    }
    stack.push(leaf);
    continue;
  }
  std::vector<RegexIR::State> res_states;
  std::vector<decltype(res_states)> union_state_list;
  bool unioned = false;
  while (!stack.empty()) {
    if (std::holds_alternative<char>(stack.top())) {
      char c = std::get<char>(stack.top());
      if (c == '|') {
        union_state_list.push_back(res_states);
        res_states.clear();
        unioned = true;
        stack.pop();
        continue;
      }
      return ResultErr("Invalid regex: no paired!");
    }
    auto state = stack.top();
    stack.pop();
    auto child = std::get<RegexIR::State>(state);
    res_states.push_back(std::move(child));
  }
  if (!unioned) {
    for (auto it = res_states.rbegin(); it != res_states.rend(); ++it) {
      ir.states.push_back(std::move(*it));
    }
  } else {
    union_state_list.push_back(res_states);
    RegexIR::Union union_state;
    for (auto it = union_state_list.begin(); it != union_state_list.end(); ++it) {
      RegexIR::Bracket bracket;
      for (auto state = it->rbegin(); state != it->rend(); ++state) {
        bracket.states.push_back(std::move(*state));
      }
      union_state.states.push_back(std::move(bracket));
    }
    ir.states.push_back(std::move(union_state));
  }
  return ir.Build();
}

class TrieFSMBuilderImpl {
 public:
  TrieFSMBuilderImpl() = default;
  std::optional<FSMWithStartEnd> Build(
      const std::vector<std::string>& patterns,
      std::vector<int32_t>* end_states,
      bool allow_overlap,
      bool add_back_edges
  );
  void AddBackEdges(FSM* fsm, int start, const std::unordered_set<int>& ends);
};

std::optional<FSMWithStartEnd> TrieFSMBuilderImpl::Build(
    const std::vector<std::string>& patterns,
    std::vector<int32_t>* end_states,
    bool allow_overlap,
    bool add_back_edges
) {
  FSM fsm(1);
  int start = 0;
  std::unordered_set<int> ends;

  if (end_states) {
    end_states->clear();
  }

  for (const auto& pattern : patterns) {
    // Check for empty patterns
    if (!allow_overlap && pattern.empty()) {
      return std::nullopt;
    }

    int current_state = 0;
    for (const auto& ch : pattern) {
      int16_t ch_int16 = static_cast<int16_t>(static_cast<uint8_t>(ch));
      int next_state = fsm.GetNextState(current_state, ch_int16);
      if (next_state == FSM::kNoNextState) {
        next_state = fsm.AddState();
        fsm.AddEdge(current_state, next_state, ch_int16, ch_int16);
      }
      current_state = next_state;
      if (!allow_overlap && ends.count(current_state) > 0) {
        return std::nullopt;
      }
    }
    if (!allow_overlap && fsm.GetEdges(current_state).size() > 0) {
      return std::nullopt;
    }
    ends.insert(current_state);
    if (end_states) {
      end_states->push_back(current_state);
    }
  }
  if (add_back_edges) {
    AddBackEdges(&fsm, start, ends);
  }

  std::vector<bool> is_end_state(fsm.NumStates(), false);
  for (const auto& end : ends) {
    is_end_state[end] = true;
  }

  return FSMWithStartEnd(fsm, start, is_end_state);
}

void TrieFSMBuilderImpl::AddBackEdges(FSM* fsm, int start, const std::unordered_set<int>& ends) {
  // Build an Aho-Corasick automaton by adding back edges.
  // When matching on the trie fails, we should go back to the start state and
  // find the next match. Back edges represent such state transitions.

  auto f_add_range_edges = [&](int node, std::set<FSMEdge, FSMEdgeRangeComparator>& cur_edges_set) {
    cur_edges_set.insert(FSMEdge(-1, -1, 0));
    cur_edges_set.insert(FSMEdge(256, 256, 0));
    XGRAMMAR_DCHECK(cur_edges_set.size() >= 2);
    for (auto it = std::next(cur_edges_set.begin()); it != cur_edges_set.end(); ++it) {
      FSMEdge prev_edge = *std::prev(it);
      XGRAMMAR_DCHECK(prev_edge.max < it->min);
      if (prev_edge.max + 1 != it->min) {
        auto new_edge = FSMEdge(prev_edge.max + 1, it->min - 1, start);
        // The new edge should be inserted before the current edge to avoid infinite loop.
        XGRAMMAR_DCHECK(new_edge < *it);
        cur_edges_set.insert(new_edge);
      }
    }

    // Remove first and last element of cur_edges_set
    XGRAMMAR_DCHECK(*cur_edges_set.begin() == FSMEdge(-1, -1, 0));
    XGRAMMAR_DCHECK(*std::prev(cur_edges_set.end()) == FSMEdge(256, 256, 0));
    cur_edges_set.erase(cur_edges_set.begin());
    cur_edges_set.erase(std::prev(cur_edges_set.end()));

    XGRAMMAR_DCHECK(cur_edges_set.begin()->min == 0);
    XGRAMMAR_DCHECK(std::prev(cur_edges_set.end())->max == 255);
  };

  for (int i = 0; i < fsm->NumStates(); i++) {
    if (i == start || ends.count(i) > 0) {
      continue;
    }
    std::vector<FSMEdge>& cur_edges = fsm->GetEdges(i);
    XGRAMMAR_DCHECK(cur_edges.size() > 0);
    std::set<FSMEdge, FSMEdgeRangeComparator> cur_edges_set(cur_edges.begin(), cur_edges.end());

    // Step 1. Add edges in the edges of the start state.
    // For start--(c)-->t, add i--(c)-->t.
    const auto& root_edges = fsm->GetEdges(start);
    for (const auto& root_edge : root_edges) {
      XGRAMMAR_DCHECK(root_edge.min == root_edge.max);
      if (cur_edges_set.count(root_edge) == 0) {
        cur_edges_set.insert(root_edge);
      }
    }

    // Step 2. Add i--(c)-->start for c not in the edge set of i.
    f_add_range_edges(i, cur_edges_set);

    // Step 3. Update the edges of i.
    cur_edges.clear();
    cur_edges.insert(cur_edges.end(), cur_edges_set.begin(), cur_edges_set.end());
  }

  // Finally, add range edges to the start state.
  std::vector<FSMEdge>& start_edges = fsm->GetEdges(start);
  std::set<FSMEdge, FSMEdgeRangeComparator> start_edges_set(start_edges.begin(), start_edges.end());
  f_add_range_edges(start, start_edges_set);
  start_edges.clear();
  start_edges.insert(start_edges.end(), start_edges_set.begin(), start_edges_set.end());
}

std::optional<FSMWithStartEnd> TrieFSMBuilder::Build(
    const std::vector<std::string>& patterns,
    std::vector<int32_t>* end_states,
    bool allow_overlap,
    bool add_back_edges
) {
  return TrieFSMBuilderImpl().Build(patterns, end_states, allow_overlap, add_back_edges);
}

}  // namespace xgrammar
