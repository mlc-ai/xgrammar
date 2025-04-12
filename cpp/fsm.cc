#include <xgrammar/fsm.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace xgrammar {
std::vector<std::pair<int, int>> HandleEscapeInClass(const std::string& regex, int start);
char HandleEscapeInString(const std::string& regex, int start);

std::unordered_set<int> CompactFSM::GetEpsilonClosure(int state) const {
  std::queue<int> queue = std::queue<int>({state});
  std::unordered_set<int> closure;
  while (!queue.empty()) {
    int current = queue.front();
    queue.pop();
    if (closure.find(current) != closure.end()) {
      continue;
    }
    closure.insert(current);
    for (const auto& edge : edges[current]) {
      if (edge.IsEpsilon()) {
        queue.push(edge.target);
      }
    }
  }
  return closure;
}

FSMEdge::FSMEdge(const short& _min, const short& _max, const int& target)
    : min(_min), max(_max), target(target) {
  if (IsCharRange() && min > max) {
    throw std::runtime_error("Invalid char range: min > max");
  }
}

bool FSMEdge::IsEpsilon() const { return min == -1 && max == -1; }

bool FSMEdge::IsRuleRef() const { return min == -1 && max != -1; }

bool FSMEdge::IsCharRange() const { return min >= 0 && max >= 0; }

short FSMEdge::GetRefRuleId() const {
  if (IsRuleRef()) {
    return max;
  } else {
    throw std::runtime_error("Not a rule reference!");
  }
}

std::unordered_set<int> FSM::GetEpsilonClosure(int state) const {
  std::queue<int> queue = std::queue<int>({state});
  std::unordered_set<int> closure;
  while (!queue.empty()) {
    int current = queue.front();
    queue.pop();
    if (closure.find(current) != closure.end()) {
      continue;
    }
    closure.insert(current);
    for (const auto& edge : edges[current]) {
      if (edge.IsEpsilon()) {
        queue.push(edge.target);
      }
    }
  }
  return closure;
}

FSM FSM::Copy() const {
  FSM copy;
  copy.edges.resize(edges.size());
  for (size_t i = 0; i < edges.size(); ++i) {
    copy.edges[i] = edges[i];
  }
  return copy;
}

FSMWithStartEnd FSMWithStartEnd::Union(const std::vector<FSMWithStartEnd>& fsms) {
  FSMWithStartEnd result;
  int node_cnt = 1;
  result.start = 0;
  // In the new FSM, we define the start state is 0.
  result.fsm.edges.push_back(std::vector<FSMEdge>());
  for (const auto& fsm_with_se : fsms) {
    result.fsm.edges[0].emplace_back(-1, -1, fsm_with_se.start + node_cnt);
    for (const auto& edges : fsm_with_se.fsm.edges) {
      result.fsm.edges.push_back(std::vector<FSMEdge>());
      for (const auto& edge : edges) {
        result.fsm.edges.back().emplace_back(edge.min, edge.max, edge.target + node_cnt);
      }
      for (const auto& end : fsm_with_se.ends) {
        result.ends.insert(end + node_cnt);
      }
    }
    node_cnt += fsm_with_se.fsm.edges.size();
  }
  return result;
}

FSMWithStartEnd FSMWithStartEnd::Not() const {
  FSMWithStartEnd result;

  // Build the DFA.
  if (!is_dfa) {
    result = TODFA();
  } else {
    result = Copy();
  }
  int node_cnt = result.fsm.edges.size();

  // Reverse all the final states.
  std::unordered_set<int> final_states;
  for (int i = 0; i < node_cnt; ++i) {
    if (result.ends.find(i) == result.ends.end()) {
      final_states.insert(i);
    }
  }
  result.ends = final_states;

  // Add all the rules in the alphabet.
  std::unordered_set<int> rules;
  for (const auto& edges : result.fsm.edges) {
    for (const auto& edge : edges) {
      if (edge.IsRuleRef()) {
        rules.insert(edge.GetRefRuleId());
      }
    }
  }

  // Add a new state to avoid the blocking.
  result.fsm.edges.push_back(std::vector<FSMEdge>());
  for (auto rule : rules) {
    result.fsm.edges.back().emplace_back(-1, rule, node_cnt);
  }
  result.fsm.edges.back().emplace_back(0, 0x00FF, node_cnt);
  result.ends.insert(node_cnt);

  for (size_t i = 0; i < fsm.edges.size(); i++) {
    const auto& node_edges = fsm.edges[i];
    std::vector<bool> char_has_edges(0x100, false);
    std::unordered_set<int> rule_has_edges;
    for (const auto& edge : node_edges) {
      if (edge.IsCharRange()) {
        for (int i = edge.min; i <= edge.max; ++i) {
          char_has_edges[i] = true;
        }
      }
      if (edge.IsRuleRef()) {
        rule_has_edges.insert(edge.GetRefRuleId());
      }
    }

    // Add the left characters to the new state.
    int interval_start = -1;
    for (int j = 0; j < 0x100; ++j) {
      if (!char_has_edges[j]) {
        // The char doesn't have any edges. Thus, we can accept it in the
        // complement FSM.
        if (interval_start == -1) {
          interval_start = j;
        }
      } else {
        if (interval_start != -1) {
          // node_cnt is the node to accept all such characters.
          result.fsm.edges[i].emplace_back(interval_start, i - 1, node_cnt);
          interval_start = -1;
        }
      }
    }
    if (interval_start != -1) {
      result.fsm.edges[i].emplace_back(interval_start, 0xFF, node_cnt);
    }

    // Add the left rules to the new state.
    for (auto rule : rules) {
      if (rule_has_edges.find(rule) == rule_has_edges.end()) {
        result.fsm.edges.back().emplace_back(-1, rule, node_cnt);
      }
    }
  }
  return result;
}

void FSM::Advance(const std::vector<int>& from, int value, std::vector<int>* result, bool is_rule)
    const {
  result->clear();
  std::queue<int> queue = std::queue<int>();
  std::unordered_set<int> visited;
  std::unordered_set<int> in_result;
  std::unordered_set<int> start_set;

  for (const auto& state : from) {
    queue.push(state);
  }
  while (!queue.empty()) {
    int current = queue.front();
    queue.pop();
    if (visited.find(current) != visited.end()) {
      continue;
    }
    visited.insert(current);
    for (const auto& edge : edges[current]) {
      if (edge.IsEpsilon()) {
        queue.push(edge.target);
        continue;
      }
      if (is_rule && edge.IsRuleRef()) {
        if (edge.GetRefRuleId() == value) {
          in_result.insert(edge.target);
        }
        continue;
      }
      if (!is_rule && edge.IsCharRange()) {
        if (value >= edge.min && value <= edge.max) {
          in_result.insert(edge.target);
        }
        continue;
      }
    }
  }
  std::unordered_set<int> result_closure;
  for (const auto& state : in_result) {
    auto closure = GetEpsilonClosure(state);
    result_closure.insert(closure.begin(), closure.end());
  }
  for (const auto& state : result_closure) {
    result->push_back(state);
  }
  return;
}

FSMWithStartEnd FSMWithStartEnd::Copy() const {
  FSMWithStartEnd copy;
  copy.is_dfa = is_dfa;
  copy.start = start;
  copy.ends = ends;
  copy.fsm = fsm.Copy();
  return copy;
}

std::string FSMWithStartEnd::Print() const {
  std::string result;
  result += "FSM(num_nodes=" + std::to_string(fsm.edges.size()) +
            ", start=" + std::to_string(start) + ", end=[";
  for (const auto& end : ends) {
    result += std::to_string(end) + ", ";
  }
  result += "], edges=[\n";
  for (int i = 0; i < int(fsm.edges.size()); ++i) {
    result += std::to_string(i) + ": [";
    const auto& edges = fsm.edges[i];
    for (int j = 0; j < static_cast<int>(fsm.edges[i].size()); ++j) {
      const auto& edge = edges[j];
      if (edge.min == edge.max) {
        result += "(" + std::to_string(edge.min) + ")->" + std::to_string(edge.target);
      } else {
        result += "(" + std::to_string(edge.min) + ", " + std::to_string(edge.max) + ")->" +
                  std::to_string(edge.target);
      }
      if (j < static_cast<int>(fsm.edges[i].size()) - 1) {
        result += ", ";
      }
    }
    result += "]\n";
  }
  result += "])";
  return result;
}

std::string CompactFSMWithStartEnd::Print() const {
  std::string result;
  result += "CompactFSM(num_nodes=" + std::to_string(fsm.edges.Size()) +
            ", start=" + std::to_string(start) + ", end=[";
  for (const auto& end : ends) {
    result += std::to_string(end) + ", ";
  }
  result += "], edges=[\n";
  for (int i = 0; i < int(fsm.edges.Size()); ++i) {
    result += std::to_string(i) + ": [";
    const auto& edges = fsm.edges[i];
    for (int j = 0; j < static_cast<int>(fsm.edges[i].size()); ++j) {
      const auto& edge = edges[j];
      if (edge.min == edge.max) {
        result += "(" + std::to_string(edge.min) + ")->" + std::to_string(edge.target);
      } else {
        result += "(" + std::to_string(edge.min) + ", " + std::to_string(edge.max) + ")->" +
                  std::to_string(edge.target);
      }
      if (j < static_cast<int>(fsm.edges[i].size()) - 1) {
        result += ", ";
      }
    }
    result += "]\n";
  }
  result += "])";
  return result;
}

CompactFSM FSM::ToCompact() {
  CompactFSM result;
  for (int i = 0; i < static_cast<int>(edges.size()); ++i) {
    std::sort(edges[i].begin(), edges[i].end(), [](const FSMEdge& a, const FSMEdge& b) {
      return a.min != b.min ? a.min < b.min : a.max < b.max;
    });
    result.edges.Insert(edges[i]);
  }
  return result;
}

FSM CompactFSM::ToFSM() {
  FSM result;
  for (int i = 0; i < edges.Size(); i++) {
    const auto& row = edges[i];
    result.edges.emplace_back(std::vector<FSMEdge>());
    for (int j = 0; j < row.size(); j++) {
      result.edges.back().push_back(row[j]);
    }
  }
  return result;
}

void CompactFSM::Advance(
    const std::vector<int>& from, int value, std::vector<int>* result, bool is_rule
) const {
  result->clear();
  std::queue<int> queue = std::queue<int>();
  std::unordered_set<int> visited;
  std::unordered_set<int> in_result;
  std::unordered_set<int> start_set;

  for (const auto& state : from) {
    queue.push(state);
  }
  while (!queue.empty()) {
    int current = queue.front();
    queue.pop();
    if (visited.find(current) != visited.end()) {
      continue;
    }
    visited.insert(current);
    for (const auto& edge : edges[current]) {
      if (edge.IsEpsilon()) {
        queue.push(edge.target);
        continue;
      }
      if (is_rule && edge.IsRuleRef()) {
        if (edge.GetRefRuleId() == value) {
          in_result.insert(edge.target);
        }
        continue;
      }
      if (!is_rule && edge.IsCharRange()) {
        if (value >= edge.min && value <= edge.max) {
          in_result.insert(edge.target);
        }
        continue;
      }
    }
  }
  std::unordered_set<int> result_closure;
  for (const auto& state : in_result) {
    auto closure = GetEpsilonClosure(state);
    result_closure.insert(closure.begin(), closure.end());
  }
  for (const auto& state : result_closure) {
    result->push_back(state);
  }
  return;
}

FSMWithStartEnd FSMWithStartEnd::TODFA() const {
  FSMWithStartEnd dfa;
  dfa.is_dfa = true;
  dfa.start = start;
  std::vector<std::unordered_set<int>> closures;
  std::unordered_set<int> rules;
  for (const auto& edges : fsm.edges) {
    for (const auto& edge : edges) {
      if (edge.IsRuleRef()) {
        rules.insert(edge.GetRefRuleId());
      }
    }
  }
  int now_process = 0;
  closures.push_back(fsm.GetEpsilonClosure(start));
  while (now_process < static_cast<int>(closures.size())) {
    std::set<int> interval_ends;
    dfa.fsm.edges.push_back(std::vector<FSMEdge>());
    // Check if the closure is a final state.
    for (const auto& node : closures[now_process]) {
      if (ends.find(node) != ends.end()) {
        dfa.ends.insert(now_process);
      }
      const auto& edges = fsm.edges[node];
      for (const auto& edge : edges) {
        if (edge.IsCharRange()) {
          interval_ends.insert(edge.min);
          interval_ends.insert(edge.max + 1);
          continue;
        }
      }
    }
    // This part is to get the all possible intervals.
    // Which can help reduce the transitions.
    using Interval = std::pair<int, int>;
    std::vector<Interval> intervals;
    intervals.reserve(interval_ends.size());
    int last = -1;
    for (const auto& end : interval_ends) {
      if (last == -1) {
        last = end;
        continue;
      }
      intervals.emplace_back(last, end - 1);
      last = end;
    }
    for (const auto& interval : intervals) {
      std::unordered_set<int> next_closure;
      for (const auto& node : closures[now_process]) {
        const auto& edges = fsm.edges[node];
        for (const auto& edge : edges) {
          if (edge.IsCharRange()) {
            if (interval.first >= edge.min && interval.second <= edge.max) {
              auto epsilon_closure = fsm.GetEpsilonClosure(edge.target);
              next_closure.insert(epsilon_closure.begin(), epsilon_closure.end());
            }
          }
        }
      }
      bool flag = false;
      for (int j = 0; j < static_cast<int>(closures.size()); j++) {
        if (closures[j] == next_closure) {
          dfa.fsm.edges[now_process].emplace_back(interval.first, interval.second, j);
          flag = true;
          break;
        }
      }
      if (!flag) {
        dfa.fsm.edges[now_process].emplace_back(interval.first, interval.second, closures.size());
        closures.push_back(next_closure);
      }
    }
    for (auto rule : rules) {
      std::unordered_set<int> next_closure;
      for (const auto& node : closures[now_process]) {
        const auto& edges = fsm.edges[node];
        for (const auto& edge : edges) {
          if (edge.IsRuleRef()) {
            if (rule == edge.GetRefRuleId()) {
              auto epsilon_closure = fsm.GetEpsilonClosure(edge.target);
              next_closure.insert(epsilon_closure.begin(), epsilon_closure.end());
            }
          }
        }
      }
      bool flag = false;
      for (int j = 0; j < static_cast<int>(closures.size()); j++) {
        if (closures[j] == next_closure) {
          dfa.fsm.edges[now_process].emplace_back(-1, rule, j);
          flag = true;
          break;
        }
      }
      if (!flag) {
        dfa.fsm.edges[now_process].emplace_back(-1, rule, closures.size());
        closures.push_back(next_closure);
      }
    }
    now_process++;
  }
  return dfa;
}

FSMWithStartEnd FSMWithStartEnd::Concatenate(const std::vector<FSMWithStartEnd>& fsms) {
  FSMWithStartEnd result;
  result.is_dfa = false;
  int node_cnt = 0;
  result.start = fsms[0].start;
  for (size_t i = 0; i < fsms.size(); i++) {
    const auto& fsm_with_se = fsms[i];
    for (const auto& edges : fsm_with_se.fsm.edges) {
      result.fsm.edges.push_back(std::vector<FSMEdge>());
      for (const auto& edge : edges) {
        result.fsm.edges.back().emplace_back(edge.min, edge.max, edge.target + node_cnt);
      }
    }
    if (i == fsms.size() - 1) {
      for (const auto& end : fsm_with_se.ends) {
        result.ends.insert(end + node_cnt);
      }
      break;
    }
    for (const auto& end : fsm_with_se.ends) {
      result.fsm.edges[end + node_cnt].emplace_back(
          -1, -1, fsm_with_se.fsm.edges.size() + node_cnt + fsms[i + 1].start
      );
    }
    node_cnt += fsm_with_se.fsm.edges.size();
  }
  return result;
}

FSMWithStartEnd FSMWithStartEnd::MakeStar() const {
  FSMWithStartEnd result;
  result.is_dfa = false;
  result.fsm = fsm.Copy();
  result.ends = ends;
  result.start = start;
  for (const auto& end : ends) {
    result.fsm.edges[end].emplace_back(-1, -1, start);
  }
  result.fsm.edges[start].emplace_back(-1, -1, *ends.begin());
  return result;
}

FSMWithStartEnd FSMWithStartEnd::MakePlus() const {
  FSMWithStartEnd result;
  result.is_dfa = false;
  result.fsm = fsm.Copy();
  result.ends = ends;
  result.start = start;
  for (const auto& end : ends) {
    result.fsm.edges[end].emplace_back(-1, -1, start);
  }
  return result;
}

FSMWithStartEnd FSMWithStartEnd::MakeOptional() const {
  FSMWithStartEnd result;
  result.is_dfa = false;
  result.fsm = fsm.Copy();
  result.ends = ends;
  result.start = start;
  result.fsm.edges[start].emplace_back(-1, -1, *ends.begin());
  return result;
}

FSMWithStartEnd RegexToFSM(const std::string& regex, int start, int end) {
  bool flag = false;
  if (end == -1) {
    end = regex.size();
  }
  FSMWithStartEnd result;
  bool quotation_mode = false;
  bool set_mode = false;
  bool not_mode = false;
  int left_middle_bracket = -1;
  int left_quote = -1;
  std::stack<int> bracket_stack;
  for (int i = start; i < end; i++) {
    // Skip the white spaces.
    if (regex[i] == ' ') {
      continue;
    }

    // Handle the not operator.
    if (regex[i] == '!') {
      if (quotation_mode || set_mode) {
        continue;
      }
      if (not_mode) {
        throw std::runtime_error("Invalid regex: nested '!' operator.");
      }
      not_mode = true;
      continue;
    }

    // Handle the escape character.
    if (regex[i] == '\\') {
      i = i + 1;
      continue;
    }

    // Handle the strings like "...".
    if (regex[i] == '\"' && !set_mode) {
      if (quotation_mode && bracket_stack.empty()) {
        FSMWithStartEnd tmp_fsm(regex.substr(left_quote, i - left_quote + 1));
        if (i < end - 1) {
          switch (regex[i + 1]) {
            case '+': {
              tmp_fsm = tmp_fsm.MakePlus();
              i = i + 1;
              break;
            }
            case '*': {
              tmp_fsm = tmp_fsm.MakeStar();
              i = i + 1;
              break;
            }
            case '?': {
              tmp_fsm = tmp_fsm.MakeOptional();
              i = i + 1;
              break;
            }
            default: {
              break;
            }
          }
        }
        if (not_mode) {
          tmp_fsm = tmp_fsm.Not();
          not_mode = false;
        }
        if (flag) {
          result = FSMWithStartEnd::Concatenate({result, tmp_fsm});
        } else {
          result = tmp_fsm;
          flag = true;
        }
      }
      if (!quotation_mode) {
        left_quote = i;
      } else {
        left_quote = -1;
      }
      quotation_mode = !quotation_mode;
      continue;
    }

    // Handle the character class like [a-zA-Z].
    if (regex[i] == '[' && !quotation_mode) {
      if (set_mode) {
        throw std::runtime_error("Invalid regex: nested set.");
      }
      left_middle_bracket = i;
      set_mode = true;
      continue;
    }
    if (regex[i] == ']' && set_mode) {
      if (left_middle_bracket == -1) {
        throw std::runtime_error("Invalid regex: unmatched ']'.");
      }
      if (bracket_stack.empty()) {
        FSMWithStartEnd tmp_fsm(regex.substr(left_middle_bracket, i - left_middle_bracket + 1));
        if (i < end - 1) {
          switch (regex[i + 1]) {
            case '+': {
              tmp_fsm = tmp_fsm.MakePlus();
              i = i + 1;
              break;
            }
            case '*': {
              tmp_fsm = tmp_fsm.MakeStar();
              i = i + 1;
              break;
            }
            case '?': {
              tmp_fsm = tmp_fsm.MakeOptional();
              i = i + 1;
              break;
            }
            default: {
              break;
            }
          }
        }
        if (not_mode) {
          tmp_fsm = tmp_fsm.Not();
          not_mode = false;
        }
        if (flag) {
          result = FSMWithStartEnd::Concatenate({result, tmp_fsm});
        } else {
          result = tmp_fsm;
          flag = true;
        }
      }
      set_mode = false;
      left_middle_bracket = -1;
      continue;
    }

    // Handle the small brackets like (a | b c*) | b.
    if (regex[i] == '(' && !quotation_mode && !set_mode) {
      bracket_stack.push(i);
      continue;
    }
    if (regex[i] == ')' && !quotation_mode && !set_mode) {
      if (bracket_stack.empty()) {
        throw std::runtime_error("Invalid regex: unmatched ')'.");
      }
      int left_bracket = bracket_stack.top();
      bracket_stack.pop();
      if (bracket_stack.empty()) {
        auto tmp_fsm = RegexToFSM(regex, left_bracket + 1, i);
        if (i < end - 1) {
          switch (regex[i + 1]) {
            case '+': {
              tmp_fsm = tmp_fsm.MakePlus();
              i = i + 1;
              break;
            }
            case '*': {
              tmp_fsm = tmp_fsm.MakeStar();
              i = i + 1;
              break;
            }
            case '?': {
              tmp_fsm = tmp_fsm.MakeOptional();
              i = i + 1;
              break;
            }
            default: {
              break;
            }
          }
        }
        if (not_mode) {
          tmp_fsm = tmp_fsm.Not();
          not_mode = false;
        }
        if (flag) {
          result = FSMWithStartEnd::Concatenate({result, tmp_fsm});
        } else {
          result = tmp_fsm;
          flag = true;
        }
      }
      continue;
    }

    // Handle the alternation operator '|'.
    if (regex[i] == '|' && !quotation_mode && !set_mode) {
      if (bracket_stack.empty()) {
        auto rhs = RegexToFSM(regex, i + 1, end);
        if (!flag) {
          throw(std::runtime_error("Invalid regex: unmatched '|'."));
        }
        result = FSMWithStartEnd::Union({result, rhs});
        return result;
      }
    }
  }
  if (quotation_mode || set_mode || !bracket_stack.empty() || not_mode) {
    throw std::runtime_error("Invalid regex: unmatched '\"' or '[' or '('. or '!'");
  }
  if (!flag) {
    throw std::runtime_error("Invalid regex: empty regex.");
  }
  return result;
}

FSMWithStartEnd::FSMWithStartEnd(const std::string& regex) {
  is_dfa = true;
  start = 0;
  auto& edges = fsm.edges;
  // Handle the regex string.
  if (regex[0] == '\"' && regex[regex.size() - 1] == '\"') {
    edges.push_back(std::vector<FSMEdge>());
    for (size_t i = 1; i < regex.size() - 1; i++) {
      if (regex[i] != '\\') {
        edges.back().emplace_back(regex[i], regex[i], edges.size());
        edges.push_back(std::vector<FSMEdge>());
        continue;
      }
      char escape_char = HandleEscapeInString(regex, i);
      edges.back().emplace_back(escape_char, escape_char, edges.size());
      edges.push_back(std::vector<FSMEdge>());
      i++;
    }
    ends.insert(edges.size() - 1);
    return;
  }
  // Handle the character class.
  if (regex[0] == '[' && regex[regex.size() - 1] == ']') {
    edges.push_back(std::vector<FSMEdge>());
    edges.push_back(std::vector<FSMEdge>());
    ends.insert(1);
    bool reverse = regex[1] == '^';
    for (size_t i = reverse ? 2 : 1; i < regex.size() - 1; i++) {
      if (regex[i] != '\\') {
        if (!(((i + 2) < regex.size() - 1) && regex[i + 1] == '-')) {
          // A single char.
          edges[0].emplace_back(regex[i], regex[i], 1);
          continue;
        }
        // Handle the char range.
        if (regex[i + 2] != '\\') {
          edges[0].emplace_back(regex[i], regex[i + 2], 1);
          i = i + 2;
          continue;
        }
        auto escaped_edges = HandleEscapeInClass(regex, i + 2);
        // Means it's not a range.
        if (escaped_edges.size() != 1 || escaped_edges[0].first != escaped_edges[0].second) {
          edges[0].emplace_back(regex[i], regex[i], 1);
          continue;
        }
        edges[0].emplace_back(regex[0], escaped_edges[0].first, 1);
        i = i + 3;
        continue;
      }
      auto escaped_edges = HandleEscapeInClass(regex, i);
      i = i + 1;
      if (escaped_edges.size() != 1 || escaped_edges[0].first != escaped_edges[0].second) {
        // It's a multi-match escape char.
        for (const auto& edge : escaped_edges) {
          edges[0].emplace_back(edge.first, edge.second, 1);
        }
        continue;
      }
      if (!(((i + 2) < regex.size() - 1) && regex[i + 1] == '-')) {
        edges[0].emplace_back(escaped_edges[0].first, escaped_edges[0].second, 1);
        continue;
      }
      if (regex[i + 2] != '\\') {
        edges[0].emplace_back(escaped_edges[0].first, regex[i + 2], 1);
        i = i + 2;
        continue;
      }
      auto rhs_escaped_edges = HandleEscapeInClass(regex, i + 2);
      if (rhs_escaped_edges.size() != 1 ||
          rhs_escaped_edges[0].first != rhs_escaped_edges[0].second) {
        edges[0].emplace_back(escaped_edges[0].first, escaped_edges[0].second, 1);
        continue;
      }
      edges[0].emplace_back(escaped_edges[0].first, rhs_escaped_edges[0].first, 1);
      i = i + 3;
      continue;
    }
    if (reverse) {
      bool has_edge[0x100];
      memset(has_edge, 0, sizeof(has_edge));
      for (const auto& edge : edges[0]) {
        for (int i = edge.min; i <= edge.max; i++) {
          has_edge[i] = true;
        }
      }
      edges[0].clear();
      int last = -1;
      for (int i = 0; i < 0x100; i++) {
        if (!has_edge[i]) {
          if (last == -1) {
            last = i;
          }
          continue;
        }
        if (last != -1) {
          edges[0].emplace_back(last, i - 1, 1);
          last = -1;
        }
      }
      if (last != -1) {
        edges[0].emplace_back(last, 0xFF, 1);
      }
    }
    return;
  }
  // TODO: The support for rules.
  throw std::runtime_error("Rules are not supported yet.");
}

FSMWithStartEnd FSMWithStartEnd::MinimizeDFA() const {
  FSMWithStartEnd now_fsm;

  // To perform the algorithm, we must make sure the FSM is
  // a DFA.
  if (!is_dfa) {
    now_fsm = TODFA();
  } else {
    now_fsm = Copy();
  }

  while (true) {
    int node_cnt = now_fsm.fsm.edges.size();
    bool mark_graph[node_cnt][node_cnt];
    std::vector<bool> is_end;
    // Initialize the mark graph.
    for (int i = 0; i < node_cnt; i++) {
      is_end.push_back(now_fsm.ends.find(i) != now_fsm.ends.end());
    }
    for (int i = 0; i < node_cnt; i++) {
      for (int j = 0; j < i; j++) {
        if (is_end[i] != is_end[j]) {
          mark_graph[i][j] = true;
        } else {
          mark_graph[i][j] = false;
        }
      }
    }
    // Check the equivalence of the states.
    bool changed = true;
    while (changed) {
      changed = false;
      for (int i = 0; i < node_cnt; i++) {
        for (int j = 0; j < i; j++) {
          if (mark_graph[i][j]) {
            continue;
          }
          auto transitions_i = now_fsm.fsm.edges[i];
          auto transitions_j = now_fsm.fsm.edges[j];
          // First, check all the actions in transtions_i, and compare them with
          // transition_j.
          for (const auto& transition_i : transitions_i) {
            if (mark_graph[i][j]) {
              break;
            }
            if (transition_i.IsRuleRef()) {
              bool is_blocked = true;
              int rule_id = transition_i.GetRefRuleId();
              for (const auto& transition_j : transitions_j) {
                if (transition_j.IsRuleRef()) {
                  if (transition_j.GetRefRuleId() == rule_id) {
                    is_blocked = false;
                    if (transition_i.target == transition_j.target) {
                      continue;
                    }
                    if (mark_graph[std::max(transition_i.target, transition_j.target)]
                                  [std::min(transition_i.target, transition_j.target)]) {
                      mark_graph[i][j] = true;
                      changed = true;
                    }
                    break;
                  }
                }
              }
              if (is_blocked) {
                mark_graph[i][j] = true;
                changed = true;
              }
              continue;
            }
            // Since it's a DFA.
            assert(!transition_i.IsEpsilon());
            int char_min = transition_i.min;
            int char_max = transition_i.max;
            bool is_blocked = true;
            for (const auto& transition_j : transitions_j) {
              if (transition_j.IsCharRange()) {
                // That means the intersection is not empty.
                if ((char_min >= transition_j.min && char_min <= transition_j.max) ||
                    (char_max >= transition_j.min && char_max <= transition_j.max)) {
                  is_blocked = false;
                  if (transition_i.target == transition_j.target) {
                    continue;
                  }
                  if (mark_graph[std::max(transition_i.target, transition_j.target)]
                                [std::min(transition_i.target, transition_j.target)]) {
                    mark_graph[i][j] = true;
                    changed = true;
                  }
                  break;
                }
              }
            }
            if (is_blocked) {
              mark_graph[i][j] = true;
              changed = true;
            }
          }
          if (mark_graph[i][j]) {
            continue;
          }
          // Now, do the same thing in reverse.
          for (const auto& transition_j : transitions_j) {
            if (mark_graph[i][j]) {
              break;
            }
            if (transition_j.IsRuleRef()) {
              bool is_blocked = true;
              int rule_id = transition_j.GetRefRuleId();
              for (const auto& transition_i : transitions_i) {
                if (transition_i.IsRuleRef()) {
                  if (transition_j.GetRefRuleId() == rule_id) {
                    is_blocked = false;
                    if (transition_i.target == transition_j.target) {
                      continue;
                    }
                    if (mark_graph[std::max(transition_i.target, transition_j.target)]
                                  [std::min(transition_i.target, transition_j.target)]) {
                      mark_graph[i][j] = true;
                      changed = true;
                    }
                    break;
                  }
                }
              }
              if (is_blocked) {
                mark_graph[i][j] = true;
                changed = true;
              }
              continue;
            }
            // Since it's a DFA.
            assert(!transition_j.IsEpsilon());
            int char_min = transition_j.min;
            int char_max = transition_j.max;
            bool is_blocked = true;
            for (const auto& transition_i : transitions_i) {
              if (transition_i.IsCharRange()) {
                // That means the intersection is not empty.
                if ((char_min >= transition_i.min && char_min <= transition_i.max) ||
                    (char_max >= transition_i.min && char_max <= transition_i.max)) {
                  is_blocked = false;
                  if (transition_i.target == transition_j.target) {
                    continue;
                  }
                  if (mark_graph[std::max(transition_i.target, transition_j.target)]
                                [std::min(transition_i.target, transition_j.target)]) {
                    mark_graph[i][j] = true;
                    changed = true;
                  }
                  break;
                }
              }
            }
            if (is_blocked) {
              mark_graph[i][j] = true;
              changed = true;
            }
          }
        }
      }
    }

    // Get the equivalence classes.
    std::vector<std::vector<int>> equivalence_classes;
    std::unordered_set<int> visited;
    for (int i = node_cnt - 1; i >= 0; i--) {
      if (visited.find(i) != visited.end()) {
        continue;
      }
      std::vector<int> equivalence_class;
      equivalence_class.push_back(i);
      for (int j = i - 1; j >= 0; j--) {
        if (!mark_graph[i][j]) {
          equivalence_class.push_back(j);
          visited.insert(j);
        }
      }
      if (equivalence_class.size() > 1) {
        equivalence_classes.push_back(equivalence_class);
      }
    }
    if (equivalence_classes.empty()) {
      break;
    }

    // Number the new nodes.
    std::unordered_map<int, int> old_to_new;
    for (size_t i = 0; i < equivalence_classes.size(); i++) {
      for (const auto& node_num : equivalence_classes[i]) {
        old_to_new[node_num] = i;
      }
    }
    int new_node_cnt = equivalence_classes.size();
    for (int i = 0; i < node_cnt; i++) {
      if (old_to_new.find(i) != old_to_new.end()) {
        continue;
      }
      old_to_new[i] = new_node_cnt++;
    }

    FSMWithStartEnd new_fsm;
    new_fsm.is_dfa = true;
    new_fsm.start = old_to_new[start];
    for (const auto& end : now_fsm.ends) {
      new_fsm.ends.insert(old_to_new[end]);
    }
    for (int i = 0; i < new_node_cnt; i++) {
      new_fsm.fsm.edges.push_back(std::vector<FSMEdge>());
    }
    std::unordered_set<int> been_built;
    for (size_t i = 0; i < now_fsm.fsm.edges.size(); i++) {
      if (been_built.find(old_to_new[i]) != been_built.end()) {
        continue;
      }
      been_built.insert(old_to_new[i]);
      for (const auto& edge : now_fsm.fsm.edges[i]) {
        new_fsm.fsm.edges[old_to_new[i]].emplace_back(edge.min, edge.max, old_to_new[edge.target]);
      }
    }
    now_fsm = new_fsm;
  }
  return now_fsm;
}

std::vector<std::pair<int, int>> HandleEscapeInClass(const std::string& regex, int start) {
  if (regex[start] != '\\') {
    throw std::runtime_error("Invalid regex: invalid escape character.");
  }
  if (int(regex.size()) <= start + 1) {
    throw std::runtime_error("Invalid regex: invalid escape character.");
  }
  std::vector<std::pair<int, int>> result;
  switch (regex[start + 1]) {
    case 'n': {
      result.emplace_back('\n', '\n');
      break;
    }
    case 't': {
      result.emplace_back('\t', '\t');
      break;
    }
    case 'r': {
      result.emplace_back('\r', '\r');
      break;
    }
    case '\\': {
      result.emplace_back('\\', '\\');
      break;
    }
    case ']': {
      result.emplace_back(']', ']');
      break;
    }
    case '0': {
      result.emplace_back('\0', '\0');
      break;
    }
    case '-': {
      result.emplace_back('-', '-');
      break;
    }
    case 'd': {
      result.emplace_back('0', '9');
      break;
    }
    case 'D': {
      result.emplace_back(0, '0' - 1);
      result.emplace_back('9' + 1, 0x00FF);
      break;
    }
    case 'w': {
      result.emplace_back('0', '9');
      result.emplace_back('a', 'z');
      result.emplace_back('A', 'Z');
      result.emplace_back('_', '_');
      break;
    }
    case 'W': {
      result.emplace_back(0, '0' - 1);
      result.emplace_back('9' + 1, 'A' - 1);
      result.emplace_back('Z' + 1, '_' - 1);
      result.emplace_back('_' + 1, 'a' - 1);
      result.emplace_back('z' + 1, 0x00FF);
      break;
    }
    case 's': {
      result.emplace_back(0, ' ');
      break;
    }
    case 'S': {
      result.emplace_back(' ' + 1, 0x00FF);
      break;
    }
    default: {
      throw std::runtime_error("Invalid regex: invalid escape character.");
    }
  }
  return result;
}

char HandleEscapeInString(const std::string& regex, int start) {
  if (regex[start] != '\\') {
    throw std::runtime_error("Invalid regex: invalid escape character.");
  }
  if (int(regex.size()) <= start + 1) {
    throw std::runtime_error("Invalid regex: invalid escape character.");
  }
  std::vector<std::pair<int, int>> result;
  switch (regex[start + 1]) {
    case 'n': {
      return '\n';
    }
    case 't': {
      return '\t';
    }
    case 'r': {
      return '\r';
    }
    case '\\': {
      return '\\';
    }
    case '\"': {
      return '\"';
    }
    case '0': {
      return '\0';
    }
    default: {
      throw std::runtime_error("Invalid regex: invalid escape character.");
    }
  }
}
}  // namespace xgrammar
