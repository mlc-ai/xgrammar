/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm.cc
 */
#include "fsm.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <list>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "support/logging.h"
#include "support/union_find_set.h"

namespace xgrammar {

std::vector<std::pair<int, int>> HandleEscapes(const std::string& regex, int start);

Result<std::pair<int, int>> CheckRepeat(const std::string& regex, int& start);

void CompactFSM::GetEpsilonClosure(
    std::unordered_set<int>* state_set, std::unordered_set<int>* result
) const {
  if (result == nullptr) {
    result = state_set;
  }
  std::queue<int> queue;
  for (const auto& state : *state_set) {
    queue.push(state);
  }
  while (!queue.empty()) {
    int current = queue.front();
    queue.pop();
    result->insert(current);
    for (const auto& edge : edges[current]) {
      if (edge.IsEpsilon()) {
        if (result->find(edge.target) != result->end()) {
          continue;
        }
        queue.push(edge.target);
      }
    }
  }
  return;
}

FSMEdge::FSMEdge(const short& _min, const short& _max, const int& target)
    : min(_min), max(_max), target(target) {
  if (IsCharRange() && min > max) {
    XGRAMMAR_DCHECK(false) << "Invalid FSMEdge: min > max. min=" << min << ", max=" << max;
  }
}

bool FSMEdge::IsEpsilon() const { return min == -1 && max == -1; }

bool FSMEdge::IsRuleRef() const { return min == -1 && max != -1; }

bool FSMEdge::IsCharRange() const { return min >= 0 && max >= 0; }

short FSMEdge::GetRefRuleId() const {
  if (IsRuleRef()) {
    return max;
  } else {
    XGRAMMAR_DCHECK(false) << "Invalid FSMEdge: not a rule reference. min=" << min
                           << ", max=" << max;
    return -1;
  }
}

void FSM::GetEpsilonClosure(std::unordered_set<int>* state_set, std::unordered_set<int>* result)
    const {
  if (result == nullptr) {
    result = state_set;
  }
  std::queue<int> queue;
  for (const auto& state : *state_set) {
    queue.push(state);
  }
  while (!queue.empty()) {
    int current = queue.front();
    queue.pop();
    result->insert(current);
    for (const auto& edge : edges[current]) {
      if (edge.IsEpsilon()) {
        if (result->find(edge.target) != result->end()) {
          continue;
        }
        queue.push(edge.target);
      }
    }
  }
  return;
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
    result = ToDFA();
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

void FSM::Advance(
    const std::vector<int>& from, int value, std::vector<int>* result, bool is_rule, bool is_closure
) const {
  result->clear();
  std::unordered_set<int> in_result;
  std::unordered_set<int> result_closure;
  std::unordered_set<int> start_set;

  for (const auto& state : from) {
    start_set.insert(state);
  }
  if (!is_closure) {
    GetEpsilonClosure(&start_set);
  }
  for (const auto& state : start_set) {
    const auto& edge_list = edges[state];
    for (const auto& edge : edge_list) {
      if (edge.IsEpsilon()) {
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
  for (const auto& state : in_result) {
    if (result_closure.find(state) != result_closure.end()) {
      continue;
    }
    std::unordered_set<int> closure;
    closure.insert(state);
    GetEpsilonClosure(&closure);
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
    const std::vector<int>& from, int value, std::vector<int>* result, bool is_rule, bool is_closure
) const {
  result->clear();
  std::unordered_set<int> in_result;
  std::unordered_set<int> result_closure;
  std::unordered_set<int> start_set;

  for (const auto& state : from) {
    start_set.insert(state);
  }
  if (!is_closure) {
    GetEpsilonClosure(&start_set);
  }
  for (const auto& state : start_set) {
    const auto& edge_list = edges[state];
    for (const auto& edge : edge_list) {
      if (edge.IsEpsilon()) {
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
  for (const auto& state : in_result) {
    if (result_closure.find(state) != result_closure.end()) {
      continue;
    }
    std::unordered_set<int> closure;
    closure.insert(state);
    GetEpsilonClosure(&closure);
    result_closure.insert(closure.begin(), closure.end());
  }
  for (const auto& state : result_closure) {
    result->push_back(state);
  }
  return;
}

FSMWithStartEnd FSMWithStartEnd::ToDFA() const {
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
  std::unordered_set<int> closure;
  closure.insert(start);
  fsm.GetEpsilonClosure(&closure);
  closures.push_back(closure);
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
              if (next_closure.find(edge.target) == next_closure.end()) {
                std::unordered_set<int> epsilon_closure;
                epsilon_closure.insert(edge.target);
                fsm.GetEpsilonClosure(&epsilon_closure);
                next_closure.insert(epsilon_closure.begin(), epsilon_closure.end());
              }
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
              if (next_closure.find(edge.target) == next_closure.end()) {
                std::unordered_set<int> epsilon_closure;
                epsilon_closure.insert(edge.target);
                fsm.GetEpsilonClosure(&epsilon_closure);
                next_closure.insert(epsilon_closure.begin(), epsilon_closure.end());
              }
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
          edges.back().emplace_back(0, 0xFF, edges.size());
        } else {
          edges.back().emplace_back(
              (unsigned char)(regex[i]), (unsigned char)(regex[i]), edges.size()
          );
        }
        edges.push_back(std::vector<FSMEdge>());
        continue;
      }
      std::vector<std::pair<int, int>> escape_vector = HandleEscapes(regex, i);
      for (const auto& escape : escape_vector) {
        edges.back().emplace_back(
            (unsigned char)(escape.first), (unsigned char)(escape.second), edges.size()
        );
      }
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
        auto escaped_edges = HandleEscapes(regex, i + 2);
        // Means it's not a range.
        if (escaped_edges.size() != 1 || escaped_edges[0].first != escaped_edges[0].second) {
          edges[0].emplace_back(regex[i], regex[i], 1);
          continue;
        }
        edges[0].emplace_back(regex[0], escaped_edges[0].first, 1);
        i = i + 3;
        continue;
      }
      auto escaped_edges = HandleEscapes(regex, i);
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
      auto rhs_escaped_edges = HandleEscapes(regex, i + 2);
      if (rhs_escaped_edges.size() != 1 ||
          rhs_escaped_edges[0].first != rhs_escaped_edges[0].second) {
        edges[0].emplace_back(escaped_edges[0].first, escaped_edges[0].second, 1);
        continue;
      }
      edges[0].emplace_back(escaped_edges[0].first, rhs_escaped_edges[0].first, 1);
      i = i + 3;
      continue;
    }
    bool has_edge[0x100];
    memset(has_edge, 0, sizeof(has_edge));
    for (const auto& edge : edges[0]) {
      for (int i = edge.min; i <= edge.max; i++) {
        has_edge[i] = true;
      }
    }
    edges[0].clear();

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
          edges[0].emplace_back(last, i - 1, 1);
          last = -1;
        }
      }
      if (last != -1) {
        edges[0].emplace_back(last, 0xFF, 1);
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
  XGRAMMAR_LOG(WARNING) << "rule is not supported yet.";
}

FSMWithStartEnd FSMWithStartEnd::MinimizeDFA() const {
  FSMWithStartEnd now_fsm;

  // To perform the algorithm, we must make sure the FSM is
  // a DFA.
  if (!is_dfa) {
    now_fsm = ToDFA();
  } else {
    now_fsm = Copy();
  }
  // Initialize the set.
  std::list<std::unordered_set<int>> blocks;
  std::list<std::unordered_set<int>> queue;
  std::unordered_set<int> not_end;
  for (size_t i = 0; i < now_fsm.fsm.edges.size(); i++) {
    if (now_fsm.ends.find(i) == now_fsm.ends.end()) {
      not_end.insert(i);
    }
  }
  queue.push_back(not_end);
  queue.push_back(now_fsm.ends);
  blocks.push_back(now_fsm.ends);
  blocks.push_back(not_end);
  std::set<int> interval_ends;
  std::unordered_set<std::pair<int, int>> intervals;
  std::unordered_set<int> rules;
  std::unordered_map<int, std::unordered_set<int>> previous_mapping;
  for (size_t i = 0; i < now_fsm.fsm.edges.size(); i++) {
    const auto& edges = now_fsm.fsm.edges[i];
    for (const auto& edge : edges) {
      if (previous_mapping.find(edge.target) == previous_mapping.end()) {
        previous_mapping[edge.target] = std::unordered_set<int>();
      }
      previous_mapping[edge.target].insert(i);
      if (edge.IsCharRange()) {
        interval_ends.insert(edge.min);
        interval_ends.insert(edge.max + 1);
        continue;
      }
      if (edge.IsRuleRef()) {
        rules.insert(edge.GetRefRuleId());
      }
    }
  }
  for (auto it = interval_ends.begin(); it != interval_ends.end(); ++it) {
    auto next_it = std::next(it);
    if (next_it != interval_ends.end()) {
      intervals.insert(std::make_pair(*it, *next_it - 1));
    }
  }

  while (!queue.empty()) {
    // Initial the alphabet.
    auto block_x = *queue.begin();
    queue.erase(queue.begin());
    std::unordered_set<int> prev_nodes;
    for (const auto& node : block_x) {
      if (previous_mapping.find(node) != previous_mapping.end()) {
        prev_nodes.insert(previous_mapping[node].begin(), previous_mapping[node].end());
      }
    }
    // Check the intervals.
    std::list<std::unordered_set<int>> blocks_copy = blocks;
    for (const auto& interval : intervals) {
      std::unordered_set<int> from_block;
      for (const auto& node : prev_nodes) {
        const auto& edges = now_fsm.fsm.edges[node];
        for (const auto& edge : edges) {
          if (block_x.find(edge.target) == block_x.end()) {
            continue;
          }
          if (edge.IsCharRange()) {
            if (interval.first >= edge.min && interval.second <= edge.max) {
              from_block.insert(node);
            }
          }
        }
      }
      for (const auto& block : blocks_copy) {
        std::unordered_set<int> intersection;
        for (const auto& prev : from_block) {
          if (block.find(prev) != block.end()) {
            intersection.insert(prev);
          }
        }
        // The intersection is empty, or the intersection == block.
        if (intersection.empty() || intersection.size() == block.size()) {
          continue;
        }
        std::unordered_set<int> difference;
        for (const auto& node : block) {
          if (intersection.find(node) == intersection.end()) {
            difference.insert(node);
          }
        }
        blocks.remove(block);
        blocks.remove(intersection);
        blocks.remove(difference);
        blocks.push_back(intersection);
        blocks.push_back(difference);
        bool found = false;
        for (auto iter = queue.begin(); iter != queue.end(); ++iter) {
          if (*iter == block) {
            found = true;
            break;
          }
        }
        if (found) {
          queue.remove(block);
          queue.push_back(intersection);
          queue.push_back(difference);
        } else {
          queue.push_back(intersection.size() < difference.size() ? intersection : difference);
        }
      }
    }
    // Do the same thing for the rules.
    blocks_copy = blocks;
    for (const auto& rule : rules) {
      std::unordered_set<int> from_block;
      for (const auto& node : prev_nodes) {
        const auto& edges = now_fsm.fsm.edges[node];
        for (const auto& edge : edges) {
          if (block_x.find(edge.target) == block_x.end()) {
            continue;
          }
          if (edge.IsRuleRef()) {
            if (rule == edge.GetRefRuleId()) {
              from_block.insert(node);
            }
          }
        }
      }
      for (const auto& block : blocks_copy) {
        std::unordered_set<int> intersection;
        for (const auto& prev : from_block) {
          if (block.find(prev) != block.end()) {
            intersection.insert(prev);
          }
        }
        // The intersection is empty, or the intersection == block.
        if (intersection.empty() || intersection.size() == block.size()) {
          continue;
        }
        std::unordered_set<int> difference;
        for (const auto& node : from_block) {
          if (intersection.find(node) == intersection.end()) {
            difference.insert(node);
          }
        }
        blocks.remove(block);
        blocks.remove(intersection);
        blocks.remove(difference);
        blocks.push_back(intersection);
        blocks.push_back(difference);
        bool found = false;
        for (auto iter = queue.begin(); iter != queue.end(); ++iter) {
          if (*iter == block) {
            found = true;
            break;
          }
        }
        if (found) {
          queue.remove(block);
          queue.push_back(intersection);
          queue.push_back(difference);
        } else {
          queue.push_back(intersection.size() < difference.size() ? intersection : difference);
        }
      }
    }
  }

  std::unordered_map<int, int> old_to_new;
  int cnt = 0;
  for (const auto& block : blocks) {
    for (const auto& node : block) {
      old_to_new[node] = cnt;
    }
    cnt++;
  }
  FSMWithStartEnd new_fsm;
  new_fsm.is_dfa = true;
  new_fsm.start = old_to_new[now_fsm.start];
  for (const auto& end : now_fsm.ends) {
    new_fsm.ends.insert(old_to_new[end]);
  }
  for (int i = 0; i < cnt; i++) {
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
  return new_fsm;
}

std::vector<std::pair<int, int>> HandleEscapes(const std::string& regex, int start) {
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

Result<FSMWithStartEnd> FSMWithStartEnd::Intersect(
    const FSMWithStartEnd& lhs, const FSMWithStartEnd& rhs, const int& num_of_nodes_limited
) {
  if (!lhs.IsLeaf() || !rhs.IsLeaf()) {
    return Result<FSMWithStartEnd>::Err(std::make_shared<Error>("Intersect only support leaf fsm!")
    );
  }
  auto lhs_dfa = lhs.ToDFA();
  auto rhs_dfa = rhs.ToDFA();
  std::unordered_set<int> rules_lhs;
  std::unordered_set<int> rules;
  std::set<int> interval_ends;
  std::vector<std::pair<int, int>> intervals;
  // This part is to build the equivalent alphabet.
  for (const auto& edges : lhs.fsm.edges) {
    for (const auto& edge : edges) {
      if (edge.IsRuleRef()) {
        rules_lhs.insert(edge.GetRefRuleId());
      }
    }
  }
  for (const auto& edges : rhs.fsm.edges) {
    for (const auto& edge : edges) {
      if (edge.IsRuleRef()) {
        if (rules_lhs.find(edge.GetRefRuleId()) != rules_lhs.end()) {
          rules.insert(edge.GetRefRuleId());
        }
      }
    }
  }
  for (const auto& edges : lhs_dfa.fsm.edges) {
    for (const auto& edge : edges) {
      if (edge.IsCharRange()) {
        interval_ends.insert(edge.min);
        interval_ends.insert(edge.max + 1);
      }
    }
  }
  for (const auto& edges : rhs_dfa.fsm.edges) {
    for (const auto& edge : edges) {
      if (edge.IsCharRange()) {
        interval_ends.insert(edge.min);
        interval_ends.insert(edge.max + 1);
      }
    }
  }
  for (auto it = interval_ends.begin(); it != interval_ends.end(); ++it) {
    auto next_it = std::next(it);
    if (next_it != interval_ends.end()) {
      intervals.emplace_back(*it, *next_it - 1);
    }
  }
  FSMWithStartEnd result;
  result.is_dfa = true;
  result.start = 0;
  std::unordered_map<std::pair<int, int>, int> state_map;
  std::unordered_set<std::pair<int, int>> visited;
  std::queue<std::pair<int, int>> queue;
  queue.push({lhs.start, rhs.start});
  result.fsm.edges.push_back(std::vector<FSMEdge>());
  state_map[{lhs.start, rhs.start}] = 0;
  while (!queue.empty()) {
    if (int(state_map.size()) > num_of_nodes_limited) {
      return Result<FSMWithStartEnd>::Err(
          std::make_shared<Error>("Intersection have too many nodes!")
      );
    }
    auto state = queue.front();
    queue.pop();
    if (visited.find(state) != visited.end()) {
      continue;
    }
    visited.insert(state);
    int lhs_state = state.first;
    int rhs_state = state.second;
    for (const auto& interval : intervals) {
      for (const auto& lhs_edge : lhs_dfa.fsm.edges[lhs_state]) {
        if (!lhs_edge.IsCharRange()) {
          continue;
        }
        if (lhs_edge.min > interval.first || lhs_edge.max < interval.second) {
          continue;
        }
        for (const auto& rhs_edge : rhs_dfa.fsm.edges[rhs_state]) {
          if (!rhs_edge.IsCharRange()) {
            continue;
          }
          if (rhs_edge.min > interval.first || rhs_edge.max < interval.second) {
            continue;
          }
          auto next_state = std::make_pair(lhs_edge.target, rhs_edge.target);
          if (state_map.find(next_state) == state_map.end()) {
            state_map[next_state] = state_map.size();
            queue.push(next_state);
            result.fsm.edges.push_back(std::vector<FSMEdge>());
          }
          result.fsm.edges[state_map[{lhs_state, rhs_state}]].emplace_back(
              interval.first, interval.second, state_map[next_state]
          );
          break;
        }
      }
    }
    for (const auto& rule : rules) {
      for (const auto& lhs_edge : lhs_dfa.fsm.edges[lhs_state]) {
        if (!lhs_edge.IsRuleRef()) {
          continue;
        }
        if (lhs_edge.GetRefRuleId() != rule) {
          continue;
        }
        for (const auto& rhs_edge : rhs_dfa.fsm.edges[rhs_state]) {
          if (!rhs_edge.IsRuleRef()) {
            continue;
          }
          if (rhs_edge.GetRefRuleId() != rule) {
            continue;
          }
          auto next_state = std::make_pair(lhs_edge.target, rhs_edge.target);
          if (state_map.find(next_state) == state_map.end()) {
            state_map[next_state] = state_map.size();
            queue.push(next_state);
            result.fsm.edges.push_back(std::vector<FSMEdge>());
          }
          result.fsm.edges[state_map[{lhs_state, rhs_state}]].emplace_back(
              -1, rule, state_map[next_state]
          );
          break;
        }
      }
    }
  }
  for (const auto& state : visited) {
    if (lhs.ends.find(state.first) != lhs.ends.end() &&
        rhs.ends.find(state.second) != rhs.ends.end()) {
      result.ends.insert(state_map[state]);
    }
  }
  return Result<FSMWithStartEnd>::Ok(result);
}

bool FSMWithStartEnd::Check(const std::string& str) const {
  std::unordered_set<int> start_states_set;
  start_states_set.insert(start);
  fsm.GetEpsilonClosure(&start_states_set);
  std::vector<int> from_states;
  std::vector<int> result_states;
  for (const auto& start_state : start_states_set) {
    from_states.push_back(start_state);
  }
  for (const auto& character : str) {
    result_states.clear();
    fsm.Advance(from_states, (unsigned char)(character), &result_states, false);
    from_states = result_states;
  }
  for (const auto& state : from_states) {
    if (ends.find(state) != ends.end()) {
      return true;
    }
  }
  return false;
}

bool CompactFSMWithStartEnd::Check(const std::string& str) const {
  std::unordered_set<int> start_states_set;
  start_states_set.insert(start);
  fsm.GetEpsilonClosure(&start_states_set);
  std::vector<int> from_states;
  std::vector<int> result_states;
  for (const auto& start_state : start_states_set) {
    from_states.push_back(start_state);
  }
  for (const auto& character : str) {
    result_states.clear();
    fsm.Advance(from_states, (unsigned char)(character), &result_states, false);
    from_states = result_states;
  }
  for (const auto& state : from_states) {
    if (ends.find(state) != ends.end()) {
      return true;
    }
  }
  return false;
}

bool FSMWithStartEnd::IsDFA() {
  if (is_dfa) {
    return true;
  }

  std::set<int> interval_ends;
  std::unordered_set<int> rules;
  for (const auto& edges : fsm.edges) {
    for (const auto& edge : edges) {
      if (edge.IsEpsilon()) {
        return false;
      }
      if (edge.IsCharRange()) {
        interval_ends.insert(edge.min);
        interval_ends.insert(edge.max + 1);
        continue;
      }
      if (edge.IsRuleRef()) {
        rules.insert(edge.GetRefRuleId());
        continue;
      }
    }
  }
  using Interval = std::pair<int, int>;
  std::unordered_set<Interval> intervals;
  for (auto it = interval_ends.begin(); it != interval_ends.end(); ++it) {
    auto next_it = std::next(it);
    if (next_it != interval_ends.end()) {
      intervals.emplace(*it, *next_it - 1);
    }
  }
  for (const auto& edges : fsm.edges) {
    for (const auto& rule : rules) {
      bool find = false;
      for (const auto& edge : edges) {
        if (edge.IsRuleRef()) {
          if (edge.GetRefRuleId() == rule) {
            if (find) {
              return false;
            }
            find = true;
          }
        }
      }
    }
    for (const auto& interval : intervals) {
      bool find = false;
      for (const auto& edge : edges) {
        if (edge.IsCharRange()) {
          if (edge.min > interval.first || edge.max < interval.second) {
            continue;
          }
          if (find) {
            return false;
          }
          find = true;
        }
      }
    }
  }
  is_dfa = true;
  return true;
}

bool FSMWithStartEnd::IsLeaf() const {
  for (const auto& edges : fsm.edges) {
    for (const auto& edge : edges) {
      if (edge.IsRuleRef()) {
        return false;
      }
    }
  }
  return true;
}

void FSMWithStartEnd::SimplifyEpsilon() {
  if (IsDFA()) {
    return;
  }
  UnionFindSet<int> union_find_set;
  std::unordered_map<int, std::unordered_set<int>> previous_nodes;
  std::unordered_set<int> has_epsilon;

  // Initialize the previous nodes, and find all the nodes that have
  // epsilon edges.
  for (size_t i = 0; i < fsm.edges.size(); i++) {
    const auto& edges = fsm.edges[i];
    for (const auto& edge : edges) {
      if (previous_nodes.find(edge.target) == previous_nodes.end()) {
        previous_nodes[edge.target] = std::unordered_set<int>();
      }
      previous_nodes[edge.target].insert(i);
      if (edge.IsEpsilon()) {
        if (edges.size() != 1) {
          has_epsilon.insert(i);
        } else {
          // a -- epsilon --> b, and a doesn't have other outward edges.
          union_find_set.Make(i);
          union_find_set.Make(edge.target);
          union_find_set.Union(i, edge.target);
        }
      }
    }
  }

  // a --> epsilon --> b, and b doesn't have other inward edges.
  for (const auto& node : has_epsilon) {
    const auto& edges = fsm.edges[node];
    for (const auto& edge : edges) {
      if (!edge.IsEpsilon()) {
        continue;
      }
      // Have other inward nodes.
      if (previous_nodes[edge.target].size() != 1) {
        continue;
      }
      bool has_other_edge = false;
      for (const auto& second_edge : edges) {
        if (second_edge.IsEpsilon()) {
          continue;
        }
        if (second_edge.target == edge.target) {
          has_other_edge = true;
          break;
        }
      }
      // The node can be merged.
      if (!has_other_edge) {
        union_find_set.Make(node);
        union_find_set.Make(edge.target);
        union_find_set.Union(node, edge.target);
      }
    }
  }

  // Merge the nodes.
  auto eq_classes = union_find_set.GetAllSets();
  if (eq_classes.empty()) {
    return;
  }
  std::unordered_map<int, int> new_to_old;
  for (size_t i = 0; i < eq_classes.size(); i++) {
    for (const auto& node : eq_classes[i]) {
      new_to_old[node] = i;
    }
  }
  int cnt = eq_classes.size();
  for (size_t i = 0; i < fsm.edges.size(); i++) {
    if (new_to_old.find(i) == new_to_old.end()) {
      new_to_old[i] = cnt;
      cnt++;
    }
  }
  RebuildFSM(new_to_old, cnt);
  return;
}

void FSMWithStartEnd::SimplifyTransition() {
  bool changed = true;
  UnionFindSet<int> union_find_set;
  while (changed) {
    union_find_set.Clear();
    std::unordered_map<int, std::unordered_set<int>> previous_nodes;
    // Initialize the previous nodes.
    for (size_t i = 0; i < fsm.edges.size(); i++) {
      const auto& edges = fsm.edges[i];
      for (const auto& edge : edges) {
        if (previous_nodes.find(edge.target) == previous_nodes.end()) {
          previous_nodes[edge.target] = std::unordered_set<int>();
        }
        previous_nodes[edge.target].insert(i);
      }
    }
    // Case 1: Like ab | ac | ad, then they can be merged into a(b | c | d).
    changed = false;
    bool change_case1 = false;
    for (const auto& edges : fsm.edges) {
      for (size_t i = 0; i < edges.size(); i++) {
        for (size_t j = i + 1; j < edges.size(); j++) {
          if (IsEndNode(edges[i].target) != IsEndNode(edges[j].target)) {
            continue;
          }
          if (edges[i].target == edges[j].target) {
            continue;
          }
          if (edges[i].max != edges[j].max || edges[i].min != edges[j].min) {
            continue;
          }
          if (previous_nodes[edges[i].target].size() != 1 ||
              previous_nodes[edges[j].target].size() != 1) {
            continue;
          }
          union_find_set.Make(edges[i].target);
          union_find_set.Make(edges[j].target);
          union_find_set.Union(edges[i].target, edges[j].target);
          change_case1 = true;
        }
      }
    }
    if (change_case1) {
      auto eq_classes = union_find_set.GetAllSets();
      std::unordered_map<int, int> old_to_new;
      for (size_t i = 0; i < eq_classes.size(); i++) {
        for (const auto& node : eq_classes[i]) {
          old_to_new[node] = i;
        }
      }
      int cnt = eq_classes.size();
      for (size_t i = 0; i < fsm.edges.size(); i++) {
        if (old_to_new.find(i) == old_to_new.end()) {
          old_to_new[i] = cnt;
          cnt++;
        }
      }
      RebuildFSM(old_to_new, cnt);
    }
    union_find_set.Clear();
    // Case 2: Like ba | ca | da, then they can be merged into (b | c | d)a.
    bool change_case2 = false;
    for (size_t i = 0; i < fsm.edges.size(); i++) {
      for (size_t j = i + 1; j < fsm.edges.size(); j++) {
        bool equivalent = true;
        for (const auto& edge_i : fsm.edges[i]) {
          bool same = false;
          for (const auto& edge_j : fsm.edges[j]) {
            if (edge_i.min == edge_j.min && edge_i.max == edge_j.max &&
                edge_i.target == edge_j.target) {
              same = true;
              break;
            }
          }
          if (!same) {
            equivalent = false;
            break;
          }
        }
        if (!equivalent) {
          continue;
        }
        for (const auto& edge_j : fsm.edges[j]) {
          bool same = false;
          for (const auto& edge_i : fsm.edges[i]) {
            if (edge_i.min == edge_j.min && edge_i.max == edge_j.max &&
                edge_i.target == edge_j.target) {
              same = true;
              break;
            }
          }
          if (!same) {
            equivalent = false;
            break;
          }
        }
        if (equivalent) {
          union_find_set.Make(i);
          union_find_set.Make(j);
          union_find_set.Union(i, j);
          change_case2 = true;
        }
      }
    }
    if (change_case2) {
      auto eq_classes = union_find_set.GetAllSets();
      std::unordered_map<int, int> old_to_new;
      for (size_t i = 0; i < eq_classes.size(); i++) {
        for (const auto& node : eq_classes[i]) {
          old_to_new[node] = i;
        }
      }
      int cnt = eq_classes.size();
      for (size_t i = 0; i < fsm.edges.size(); i++) {
        if (old_to_new.find(i) == old_to_new.end()) {
          old_to_new[i] = cnt;
          cnt++;
        }
      }
      RebuildFSM(old_to_new, cnt);
    }
    changed = change_case1 || change_case2;
  }
  return;
}

void FSMWithStartEnd::RebuildFSM(
    std::unordered_map<int, int>& old_to_new, const int& new_node_cnt
) {
  start = old_to_new[start];
  decltype(ends) new_ends;
  for (const auto& end : ends) {
    new_ends.insert(old_to_new[end]);
  }
  ends = new_ends;
  decltype(fsm.edges) new_edges;
  struct Compare {
    bool operator()(const FSMEdge& lhs, const FSMEdge& rhs) const {
      if (lhs.min != rhs.min) {
        return lhs.min < rhs.min;
      }
      if (lhs.max != rhs.max) {
        return lhs.max < rhs.max;
      }
      return lhs.target < rhs.target;
    }
  };
  std::vector<std::set<FSMEdge, Compare>> new_edges_set;
  new_edges_set.resize(new_node_cnt);
  new_edges.resize(new_node_cnt);
  for (size_t i = 0; i < fsm.edges.size(); i++) {
    const auto& edges = fsm.edges[i];
    for (const auto& edge : edges) {
      if (edge.IsEpsilon() && old_to_new[i] == old_to_new[edge.target]) {
        continue;
      }
      new_edges_set[old_to_new[i]].insert({edge.min, edge.max, old_to_new[edge.target]});
    }
  }
  for (size_t i = 0; i < new_edges_set.size(); i++) {
    for (const auto& edge : new_edges_set[i]) {
      new_edges[i].emplace_back(edge.min, edge.max, edge.target);
    }
  }
  fsm.edges = new_edges;
  return;
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Leaf& node) const {
  FSMWithStartEnd result(node.regex);
  return Result<FSMWithStartEnd>::Ok(result);
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Union& node) const {
  std::vector<FSMWithStartEnd> fsm_list;
  for (const auto& child : node.nodes) {
    auto visited = std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, child);
    if (visited.IsErr()) {
      return visited;
    }
    fsm_list.push_back(visited.Unwrap());
  }
  if (fsm_list.size() <= 1) {
    return Result<FSMWithStartEnd>::Err(std::make_shared<Error>("Invalid union"));
  }
  return Result<FSMWithStartEnd>::Ok(FSMWithStartEnd::Union(fsm_list));
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Symbol& node) const {
  if (node.node.size() != 1) {
    return Result<FSMWithStartEnd>::Err(std::make_shared<Error>("Invalid symbol"));
  }
  Result<FSMWithStartEnd> child =
      std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, node.node[0]);
  if (child.IsErr()) {
    return child;
  }
  FSMWithStartEnd result;
  switch (node.symbol) {
    case RegexIR::RegexSymbol::plus: {
      result = child.Unwrap().MakePlus();
      break;
    }
    case RegexIR::RegexSymbol::star: {
      result = child.Unwrap().MakeStar();
      break;
    }
    case RegexIR::RegexSymbol::optional: {
      result = child.Unwrap().MakeOptional();
      break;
    }
  }
  return Result<FSMWithStartEnd>::Ok(result);
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Bracket& node) const {
  std::vector<FSMWithStartEnd> fsm_list;
  for (const auto& child : node.nodes) {
    auto visited = std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, child);
    if (visited.IsErr()) {
      return visited;
    }
    fsm_list.push_back(visited.Unwrap());
  }
  if (fsm_list.empty()) {
    return Result<FSMWithStartEnd>::Err(std::make_shared<Error>("Invalid bracket"));
  }
  return Result<FSMWithStartEnd>::Ok(FSMWithStartEnd::Concatenate(fsm_list));
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Repeat& node) const {
  if (node.nodes.size() != 1) {
    return Result<FSMWithStartEnd>::Err(std::make_shared<Error>("Invalid repeat"));
  }
  Result<FSMWithStartEnd> child =
      std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, node.nodes[0]);
  if (child.IsErr()) {
    return child;
  }
  FSMWithStartEnd result;
  result = child.Unwrap();
  std::unordered_set<int> new_ends;
  if (node.lower_bound == 1) {
    for (const auto& end : result.ends) {
      new_ends.insert(end);
    }
  }
  // Handling {n,}
  if (node.upper_bound == RegexIR::REPEATNOUPPERBOUND) {
    for (int i = 2; i < node.lower_bound; i++) {
      result = FSMWithStartEnd::Concatenate(std::vector<FSMWithStartEnd>{result, child.Unwrap()});
    }
    int one_of_end_node = *result.ends.begin();
    result = FSMWithStartEnd::Concatenate(std::vector<FSMWithStartEnd>{result, child.Unwrap()});
    for (const auto& end : result.ends) {
      result.fsm.edges[end].emplace_back(-1, -1, one_of_end_node);
    }
    return Result<FSMWithStartEnd>::Ok(result);
  }
  // Handling {n, m} or {n}
  for (int i = 2; i <= node.upper_bound; i++) {
    result = FSMWithStartEnd::Concatenate(std::vector<FSMWithStartEnd>{result, child.Unwrap()});
    if (i >= node.lower_bound) {
      for (const auto& end : result.ends) {
        new_ends.insert(end);
      }
    }
  }
  result.ends = new_ends;
  return Result<FSMWithStartEnd>::Ok(result);
}

Result<FSMWithStartEnd> RegexToFSM(const std::string& regex) {
  RegexIR ir;
  using IRNode = std::variant<RegexIR::Node, char>;
  // We use a stack to store the nodes.
  std::stack<IRNode> stack;
  int left_middle_bracket = -1;
  for (size_t i = 0; i < regex.size(); i++) {
    if (i == 0 && regex[i] == '^') {
      continue;
    }
    if (i == regex.size() - 1 && regex[i] == '$') {
      continue;
    }
    // Handle The class.
    if (regex[i] == '[') {
      if (left_middle_bracket != -1) {
        return Result<FSMWithStartEnd>::Err(std::make_shared<Error>("Nested middle bracket!"));
      }
      left_middle_bracket = i;
      continue;
    }
    if (regex[i] == ']') {
      if (left_middle_bracket == -1) {
        return Result<FSMWithStartEnd>::Err(std::make_shared<Error>("Invalid middle bracket!"));
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
        return Result<FSMWithStartEnd>::Err(
            std::make_shared<Error>("Invalid regex: no node before operator!")
        );
      }
      auto node = stack.top();
      if (std::holds_alternative<char>(node)) {
        return Result<FSMWithStartEnd>::Err(
            std::make_shared<Error>("Invalid regex: no node before operator!")
        );
      }
      stack.pop();
      auto child = std::get<RegexIR::Node>(node);
      RegexIR::Symbol symbol;
      symbol.node.push_back(child);
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
      if (i < regex.size() - 2 && regex[i] == '(' && regex[i + 1] == '?' && regex[i + 2] == ':') {
        i += 2;
        continue;
      }
      if (i < regex.size() - 2 && regex[i] == '(' && regex[i + 1] == '?' &&
          (regex[i + 2] == '!' || regex[i + 2] == '=')) {
        i += 2;
        // TODO(Linzhang Li): Handling the lookahead.
        continue;
      }
      continue;
    }
    if (regex[i] == ')') {
      std::stack<IRNode> nodes;
      bool paired = false;
      bool unioned = false;
      while ((!stack.empty()) && (!paired)) {
        auto node = stack.top();
        stack.pop();
        if (std::holds_alternative<char>(node)) {
          char c = std::get<char>(node);
          if (c == '(') {
            paired = true;
            break;
          }
          if (c == '|') {
            unioned = true;
          }
          nodes.push(node);
        } else {
          nodes.push(node);
        }
      }
      if (!paired) {
        return Result<FSMWithStartEnd>::Err(
            std::make_shared<Error>("Invalid regex: no paired bracket!" + std::to_string(__LINE__))
        );
      }
      if (nodes.empty()) {
        continue;
      }
      if (!unioned) {
        RegexIR::Bracket bracket;
        while (!nodes.empty()) {
          auto node = nodes.top();
          nodes.pop();
          auto child = std::get<RegexIR::Node>(node);
          bracket.nodes.push_back(child);
        }
        stack.push(bracket);
      } else {
        RegexIR::Union union_node;
        RegexIR::Bracket bracket;
        while (!nodes.empty()) {
          auto node = nodes.top();
          nodes.pop();
          if (std::holds_alternative<char>(node)) {
            char c = std::get<char>(node);
            if (c == '|') {
              union_node.nodes.push_back(bracket);
              bracket.nodes.clear();
              continue;
            }
            return Result<FSMWithStartEnd>::Err(std::make_shared<Error>(
                "Invalid regex: no paired bracket!" + std::to_string(__LINE__)
            ));
          }
          if (std::holds_alternative<RegexIR::Node>(node)) {
            auto child = std::get<RegexIR::Node>(node);
            bracket.nodes.push_back(child);
            continue;
          }
          return Result<FSMWithStartEnd>::Err(std::make_shared<Error>(
              "Invalid regex: no paired bracket!" + std::to_string(__LINE__)
          ));
        }
        union_node.nodes.push_back(bracket);
        stack.push(union_node);
      }
      continue;
    }
    if (regex[i] == '{') {
      if (stack.empty()) {
        return Result<FSMWithStartEnd>::Err(
            std::make_shared<Error>("Invalid regex: no node before repeat!")
        );
      }
      auto node = stack.top();
      if (std::holds_alternative<char>(node)) {
        return Result<FSMWithStartEnd>::Err(
            std::make_shared<Error>("Invalid regex: no node before repeat!")
        );
      }
      stack.pop();
      auto bounds_result = CheckRepeat(regex, i);
      if (bounds_result.IsErr()) {
        return Result<FSMWithStartEnd>::Err(bounds_result.UnwrapErr());
      }
      auto child = std::get<RegexIR::Node>(node);
      RegexIR::Repeat repeat;
      repeat.lower_bound = bounds_result.Unwrap().first;
      repeat.upper_bound = bounds_result.Unwrap().second;
      repeat.nodes.push_back(child);
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
  std::vector<RegexIR::Node> res_nodes;
  std::vector<decltype(res_nodes)> union_node_list;
  bool unioned = false;
  while (!stack.empty()) {
    if (std::holds_alternative<char>(stack.top())) {
      char c = std::get<char>(stack.top());
      if (c == '|') {
        union_node_list.push_back(res_nodes);
        res_nodes.clear();
        unioned = true;
        stack.pop();
        continue;
      }
      return Result<FSMWithStartEnd>::Err(std::make_shared<Error>("Invalid regex: no paired!"));
    }
    auto node = stack.top();
    stack.pop();
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
  return ir.Build();
}

Result<FSMWithStartEnd> RegexIR::Build() const {
  if (nodes.empty()) {
    FSMWithStartEnd result;
    result.is_dfa = false;
    result.start = 0;
    result.fsm.edges.push_back(std::vector<FSMEdge>());
    return Result<FSMWithStartEnd>::Ok(result);
  }
  std::vector<FSMWithStartEnd> fsm_list;
  for (const auto& node : nodes) {
    auto visited = std::visit([&](auto&& arg) { return visit(arg); }, node);
    if (visited.IsErr()) {
      return visited;
    }
    fsm_list.push_back(visited.Unwrap());
  }
  return Result<FSMWithStartEnd>::Ok(FSMWithStartEnd::Concatenate(fsm_list));
}

Result<std::pair<int, int>> CheckRepeat(const std::string& regex, size_t& start) {
  if (regex[start] != '{') {
    return Result<std::pair<int, int>>::Err(std::make_shared<Error>("Invalid regex: invalid repeat!"
    ));
  }
  start++;
  int lower_bound = 0;
  int upper_bound = 0;
  while (start < regex.size() && regex[start] == ' ') {
    start++;
  }
  while (start < regex.size() && regex[start] >= '0' && regex[start] <= '9') {
    lower_bound = lower_bound * 10 + (regex[start] - '0');
    start++;
  }
  while (start < regex.size() && regex[start] == ' ') {
    start++;
  }
  if (start >= regex.size() || (regex[start] != ',' && regex[start] != '}')) {
    return Result<std::pair<int, int>>::Err(std::make_shared<Error>("Invalid regex: invalid repeat!"
    ));
  }
  if (regex[start] == '}') {
    upper_bound = lower_bound;
  } else {
    start++;
    while (start < regex.size() && regex[start] == ' ') {
      start++;
    }
    if (start < regex.size() && regex[start] == '}') {
      upper_bound = RegexIR::REPEATNOUPPERBOUND;
      return Result<std::pair<int, int>>::Ok(std::make_pair(lower_bound, upper_bound));
    }
    while (start < regex.size() && regex[start] >= '0' && regex[start] <= '9') {
      upper_bound = upper_bound * 10 + (regex[start] - '0');
      start++;
    }
    while (start < regex.size() && regex[start] == ' ') {
      start++;
    }
    if (start >= regex.size() || regex[start] != '}') {
      return Result<std::pair<int, int>>::Err(
          std::make_shared<Error>("Invalid regex: invalid repeat!")
      );
    }
  }
  return Result<std::pair<int, int>>::Ok(std::make_pair(lower_bound, upper_bound));
}

void FSMWithStartEnd::GetPossibleRules(const int& state, std::unordered_set<int>* rules) const {
  rules->clear();
  for (const auto& edge : fsm.edges[state]) {
    if (edge.IsRuleRef()) {
      rules->insert(edge.GetRefRuleId());
    }
  }
  return;
}

void CompactFSMWithStartEnd::GetPossibleRules(const int& state, std::unordered_set<int>* rules)
    const {
  rules->clear();
  for (const auto& edge : fsm.edges[state]) {
    if (edge.IsRuleRef()) {
      rules->insert(edge.GetRefRuleId());
    }
  }
  return;
}

std::ostream& operator<<(std::ostream& os, const FSMWithStartEnd& fsm) {
  os << "FSM(num_nodes=" << fsm.NumNodes() << ", start=" << fsm.StartNode() << ", end=[";
  for (auto end = fsm.ends.begin(); end != fsm.ends.end(); ++end) {
    os << *end;
    if (std::next(end) != fsm.ends.end()) {
      os << ", ";
    }
  }
  os << "], edges=[\n";
  for (int i = 0; i < fsm.NumNodes(); ++i) {
    os << i << ": [";
    const auto& edges = fsm.fsm.edges[i];
    for (int j = 0; j < static_cast<int>(edges.size()); ++j) {
      const auto& edge = edges[j];
      if (edge.min == edge.max) {
        os << "(" << edge.min << ")->" << edge.target;
      } else {
        os << "(" << edge.min << ", " << edge.max << ")->" << edge.target;
      }
      if (j < static_cast<int>(edges.size()) - 1) {
        os << ", ";
      }
    }
    os << "]\n";
  }
  os << "])";
  return os;
}

FSMWithStartEnd BuildTrie(
    const std::vector<std::string>& patterns, std::vector<int32_t>* end_nodes
) {
  FSMWithStartEnd fsm(1);
  fsm.SetStartNode(0);
  if (end_nodes) {
    end_nodes->clear();
  }
  for (const auto& pattern : patterns) {
    int current_node = 0;
    for (const auto& ch : pattern) {
      int16_t ch_int16 = static_cast<int16_t>(static_cast<uint8_t>(ch));
      int next_node = fsm.Transition(current_node, ch_int16);
      if (next_node == FSMWithStartEnd::NO_TRANSITION) {
        next_node = fsm.AddNode();
        fsm.AddEdge(current_node, next_node, ch_int16, ch_int16);
      }
      current_node = next_node;
    }
    fsm.AddEndNode(current_node);
    if (end_nodes) {
      end_nodes->push_back(current_node);
    }
  }
  return fsm;
}

}  // namespace xgrammar
