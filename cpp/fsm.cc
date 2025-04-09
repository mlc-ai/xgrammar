#include <xgrammar/fsm.h>

#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

namespace xgrammar {

FSMEdge::FSMEdge(const short& _min, const short& _max, const int& target)
    : min(_min), max(_max), target(target) {
  if (IsCharRange() && min > max) {
    throw std::runtime_error("Invalid char range: min > max");
  }
}

bool FSMEdge::IsEpsilon() const { return min == -1 && max == -1; }

bool FSMEdge::IsRuleRef() const { return min == -1; }

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
      node_cnt += fsm_with_se.fsm.edges.size();
    }
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

  for (const auto& node_edges : fsm.edges) {
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
    for (int i = 0; i < 0x100; ++i) {
      if (!char_has_edges[i]) {
        // The char doesn't have any edges. Thus, we can accept it in the
        // complement FSM.
        if (interval_start == -1) {
          interval_start = i;
        }
      } else {
        if (interval_start != -1) {
          // node_cnt is the node to accept all such characters.
          result.fsm.edges.back().emplace_back(interval_start, i - 1, node_cnt);
          interval_start = -1;
        }
      }
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
  std::queue<int> epsilon_queue = std::queue<int>();
  std::unordered_set<int> visited;
  std::unordered_set<int> in_result;

  for (const auto& state : from) {
    if (visited.find(state) != visited.end()) {
      continue;
    }
    visited.insert(state);
    for (const auto& edge : edges[state]) {
      if (edge.IsEpsilon()) {
        epsilon_queue.push(edge.target);
        continue;
      }
      if (is_rule && edge.IsRuleRef()) {
        if (edge.GetRefRuleId() == value) {
          if (in_result.find(edge.target) == in_result.end()) {
            result->push_back(edge.target);
            in_result.insert(edge.target);
          }
        }
        continue;
      }
      if ((!is_rule) && edge.IsCharRange()) {
        if (value >= edge.min && value <= edge.max) {
          if (in_result.find(edge.target) == in_result.end()) {
            result->push_back(edge.target);
            in_result.insert(edge.target);
          }
        }
        continue;
      }
    }
  }

  while (!epsilon_queue.empty()) {
    int current = epsilon_queue.front();
    epsilon_queue.pop();
    if (visited.find(current) != visited.end()) {
      continue;
    }
    visited.insert(current);
    for (const auto& edge : edges[current]) {
      if (edge.IsEpsilon()) {
        epsilon_queue.push(edge.target);
        continue;
      }
      if (is_rule && edge.IsRuleRef()) {
        if (edge.GetRefRuleId() == value) {
          if (in_result.find(edge.target) == in_result.end()) {
            result->push_back(edge.target);
            in_result.insert(edge.target);
          }
        }
        continue;
      }
      if ((!is_rule) && edge.IsCharRange()) {
        if (value >= edge.min && value <= edge.max) {
          if (in_result.find(edge.target) == in_result.end()) {
            result->push_back(edge.target);
            in_result.insert(edge.target);
          }
        }
        continue;
      }
    }
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
    for (int j = 0; j < row.size(); i++) {
      result.edges.back().push_back(row[j]);
    }
  }
  return result;
}

}  // namespace xgrammar
