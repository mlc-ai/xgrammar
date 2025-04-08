#include <xgrammar/fsm.h>

#include <queue>
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

}  // namespace xgrammar
