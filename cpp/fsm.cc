#include <xgrammar/fsm.h>

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
}  // namespace xgrammar
