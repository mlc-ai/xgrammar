/*!
 *  Copyright (c) 2023 by Contributors
 * \file xgrammar/fsm.h
 */
#ifndef XGRAMMAR_FSM_H_
#define XGRAMMAR_FSM_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "../cpp/support/csr_array.h"
#include "../cpp/support/utils.h"

namespace xgrammar {

struct FSMEdge {
  /*
    The min and max are used to represent the range of characters.
    When min == -1 and max == -1, it means the edge is an epsilon transition.
    When min == -1 and max >= 0, then max represents the rule id.
    When min >= 0 and max >= 0, then it represents a range of characters.
    target is the target state id.
  */
  short min, max;
  int target;
  FSMEdge(const short& _min, const short& _max, const int& target);
  bool IsEpsilon() const;
  bool IsRuleRef() const;
  short GetRefRuleId() const;
  bool IsCharRange() const;
};
class CompactFSM;
class FSM {
 private:
  std::unordered_set<int> GetEpsilonClosure(int state) const;

 public:
  using Edge = FSMEdge;
  CompactFSM ToCompact() const;
  void Advance(
      const std::vector<int>& from, int value, std::vector<int>* result, bool is_rule = false
  ) const;
  FSM Copy() const;
  std::string Print() const;
  // The interanl states are also public
  std::vector<std::vector<Edge>> edges;
  friend class FSMWithStartEnd;
};

class FSMWithStartEnd {
 public:
  bool is_dfa = false;
  FSM fsm;
  int start;
  std::unordered_set<int> ends;
  FSMWithStartEnd TODFA() const;
  FSMWithStartEnd Not();
  FSMWithStartEnd MinimizeDFA();
  FSMWithStartEnd Copy() const;
  static FSMWithStartEnd Intersect(const FSMWithStartEnd& lhs, const FSMWithStartEnd& rhs);
  static FSMWithStartEnd Union(const std::vector<FSMWithStartEnd>& fsms);
};

class CompactFSM {
 public:
  void Advance(const std::vector<int>& from, int char_value, std::vector<int>* result);
  FSM ToFSM();
  std::string Print();
  // The interanl states are also public
  using Edge = FSMEdge;
  CSRArray<Edge> edges;
};

FSMWithStartEnd RegexToFSM(const std::string& regex);
}  // namespace xgrammar

#endif  // XGRAMMAR_FSM_H_
