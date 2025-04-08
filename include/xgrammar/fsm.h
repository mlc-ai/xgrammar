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
#include <vector>

#include "../cpp/support/csr_array.h"
#include "../cpp/support/utils.h"

namespace xgrammar {

class FSMEdge {
  /*
    The min and max are used to represent the range of characters.
    When min == -1 and max == -1, it means the edge is an epsilon transition.
    When min == -1 and max >= 0, then max represents the rule id.
    When min >= 0 and max >= 0, then it represents a range of characters.
    target is the target state id.
  */
 private:
  short min, max;
  int target;

 public:
  FSMEdge(const short& _min, const short& _max, const int& target);
  bool IsEpsilon() const;
  bool IsRuleRef() const;
  short GetRefRuleId() const;
  bool IsCharRange() const;
};
class CompactFSM;
class FSM {
 public:
  using Edge = FSMEdge;
  static FSM Intersect(const FSM& lhs, const FSM& rhs);
  static FSM Union(const std::vector<FSM>& fsms);
  FSM Not();
  FSM ToDFA();
  FSM MinimizeDFA();

  CompactFSM ToCompact() const;
  void Advance(const std::vector<int>& from, int char_value, std::vector<int>* result) const;
  FSM Copy() const;
  std::string Print() const;
  // The interanl states are also public
  std::vector<std::vector<Edge>> edges;
};

class FSMWithStartEnd {
 public:
  FSM fsm;
  int start;
  std::vector<int> ends;
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
