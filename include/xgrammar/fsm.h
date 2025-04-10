/*!
 *  Copyright (c) 2023 by Contributors
 * \file xgrammar/fsm.h
 */
#ifndef XGRAMMAR_FSM_H_
#define XGRAMMAR_FSM_H_

#include <unordered_set>
#include <vector>

#include "../cpp/support/csr_array.h"

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
  /*!
    \brief Check if the edge is an epsilon transition.
  */
  bool IsEpsilon() const;
  /*!
    \brief Check if the edge is a rule reference.
  */
  bool IsRuleRef() const;
  /*!
    \brief Get the rule id of the edge.
    \return The rule id of the edge.
    \throw std::runtime_error if the edge is not a rule reference.
  */
  short GetRefRuleId() const;
  /*!
    \brief Check if the edge is a character range.
  */
  bool IsCharRange() const;
};
class CompactFSM;
class FSM {
 private:
  /*!
    \brief Get the epsilon closure of a state.
    \param state The current state id.
    \return The epsilon closure of the state.
  */
  std::unordered_set<int> GetEpsilonClosure(int state) const;

 public:
  using Edge = FSMEdge;
  /*!
    \brief Transform a FSM to a compact FSM.
    \return The compact FSM.
  */
  CompactFSM ToCompact();
  /*!
    \brief Advance the FSM to the next state.
    \param from The current states.
    \param value The input value.
    \param result The next states, which can be seen as the result of the transition.
    \param is_rule Whether the input value is a rule id.
  */
  void Advance(
      const std::vector<int>& from, int value, std::vector<int>* result, bool is_rule = false
  ) const;
  /*!
    \brief Return a copy of the FSM.
  */
  FSM Copy() const;
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

  /*!
    \brief Assume the FSM accepts rule1, then the FSM will accept rule1*.
    \return The FSM that accepts rule1*.
  */
  FSMWithStartEnd MakeStar() const;
  /*!
    \brief Assume the FSM accepts rule1, then the FSM will accept rule1+.
    \return The FSM that accepts rule1+.
  */
  FSMWithStartEnd MakePlus() const;
  /*!
    \brief Assume the FSM accepts rule1, then the FSM will accept rule1?.
    \return The FSM that accepts rule1?.
  */
  FSMWithStartEnd MakeOptional() const;
  /*!
    \brief Transform the FSM to a DFA.
    \return The DFA.
  */
  FSMWithStartEnd TODFA() const;
  /*!
    \brief Transform the FSM to accept the complement of the language.
    \return The complement FSM.
  */
  FSMWithStartEnd Not() const;
  /*!
    \brief Minimize the DFA.
    \return The minimized DFA.
  */
  FSMWithStartEnd MinimizeDFA() const;
  /*!
    \brief Return a copy of the FSM.
    \return The copy of the FSM.
  */
  FSMWithStartEnd Copy() const;
  /*!
    \brief Print the FSM.
    \return The string representation of the FSM.
  */
  std::string Print() const;
  /*!
    \brief Intersect the FSMs.
    \param lhs The left FSM.
    \param rhs The right FSM.
    \return The intersection of the FSMs.
  */
  static FSMWithStartEnd Intersect(const FSMWithStartEnd& lhs, const FSMWithStartEnd& rhs);
  /*!
    \brief Union the FSMs.
    \param fsms The FSMs to be unioned.
    \return The union of the FSMs.
  */
  static FSMWithStartEnd Union(const std::vector<FSMWithStartEnd>& fsms);

  /*!
    \brief Concatenate the FSMs.
    \param fsms The FSMs to be concatenated, which should be in order.
    \return The concatenation of the FSMs.
  */
  static FSMWithStartEnd Concatenate(const std::vector<FSMWithStartEnd>& fsms);
};

class CompactFSM {
 public:
  /*!
   \brief Advance the FSM to the next state.
   \param from The current states.
   \param value The input value.
   \param result The next states, which can be seen as the result of the transition.
   \param is_rule Whether the input value is a rule id.
  */
  void Advance(
      const std::vector<int>& from, int value, std::vector<int>* result, bool is_rule = false
  ) const;
  /*!
    \brief Transform the compact FSM to a FSM.
    \return The FSM.
  */
  FSM ToFSM();
  // The interanl states are also public
  using Edge = FSMEdge;
  CSRArray<Edge> edges;
};

class CompactFSMWithStartEnd {
 public:
  bool is_dfa = false;
  CompactFSM fsm;
  int start;
  std::unordered_set<int> ends;
  /*!
  \brief Print the FSM.
  \return The string representation of the FSM.
*/
  std::string Print() const;
};

/*!
  \brief Converts a regex string to a FSM. The parsing range is [start, end).
  \param regex The regex string.
  \param start The current processing character index in the regex string.
  \param end The end character index in the regex string, -1 means the end of the string.
  \return The FSM with start and end states.
*/
FSMWithStartEnd RegexToFSM(const std::string& regex, int start = 0, int end = -1);
}  // namespace xgrammar

#endif  // XGRAMMAR_FSM_H_
