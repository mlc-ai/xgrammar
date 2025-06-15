/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm.h
 * \note For functions accepting a pointer to a container as result, the container will be cleared
 * before the result is stored.
 */
#ifndef XGRAMMAR_FSM_H_
#define XGRAMMAR_FSM_H_

#include <xgrammar/object.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "picojson.h"
#include "support/csr_array.h"
#include "support/reflection.h"

namespace xgrammar {

/*!
 * \brief The edge of a FSM.
 */
struct FSMEdge {
  /*!
   * \brief The min and max are used to represent the range of characters.
   * \details When min == -1 and max == -1, it means the edge is an epsilon transition.
   * When min == -1 and max >= 0, then max represents the rule id.
   * When min >= 0 and max >= 0, then it represents a range of characters.
   */
  short min, max;

  /*!
   * \brief The target state id of the edge.
   */
  int target;

  FSMEdge(short min, short max, int target) : min(min), max(max), target(target) {
    XGRAMMAR_DCHECK(!IsCharRange() || min <= max)
        << "Invalid FSMEdge: min > max. min=" << min << ", max=" << max;
  }

  // for serialization only
  FSMEdge() = default;

  /*!
   * \brief Compare the edges. Used to sort the edges in the FSM.
   */
  // TODO(yixin): consider combining the fields to a single int64_t for better efficiency
  friend bool operator==(const FSMEdge& lhs, const FSMEdge& rhs) {
    return std::make_tuple(lhs.min, lhs.max, lhs.target) ==
           std::make_tuple(rhs.min, rhs.max, rhs.target);
  }

  /*!
   * \brief Compare the edges. Used to sort the edges in the FSM.
   */
  friend bool operator<(const FSMEdge& lhs, const FSMEdge& rhs) {
    return std::make_tuple(lhs.min, lhs.max, lhs.target) <
           std::make_tuple(rhs.min, rhs.max, rhs.target);
  }

  /*!
   * \brief Check if the edge is a character range.
   */
  bool IsCharRange() const { return min >= 0 && max >= 0; }

  /*!
   * \brief Check if the edge is an epsilon transition.
   */
  bool IsEpsilon() const { return min == -1 && max == -1; }

  /*!
   * \brief Check if the edge is a rule reference.
   */
  bool IsRuleRef() const { return min == -1 && max >= 0; }

  /*!
   * \brief Get the rule id of the edge.
   * \return The rule id of the edge. -1 if the edge is not a rule reference.
   */
  int GetRefRuleId() const { return IsRuleRef() ? max : -1; }
};

XGRAMMAR_MEMBER_ARRAY(FSMEdge, &FSMEdge::min, &FSMEdge::max, &FSMEdge::target);

}  // namespace xgrammar

namespace std {

/*!
 * \brief Hash function for FSMEdge.
 */
template <>
struct hash<xgrammar::FSMEdge> {
  size_t operator()(const xgrammar::FSMEdge& edge) const {
    return std::hash<std::tuple<short, short, int>>()(
        std::make_tuple(edge.min, edge.max, edge.target)
    );
  }
};

}  // namespace std

namespace xgrammar {

class CompactFSM;

/*!
 * \brief FSM is a class that represents a finite state machine, could be a DFA or an NFA.
 * \details It's mutable, which means you can add edges and states to it.
 */
class FSM {
 public:
  /*!
   * \brief Construct an FSM with a given number of states.
   * \param num_states The number of states in the FSM.
   */
  FSM(int num_states = 0);

  /*!
   * \brief Construct an FSM with a given set of edges.
   */
  FSM(const std::vector<std::vector<FSMEdge>>& edges);

  /*!
   * \brief Construct an FSM with a given set of edges.
   */
  FSM(std::vector<std::vector<FSMEdge>>&& edges);

  /****************** FSM Visitors ******************/

  /*!
   * \brief Get the number of states in the FSM.
   * \return The number of states in the FSM.
   */
  int NumStates() const;

  const std::vector<std::vector<FSMEdge>>& GetEdges() const;

  const std::vector<FSMEdge>& GetEdges(int state) const;

  std::string PrintEdges() const;

  /****************** FSM Traversal Visitors ******************/

  inline static constexpr int kNoNextState = -1;

  /*!
   * \brief Advance the FSM from a given state based on an input character. If there are multiple
   * transitions, the first one will be returned.
   * \param from The source state to transition from.
   * \param character The input character.
   * \return The target state if a valid transition exists, kNoNextState otherwise.
   */
  int GetNextState(int from, int16_t character) const;

  /*!
   * \brief Advance the FSM to the next state.
   * \param from The current states.
   * \param value The input value.
   * \param result The possible next states. The result is cleared at the beginning.
   * \param value_is_rule Whether the input value is a rule id.
   * \param from_is_closure Whether from is an epsilon closure.
   */
  void Advance(
      const std::unordered_set<int>& from,
      int value,
      std::unordered_set<int>* result,
      bool value_is_rule = false,
      bool from_is_closure = false
  ) const;

  /*!
   * \brief Get all the possible rule numbers for a given state.
   * \param state_num The state number.
   * \param rules The set of possible rule numbers. The result is cleared at the beginning.
   */
  void GetPossibleRules(const int& state_num, std::unordered_set<int>* rules) const;

  /*!
   * \brief Get the epsilon closure of a set of states, i.e. those can be reached by epsilon
   * transitions.
   * \param state_set The states in the epsilon closure. The result is not cleared.
   */
  void GetEpsilonClosure(std::unordered_set<int>* state_set) const;

  /*!
   * \brief Get the reachable states from a set of states.
   * \param from The current states.
   * \param result The reachable states. The result is cleared at the beginning.
   */
  void GetReachableStates(const std::vector<int>& from, std::unordered_set<int>* result) const;

  /****************** FSM Mutators ******************/

  /*!
   * \brief Adds a new state to the FSM.
   * \return The index of the newly added state.
   */
  int AddState();

  /*!
   * \brief Adds a transition edge between states with a character range.
   * \param from The source state.
   * \param to The target state.
   * \param min_ch The minimum character in the range (inclusive).
   * \param max_ch The maximum character in the range (inclusive).
   */
  void AddEdge(int from, int to, int16_t min_ch, int16_t max_ch);

  /*!
   * \brief Add an epsilon transition between two states.
   * \param from The source state.
   * \param to The target state.
   */
  void AddEpsilonEdge(int from, int to);

  /*!
   * \brief Add a rule reference edge between states.
   * \param from The source state.
   * \param to The target state.
   * \param rule_id The rule id to reference.
   */
  void AddRuleEdge(int from, int to, int16_t rule_id);

  /*!
   * \brief Add a whole FSM to the current FSM.
   * \param fsm The FSM to be added.
   * \param state_mapping The mapping from the state ids of the added FSM to the new ids in the
   * current FSM. The result is cleared at the beginning.
   */
  void AddFSM(const FSM& fsm, std::unordered_map<int, int>* state_mapping = nullptr);

  /****************** FSM Construction Algorithms ******************/

  /*!
    \brief Return a copy of the FSM.
  */
  FSM Copy() const;

  /*!
   * \brief Rebuild the FSM with the new state ids.
   * \param state_mapping The mapping from the old state ids to the new state ids.
   * \param new_num_states The new number of states.
   * \return The rebuilt FSM.
   */
  FSM RebuildWithMapping(std::unordered_map<int, int>& state_mapping, int new_num_states);

  /*!
   * \brief Transform a FSM to a compact FSM.
   * \return The compact FSM.
   */
  CompactFSM ToCompact();

  XGRAMMAR_DEFINE_PIMPL_METHODS(FSM);
};

/*!
 * \brief CompactFSM is the compact from of FSM.
 * \details It uses CSRArray to store the edges, ensuring memory contiguity. It sorts all outgoing
 * edges from a node according to their min and max values, so traversal can be faster.
 *
 * CompactFSM is immutable. If you need to modify a CompactFSM, you need to convert it to a FSM
 * first, and convert it back after modification.
 *
 * It share the same set of visitor methods with FSM.
 */
class CompactFSM {
 public:
  // for serialization only
  CompactFSM() = default;

  CompactFSM(const CSRArray<FSMEdge>& edges);

  CompactFSM(CSRArray<FSMEdge>&& edges);

  /****************** CompactFSM Visitors ******************/

  /*!
   * \brief Get the number of states in the FSM.
   * \return The number of states in the FSM.
   */
  int NumStates() const;

  const CSRArray<FSMEdge>& GetEdges() const;

  CSRArray<FSMEdge>::Row GetEdges(int state) const;

  std::string PrintEdges() const;

  friend std::size_t MemorySize(const CompactFSM& self);

  /****************** CompactFSM Traversal Visitors ******************/

  inline static constexpr int kNoNextState = -1;

  /*!
   * \brief Advance the FSM from a given state based on an input character. If there are multiple
   * transitions, the first one will be returned.
   * \param from The source state to transition from.
   * \param character The input character.
   * \return The target state if a valid transition exists, kNoNextState otherwise.
   */
  int GetNextState(int from, int16_t character) const;

  /*!
   * \brief Advance the FSM to the next state.
   * \param from The current states.
   * \param value The input value.
   * \param result The possible next states. The result is cleared at the beginning.
   * \param value_is_rule Whether the input value is a rule id.
   * \param from_is_closure Whether from is an epsilon closure.
   */
  void Advance(
      const std::unordered_set<int>& from,
      int value,
      std::unordered_set<int>* result,
      bool value_is_rule = false,
      bool from_is_closure = false
  ) const;

  /*!
   * \brief Get all the possible rule numbers for a given state.
   * \param state_num The state number.
   * \param rules The set of possible rule numbers. The result is cleared at the beginning.
   */
  void GetPossibleRules(const int& state_num, std::unordered_set<int>* rules) const;

  /*!
   * \brief Get the epsilon closure of a set of states, i.e. those can be reached by epsilon
   * transitions.
   * \param state_set The states in the epsilon closure. The result is not cleared.
   */
  void GetEpsilonClosure(std::unordered_set<int>* state_set) const;

  /*!
   * \brief Get the reachable states from a set of states.
   * \param from The current states.
   * \param result The reachable states. The result is cleared at the beginning.
   */
  void GetReachableStates(const std::vector<int>& from, std::unordered_set<int>* result) const;

  /****************** CompactFSM Construction Algorithms ******************/

  /*!
   * \brief Transform the compact FSM to a FSM.
   * \return The FSM.
   */
  FSM ToFSM() const;

  picojson::value SerializeJSONValue() const;
  friend void DeserializeJSONValue(CompactFSM& fsm, const picojson::value& v);

  XGRAMMAR_DEFINE_PIMPL_METHODS(CompactFSM);
};

class CompactFSMWithStartEnd;

/*!
 * \brief The base class for FSMWithStartEnd and CompactFSMWithStartEnd. It defines the
 * common constructor and visitor methods.
 */
template <typename FSMType>
class FSMWithStartEndBase {
  static_assert(
      std::is_same_v<FSMType, FSM> || std::is_same_v<FSMType, CompactFSM>,
      "FSMType must be FSM or CompactFSM"
  );

 public:
  // for serialization only
  FSMWithStartEndBase() = default;

  /*! \brief Constructs an FSMWithStartEnd with a given FSM, start state, and end states. */
  FSMWithStartEndBase(
      const FSMType& fsm, int start, const std::unordered_set<int>& ends, bool is_dfa = false
  )
      : fsm_(fsm), start_(start), ends_(ends), is_dfa_(is_dfa) {}

  /****************** Member Accessors and Mutators ******************/

  /*! \brief Returns the underlying FSM. */
  const FSMType& GetFSM() const { return fsm_; }

  /*! \brief Returns the start state of the FSM. */
  int GetStart() const { return start_; }

  /*! \brief Returns the end states of the FSM. */
  const std::unordered_set<int>& GetEnds() const { return ends_; }

  /*!
   * \brief Checks if a given state is an end/accepting state.
   * \param state The state to check.
   * \return True if the state is an end state, false otherwise.
   */
  bool IsEndState(int state) const {
    return std::any_of(ends_.begin(), ends_.end(), [state](int end_state) {
      return end_state == state;
    });
  }

  /*!
   * \brief Sets the start state of the FSM.
   * \param state The state to set as the start state.
   */
  void SetStartState(int state) {
    XGRAMMAR_DCHECK(state < NumStates());
    start_ = state;
  }

  /*!
   * \brief Adds an end/accepting state to the FSM.
   * \param state The state to add as an end state.
   */
  void AddEndState(int state) {
    XGRAMMAR_DCHECK(state < NumStates());
    ends_.insert(state);
  }

  /*!
   * \brief Sets the end states of the FSM.
   * \param ends The new end states.
   */
  void SetEndStates(const std::unordered_set<int>& ends) { ends_ = ends; }

  /*! \brief Returns the total number of states in the FSM. */
  int NumStates() const { return fsm_.NumStates(); }

  /*!
   * \brief Access the methods of the underlying FSM.
   */
  FSMType* operator->() { return &fsm_; }

  /*!
   * \brief Access the methods of the underlying FSM.
   */
  const FSMType* operator->() const { return &fsm_; }

  /****************** FSM Traversal Algorithms ******************/

  /*!
   * \brief Check if the FSM accepts the string.
   * \param str The input string.
   * \return True if the FSM accepts the string, false otherwise.
   */
  bool AcceptString(const std::string& str) const;

  /*!
   * \brief Get the reachable states from the start state.
   * \param result The reachable states. The result is cleared at the beginning.
   */
  void GetReachableStates(std::unordered_set<int>* result) const;

  /*!
   * \brief Check if the FSM is a leaf FSM.
   * \return True if the FSM is a leaf FSM, false otherwise.
   */
  bool IsLeaf() const;

 protected:
  /*! \brief The underlying finite state machine. */
  FSMType fsm_;
  /*! \brief The start state of the FSM. */
  int start_;
  /*! \brief The set of accepting/end states. */
  std::unordered_set<int> ends_;
  /*! \brief Whether this FSM is a deterministic finite automaton. */
  bool is_dfa_ = false;

  friend struct member_trait<CompactFSMWithStartEnd>;
};

/*!
 * \brief FSMWithStartEnd represents a FSM with start and end states.
 * \details It stores a pointer to a FSM, a start state, and a set of end states. Multiple
 * FSMWithStartEnd can share the same FSM. It also provides a set of methods to construct FSMs.
 */
class FSMWithStartEnd : public FSMWithStartEndBase<FSM> {
 public:
  using FSMWithStartEndBase<FSM>::FSMWithStartEndBase;

  /*!
   * \brief Return a copy of the FSMWithStartEnd.
   */
  FSMWithStartEnd Copy() const;

  /*!
   * \brief Print the FSM.
   * \return The string representation of the FSMWithStartEnd.
   */
  std::string Print() const;

  friend std::ostream& operator<<(std::ostream& os, const FSMWithStartEnd& fsm);

  /****************** FSM Construction Algorithms ******************/

  /*!
   * \brief Return a new FSM representing FSM*
   * \return The FSM that accepts FSM*.
   */
  FSMWithStartEnd Star() const;

  /*!
   * \brief Return a new FSM representing rule1+.
   * \return The FSM that accepts rule1+.
   */
  FSMWithStartEnd Plus() const;

  /*!
   * \brief Return a new FSM representing rule1?.
   * \return The FSM that accepts rule1?.
   */
  FSMWithStartEnd Optional() const;

  /*!
   * \brief Return a new FSM representing the complement of the language.
   * \return The complement FSM.
   */
  FSMWithStartEnd Not() const;

  /*!
   * \brief Intersect the FSMs.
   * \param lhs The left FSM.
   * \param rhs The right FSM.
   * \return The intersection of the FSMs.
   */
  static Result<FSMWithStartEnd> Intersect(
      const FSMWithStartEnd& lhs, const FSMWithStartEnd& rhs, const int& num_of_states_limited = 1e6
  );

  /*!
   * \brief Union the FSMs.
   * \param fsms The FSMs to be unioned.
   * \return The union of the FSMs.
   */
  static FSMWithStartEnd Union(const std::vector<FSMWithStartEnd>& fsms);

  /*!
   * \brief Concatenate the FSMs.
   * \param fsms The FSMs to be concatenated, which should be in order.
   * \return The concatenation of the FSMs.
   */
  static FSMWithStartEnd Concat(const std::vector<FSMWithStartEnd>& fsms);

  /*!
   * \brief Check if the FSM is a DFA.
   * \return True if the FSM is a DFA, false otherwise.
   */
  bool IsDFA();

  /*!
   * \brief Merge some states by removing some epsilon transitions.
   * \details If a --\epsilon--> b, and either 1) b doesn't have any other inward edges, or
   * 2) a doesn't have any other outward edges, we can merge a and b.
   */
  FSMWithStartEnd SimplifyEpsilon() const;

  /*!
   * \brief Merge equivalent states in the FSM.
   * \details If two states are 1) pointed to by edges with the same label from the same state, and
   * 2) they are not pointed to by other edges, then we can merge them.
   * \example n0 --(c)--> n1, n0 --(c)--> n2, then we can merge n1 and n2.
   */
  FSMWithStartEnd MergeEquivalentSuccessors() const;

  /*!
   * \brief Transform the FSM to a DFA.
   * \return The DFA.
   */
  FSMWithStartEnd ToDFA() const;

  /*!
   * \brief Minimize the DFA.
   * \return The minimized DFA.
   */
  FSMWithStartEnd MinimizeDFA() const;

  /*!
   * \brief Rebuild the FSM with the new state ids.
   * \param state_mapping The mapping from old state ids to new state ids.
   * \param new_num_states The new number of states.
   */
  FSMWithStartEnd RebuildWithMapping(
      std::unordered_map<int, int>& state_mapping, int new_num_states
  );

  /*!
   * \brief Transform the FSMWithStartEnd to a CompactFSMWithStartEnd.
   * \return The CompactFSMWithStartEnd.
   */
  CompactFSMWithStartEnd ToCompact();
};

/*!
 * \brief A class that represents a compact-form FSM with a start state and a set of end states.
 * \details CompactFSMWithStartEnd stores a pointer to a CompactFSM, a start state, and a set of end
 * states. Multiple CompactFSMWithStartEnd can share the same CompactFSM. It share the same set of
 * visitor methods with FSMWithStartEnd.
 */
class CompactFSMWithStartEnd : public FSMWithStartEndBase<CompactFSM> {
 public:
  using FSMWithStartEndBase<CompactFSM>::FSMWithStartEndBase;

  // for serialization only
  CompactFSMWithStartEnd() = default;

  /*!
   * \brief Print the FSM.
   * \return The string representation of the FSM.
   */
  std::string Print() const;

  friend std::ostream& operator<<(std::ostream& os, const CompactFSMWithStartEnd& fsm);

  /*!
   * \brief Get the memory size of the CompactFSMWithStartEnd.
   * \param self The CompactFSMWithStartEnd.
   * \return The memory size of the CompactFSMWithStartEnd.
   */
  friend std::size_t MemorySize(const CompactFSMWithStartEnd& self);

  /*!
   * \brief Transform the CompactFSMWithStartEnd to a FSMWithStartEnd.
   * \return The FSMWithStartEnd.
   */
  FSMWithStartEnd ToFSM() const;
};

XGRAMMAR_MEMBER_ARRAY(
    CompactFSMWithStartEnd,
    &CompactFSMWithStartEnd::fsm_,
    &CompactFSMWithStartEnd::start_,
    &CompactFSMWithStartEnd::ends_,
    &CompactFSMWithStartEnd::is_dfa_
);

/****************** FSMWithStartEndBase Template Implementation ******************/

template <typename FSMType>
inline bool FSMWithStartEndBase<FSMType>::AcceptString(const std::string& str) const {
  std::unordered_set<int> start_states{start_};
  fsm_.GetEpsilonClosure(&start_states);
  std::unordered_set<int> result_states;
  for (const auto& character : str) {
    result_states.clear();
    fsm_.Advance(
        start_states, static_cast<int>(static_cast<unsigned char>(character)), &result_states, false
    );
    if (result_states.empty()) {
      return false;
    }
    start_states = result_states;
  }
  return std::any_of(start_states.begin(), start_states.end(), [&](int state) {
    return ends_.find(state) != ends_.end();
  });
}

template <typename FSMType>
inline void FSMWithStartEndBase<FSMType>::GetReachableStates(std::unordered_set<int>* result
) const {
  return fsm_.GetReachableStates({start_}, result);
}

template <typename FSMType>
inline bool FSMWithStartEndBase<FSMType>::IsLeaf() const {
  std::unordered_set<int> reachable_states;
  GetReachableStates(&reachable_states);
  for (const auto& state : reachable_states) {
    for (const auto& edge : fsm_.GetEdges(state)) {
      if (edge.IsRuleRef()) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace xgrammar

#endif  // XGRAMMAR_FSM_H_
