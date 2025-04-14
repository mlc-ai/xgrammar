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
    \param result The epsilon closure of the state.
  */
  void GetEpsilonClosure(int state, std::unordered_set<int>* result) const;

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

  FSM() = default;

  friend class FSMWithStartEnd;
};

class FSMWithStartEnd {
 public:
  bool is_dfa = false;
  FSM fsm;
  int start;
  std::unordered_set<int> ends;

  /*!
  \brief Construct a FSM from a regex string.
  \details The regex string should only be the format like "abx" or [a-c0-9].
  \details Any symbols like "a|b" or "a*b" are not supported.
  \param regex The regex string.
*/
  FSMWithStartEnd(const std::string& regex);

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
  FSMWithStartEnd ToDFA() const;
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
  /*!
    \brief Check if the FSM accepts the string.
    \param str The input string.
    \return True if the FSM accepts the string, false otherwise.
  */
  bool Check(const std::string& str) const;

  /*! \brief Constructs an FSM with the specified number of nodes. */
  FSMWithStartEnd(int num_nodes = 0, bool is_dfa = false) : is_dfa(is_dfa) {
    for (int i = 0; i < num_nodes; ++i) {
      fsm.edges.emplace_back();
    }
  }

  /********************** Accessors **********************/

  inline static constexpr int NO_TRANSITION = -1;

  /*!
   * \brief Transitions from a given state based on an input character.
   * \param from The source state to transition from.
   * \param character The input character.
   * \return The target state if a valid transition exists, -1 otherwise.
   */
  int Transition(int from, int16_t character) const {
    auto& edges = fsm.edges[from];
    for (const auto& edge : edges) {
      if (edge.min <= character && edge.max >= character) {
        return edge.target;
      }
    }
    return NO_TRANSITION;
  }

  /*! \brief Returns the start node of the FSM. */
  int StartNode() const { return start; }

  /*!
   * \brief Checks if a given node is an end/accepting state.
   * \param node The node to check.
   * \return True if the node is an end state, false otherwise.
   */
  bool IsEndNode(int node) const {
    return std::any_of(ends.begin(), ends.end(), [node](int end_node) { return end_node == node; });
  }

  /*! \brief Returns the total number of nodes in the FSM. */
  int NumNodes() const { return fsm.edges.size(); }

  /********************** Modifiers **********************/

  /*!
   * \brief Adds a transition edge between states with a character range.
   * \param from The source state.
   * \param to The target state.
   * \param min_ch The minimum character in the range (inclusive).
   * \param max_ch The maximum character in the range (inclusive).
   */
  void AddEdge(int from, int to, int16_t min_ch, int16_t max_ch) {
    fsm.edges[from].push_back({min_ch, max_ch, to});
  }

  /*!
   * \brief Adds a new node to the FSM.
   * \return The index of the newly added node.
   */
  int AddNode() {
    fsm.edges.emplace_back();
    return fsm.edges.size() - 1;
  }

  /*!
   * \brief Sets the start node of the FSM.
   * \param node The node to set as the start node.
   */
  void SetStartNode(int node) { start = node; }

  /*!
   * \brief Adds an end/accepting node to the FSM.
   * \param node The node to add as an end node.
   */
  void AddEndNode(int node) { ends.insert(node); }

  /*!
  \brief Check if the FSM is a DFA.
  \return True if the FSM is a DFA, false otherwise.
  */
  bool IsDFA();

  /*!
    \brief Check if the FSM is a leaf FSM.
    \return True if the FSM is a leaf FSM, false otherwise.
  */
  bool IsLeaf() const;

  /*!
  \brief Merge some nodes by removing some epsilon transitions.
  \details For example, a -- \epsilon --> b, and b doesn't have
  \details any other inward edges, then we can merge the two nodes.
*/
  void SimplifyEpsilon();

  /*!
   \brief Merge some nodes which are approximately the same.
   \details Actually, if two nodes have the same outward edges,
   \details or the same inward edges, then we can merge them.
  */
  void SimplifyTransition();

  friend std::ostream& operator<<(std::ostream& os, const FSMWithStartEnd& fsm);
};

class CompactFSM {
 private:
  /*!
    \brief Get the epsilon closure of a state.
    \param state The current state id.
    \param result The epsilon closure of the state.
  */
  void GetEpsilonClosure(int state, std::unordered_set<int>* result) const;

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
  // The internal states are also public
  using Edge = FSMEdge;

  CSRArray<Edge> edges;
  friend class CompactFSMWithStartEnd;
};

class CompactFSMWithStartEnd {
 public:
  bool is_dfa = false;
  CompactFSM fsm;
  int start;
  std::unordered_set<int> ends;
  using Edge = FSMEdge;
  /*!
    \brief Print the FSM.
    \return The string representation of the FSM.
  */
  std::string Print() const;
  /*!
    \brief Check if the FSM accepts the string.
    \param str The input string.
    \return True if the FSM accepts the string, false otherwise.
  */
  bool Check(const std::string& str) const;
  inline static constexpr int NO_TRANSITION = -1;
  int Transition(int from, int16_t character) const {
    auto edges = fsm.edges[from];
    // TODO(yixin): test correctness for both cases
    if (edges.size() <= 16) {
      for (const auto& edge : edges) {
        if (edge.min > character) {
          return NO_TRANSITION;
        } else if (edge.max >= character) {
          return edge.target;
        }
      }
      return NO_TRANSITION;
    } else {
      auto it = std::lower_bound(
          edges.begin(),
          edges.end(),
          character,
          [](const Edge& edge, int16_t character) { return edge.min <= character; }
      );
      if (it != edges.end() && it->min <= character) {
        return it->target;
      }
      return NO_TRANSITION;
    }
  }

  /*! \brief Returns the start node of the FSM. */
  int StartNode() const { return start; }

  /*!
   * \brief Checks if a given node is an end/accepting state.
   * \param node The node to check.
   * \return True if the node is an end state, false otherwise.
   */
  bool IsEndNode(int node) const {
    return std::any_of(ends.begin(), ends.end(), [node](int end_node) { return end_node == node; });
  }

  /*! \brief Returns the total number of nodes in the FSM. */
  int NumNodes() const { return fsm.edges.Size(); }

  friend std::ostream& operator<<(std::ostream& os, const CompactFSM& fsm);

  friend std::size_t MemorySize(const CompactFSMWithStartEnd& self) {
    return MemorySize(self.fsm.edges) + MemorySize(self.ends);
  }
};

/*!
  \brief Converts a regex string to a FSM. The parsing range is [start, end).
  \param regex The regex string.
  \param start The current processing character index in the regex string.
  \param end The end character index in the regex string, -1 means the end of the string.
  \return The FSM with start and end states.
*/
FSMWithStartEnd RegexToFSM(const std::string& regex, int start = 0, int end = -1);

inline std::ostream& operator<<(std::ostream& os, const FSMWithStartEnd& fsm) {
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

inline FSMWithStartEnd BuildTrie(
    const std::vector<std::string>& patterns, std::vector<int32_t>* end_nodes = nullptr
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

#endif  // XGRAMMAR_FSM_H_
