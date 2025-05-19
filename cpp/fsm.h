/*!
 *  Copyright (c) 2023 by Contributors
 * \file xgrammar/fsm.h
 */
#ifndef XGRAMMAR_FSM_H_
#define XGRAMMAR_FSM_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "../cpp/support/csr_array.h"
#include "support/utils.h"

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
 public:
  using Edge = FSMEdge;

  /*!
    \brief Transform a FSM to a compact FSM.
    \return The compact FSM.
  */
  CompactFSM ToCompact() const;

  /*!
    \brief Advance the FSM to the next state.
    \param from The current states.
    \param value The input value.
    \param result The next states, which can be seen as the result of the
    transition.
    \param is_closure Whether from is an epsilon closure.
    \param is_rule Whether the input value is a rule id.
  */
  void Advance(
      const std::vector<int>& from,
      int value,
      std::vector<int>* result,
      bool is_rule = false,
      bool is_closure = false
  ) const;

  /*!
    \brief Get the epsilon closure of a state.
    \param state_set The current states.
    \param result The epsilon closure of the state. If nullptr,
           then the result will be stored in state_set.
  */
  void GetEpsilonClosure(
      std::unordered_set<int>* state_set, std::unordered_set<int>* result = nullptr
  ) const;

  /*!
    \brief Return a copy of the FSM.
  */
  FSM Copy() const;

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
    \brief Rebuild the FSM with the new state ids.
    \param old_to_new The mapping from old state ids to new state ids.
  */
  void RebuildFSM(std::unordered_map<int, int>& old_to_new, const int& new_node_cnt);

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
  static Result<FSMWithStartEnd> Intersect(
      const FSMWithStartEnd& lhs, const FSMWithStartEnd& rhs, const int& num_of_nodes_limited = 1e6
  );

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
    fsm.edges.resize(num_nodes);
  }

  inline static constexpr int NO_TRANSITION = -1;

  /*!
   * \brief Transitions from a given state based on an input character.
   * \param from The source state to transition from.
   * \param character The input character.
   * \return The target state if a valid transition exists, -1 otherwise.
   */
  int LegacyTransitionOnDFA(int from, int16_t character) const {
    auto& edges = fsm.edges[from];
    for (const auto& edge : edges) {
      if (edge.min <= character && edge.max >= character) {
        return edge.target;
      }
    }
    return NO_TRANSITION;
  }

  /*!
  \brief Transition the FSM.
  \param from The current states. It should be a epsilon closure.
  \param character The input character, or the rule id.
  \param result The next states set. It will return an epsilon closure.
  \param is_rule Whether the input character is a rule id.
  */
  void Transition(
      const std::unordered_set<int>& from,
      int16_t character,
      std::unordered_set<int>* result,
      bool is_rule = false
  ) const;

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

  /*!
    \brief Get all the possible rule numbers for a given node.
    \param node_num The node number.
    \param rules The set of possible rule numbers.
  */
  void GetPossibleRules(const int& node_num, std::unordered_set<int>* rules) const;

  /*!
    \brief Accept all the unicode characters in the FSM.
    \param from The start node of the fsm.
    \param to The destination node of the fsm.
  */
  void AcceptAllUnicodeCharacters(const int& from_node, const int& to_node);

 private:
  /*!
     \brief Build the character class for the FSM.
     \param char_class_edges The character class edges.
     \param is_negative Whether the character class is negative.
     \param start_node The start node of the character class.
     \param end_node The end node of the character class.
  */
  void BuildCharacterClass(
      std::vector<std::pair<uint32_t, uint32_t>>& char_class_edges,
      bool is_negative,
      int start_node = 0,
      int end_node = 1
  );

  /*!
    \brief Add a unicode character edge in the FSM.
    \param min_ch The minimum unicode character.
    \param max_ch The maximum unicode character.
    \param start_node The start node of the edge.
    \param end_node The end node of the edge.
  */
  void AddUnicodeEdge(uint32_t min_ch, uint32_t max_ch, int start_node, int end_node);

  friend std::ostream& operator<<(std::ostream& os, const FSMWithStartEnd& fsm);
};

class CompactFSM {
 public:
  /*!
    \brief Get the epsilon closure of a state.
    \param state_set The current states.
    \param result The epsilon closure of the state. If nullptr,
           then the result will be stored in state_set.
  */
  void GetEpsilonClosure(
      std::unordered_set<int>* state_set, std::unordered_set<int>* result = nullptr
  ) const;

  /*!
   \brief Advance the FSM to the next state.
   \param from The current states.
   \param value The input value.
   \param result The next states, which can be seen as the result of the
   transition.
   \param is_closure Whether from is an epsilon closure.
   \param is_rule Whether the input value is a rule id.
  */
  void Advance(
      const std::vector<int>& from,
      int value,
      std::vector<int>* result,
      bool is_closure = false,
      bool is_rule = false
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

  /*!
    \brief Get all the possible rule numbers for a given node.
    \param node_num The node number.
    \param rules The set of possible rule numbers.s
  */
  void GetPossibleRules(const int& node_num, std::unordered_set<int>* rules) const;
};

class RegexIR {
 public:
  struct Leaf;

  struct Symbol;

  struct Union;

  struct Bracket;

  struct Repeat;

  struct RuleRef;

  static constexpr int KRepeatNoUpperBound = -1;

  using Node = std::variant<Leaf, Symbol, Union, Bracket, Repeat, RuleRef>;

  // This struct is used to store the string in regex, or
  // the character class in regex.
  struct Leaf {
    std::string regex;
    bool is_literal = false;
  };

  // This struct is used to store the symbol in regex, i.e.
  // +, *, ?
  enum class RegexSymbol {
    star,
    plus,
    optional,
  };

  struct Bracket {
    std::vector<Node> nodes;
  };

  struct Symbol {
    RegexSymbol symbol;
    std::shared_ptr<Node> node = nullptr;
  };

  // This struct is used to represent a union symbol.
  struct Union {
    std::vector<Node> nodes;
  };

  struct Repeat {
    std::shared_ptr<Node> node = nullptr;
    int lower_bound = 0;
    int upper_bound = 0;
  };

  struct LookAhead {
    bool is_positive;
    std::vector<Node> nodes;
  };

  struct RuleRef {
    int rule_id;
  };

  // This struct is used to represent a bracket in regex.
  std::vector<Node> nodes;

  /*!
    \brief Constructs a NFA from the regex IR.
  */
  Result<FSMWithStartEnd> Build() const;

  /*!
    \brief the visit function for the variant.
  */
  Result<FSMWithStartEnd> visit(const Leaf& node) const;

  Result<FSMWithStartEnd> visit(const Symbol& node) const;

  Result<FSMWithStartEnd> visit(const Union& node) const;

  Result<FSMWithStartEnd> visit(const Bracket& node) const;

  Result<FSMWithStartEnd> visit(const Repeat& node) const;

  Result<FSMWithStartEnd> visit(const LookAhead& node) const;

  Result<FSMWithStartEnd> visit(const RuleRef& node) const;
};

/*!
  \brief Check repeat in regex. i.e {...} and {...,...}
  \param regex The regex string.
  \param start The start position of the repeat. i.e. regex[start] == '{'.
         After the function, start will be the position of '}'.
  \return The repeat range.
*/
Result<std::pair<int, int>> CheckRepeat(const std::string& regex, size_t& start);

/*!
  \brief Handle escape characters.
  \param regex the corresponding string.
  \param start the pos escape characters start.
  \return int: the length of the escape characters.
  \return std::vector<std::pair<int, int>>: the stored escape character ranges.
*/
std::pair<int, std::vector<std::pair<uint32_t, uint32_t>>> HandleEscapes(
    const std::string& regex, int start
);

/*!
  \brief Build a FSM from a list of patterns.
  \param patterns The patterns to be built.
  \param end_nodes The end nodes of the FSM.
  \return The FSM with start and end states.
*/
FSMWithStartEnd BuildTrie(
    const std::vector<std::string>& patterns, std::vector<int32_t>* end_nodes = nullptr
);

std::ostream& operator<<(std::ostream& os, const FSMWithStartEnd& fsm);

struct FSMState {
  /*! \brief The id of the fsm. */
  int32_t fsm_id = 0;

  /*! \brief The id of the node in the corresponding fsm. */
  int32_t node_id = 0;

  /*! \brief The input position. */
  int32_t input_pos = 0;

  bool operator<(const FSMState& other) const {
    if (fsm_id != other.fsm_id) {
      return fsm_id < other.fsm_id;
    }
    if (node_id != other.node_id) {
      return node_id < other.node_id;
    }
    return input_pos < other.input_pos;
  }

  FSMState() = default;

  FSMState(int32_t fsm_id, int32_t node_id, int32_t input_pos)
      : fsm_id(fsm_id), node_id(node_id), input_pos(input_pos) {}

  static FSMState RuleState(int32_t fsm_id) { return FSMState(fsm_id, -1, -1); }

  std::string ToString() const {
    return "FSMState(fsm_id=" + std::to_string(fsm_id) + ", node_id=" + std::to_string(node_id) +
           ", input_pos=" + std::to_string(input_pos) + ")";
  }
};

class FSMGroup {
 protected:
  /*! \brief It's a mapping from the rule_name to the fsm_id. */
  std::unordered_map<std::string, int32_t> rule_name_to_id_;

  /*! \brief The mapping stores the lookahead fsms. */
  std::unordered_map<int, FSMWithStartEnd> lookahead_fsms_;

  /*! \brief The vector stores the rule names. rule_names[i] stores the name
  of the rule with the fsm_id = i. */
  std::vector<std::string> rule_names_;

  /*! \brief The vector stores the FSMs. */
  std::vector<FSMWithStartEnd> fsms_;

  /*! \brief The id of the root rule. */
  int32_t root_rule_id_;

  /*! \brief Build the mapping from the rule_name to the rule_id.
      \param rule_names A series of names of the rules, repeatation is allowed.
      \param root_rule The name of the root rule.
      \return If the mapping is built successfully, return true; false otherwise. */
  bool BuildNameIdMap(const std::vector<std::string>& rule_names, const std::string& root_rule);

 public:
  FSMGroup() = default;

  /*! \brief Get the size of the FSMGroup. i.e. How many rules are there in the FSMGroup. */
  size_t Size() const { return fsms_.size(); }

  /*! \brief Get the root rule name. */
  const std::string& GetRootRuleName() const { return rule_names_[root_rule_id_]; }

  /*! \brief Get the rule name from the rule id. */
  int32_t GetRuleID(const std::string& rule_name) const { return rule_name_to_id_.at(rule_name); }

  /*! \brief Get the FSM from the rule id. */
  const FSMWithStartEnd& GetFSM(int32_t rule_id) const { return fsms_[rule_id]; }

  std::optional<FSMWithStartEnd> GetLookaheadFSM(int32_t rule_id) const {
    auto it = lookahead_fsms_.find(rule_id);
    if (it != lookahead_fsms_.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  /*! \brief Simplify the FSMGroup. */
  void Simplify();

  /*! \brief Convert the FSMGroup to a minimized DFA.
      \param maximum_nodes The maximum number of nodes in the current NFA.
      \details If the number of nodes of a FSM exceeds the maximum number,
      the function won't do anything, otherwise, the function will convert the NFA
      to a DFA, and then minimize the DFA. */
  void ToMinimizedDFA(int32_t maximum_nodes = 50);

  /*! \brief Get the FSMGroup from the grammar.
      \param grammar The given grammar.
      \param root_rule The root rule.
      \return The FSMGroup. */
  friend Result<FSMGroup> GrammarToFSMs(const std::string& grammar, std::string root_rule);
  friend std::ostream& operator<<(std::ostream& os, const FSMGroup& fsm_group) {
    os << "FSMGroup: " << std::endl;
    os << "size: " << fsm_group.fsms_.size() << std::endl;
    os << "root_rule_id: " << fsm_group.root_rule_id_ << std::endl;
    for (size_t i = 0; i < fsm_group.fsms_.size(); ++i) {
      os << "FSM " << i << ":" << std::endl;
      os << fsm_group.fsms_[i] << std::endl;
    }
    if (!fsm_group.lookahead_fsms_.empty()) {
      os << "lookahead_fsms: " << std::endl;
      for (const auto& [id, fsm] : fsm_group.lookahead_fsms_) {
        os << "lookahed FSM " << id << ":" << std::endl;
        os << fsm << std::endl;
      }
    }
    return os;
  }
  friend class CompactFSMGroup;
};

class CompactFSMGroup {
 private:
  /*! \brief It's a mapping from the rule_name to the fsm_id. */
  std::unordered_map<std::string, int32_t> rule_name_to_id_;

  /*! \brief The vector stores the rule names. rule_names[i] stores the name
  of the rule with the fsm_id = i. */
  std::vector<std::string> rule_names_;

  /*! \brief The vector stores the FSMs. */
  std::vector<CompactFSMWithStartEnd> fsms_;

  /*! \brief The id of the root rule. */
  int32_t root_rule_id_;

 public:
  CompactFSMGroup(const FSMGroup& fsm_group) {
    rule_name_to_id_ = fsm_group.rule_name_to_id_;
    rule_names_ = fsm_group.rule_names_;
    fsms_.reserve(fsm_group.fsms_.size());
    for (const auto& fsm : fsm_group.fsms_) {
      CompactFSMWithStartEnd compact_fsm;
      compact_fsm.fsm = fsm.fsm.ToCompact();
      compact_fsm.start = fsm.start;
      compact_fsm.ends = fsm.ends;
      compact_fsm.is_dfa = fsm.is_dfa;
      fsms_.emplace_back(std::move(compact_fsm));
    }
    root_rule_id_ = fsm_group.root_rule_id_;
  }

  /*! \brief Get the size of the FSMGroup. i.e. How many rules are there in the FSMGroup. */
  size_t Size() const { return fsms_.size(); }

  /*! \brief Get the root rule name. */
  const std::string& GetRootRuleName() const { return rule_names_[root_rule_id_]; }
};

}  // namespace xgrammar

#endif  // XGRAMMAR_FSM_H_
