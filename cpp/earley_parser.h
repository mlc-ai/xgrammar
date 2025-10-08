/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/earley_parser.h
 * \brief The header for the definition of the Earley parser.
 */

#ifndef XGRAMMAR_EARLEY_PARSER_H_
#define XGRAMMAR_EARLEY_PARSER_H_
#include <cstdint>
#include <ostream>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>

#include "grammar_impl.h"
#include "support/compact_2d_array.h"
#include "support/utils.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

/*!
 * \brief The state of the Earley parser.
 * In the implementation, a rule can only be a kchoices or a ktagdispatch.
 * A kchoices rule must be composed of some ksequence rules, or a kemptyrule.
 * In the ksequence, every element in the sequence must be a kbytestring, a
 * kcharacterclass, a kcharacterclassstar, or a rule reference.
 *
 * - rule_id: The id of the rule.
 * - sequence_id: The id of the sequence in the rule.
 * - element_id: The id of the element in the sequence, or the id of the node in
 *   the tag dispatch fsm.
 * - rule_start_pos: The id of the parent node in the Earley parser. i.e. the rule
 *   is predicted from the k-th character.
 * - sub_element_id: The id of the sub element in the current element, i.e.:
 *   - kbytestring: the id of the byte in the string.
 *   - kcharacterclass: How many bytes are left to be read in the utf8 character.
 *   - kcharacterclassstar: How many bytes are left to be read in the utf8 character.
 */
struct ParserState {
  constexpr ParserState() = default;

  constexpr ParserState(
      const int32_t& rule_id,
      const int32_t& sequence_id,
      const int32_t& element_id,
      const int32_t& rule_start_pos,
      const int32_t& sub_element_id,
      const int32_t& repeat_count = 0
  )
      : rule_id(rule_id),
        sequence_id(sequence_id),
        element_id(element_id),
        rule_start_pos(rule_start_pos),
        sub_element_id(sub_element_id),
        repeat_count(repeat_count) {}

  /*!
   * \brief A sequence_id value of kUnexpandedRuleStartSequenceId means a rule hasn't been
   * expanded.
   */
  static constexpr int32_t kUnexpandedRuleStartSequenceId = 128000;

  /*!
   * \brief A parent_id value of kNoParent means this ParserState is the root of the parsing stack.
   */
  static constexpr int32_t kNoPrevInputPos = -1;

  /*! \brief A sequence_id value of kInvalid means the ParserState is invalid. */
  static constexpr int32_t kInvalidSequenceId = -1;

  /*! \brief The rule's id. */
  int32_t rule_id = -1;

  /*! \brief Which choice in this rule is selected. */
  int32_t sequence_id = -1;

  /*!
   * \brief Which element of the choice sequence is to be visited. When the current sequence is
   * a tag dispatch rule, this element id is the current node.
   */
  int32_t element_id = -1;

  /*! \brief The position of the state, i.e. from which position, the rule starts. */
  int32_t rule_start_pos = -1;

  /*! \brief The id of the sub element in the current selement of the sequence. */
  int32_t sub_element_id = 0;

  /*! \brief The number of times the element is repeated. It will be used in kRepeat.*/
  int32_t repeat_count = 0;

  /*! \brief The element is invalid when sequence_id is -1. */
  bool IsInvalid() const { return sequence_id == -1; }

  static ParserState GetInvalidState() { return {-1, -1, -1, -1, -1}; }

  bool operator==(const ParserState& other) const {
    return rule_id == other.rule_id && sequence_id == other.sequence_id &&
           element_id == other.element_id && sub_element_id == other.sub_element_id;
  }

  bool operator<(const ParserState& other) const {
    if (rule_id != other.rule_id) return rule_id < other.rule_id;
    if (sequence_id != other.sequence_id) return sequence_id < other.sequence_id;
    if (element_id != other.element_id) return element_id < other.element_id;
    if (rule_start_pos != other.rule_start_pos) return rule_start_pos < other.rule_start_pos;
    if (sub_element_id != other.sub_element_id) return sub_element_id < other.sub_element_id;
    return repeat_count < other.repeat_count;
  }

  friend std::ostream& operator<<(std::ostream& os, const ParserState& state) {
    os << state.ToString();
    return os;
  }

  std::string ToString() const {
    return "ParserState(rule_id=" + std::to_string(rule_id) +
           ", sequence_id=" + std::to_string(sequence_id) +
           ", element_id=" + std::to_string(element_id) +
           ", rule_start_pos=" + std::to_string(rule_start_pos) +
           ", sub_element_id=" + std::to_string(sub_element_id) + ")";
  }
};

XGRAMMAR_MEMBER_ARRAY(
    ParserState,
    &ParserState::rule_id,
    &ParserState::sequence_id,
    &ParserState::element_id,
    &ParserState::rule_start_pos,
    &ParserState::sub_element_id,
    &ParserState::repeat_count
);

/*!
 * \brief When getting the mask of the state, we don't need to consider the rule_start_pos.
 */
class StateHashForCache {
 public:
  size_t operator()(const ParserState& state) const {
    return HashCombine(state.rule_id, state.sequence_id, state.element_id, state.sub_element_id);
  }
};

/*!
 * \brief When matching the state, we need to consider the rule_start_pos, since if two states
 * don't have the same rule_start_pos, they are not the same state.
 */
class StateEqualForParsing {
 public:
  bool operator()(const ParserState& lhs, const ParserState& rhs) const {
    return lhs.rule_id == rhs.rule_id && lhs.sequence_id == rhs.sequence_id &&
           lhs.element_id == rhs.element_id && lhs.rule_start_pos == rhs.rule_start_pos &&
           lhs.sub_element_id == rhs.sub_element_id && lhs.repeat_count == rhs.repeat_count;
  }
};

/*!
 * \brief This class is used to hash the ParserState for parsing.
 * If two ParserStates don't have the same rule_start_pos, they are not the same state.
 */
class StateHashForParsing {
 public:
  size_t operator()(const ParserState& state) const {
    return HashCombine(
        state.rule_id,
        state.sequence_id,
        state.element_id,
        state.rule_start_pos,
        state.sub_element_id,
        state.repeat_count
    );
  }
};

/*! \brief This class is used to detect the repeated states. */
class RepeatDetector {
 private:
  const int transition_threshold_;

  std::vector<ParserState> visited_vector_;

  std::unordered_set<ParserState, StateHashForParsing, StateEqualForParsing> visited_set_;

  int size_ = 0;

 public:
  RepeatDetector(const int transition_threshold = 50)
      : transition_threshold_(transition_threshold), size_(0) {
    visited_vector_.resize(transition_threshold_);
  }

  /*!
   * \brief Check if the element is visited.
   * \return True if visited, false otherwise.
   */
  bool IsVisited(const ParserState& state) const;

  /*!
   * \brief Add the state into the visited states.
   * \param state The state to be added.
   */
  void Insert(const ParserState& state);

  /*! \brief Reset the detector. */
  void Clear();
};

class EarleyParser {
  /*!
   * \brief Here is an article about Earley Parser.
   * https://en.wikipedia.org/wiki/Earley_parser#Pseudocode
   * We divide the parser states into three categories:
   * - Scanable (which will be stored in scanable_state_history_).
   * - Predictable(If it predict a new rule successfully, then it will be stored in
   * rule_id_to_completable_states).
   * - completable(which can perform a completion operation).
   * A state will be stored in rule_id_to_completable_states_ if it can be completed,
   * and it will be stored in scanable_state_history_ if it can be scanned. Otherwise,
   * it will be discarded.
   */
 protected:
  using GrammarExpr = Grammar::Impl::GrammarExpr;

  /*! \brief The grammar to be parsed. */
  Grammar grammar_;

  /*! \brief In this round of advancing, check if the stop token can be accepted. */
  bool tmp_accept_stop_token_ = false;

  /*! \brief store when accepting i characters, if the stop token can be accepted. */
  std::vector<bool> is_completed_;

  /*!
   * \brief rule_id_to_completable_states[i][j] is the i pos j rule_id states. Earley
   * parser needs it to complete.
   */
  Compact2DArray<std::pair<int32_t, ParserState>> rule_id_to_completable_states_;

  /*!
   * \brief The states history. state_stack[i] is a vector storing the states after accepting the
   * input[i-1].
   */
  Compact2DArray<ParserState> scanable_state_history_;

  /*!
   * \brief A temperate vector only used in Advance, used to add states in the
   * scanable_state_history.
   */
  std::vector<ParserState> tmp_states_to_be_added_;

  /*! \brief It's the processing queue of the earley parser. */
  std::queue<ParserState> tmp_process_state_queue_;

  /*! \brief The class is used to check if a state has been added into the queue. */
  RepeatDetector tmp_states_visited_in_queue_;

  /*! \brief Check if the stop token is accepted. */
  bool stop_token_is_accepted_ = false;

  /*!
   * \brief Check if the state has been added into the queue.
   * \param state The state to check.
   * \return True if in the vector, false otherwise.
   */
  bool IsStateVisitedInQueue(const ParserState& state) const {
    return tmp_states_visited_in_queue_.IsVisited(state);
  }

  /*!
   * \brief The scanning operation of the Earley parser. Put the new states in the queue.
   */
  void Scan(const ParserState& state, const uint8_t ch);

  /*!
   * \brief The completion operation of the Earley parser.
   * \param state The state to be completed.
   * \param debug_print Whether to print the debug information.
   * \details The reason is that if the state can't be scanned, then
   * add it into the next states is useless. Moreover, the end
   * of the grammar is used to check if the grammar is completed,
   * so it should be added into the next states.
   */
  void Complete(const ParserState& state, bool debug_print = false);

  /*!
   * \brief The prediction operation of the Earley parser.
   * \param state The state to be predicted.
   * \param debug_print Whether to print the debug information.
   * \return First: If the state scanable, or the state is the end of the grammar,
   * then return true, otherwise return false.
   * \return Second: If the state is completable, then return true, otherwise return false.
   */
  std::pair<bool, bool> Predict(const ParserState& state, bool debug_print = false);

  /*!
   * \brief Handle the unexpanded rule, used for pushing initial state.
   * \param state The state to be handled.
   * \return True if the rule is unexpanded, false otherwise.
   */
  bool ExpandAndEnqueueUnexpandedState(const ParserState& state);

  /*!
   * \brief Expand the rule, used for RuleRef and kTagDispatch.
   * \param state The state to be expanded, which is the parent state.
   * The type of the state is kTagDispatch or kSequence. Moreover, the
   * element of the sequence should be a rule reference; the node in
   * the kTagDispatch should be an end node.
   * \param grammar_expr The grammar expression to be expanded.
   * \param sub_grammar_expr The sub grammar expression to be expanded, especially
   * when the rule is a kSequence, and the sub rule is a kRuleRef.
   * \param debug_print Whether to print the debug information.
   */
  void ExpandNextRuleRefElement(
      const ParserState& state,
      const GrammarExpr& grammar_expr,
      const GrammarExpr* sub_grammar_expr,
      bool debug_print = false
  );

  /*!
   * \brief Expand the rule, used for RuleRef and kTagDispatch.
   * \param state The state to be expanded, and it's should be on the FSM.
   * \param debug_print Whether to print the debug information.
   */
  void ExpandNextRuleRefElementOnFSM(const ParserState& state, bool debug_print = false);

  /*!
   * \brief Advance the parser to the next state, with the sub sequence is kCharacterClass.
   * \param state The state to be advanced.
   * \param ch The character to be advanced.
   * \param sub_sequence The sub sequence to be checked.
   * \return The next state, Invalid state if the character is not accepted.
   */
  void AdvanceCharacterClass(
      const ParserState& state, const uint8_t ch, const GrammarExpr& sub_sequence
  );

  /*!
   * \brief Advance the parser to the next state, with the sub sequence is kByteString.
   * \param state The state to be advanced.
   * \param ch The character to be advanced.
   * \param sub_sequence The sub sequence to be checked.
   * \return The next state, Invalid state if the character is not accepted.
   */
  void AdvanceByteString(
      const ParserState& state, const uint8_t ch, const GrammarExpr& sub_sequence
  );

  /*!
   * \brief Advance the parser to the next state, with the sub sequence is kCharacterClassStar.
   * \param state The state to be advanced.
   * \param ch The character to be advanced.
   * \param sub_sequence The sub sequence to be checked.
   * \return The next state, Invalid state if the character is not accepted.
   */
  void AdvanceCharacterClassStar(
      const ParserState& state, const uint8_t ch, const GrammarExpr& sub_sequence
  );

  /*!
   * \brief Advance the parser to the next state, with the sequence is kTagDispatch.
   * \param state The state to be advanced.
   * \param ch The character to be advanced.
   * \param cur_sequence The sequence of the current state.
   * \return The next state, Invalid state if the character is not accepted.
   */
  void AdvanceFsm(const ParserState& state, const uint8_t ch);

  /*!
   * \brief Enqueue the state into the queue.
   * \param state The state to be enqueued.
   * \details The state is enqueued if it is not visited in the queue.
   */
  void Enqueue(const ParserState& state) {
    if (!IsStateVisitedInQueue(state)) {
      tmp_process_state_queue_.push(state);
      tmp_states_visited_in_queue_.Insert(state);
    }
  }

  /*!
   * \brief Enqueue the state into the queue, without prediction and completion.
   * \param state The state to be enqueued.
   */
  void EnqueueWithoutProcessing(const ParserState& state) {
    if (!IsStateVisitedInQueue(state)) {
      tmp_states_visited_in_queue_.Insert(state);
      tmp_states_to_be_added_.push_back(state);
    }
  }

 public:
  /*!
   * \brief Constructor of the Earley parser.
   * \param grammar The grammar to be parsed.
   * \param initial_state The initial state to be pushed into the parser.
   */
  EarleyParser(
      const Grammar& grammar, const ParserState& initial_state, const bool need_expand = true
  );

  /*!
   * \brief From the current states, advance to the next state.
   * \param ch The character to be advanced.
   * \param debug_print Whether to print the debug information.
   * \return True if the character is accepted, false otherwise.
   * \note If the character isn't accepted, then the states won't be changed.
   */
  bool Advance(const uint8_t ch, bool debug_print = false);

  /*!
   * \brief Remove the newly added states.
   * \param count The number of states to be removed.
   */
  void PopLastStates(int32_t count = 1);

  /*!
   * \brief Check whether any of the multiple states stored in the parser has already completed.
   * \note Since the parser contains multiple parallel states, some may have already completed,
   * while others might still be able to accept more characters.
   * \return True if the root rule is completed, false otherwise.
   */
  bool IsCompleted() const;

  /*!
   * \brief Push the initial state into the Earley parser.
   * \param state The initial state to be pushed.
   */
  void PushStateAndExpand(const ParserState& state);

  /*!
   * \brief Reset the parser.
   * \note This function is used to reset the parser, and initialize the
   * parser with the root rule.
   */
  void Reset();

  /*!
   * \brief Get the current scanable states.
   * \return The scanable states.
   */
  std::vector<ParserState> GetLatestScanableStates() const {
    std::vector<ParserState> latest_states;
    for (const auto& state : scanable_state_history_[scanable_state_history_.size() - 1]) {
      latest_states.push_back(state);
    }
    return latest_states;
  }

  /*!
   * \brief Push one state to check if it can accept the token.
   * \param state The state to be pushed.
   */
  void PushOneStateToCheck(const ParserState& state) {
    rule_id_to_completable_states_.PushBack(std::vector<std::pair<int32_t, ParserState>>());
    is_completed_.push_back(is_completed_.back());
    scanable_state_history_.PushBack(&state, 1);
    return;
  }

  std::string PrintStates() const {
    std::string result;
    result += "There are " + std::to_string(scanable_state_history_.size()) +
              " steps in history. Last step: [\n";
    for (const auto& state : scanable_state_history_[scanable_state_history_.size() - 1]) {
      result += state.ToString() + ", \n";
    }
    result += "]";
    return result;
  }
};

}  // namespace xgrammar

#endif  // XGRAMMAR_EARLEY_PARSER_H_
