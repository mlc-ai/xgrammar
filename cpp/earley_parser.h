/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/earley_parser.h
 * \brief The header for the definition of the Earley parser.
 */

#ifndef XGRAMMAR_EARLEY_PARSER_H_
#define XGRAMMAR_EARLEY_PARSER_H_
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
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
      const int32_t& repeat_count = 0,
      const int32_t& partial_codepoint = 0
  )
      : rule_id(rule_id),
        sequence_id(sequence_id),
        element_id(element_id),
        rule_start_pos(rule_start_pos),
        sub_element_id(sub_element_id),
        repeat_count(repeat_count),
        partial_codepoint(partial_codepoint) {}

  /*!
   * \brief A rule_start_pos value of kNoPrevInputPos means this ParserState is the root of the
   * parsing stack.
   */
  static constexpr int32_t kNoPrevInputPos = -1;

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

  /*! \brief The id of the sub element in the current element of the sequence. */
  int32_t sub_element_id = 0;

  /*! \brief The number of times the element is repeated. It will be used in kRepeat.*/
  int32_t repeat_count = 0;

  /*! \brief Partial codepoint accumulated during UTF-8 decoding for positive character classes. */
  int32_t partial_codepoint = 0;

  /*!
   * \brief Lexicographic order over all fields. It is only used to sort the states for
   * deterministic serialization, and is not needed during parsing.
   */
  bool operator<(const ParserState& other) const {
    if (rule_id != other.rule_id) return rule_id < other.rule_id;
    if (sequence_id != other.sequence_id) return sequence_id < other.sequence_id;
    if (element_id != other.element_id) return element_id < other.element_id;
    if (rule_start_pos != other.rule_start_pos) return rule_start_pos < other.rule_start_pos;
    if (sub_element_id != other.sub_element_id) return sub_element_id < other.sub_element_id;
    if (repeat_count != other.repeat_count) return repeat_count < other.repeat_count;
    return partial_codepoint < other.partial_codepoint;
  }

  friend std::ostream& operator<<(std::ostream& os, const ParserState& state) {
    os << state.ToString();
    return os;
  }

  std::string ToString() const {
    std::string result = "ParserState(rule_id=" + std::to_string(rule_id) +
                         ", sequence_id=" + std::to_string(sequence_id) +
                         ", element_id=" + std::to_string(element_id) +
                         ", rule_start_pos=" + std::to_string(rule_start_pos) +
                         ", sub_element_id=" + std::to_string(sub_element_id);
    if (repeat_count != 0) {
      result += ", repeat_count=" + std::to_string(repeat_count);
    }
    if (partial_codepoint != 0) {
      result += ", partial_codepoint=" + std::to_string(partial_codepoint);
    }
    result += ")";
    return result;
  }
};

XGRAMMAR_MEMBER_ARRAY(
    ParserState,
    &ParserState::rule_id,
    &ParserState::sequence_id,
    &ParserState::element_id,
    &ParserState::rule_start_pos,
    &ParserState::sub_element_id,
    &ParserState::repeat_count,
    &ParserState::partial_codepoint
);

/*!
 * \brief Hash of a state used as the key of the adaptive token mask cache. The token mask of a
 * state does not depend on rule_start_pos, repeat_count or partial_codepoint, so they are
 * ignored. Pairs with StateEqualForCache.
 */
class StateHashForCache {
 public:
  size_t operator()(const ParserState& state) const {
    return HashCombine(state.rule_id, state.sequence_id, state.element_id, state.sub_element_id);
  }
};

/*!
 * \brief Equality of states used as the key of the adaptive token mask cache. Compares the same
 * fields as StateHashForCache hashes.
 */
class StateEqualForCache {
 public:
  bool operator()(const ParserState& lhs, const ParserState& rhs) const {
    return lhs.rule_id == rhs.rule_id && lhs.sequence_id == rhs.sequence_id &&
           lhs.element_id == rhs.element_id && lhs.sub_element_id == rhs.sub_element_id;
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
           lhs.sub_element_id == rhs.sub_element_id && lhs.repeat_count == rhs.repeat_count &&
           lhs.partial_codepoint == rhs.partial_codepoint;
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
        state.repeat_count,
        state.partial_codepoint
    );
  }
};

/*!
 * \brief Immutable Earley-parser metadata derived solely from an optimized grammar.
 *
 * A compiled grammar creates thousands of short-lived EarleyParser instances while building
 * adaptive token masks. Keeping these tables in each parser repeats the same state classification,
 * nullable-rule initialization, and deterministic byte-transition construction for every mask
 * task. Build them once and share them between compile-time tasks and runtime matchers.
 */
struct EarleyParserGrammarMetadata {
  enum FsmStateFlag : uint8_t {
    kFsmStateInitialized = 1 << 0,
    kFsmStateScanable = 1 << 1,
    kFsmStateNonTerminal = 1 << 2,
    kFsmStateEnd = 1 << 3,
    kFsmStateHasEdges = 1 << 4,
  };

  /*! \brief Precomputed FSM state properties, indexed by complete-FSM state id. */
  std::vector<uint8_t> fsm_state_flags;

  /*! \brief Whether each rule can match the empty string. */
  std::vector<uint8_t> rule_is_nullable;

  /*!
   * \brief Whether a rule or any transitively referenced rule can observe an atomic token ID.
   */
  std::vector<uint8_t> rule_has_atomic_token_edges;

  /*!
   * \brief State-to-table mapping for deterministic states with several character edges.
   * A value of -1 means AdvanceFsm should scan the state's edges.
   */
  std::vector<int32_t> deterministic_byte_transition_ids;

  /*! \brief Dense byte transitions selected by deterministic_byte_transition_ids. */
  std::vector<std::array<int32_t, 256>> deterministic_byte_transitions;

  EarleyParserGrammarMetadata() = default;
  explicit EarleyParserGrammarMetadata(const Grammar& grammar);

  friend std::size_t MemorySize(const EarleyParserGrammarMetadata& metadata) {
    return MemorySize(metadata.fsm_state_flags) + MemorySize(metadata.rule_is_nullable) +
           MemorySize(metadata.rule_has_atomic_token_edges) +
           MemorySize(metadata.deterministic_byte_transition_ids) +
           MemorySize(metadata.deterministic_byte_transitions);
  }
};

/*! \brief This class is used to detect the repeated states. */
class RepeatDetector {
 private:
  const int transition_threshold_;

  std::vector<ParserState> visited_vector_;

  std::unordered_set<ParserState, StateHashForParsing, StateEqualForParsing> visited_set_;

  int size_ = 0;
  bool using_set_ = false;

  const ParserState* InsertInSet(const ParserState& state);

  void ClearSet();

 public:
  RepeatDetector(const int transition_threshold = 50)
      : transition_threshold_(transition_threshold), size_(0) {
    visited_vector_.resize(transition_threshold_);
  }

  /*! \brief Insert a state only if absent and return its stable address, or nullptr. */
  const ParserState* InsertIfAbsent(const ParserState& state) {
    if (!using_set_ && size_ < transition_threshold_) {
      for (int i = 0; i < size_; ++i) {
        if (StateEqualForParsing()(state, visited_vector_[i])) {
          return nullptr;
        }
      }
      visited_vector_[size_] = state;
      return &visited_vector_[size_++];
    }
    return InsertInSet(state);
  }

  /*! \brief Insert a copy of an FSM state with a new element id. */
  const ParserState* InsertFsmTransitionIfAbsent(
      const ParserState& state, int32_t target_element_id
  ) {
    if (!using_set_ && size_ < transition_threshold_) {
      for (int i = 0; i < size_; ++i) {
        const ParserState& existing = visited_vector_[i];
        if (existing.rule_id == state.rule_id && existing.sequence_id == state.sequence_id &&
            existing.element_id == target_element_id &&
            existing.rule_start_pos == state.rule_start_pos &&
            existing.sub_element_id == state.sub_element_id &&
            existing.repeat_count == state.repeat_count &&
            existing.partial_codepoint == state.partial_codepoint) {
          return nullptr;
        }
      }
      ParserState* inserted = &visited_vector_[size_++];
      *inserted = state;
      inserted->element_id = target_element_id;
      return inserted;
    }
    ParserState transitioned = state;
    transitioned.element_id = target_element_id;
    return InsertInSet(transitioned);
  }

  /*! \brief Reset the detector. */
  void Clear() {
    if (using_set_) {
      ClearSet();
    }
    size_ = 0;
    using_set_ = false;
  }
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

  /*! \brief Direct access to the shared complete-FSM edge table. */
  const Compact2DArray<FSMEdge>* complete_fsm_edges_;

  /*! \brief Grammar-only metadata shared by all parsers for one compiled grammar. */
  const EarleyParserGrammarMetadata* grammar_metadata_;

  /*! \brief In this round of advancing, check if the stop token can be accepted. */
  bool tmp_accept_stop_token_ = false;

  /*! \brief store when accepting i characters, if the stop token can be accepted. */
  std::vector<uint8_t> is_completed_;

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
   * \brief A temporary vector only used in Advance, used to add states in the
   * scanable_state_history.
   */
  std::vector<const ParserState*> tmp_states_to_be_added_;

  /*! \brief Stable pointers to visited states awaiting prediction/completion. */
  std::queue<const ParserState*> tmp_process_state_queue_;

  /*! \brief The class is used to check if a state has been added into the queue. */
  RepeatDetector tmp_states_visited_in_queue_;

  /*! \brief Check if the stop token is accepted. */
  bool stop_token_is_accepted_ = false;

  using FsmStateFlag = EarleyParserGrammarMetadata::FsmStateFlag;
  static constexpr uint8_t kFsmStateInitialized = EarleyParserGrammarMetadata::kFsmStateInitialized;
  static constexpr uint8_t kFsmStateScanable = EarleyParserGrammarMetadata::kFsmStateScanable;
  static constexpr uint8_t kFsmStateNonTerminal = EarleyParserGrammarMetadata::kFsmStateNonTerminal;
  static constexpr uint8_t kFsmStateEnd = EarleyParserGrammarMetadata::kFsmStateEnd;
  static constexpr uint8_t kFsmStateHasEdges = EarleyParserGrammarMetadata::kFsmStateHasEdges;

  struct ByteTransitionCacheSlot {
    int32_t state_id = -1;
    const std::array<int32_t, 256>* targets = nullptr;
  };

  /*! \brief Parser-local pointer front-end for the shared deterministic transition tables. */
  std::array<ByteTransitionCacheSlot, 64> byte_transition_direct_cache_{};

  const std::array<int32_t, 256>* GetDeterministicByteTransitions(int32_t state_id) {
    auto& slot = byte_transition_direct_cache_
        [static_cast<uint32_t>(state_id) % byte_transition_direct_cache_.size()];
    if (slot.state_id == state_id) {
      return slot.targets;
    }
    const int32_t transition_id = grammar_metadata_->deterministic_byte_transition_ids[state_id];
    const auto* targets = transition_id >= 0
                              ? &grammar_metadata_->deterministic_byte_transitions[transition_id]
                              : nullptr;
    slot = {state_id, targets};
    return targets;
  }

  /*! \brief Return precomputed properties for a state in a per-rule FSM. */
  uint8_t GetFsmStateFlags(int32_t rule_id, int32_t state_id) const {
    XGRAMMAR_DCHECK(rule_id >= 0 && rule_id < grammar_->NumRules());
    XGRAMMAR_DCHECK(
        state_id >= 0 && state_id < static_cast<int32_t>(grammar_metadata_->fsm_state_flags.size())
    );
    return grammar_metadata_->fsm_state_flags[state_id];
  }

  bool IsRuleNullable(int32_t rule_id) const {
    return grammar_metadata_->rule_is_nullable[rule_id] != 0;
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

  /*! \brief The initial state expanded from the root rule of the grammar. */
  ParserState RootInitialState() const;

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
   * \note The advanced states are enqueued; nothing is enqueued if the character is not accepted.
   */
  void AdvanceCharacterClass(
      const ParserState& state, const uint8_t ch, const GrammarExpr& sub_sequence
  );

  /*!
   * \brief Advance the parser to the next state, with the sub sequence is kByteString.
   * \param state The state to be advanced.
   * \param ch The character to be advanced.
   * \param sub_sequence The sub sequence to be checked.
   * \note The advanced states are enqueued; nothing is enqueued if the character is not accepted.
   */
  void AdvanceByteString(
      const ParserState& state, const uint8_t ch, const GrammarExpr& sub_sequence
  );

  /*!
   * \brief Advance the parser to the next state, with the sub sequence is kCharacterClassStar.
   * \param state The state to be advanced.
   * \param ch The character to be advanced.
   * \param sub_sequence The sub sequence to be checked.
   * \note The advanced states are enqueued; nothing is enqueued if the character is not accepted.
   */
  void AdvanceCharacterClassStar(
      const ParserState& state, const uint8_t ch, const GrammarExpr& sub_sequence
  );

  /*!
   * \brief Advance the parser to the next state, with the sequence is kTagDispatch.
   * \param state The state to be advanced.
   * \param ch The character to be advanced.
   * \note The advanced states are enqueued; nothing is enqueued if the character is not accepted.
   */
  void AdvanceFsm(const ParserState& state, const uint8_t ch);

  /*!
   * \brief Scan a token edge: check if token_id matches any kToken or kExcludeToken edge from
   * state.
   */
  void ScanAtomicToken(const ParserState& state, int32_t token_id);

  /*!
   * \brief Advance the parser by accepting a whole token via kToken/kExcludeToken edges.
   * \param token_id The token ID to accept.
   * \param debug_print Whether to print debug info.
   * \return True if any state advanced, false otherwise.
   */
  bool AdvanceAtomicToken(int32_t token_id, bool debug_print = false);

  /*!
   * \brief Enqueue the state into the queue.
   * \param state The state to be enqueued.
   * \details The state is enqueued if it is not visited in the queue.
   */
  void Enqueue(const ParserState& state) {
    if (const ParserState* inserted = tmp_states_visited_in_queue_.InsertIfAbsent(state)) {
      tmp_process_state_queue_.push(inserted);
    }
  }

  /*!
   * \brief Enqueue the state into the queue, without prediction and completion.
   * \param state The state to be enqueued.
   */
  void EnqueueWithoutProcessing(const ParserState& state) {
    if (const ParserState* inserted = tmp_states_visited_in_queue_.InsertIfAbsent(state)) {
      tmp_states_to_be_added_.push_back(inserted);
    }
  }

  void EnqueueFsmTransition(const ParserState& state, int32_t target_element_id) {
    if (const ParserState* inserted =
            tmp_states_visited_in_queue_.InsertFsmTransitionIfAbsent(state, target_element_id)) {
      tmp_process_state_queue_.push(inserted);
    }
  }

  void EnqueueFsmTransitionWithoutProcessing(const ParserState& state, int32_t target_element_id) {
    if (const ParserState* inserted =
            tmp_states_visited_in_queue_.InsertFsmTransitionIfAbsent(state, target_element_id)) {
      tmp_states_to_be_added_.push_back(inserted);
    }
  }

 public:
  /*!
   * \brief Constructor of the Earley parser.
   * \param grammar The grammar to be parsed. It must be optimized.
   * \param initial_state The state to start parsing from. If not provided, parsing starts
   * from the root rule of the grammar.
   */
  EarleyParser(
      const Grammar& grammar,
      const EarleyParserGrammarMetadata& grammar_metadata,
      std::optional<ParserState> initial_state = std::nullopt,
      bool need_expand = true
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
    rule_id_to_completable_states_.PushBackEmpty();
    is_completed_.push_back(is_completed_.back());
    scanable_state_history_.PushBack(&state, 1);
    return;
  }

  /*! \brief Push several already-expanded states as one parser frontier. */
  void PushStatesToCheck(const std::vector<ParserState>& states) {
    XGRAMMAR_DCHECK(!states.empty());
    rule_id_to_completable_states_.PushBackEmpty();
    is_completed_.push_back(is_completed_.back());
    scanable_state_history_.PushBack(states);
  }

  bool HasLatestScanableStates() const {
    return scanable_state_history_[scanable_state_history_.size() - 1].size() != 0;
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
