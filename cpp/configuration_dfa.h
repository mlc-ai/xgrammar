/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/configuration_dfa.h
 * \brief A DFA over the configurations of the Earley parser, built on demand, and a cursor that
 * runs a parser through it.
 */

#ifndef XGRAMMAR_CONFIGURATION_DFA_H_
#define XGRAMMAR_CONFIGURATION_DFA_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "earley_parser.h"
#include "grammar_impl.h"
#include "support/logging.h"

namespace xgrammar {

/*!
 * \brief A DFA whose states are Earley parser configurations and whose transitions are bytes.
 * \details Checking the tokens of a vocabulary brings the parser to the same configurations
 * again and again: tokens share substrings, and so do the positions they are checked at. For
 * the grammars accepted by Supports(), the configuration (EarleyParser::SerializeConfiguration)
 * determines which bytes the parser accepts next and which configuration each of them leads
 * to. A transition is therefore computed by the parser once, and is a table lookup after that.
 *
 * The DFA only stores states and transitions. ConfigurationDfaCursor fills it in. A state is
 * identified by its whole serialized configuration, not by a hash of it, and takes 1 KiB of
 * transitions plus the configuration, which is usually a few hundred bytes.
 */
class ConfigurationDfa {
 public:
  /*! \brief The transition has not been computed yet. */
  static constexpr int32_t kUnknown = -2;
  /*! \brief The parser does not accept the byte. */
  static constexpr int32_t kRejected = -1;

  /*! \param max_states The number of states after which no more states are added. */
  explicit ConfigurationDfa(int32_t max_states) : max_states_(max_states) {}

  /*!
   * \brief Whether the configuration of a parser of this grammar determines its later scans.
   * \details That is not the case for token and character budgets, token edges, captures, lazy
   * rules, suffix/stop rules and tag dispatch: with them, the parser also reads state that is
   * not part of the configuration.
   */
  static bool Supports(const Grammar& grammar) {
    using GrammarExprType = Grammar::Impl::GrammarExprType;
    if (grammar->has_token_edges) return false;
    for (int32_t rule_id = 0; rule_id < grammar->NumRules(); ++rule_id) {
      const auto& rule = grammar->GetRule(rule_id);
      if (rule.max_tokens >= 0 || rule.max_chars >= 0 || rule.is_lazy ||
          !rule.capture_name.empty() || grammar->GetSuffixStopInfo(rule_id) != nullptr ||
          grammar->GetGrammarExpr(rule.body_expr_id).type == GrammarExprType::kTagDispatch) {
        return false;
      }
      // has_token_edges only covers the rule bodies. A lookahead assertion can contain token
      // elements too, and the parser reports that it reached one through a flag that a table
      // lookup would not set.
      if (rule.lookahead_assertion_id == -1) continue;
      for (int32_t element_id : grammar->GetGrammarExpr(rule.lookahead_assertion_id)) {
        const auto element_type = grammar->GetGrammarExpr(element_id).type;
        if (element_type == GrammarExprType::kToken ||
            element_type == GrammarExprType::kExcludeToken) {
          return false;
        }
      }
    }
    return true;
  }

  int32_t NumStates() const { return static_cast<int32_t>(is_completed_.size()); }

  int32_t MaxStates() const { return max_states_; }

  void Clear() {
    state_ids_.clear();
    transitions_.clear();
    is_completed_.clear();
    lookahead_states_.clear();
  }

  /*!
   * \brief Find the state of the parser's current configuration, adding it if it is new.
   * \return The state, or kUnknown if the configuration is too large to serialize or the DFA
   * already has MaxStates() states.
   */
  int32_t FindOrAddState(const EarleyParser& parser) {
    if (!parser.SerializeConfiguration(&key_)) return kUnknown;
    auto it = state_ids_.find(key_);
    if (it != state_ids_.end()) return it->second;
    if (NumStates() >= max_states_) return kUnknown;
    const int32_t state = NumStates();
    state_ids_.emplace(key_, state);
    transitions_.resize(transitions_.size() + 256, kUnknown);
    is_completed_.push_back(parser.IsCompleted());
    return state;
  }

  /*! \brief Whether the parser is completed in this state. */
  bool IsCompleted(int32_t state) const { return is_completed_[state]; }

  /*! \brief The state after a byte, kRejected, or kUnknown. */
  int32_t GetTransition(int32_t state, uint8_t byte) const {
    return transitions_[state * 256 + byte];
  }

  void SetTransition(int32_t state, uint8_t byte, int32_t next_state) {
    transitions_[state * 256 + byte] = next_state;
  }

  /*!
   * \brief The state right after a lookahead assertion is pushed, by the id of the assertion.
   * The pushed state has no parent row, so this state does not depend on what is below it.
   * \return kUnknown if it has not been recorded.
   */
  int32_t GetLookaheadState(int32_t lookahead_assertion_id) const {
    auto it = lookahead_states_.find(lookahead_assertion_id);
    return it == lookahead_states_.end() ? kUnknown : it->second;
  }

  void SetLookaheadState(int32_t lookahead_assertion_id, int32_t state) {
    lookahead_states_[lookahead_assertion_id] = state;
  }

 private:
  int32_t max_states_;
  std::unordered_map<std::string, int32_t> state_ids_;
  /*! \brief transitions_[state * 256 + byte]. */
  std::vector<int32_t> transitions_;
  std::vector<uint8_t> is_completed_;
  std::unordered_map<int32_t, int32_t> lookahead_states_;
  /*! \brief The key of the configuration being looked up. A member to reuse its storage. */
  std::string key_;
};

/*!
 * \brief Advances and rolls back an Earley parser through a ConfigurationDfa.
 * \details The cursor offers the parser operations that the token loops use: Advance, Pop and
 * IsCompleted. It keeps the DFA state after every byte, and it answers from the DFA as long as
 * the transitions are known. The parser is left behind meanwhile. When a transition is not
 * known, the cursor first replays on the parser the bytes it has not seen, then lets the parser
 * compute the transition and records it.
 *
 * If a configuration cannot be added to the DFA, the cursor leaves the DFA: the parser is up to
 * date at that point, and every later operation goes straight to it. A cursor that was started
 * without a DFA works the same way from the beginning. The results are the same in all cases;
 * only the amount of parser work differs.
 */
class ConfigurationDfaCursor {
 public:
  /*!
   * \brief Start at the parser's current configuration.
   * \param parser The parser. Until the cursor has popped everything it advanced, the parser
   * must only be changed through the cursor.
   * \param dfa The DFA to use, or nullptr to work on the parser alone.
   */
  void Start(EarleyParser* parser, ConfigurationDfa* dfa) {
    parser_ = parser;
    dfa_ = nullptr;
    states_.clear();
    operations_.clear();
    num_parser_operations_ = 0;
    if (dfa == nullptr) return;
    // The entries only save work, so the DFA can be emptied at any time. Doing it here keeps
    // it within its bound and leaves room for the states of this run.
    if (dfa->NumStates() > dfa->MaxStates() / 2) dfa->Clear();
    const int32_t start_state = dfa->FindOrAddState(*parser);
    if (start_state == ConfigurationDfa::kUnknown) return;
    dfa_ = dfa;
    states_.push_back(start_state);
  }

  /*! \brief Whether the cursor is answering from a DFA. False once it has left the DFA. */
  bool IsUsingDfa() const { return dfa_ != nullptr; }

  /*! \brief Advance by one byte, like EarleyParser::Advance. */
  bool Advance(uint8_t byte) {
    if (dfa_ == nullptr) return parser_->Advance(byte);
    int32_t next_state = dfa_->GetTransition(states_.back(), byte);
    if (next_state == ConfigurationDfa::kUnknown) {
      SyncParser();
      if (!parser_->Advance(byte)) {
        next_state = ConfigurationDfa::kRejected;
      } else {
        next_state = dfa_->FindOrAddState(*parser_);
        if (next_state == ConfigurationDfa::kUnknown) {
          // The parser has taken this byte and everything before it, so it continues alone.
          dfa_ = nullptr;
          return true;
        }
        ++num_parser_operations_;
      }
      dfa_->SetTransition(states_.back(), byte, next_state);
    }
    if (next_state == ConfigurationDfa::kRejected) return false;
    states_.push_back(next_state);
    operations_.push_back(byte);
    return true;
  }

  /*!
   * \brief Push the initial state of a lookahead assertion, like
   * EarleyParser::PushStateAndExpand. It counts as one operation for Pop.
   */
  void PushLookahead(const ParserState& lookahead_state) {
    if (dfa_ == nullptr) {
      parser_->PushStateAndExpand(lookahead_state);
      return;
    }
    XGRAMMAR_DCHECK(
        lookahead_state.rule_id == -1 &&
        lookahead_state.rule_start_pos == ParserState::kNoPrevInputPos
    );
    lookahead_state_ = lookahead_state;
    int32_t next_state = dfa_->GetLookaheadState(lookahead_state.sequence_id);
    if (next_state == ConfigurationDfa::kUnknown) {
      SyncParser();
      parser_->PushStateAndExpand(lookahead_state);
      next_state = dfa_->FindOrAddState(*parser_);
      if (next_state == ConfigurationDfa::kUnknown) {
        dfa_ = nullptr;
        return;
      }
      ++num_parser_operations_;
      dfa_->SetLookaheadState(lookahead_state.sequence_id, next_state);
    }
    states_.push_back(next_state);
    operations_.push_back(kPushLookahead);
  }

  /*! \brief Undo the latest operations, like EarleyParser::PopLastStates. */
  void Pop(int32_t count) {
    if (dfa_ == nullptr) {
      parser_->PopLastStates(count);
      return;
    }
    XGRAMMAR_DCHECK(count >= 0 && count <= static_cast<int32_t>(operations_.size()));
    states_.resize(states_.size() - count);
    operations_.resize(operations_.size() - count);
    if (num_parser_operations_ > operations_.size()) {
      parser_->PopLastStates(static_cast<int32_t>(num_parser_operations_ - operations_.size()));
      num_parser_operations_ = operations_.size();
    }
  }

  /*! \brief Like EarleyParser::IsCompleted. */
  bool IsCompleted() const {
    return dfa_ == nullptr ? parser_->IsCompleted() : dfa_->IsCompleted(states_.back());
  }

 private:
  /*! \brief Replay on the parser the operations it has not seen yet. */
  void SyncParser() {
    for (; num_parser_operations_ < operations_.size(); ++num_parser_operations_) {
      const int16_t operation = operations_[num_parser_operations_];
      if (operation == kPushLookahead) {
        parser_->PushStateAndExpand(lookahead_state_);
      } else {
        // The byte was accepted from an equal configuration, so it is accepted here too.
        [[maybe_unused]] const bool accepted = parser_->Advance(static_cast<uint8_t>(operation));
        XGRAMMAR_DCHECK(accepted);
      }
    }
  }

  /*! \brief The operation that pushes lookahead_state_. The other operations are bytes. */
  static constexpr int16_t kPushLookahead = 256;

  EarleyParser* parser_ = nullptr;
  ConfigurationDfa* dfa_ = nullptr;
  /*! \brief states_[i + 1] is the DFA state after operations_[i]. states_[0] is the start. */
  std::vector<int32_t> states_;
  std::vector<int16_t> operations_;
  /*! \brief The parser has performed the first num_parser_operations_ of operations_. */
  size_t num_parser_operations_ = 0;
  /*! \brief The lookahead state on the stack. There is at most one at a time. */
  ParserState lookahead_state_;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_CONFIGURATION_DFA_H_
