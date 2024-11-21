/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_matcher_base.h
 * \brief The base class of GrammarMatcher. It implements a character-based matching automata.
 */
#ifndef XGRAMMAR_MATCHER_BASE_H_
#define XGRAMMAR_MATCHER_BASE_H_

#include <xgrammar/grammar.h>

#include <algorithm>
#include <string>
#include <vector>

#include "grammar_data_structure.h"
#include "matcher_data_structure.h"

namespace xgrammar {

/*!
 * \brief The base class of GrammarMatcher. It implements a character-based matching
 * automata, and supports accepting a character, rolling back by character, etc.
 */
class GrammarMatcherBase {
 protected:
  using RuleExpr = Grammar::Impl::RuleExpr;
  using RuleExprType = Grammar::Impl::RuleExprType;

 public:
  /*!
   * \brief Construct a GrammarMatcherBase with the given grammar and initial rule position.
   * \param grammar The grammar to match.
   * \param init_rule_position The initial rule position. If not specified, the root rule will be
   * used.
   * \param expand_init_rule_position Whether to expand the initial rule position to all possible
   * locations. See ExpandRulePosition.
   */
  GrammarMatcherBase(
      const Grammar& grammar,
      RulePosition init_rule_position = kInvalidRulePosition,
      bool expand_init_rule_position = true
  )
      : grammar_(grammar), persistent_stack_(grammar), stack_tops_history_(&persistent_stack_) {
    PushInitialState(init_rule_position, expand_init_rule_position);
  }

  /*! \brief Accept one character. */
  bool AcceptChar(uint8_t char_value, bool debug_print = false);

  /*! \brief Check if the end of the root rule is reached. If so, the stop token can be accepted. */
  bool CanReachEnd() const;

  /*! \brief Rollback the matcher to a previous state by the number of characters. */
  void RollbackChars(int rollback_cnt);

  /*! \brief Discard the earliest history by the number of characters. */
  void DiscardEarliestChars(int discard_cnt);

  /*! \brief Print the stack state. */
  std::string PrintStackState(int steps_behind_latest = 0) const;

 protected:
  // Push an initial stack state according to the given rule position.
  // If init_rule_position is kInvalidRulePosition, init the stack with the root rule.
  void PushInitialState(RulePosition init_rule_position, bool expand_init_rule_position);

  // Check if the character is accepted by the current rule position.
  bool CheckIfAccepted(const RulePosition& rule_position, uint8_t char_value) const;

  /*!
   * \brief Find the next position in the rule. If the next position is at the end of the rule,
   * and consider_parent is true, will iteratively find the next position in the parent rule.
   * \param rule_position The current position.
   * \param consider_parent Whether to consider the parent position if the current position is
   * at the end of the rule.
   * \returns (success, next_rule_position), indicating if the iteration is successful and the
   * next rule position.
   */
  std::pair<bool, RulePosition> GetNextPositionInSequence(
      const RulePosition& rule_position, bool consider_parent
  ) const;

  // Return the updated rule position after accepting the char
  RulePosition UpdatePositionWithChar(const RulePosition& rule_position, uint8_t char_value) const;

  /*!
   * \brief Expand the given rule position to all possible positions approachable in the grammar.
   * The expanded positions must refers to an element (CharacterClass or CharacterClassStar or
   * ByteString) in a rule. Push all new positions into new_stack_tops.
   * \example
   * A ::= "a" B [a-z]* "c"
   * B ::= "b" | ""
   *
   * Input position: (rule=A, position=B)
   * Approachable positions: (rule=B, position="b"), (rule=A, position=[a-z]*),
   * (rule=A, position="c"), since B and [a-z]* can be empty.
   * \param cur_rule_position The current rule position.
   * \param new_stack_tops The vector to store the new stack tops.
   * \param consider_parent Whether consider expanding the elements in the parent rule. Useful for
   * inner recursion.
   * \param first_id_if_inserted An optimization. When cur_rule_position is already inserted to
   * the state tree, pass its id to avoid inserting it again. -1 (ignore it) by default.
   * \return Whether the end of the rule can be reached. Useful for inner recursion.
   */
  bool ExpandRulePosition(
      RulePosition cur_rule_position,
      std::vector<int32_t>* new_stack_tops,
      bool consider_parent = true,
      int32_t first_id_if_inserted = -1
  );

  // The matched grammar.
  Grammar grammar_;
  // The tree storing all states
  PersistentStack persistent_stack_;
  // The tracked history of stack tops (each stack top refers to a node in the tree).
  // We store the stack tops in different steps in the history to support rollback.
  StackTopsHistory stack_tops_history_;

  // Temporary data for AcceptChar, PushInitialState, etc to store new stacks.
  // They are stored here to avoid repeated allocation.
  std::vector<int32_t> tmp_new_stack_tops_;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_MATCHER_BASE_H_
