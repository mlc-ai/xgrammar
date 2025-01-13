/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_matcher_base.h
 * \brief The base class of GrammarMatcher. It implements a character-based matching automata.
 */
#ifndef XGRAMMAR_GRAMMAR_MATCHER_BASE_H_
#define XGRAMMAR_GRAMMAR_MATCHER_BASE_H_

#include <xgrammar/grammar.h>

#include <algorithm>
#include <string>
#include <vector>

#include "grammar_data_structure.h"
#include "persistent_stack.h"

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
   * \brief Construct a GrammarMatcherBase with the given grammar and initial stack element.
   * \param grammar The grammar to match.
   * \param init_stack_element The initial stack element. If not specified, the root rule will be
   * used.
   * \param expand_init_stack_element Whether to expand the initial stack element to all possible
   * locations. See ExpandEquivalentStackElements.
   */
  GrammarMatcherBase(
      const Grammar& grammar,
      StackElement init_stack_element = kInvalidStackElement,
      bool expand_init_stack_element = true
  )
      : grammar_(grammar), persistent_stack_(grammar), stack_tops_history_(&persistent_stack_) {
    PushInitialState(init_stack_element, expand_init_stack_element);
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
  // Push an initial stack state according to the given stack element.
  // If init_stack_element is kInvalidStackElement, init the stack with the root rule.
  void PushInitialState(StackElement init_stack_element, bool expand_init_stack_element);

  // Check if the character is accepted by the current stack element.
  bool CheckIfAccepted(const StackElement& stack_element, uint8_t char_value) const;

  /*!
   * \brief Move to the next position in the grammar. It could be the next position in the current
   * rule, or the next position in the parent rule if the current position is at the end of the
   * rule. Does not change the underlying persistent stack.
   * \param stack_element The current stack element.
   * \param consider_parent Whether to consider the parent position if the current position is
   * at the end of the rule.
   * \returns (success, next_stack_element), indicating if the iteration is successful and the
   * next stack element.
   */
  StackElement MoveToNextPosition(StackElement stack_element);

  // Return the updated stack element after accepting the char
  StackElement AdvanceStackElementWithChar(const StackElement& stack_element, uint8_t char_value);

  /*!
   * \brief Expand the given stack element to all possible positions approachable in the grammar.
   * The expanded positions must refers to an element (CharacterClass or CharacterClassStar or
   * ByteString) in a rule. Push all new positions into new_stack_tops.
   * \example
   * A ::= "a" B [a-z]* "c"
   * B ::= "b" | ""
   *
   * Input position: (rule=A, position=B)
   * Approachable positions: (rule=B, position="b"), (rule=A, position=[a-z]*),
   * (rule=A, position="c"), since B and [a-z]* can be empty.
   * \param cur_stack_element The current stack element.
   * \param new_stack_tops The vector to store the new stack tops.
   * \param is_inner_recursion Whether this is an inner recursion. The inner recursion will not
   * consider the parent rule.
   * \param first_id_if_inserted An optimization. When cur_stack_element is already inserted to
   * the state tree, pass its id to avoid inserting it again. -1 (ignore it) by default.
   * \return Whether the end of the rule can be reached. Useful for inner recursion.
   */
  void ExpandEquivalentStackElements(
      StackElement cur_stack_element,
      std::vector<int32_t>* new_stack_tops,
      int32_t cur_stack_element_id = -1,
      bool consider_parent = true
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

#endif  // XGRAMMAR_GRAMMAR_MATCHER_BASE_H_
