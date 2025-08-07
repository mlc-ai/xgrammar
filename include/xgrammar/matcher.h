/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/matcher.h
 * \brief The header for the matcher.
 */

#ifndef XGRAMMAR_MATCHER_H_
#define XGRAMMAR_MATCHER_H_

#include <dlpack/dlpack.h>
#include <xgrammar/compiler.h>
#include <xgrammar/object.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace xgrammar {

int32_t GetBitmaskSize(int vocab_size);

DLDataType GetBitmaskDLType();

void _DebugGetMaskedTokensFromBitmask(
    std::vector<int>* rejected_tokens, const DLTensor& token_bitmask, int vocab_size, int index = 0
);

std::pair<bool, int> _IsSingleTokenBitmask(const DLTensor& bitmask, int vocab_size, int index);

void ApplyTokenBitmaskInplaceCPU(
    DLTensor* logits,
    const DLTensor& bitmask,
    int vocab_size = -1,
    std::optional<std::vector<int>> indices = std::nullopt
);

/*!
 * \brief A stateful matcher to match tokens to the specified BNF grammar. This class is the core
 * logic of the grammar-guided generation.
 *
 * \details This class implements the non-deterministic pushdown automaton (NPDA) matching algorithm
 * to match characters to a BNF grammar. It keep track of the current state of the matching process
 * by maintaining several stacks internally as possible paths in the NPDA. It also supports
 * backtracking.
 *
 * It is particularly capable of finding the set of tokens that are acceptable for the next step
 * and storing them in a bitmask. This aids in grammar-guided generation.
 *
 * \example
 * \code
 * Tokenizer tokenizer = ...;
 * auto compiled_grammar = GrammarMatcher::CreateCompiledGrammar(grammar,
 *                                                        tokenizer->PostProcessedVocab());
 * GrammarMatcher matcher(compiled_grammar, 10);
 * matcher->AcceptToken(67);
 *
 * // Construct a DLTensor with shape (tokenizer.GetVocabSize() + 31) / 32, and dtype int32.
 * DLTensor next_token_bitmask = ...;
 * matcher->FillNextTokenBitmask(&next_token_bitmask);
 *
 * // Rollback is supported
 * matcher->Rollback(1);
 * \endcode
 */
class GrammarMatcher {
 public:
  /*!
   * \brief Construct a GrammarMatcher from the preprocessing result of type
   * CompiledGrammar.
   * \param compiled_grammar The compiled grammar. It is obtained through
   * CreateCompiledGrammar as a result of preprocessing the grammar and tokenizer.
   */
  GrammarMatcher(
      const CompiledGrammar& compiled_grammar,
      std::optional<std::vector<int>> override_stop_tokens = std::nullopt,
      bool terminate_without_stop_token = false,
      int max_rollback_tokens = -1
  );

  /*!
   * \brief Accept one token and update the state of the matcher.
   * \param token_id The id of the token to accept.
   * \return Whether the token is accepted.
   * \note Termination state.
   * When the end of the root rule is reached, the matcher can only accept the stop token.
   * The matcher is terminated after accepting the stop token, i.e. no AcceptToken or
   * FindNextTokenMask operations can be performed. The termination state can be canceled
   * using Rollback().
   */
  bool AcceptToken(int32_t token_id, bool debug_print = false);

  /*!
   * \brief Accept a string and update the state of the matcher. The whole string is considered
   * as one step in rollback. It is used to complement the functionality of AcceptToken, and
   * AcceptToken should always be used to accept tokens.
   * \param input_str The string to be accepted.
   * \param debug_print Whether to print information about the internal state of the matcher.
   * \return Whether the string is accepted.
   */
  bool AcceptString(const std::string& input_str, bool debug_print = false);

  /*!
   * \brief Get the set of tokens that are acceptable for the next step and store them in a
   * bitmask.
   * \param next_token_bitmask The bitmask to store the result. The bitmask must be pre-allocated
   * and with shape (GetBitmaskSize(),) and dtype int32.
   * \return Whether the bitmask need to be applied (not all-true).
   */
  bool FillNextTokenBitmask(DLTensor* next_token_bitmask, int index = 0, bool debug_print = false);

  /*!
   * \brief Find the jump-forward string for jump-forward decoding. This is the longest string that
   will be valid according to the current syntax.
   * \note This method does not change the grammar state.
   */
  std::string FindJumpForwardString();

  /*!
   * \brief Rollback the matcher to a previous state.
   * \param num_tokens The number of tokens to rollback. It cannot exceed the current number of
   * steps, nor can it exceed the specified maximum number of rollback tokens.
   */
  void Rollback(int num_tokens = 1);

  /*!
   * \brief Check if the matcher has accepted the stop token and terminated.
   * \sa AcceptToken
   */
  bool IsTerminated() const;

  /*! \brief Reset the matcher to the initial state. */
  void Reset();

  /*! \brief Get the maximum number of rollback tokens allowed. */
  int GetMaxRollbackTokens() const;

  const std::vector<int>& GetStopTokenIds() const;

  /*! \brief Print the internal state of the matcher. This is only used for debugging. The
   * representation of the internal state is subject to change.
   */
  std::string _DebugPrintInternalState() const;

  XGRAMMAR_DEFINE_PIMPL_METHODS(GrammarMatcher);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_MATCHER_H_
