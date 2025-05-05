/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/compiled_grammar_data_structure.h
 * \brief The header for the data structures of the compiled grammar.
 */
#ifndef XGRAMMAR_COMPILED_GRAMMAR_DATA_STRUCTURE_H_
#define XGRAMMAR_COMPILED_GRAMMAR_DATA_STRUCTURE_H_

#include <xgrammar/grammar.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// matcher_data_structure.h is included to use StackElement
#include "persistent_stack.h"
#include "support/dynamic_bitset.h"
#include "support/utils.h"

namespace xgrammar {

/******************* CompiledGrammar Datastructures *******************/

/*!
 * \brief Preprocessed information, for a given specific StackElement, divides the token set
 * into three categories: accepted, rejected, and uncertain.
 * Accepted: tokens that can be determined by the current StackElement to be acceptable
 * Rejected: tokens that can be determined by the current StackElement to be unacceptable
 * Uncertain: tokens that need the state of the parent StackElements to determine if acceptable
 *
 * \note uncertain indices are stored directly. Accepted / rejected indices have three ways to
 * store to reduce memory and computation usage. See StoreType.
 * \note These indices are the indices of sorted_decoded_vocab in the CompiledGrammar
 * object, instead of the token ids. That helps the matching process.
 */
struct AdaptiveTokenMask {
  enum class StoreType {
    // Only store all accepted token indices. Then rejected indices = all_indices - accepted_indices
    // - uncertain_indices. This is useful when |accepted_indices| < |rejected_indices|.
    kAccepted = 0,
    // Only store all accepted token indices. Then accepted indices = all_indices - rejected_indices
    // - uncertain_indices. This is useful when |accepted_indices| > |rejected_indices|.
    kRejected = 1,
    // Store all accepted token indices in a bitset. This is useful when both |accepted_indices| and
    // |rejected_indices| are large.
    kAcceptedBitset = 2
  };
  StoreType store_type;

  static constexpr int USE_BITSET_THRESHOLD = 200;

  std::vector<int32_t> accepted_indices;
  std::vector<int32_t> rejected_indices;
  DynamicBitset accepted_bitset;

  std::vector<int32_t> uncertain_indices;

  AdaptiveTokenMask() = default;

  AdaptiveTokenMask(
      size_t vocab_size,
      const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
      const std::vector<int32_t>& accepted_indices,
      const std::vector<int32_t>& rejected_indices,
      const std::vector<int32_t>& uncertain_indices
  );

  std::string Print(const TokenizerInfo& tokenizer_info) const;

  std::size_t MemorySize() const;
  friend std::size_t MemorySize(const AdaptiveTokenMask& mask);
};

/*!
 * \brief All information that we need to match tokens in the tokenizer to the specified grammar.
 * It is the result of preprocessing.
 * \sa xgrammar::GrammarMatcher
 */
class CompiledGrammar::Impl {
 public:
  /******************* The grammar and tokenizer info *******************/

  /*! \brief The grammar for the GrammarMatcher. */
  Grammar grammar;

  /*! \brief The tokenizer information. */
  TokenizerInfo tokenizer_info;

  /******************* The adaptive token mask cache *******************/

  struct StackElementEqual {
    std::size_t operator()(const StackElement& lhs, const StackElement& rhs) const noexcept {
      return lhs.sequence_id == rhs.sequence_id && lhs.element_id == rhs.element_id &&
             lhs.left_utf8_bytes == rhs.left_utf8_bytes &&
             lhs.element_in_string == rhs.element_in_string;
    }
  };

  struct StackElementHash {
    std::size_t operator()(const StackElement& stack_element) const noexcept {
      return HashCombine(
          stack_element.sequence_id,
          stack_element.element_id,
          stack_element.left_utf8_bytes,
          stack_element.element_in_string
      );
    }
  };

  /*! \brief Mapping from the stack top element to the adaptive token mask. */
  std::unordered_map<StackElement, AdaptiveTokenMask, StackElementHash, StackElementEqual>
      adaptive_token_mask_cache;

  Grammar GetGrammar() const { return grammar; }

  TokenizerInfo GetTokenizerInfo() const { return tokenizer_info; }

  std::size_t MemorySize() const;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_COMPILED_GRAMMAR_DATA_STRUCTURE_H_
