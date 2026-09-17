/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/compiled_grammar_impl.h
 * \brief The header for the data structures of the compiled grammar.
 */
#ifndef XGRAMMAR_COMPILED_GRAMMAR_IMPL_H_
#define XGRAMMAR_COMPILED_GRAMMAR_IMPL_H_

#include <xgrammar/grammar.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "earley_parser.h"
#include "support/dynamic_bitset.h"
#include "support/reflection.h"
#include "xgrammar/compiler.h"
#include "xgrammar/exception.h"

namespace xgrammar {

/******************* CompiledGrammar Datastructures *******************/

/*!
 * \brief Preprocessed information, for a given specific ParserState, divides the token set
 * into three categories: accepted, rejected, and uncertain.
 * Accepted: tokens that can be determined by the current ParserState to be acceptable
 * Rejected: tokens that can be determined by the current ParserState to be unacceptable
 * Uncertain: tokens that need the state of the parent ParserStates to determine if acceptable
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
    // Only store all rejected token indices. Then accepted indices = all_indices - rejected_indices
    // - uncertain_indices. This is useful when |accepted_indices| > |rejected_indices|.
    kRejected = 1,
    // Store all accepted token indices in a bitset. This is useful when both |accepted_indices| and
    // |rejected_indices| are large.
    kAcceptedBitset = 2
  };
  StoreType store_type;

  static constexpr int USE_BITSET_THRESHOLD = 1000;

  std::vector<int32_t> accepted_indices;
  std::vector<int32_t> rejected_indices;
  DynamicBitset accepted_bitset;

  std::vector<int32_t> uncertain_indices;

  /*!
   * \brief Fast path for the body state of a counted repetition of a single character class.
   *
   * `repeat_interior_bitsets[i]` holds every token whose bytes decode to a non-empty sequence of
   * codepoints accepted by that class and whose codepoint count is at most
   * `repeat_interior_char_counts[i]`; the counts are strictly increasing. Consuming such a token
   * keeps the parser inside the repetition and costs one repetition per codepoint, so the token is
   * legal exactly when the repetition budget left in the parent state covers its codepoint count.
   * That budget is not part of the compiled state, so the matcher reads it from the parse history
   * and picks the matching bitset; the mask itself stays count-independent.
   *
   * These tokens are removed from the mask's own accepted/rejected/uncertain classes, which keeps
   * over-budget tokens classified exactly as the Earley replay would have classified them.
   *
   * This is derived data: it is deliberately not part of the serialized form (see
   * XGRAMMAR_MEMBER_TABLE below) and is recomputed by PopulateRepeatInteriorBitsets whenever a
   * compiled grammar is built or deserialized.
   */
  std::vector<int32_t> repeat_interior_char_counts;
  std::vector<DynamicBitset> repeat_interior_bitsets;

  /*! \brief Default constructor. Only for deserialization. */
  AdaptiveTokenMask() = default;

  AdaptiveTokenMask(
      size_t vocab_size,
      const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
      const std::vector<int32_t>& accepted_indices,
      const std::vector<int32_t>& rejected_indices,
      const std::vector<int32_t>& uncertain_indices
  );

  AdaptiveTokenMask(
      size_t vocab_size,
      const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
      const std::vector<int32_t>& accepted_indices,
      const std::vector<int32_t>& uncertain_indices
  );

  std::string Print(const TokenizerInfo& tokenizer_info) const;

  friend std::size_t MemorySize(const AdaptiveTokenMask& mask) {
    return MemorySize(mask.uncertain_indices) + MemorySize(mask.accepted_indices) +
           MemorySize(mask.rejected_indices) + MemorySize(mask.accepted_bitset) +
           MemorySize(mask.repeat_interior_char_counts) +
           MemorySize(mask.repeat_interior_bitsets);
  }
};

XGRAMMAR_MEMBER_TABLE(
    AdaptiveTokenMask,
    "store_type",
    &AdaptiveTokenMask::store_type,
    "accepted_indices",
    &AdaptiveTokenMask::accepted_indices,
    "rejected_indices",
    &AdaptiveTokenMask::rejected_indices,
    "accepted_bitset",
    &AdaptiveTokenMask::accepted_bitset,
    "uncertain_indices",
    &AdaptiveTokenMask::uncertain_indices
);

/*!
 * \brief All information that we need to match tokens in the tokenizer to the specified grammar.
 * It is the result of preprocessing.
 * \sa xgrammar::GrammarMatcher
 */
class CompiledGrammar::Impl {
 public:
  /*! \brief The grammar for the GrammarMatcher. */
  Grammar grammar{NullObj{}};

  /*! \brief The tokenizer information. */
  TokenizerInfo tokenizer_info{NullObj{}};

  /*! \brief Default constructor. */
  Impl() = default;

  /*! \brief Mapping from the parser state to the adaptive token mask. */
  std::unordered_map<ParserState, AdaptiveTokenMask, StateHashForCache, StateEqualForCache>
      adaptive_token_mask_cache;

  Grammar GetGrammar() const { return grammar; }

  TokenizerInfo GetTokenizerInfo() const { return tokenizer_info; }

  friend struct member_trait<Impl>;
  friend picojson::value SerializeJSONValue(const Impl& impl);
  friend std::optional<SerializationError> DeserializeJSONValue(
      CompiledGrammar::Impl* impl,
      const picojson::value& json_value,
      const TokenizerInfo& tokenizer_info
  );
  friend std::size_t MemorySize(const Impl& impl);
};

/*!
 * \brief Compute the repeat-interior fast path data for every mask in `cache`.
 *
 * For the body state of a rule whose body is a single character class, record the tokens that
 * stay inside that class, grouped by codepoint count, and take them out of the mask's own
 * accepted/rejected/uncertain classes so that the matcher can decide them with a budget check.
 */
void PopulateRepeatInteriorBitsets(
    const Grammar& grammar,
    const TokenizerInfo& tokenizer_info,
    std::unordered_map<ParserState, AdaptiveTokenMask, StateHashForCache, StateEqualForCache>* cache
);

XGRAMMAR_MEMBER_TABLE(
    CompiledGrammar::Impl,
    "grammar",
    &CompiledGrammar::Impl::grammar,
    "tokenizer_info",
    &CompiledGrammar::Impl::tokenizer_info,
    "adaptive_token_mask_cache",
    &CompiledGrammar::Impl::adaptive_token_mask_cache
);

}  // namespace xgrammar

#endif  // XGRAMMAR_COMPILED_GRAMMAR_IMPL_H_
