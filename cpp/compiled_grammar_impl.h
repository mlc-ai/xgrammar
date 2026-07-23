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
  static constexpr int UNCERTAIN_BITSET_THRESHOLD = 1024;

  std::vector<int32_t> accepted_indices;
  std::vector<int32_t> rejected_indices;
  DynamicBitset accepted_bitset;

  std::vector<int32_t> uncertain_indices;
  // Derived, non-serialized data for batching large runtime uncertain-token operations.
  DynamicBitset uncertain_bitset;
  std::vector<int32_t> uncertain_lcp_with_previous;
  int32_t max_uncertain_token_length = 0;

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

  AdaptiveTokenMask(
      size_t vocab_size,
      const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
      std::vector<int32_t>&& accepted_indices,
      std::vector<int32_t>&& uncertain_indices
  );

  void RebuildDerivedData(
      size_t vocab_size, const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab
  );

  std::string Print(const TokenizerInfo& tokenizer_info) const;

  friend std::size_t MemorySize(const AdaptiveTokenMask& mask) {
    return MemorySize(mask.uncertain_indices) + MemorySize(mask.accepted_indices) +
           MemorySize(mask.rejected_indices) + MemorySize(mask.accepted_bitset) +
           MemorySize(mask.uncertain_bitset) + MemorySize(mask.uncertain_lcp_with_previous);
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

#ifdef XGRAMMAR_PROFILE_COMPILE
/*!
 * \brief Profile-only cost of computing one structurally deduplicated adaptive-mask task.
 *
 * Several ParserStates can share one task result. Runtime coverage is therefore measured both by
 * state (stored mask bytes) and by task group (compile work that a lazy compiler would execute).
 */
struct AdaptiveMaskCompileProfile {
  ParserState representative_state;
  uint64_t mask_cpu_us = 0;
  uint64_t possible_tokens = 0;
  uint64_t rule_cache_hits = 0;
  uint64_t first_byte_cache_reused_tokens = 0;
  uint64_t saturated_cache_reused_tokens = 0;
  uint64_t token_edge_skipped_tokens = 0;
  uint64_t speculative_accepted_tokens = 0;
  uint64_t subtree_pruned_tokens = 0;
  uint64_t parser_simulated_tokens = 0;
  uint64_t parser_naive_token_bytes = 0;
  uint64_t parser_advance_calls = 0;
  uint64_t parser_failed_advance_calls = 0;
  uint64_t mask_bytes = 0;
  uint64_t state_count = 0;
};
#endif

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

  /*! \brief Grammar-only Earley metadata shared by compile-time tasks and runtime matchers. */
  EarleyParserGrammarMetadata earley_parser_metadata;

  /*! \brief Default constructor. */
  Impl() = default;

  /*! \brief Structurally deduplicated adaptive token masks. */
  std::vector<AdaptiveTokenMask> adaptive_token_masks;

  /*! \brief Mapping from each parser state to an entry in adaptive_token_masks. */
  std::unordered_map<ParserState, uint32_t, StateHashForCache, StateEqualForCache>
      adaptive_token_mask_ids;

#ifdef XGRAMMAR_PROFILE_COMPILE
  /*! \brief Per-task compile work, indexed by the structurally deduplicated task-group id. */
  std::vector<AdaptiveMaskCompileProfile> adaptive_mask_compile_profiles;
#endif

  const AdaptiveTokenMask* FindAdaptiveTokenMask(const ParserState& state) const {
    const auto mask_id_it = adaptive_token_mask_ids.find(state);
    if (mask_id_it == adaptive_token_mask_ids.end()) {
      return nullptr;
    }
    XGRAMMAR_DCHECK(mask_id_it->second < adaptive_token_masks.size());
    return &adaptive_token_masks[mask_id_it->second];
  }

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

XGRAMMAR_MEMBER_TABLE(
    CompiledGrammar::Impl,
    "grammar",
    &CompiledGrammar::Impl::grammar,
    "tokenizer_info",
    &CompiledGrammar::Impl::tokenizer_info,
    "adaptive_token_masks",
    &CompiledGrammar::Impl::adaptive_token_masks,
    "adaptive_token_mask_ids",
    &CompiledGrammar::Impl::adaptive_token_mask_ids
);

}  // namespace xgrammar

#endif  // XGRAMMAR_COMPILED_GRAMMAR_IMPL_H_
