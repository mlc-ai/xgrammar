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
#include <memory>
#include <mutex>
#include <optional>
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

class RuleLevelCache;
class CharacterClassTokenSummaryCache;

/******************* CompiledGrammar Datastructures *******************/

XGRAMMAR_MEMBER_TABLE(
    EarleyParserFeatures,
    "fsm_state_flags",
    &EarleyParserFeatures::fsm_state_flags,
    "rule_is_nullable",
    &EarleyParserFeatures::rule_is_nullable,
    "rule_is_context_independent",
    &EarleyParserFeatures::rule_is_context_independent,
    "has_budget_rules",
    &EarleyParserFeatures::has_budget_rules,
    "has_char_budget_rules",
    &EarleyParserFeatures::has_char_budget_rules,
    "capture_tracking",
    &EarleyParserFeatures::capture_tracking,
    "has_hidden_capture_rules",
    &EarleyParserFeatures::has_hidden_capture_rules
);

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
           MemorySize(mask.rejected_indices) + MemorySize(mask.accepted_bitset);
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

struct CharacterClassTokenSummary {
  int32_t sorted_vocab_index;
  int32_t locally_consumed_characters;
  bool consumed_whole_token;
  bool has_completed_character_prefix;
};

struct RepeatedCharacterClassTokenMask {
  AdaptiveTokenMask adaptive_token_mask;
  DynamicBitset accepted_prefix_tokens;

  friend std::size_t MemorySize(const RepeatedCharacterClassTokenMask& mask) {
    return MemorySize(mask.adaptive_token_mask) + MemorySize(mask.accepted_prefix_tokens);
  }
};

/*!
 * \brief Manages the adaptive token masks of a compiled grammar. In eager mode (the default),
 * all masks are precomputed at compile time. In dynamic mode, masks are generated and cached on
 * first use.
 */
class TokenMaskCache {
 public:
  /*! \brief Construct an eager cache. Only for deserialization. */
  TokenMaskCache() = default;

  /*!
   * \brief Construct the cache with the mask generation mode.
   * \param dynamic Whether missing token masks should be generated on first use.
   */
  explicit TokenMaskCache(bool dynamic) : dynamic_(dynamic) {}

  /*! \brief Configure cross-grammar rule mask sharing for dynamic generation. */
  void SetRuleLevelCache(std::shared_ptr<RuleLevelCache> rule_level_cache) {
    rule_level_cache_ = std::move(rule_level_cache);
  }

  /*! \brief Whether missing token masks should be generated on first use. */
  bool IsDynamic() const { return dynamic_; }

  /*! \brief Insert a precomputed mask during eager compilation. Not thread-safe; the compiler
   * synchronizes concurrent insertions itself. */
  void Insert(const ParserState& state, AdaptiveTokenMask mask) { masks_[state] = std::move(mask); }

  /*! \brief Return the mask for the state. In dynamic mode, generate and cache it on miss. */
  const AdaptiveTokenMask& Get(
      const ParserState& state,
      bool is_root_rule,
      const Grammar& grammar,
      const TokenizerInfo& tokenizer_info,
      const EarleyParserFeatures& features
  );

 private:
  /*! \brief Return the tag-dispatch bitset for a rule, computing and caching it on first use. */
  const DynamicBitset* GetTagDispatchSecondSlicingBitset(
      int32_t rule_id, const Grammar& grammar, const TokenizerInfo& tokenizer_info
  );

  /*! \brief Whether missing token masks should be generated on first use. */
  bool dynamic_{false};

  /*! \brief Rule masks shared by grammars compiled with the same compiler. */
  std::shared_ptr<RuleLevelCache> rule_level_cache_;

  /*! \brief Mapping from the parser state to the adaptive token mask. */
  std::unordered_map<ParserState, AdaptiveTokenMask, StateHashForCache, StateEqualForCache> masks_;

  /*! \brief Tag-dispatch data computed on first use in dynamic mode. */
  std::unordered_map<int32_t, DynamicBitset> tag_dispatch_rule_id_to_second_slicing_bitset_;

  /*! \brief Protects on-demand token mask lookup and insertion. */
  mutable std::mutex mutex_;

  friend struct member_trait<TokenMaskCache>;
  friend picojson::value SerializeJSONValue(const CompiledGrammar::Impl& impl);
  friend std::optional<SerializationError> DeserializeJSONValue(
      CompiledGrammar::Impl* impl,
      const picojson::value& json_value,
      const TokenizerInfo& tokenizer_info
  );

  friend std::size_t MemorySize(const TokenMaskCache& token_mask_cache) {
    std::lock_guard<std::mutex> lock(token_mask_cache.mutex_);
    return MemorySize(token_mask_cache.masks_) +
           MemorySize(token_mask_cache.tag_dispatch_rule_id_to_second_slicing_bitset_);
  }
};

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

  /*! \brief Default constructor. Only for deserialization. */
  Impl() = default;

  /*! \brief Construct with the token mask generation mode. \sa TokenMaskCache */
  explicit Impl(bool enable_dynamic_compilation) : token_mask_cache(enable_dynamic_compilation) {}

  /*! \brief The adaptive token masks, precomputed or generated on first use. */
  TokenMaskCache token_mask_cache;

  /*! \brief Grammar-wide flags and nullable rules shared by Earley parsers. */
  EarleyParserFeatures earley_parser_features;

  /*! \brief Character-class token summaries shared by grammars from the same compiler. */
  std::shared_ptr<CharacterClassTokenSummaryCache> character_class_token_summary_cache;

  /*! \brief Repeated character-class masks shared by matchers using this compiled grammar. */
  std::unordered_map<uint64_t, RepeatedCharacterClassTokenMask>
      repeated_character_class_token_masks;

  /*! \brief Protects repeated character-class summary and mask generation. */
  mutable std::mutex repeated_character_class_cache_mutex;

  const RepeatedCharacterClassTokenMask& GetRepeatedCharacterClassTokenMask(
      int32_t character_class_expression_id, int32_t character_limit
  );

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

XGRAMMAR_MEMBER_TABLE(TokenMaskCache, "adaptive_token_mask_cache", &TokenMaskCache::masks_);

XGRAMMAR_MEMBER_TABLE(
    CompiledGrammar::Impl,
    "grammar",
    &CompiledGrammar::Impl::grammar,
    "tokenizer_info",
    &CompiledGrammar::Impl::tokenizer_info,
    "token_mask_cache",
    &CompiledGrammar::Impl::token_mask_cache
);

}  // namespace xgrammar

#endif  // XGRAMMAR_COMPILED_GRAMMAR_IMPL_H_
