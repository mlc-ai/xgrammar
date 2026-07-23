/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/compiler.cc
 */

#include <xgrammar/compiler.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "compiled_grammar_impl.h"
#include "earley_parser.h"
#include "fsm.h"
#include "grammar_functor.h"
#include "grammar_impl.h"
#include "support/dynamic_bitset.h"
#include "support/encoding.h"
#include "support/int_set.h"
#include "support/logging.h"
#include "support/thread_pool.h"
#include "support/thread_safe_cache.h"
#include "support/utils.h"
#include "tokenizer_info_impl.h"
#include "xgrammar/grammar.h"
#include "xgrammar/tokenizer_info.h"

namespace xgrammar {

/************** AdaptiveTokenMaskCache Generator **************/

/*!
 * \brief A compilation-local cache for one first-byte subtree of the tokenizer vocabulary.
 *
 * Two scanable FSM states can have different full reachable grammars while consuming a particular
 * first byte into structurally identical continuations. In that case every token with this first
 * byte has the same adaptive-mask classification. Caching that vocabulary bucket avoids replaying
 * the common suffix language for every node of generated exclusion tries.
 */
class FirstByteTokenMaskCache {
 public:
  struct Key {
    uint8_t first_byte;
    bool collect_rejected;
    std::vector<Grammar::Impl::FSMStateCacheKey> continuations;

    bool operator==(const Key& other) const {
      return first_byte == other.first_byte && collect_rejected == other.collect_rejected &&
             continuations == other.continuations;
    }
  };

  struct Result {
    std::vector<int32_t> accepted_indices;
    std::vector<int32_t> rejected_indices;
    std::vector<int32_t> uncertain_indices;
    size_t covered_max_suffix_bytes = std::numeric_limits<size_t>::max();
  };

  struct KeyHash {
    size_t operator()(const Key& key) const {
      uint64_t result = HashCombine(key.first_byte, key.collect_rejected);
      for (const auto& continuation : key.continuations) {
        HashCombineBinary(result, continuation.hash);
        HashCombineBinary(result, continuation.state_count);
        HashCombineBinary(result, continuation.edge_count);
      }
      return result;
    }
  };

  explicit FirstByteTokenMaskCache(size_t max_memory_bytes) : max_memory_bytes_(max_memory_bytes) {}

#ifdef XGRAMMAR_PROFILE_COMPILE
  ~FirstByteTokenMaskCache() {
    XGRAMMAR_LOG(INFO) << "FirstByteTokenMaskCacheProfile(hits=" << profile_hits_
                       << ", misses=" << profile_misses_
                       << ", optional_chain_queries=" << profile_optional_chain_queries_
                       << ", optional_chain_hits=" << profile_optional_chain_hits_
                       << ", fixed_tail_queries=" << profile_fixed_tail_queries_
                       << ", fixed_tail_hits=" << profile_fixed_tail_hits_
                       << ", additions=" << profile_additions_
                       << ", replacements=" << profile_replacements_
                       << ", entries=" << cache_.size() << ", bytes=" << memory_bytes_ << ")";
  }

#endif

  std::shared_ptr<const Result> Get(const Key& key) {
    std::lock_guard<std::mutex> lock(mutex_);
#ifdef XGRAMMAR_PROFILE_COMPILE
    const bool is_optional_chain =
        !key.continuations.empty() && key.continuations[0].state_count == -1;
    const bool is_fixed_tail = !key.continuations.empty() && key.continuations[0].state_count == -2;
    profile_optional_chain_queries_ += is_optional_chain;
    profile_fixed_tail_queries_ += is_fixed_tail;
#endif
    const auto it = cache_.find(key);
    if (it == cache_.end()) {
#ifdef XGRAMMAR_PROFILE_COMPILE
      ++profile_misses_;
#endif
      return nullptr;
    }
#ifdef XGRAMMAR_PROFILE_COMPILE
    ++profile_hits_;
    profile_optional_chain_hits_ += is_optional_chain;
    profile_fixed_tail_hits_ += is_fixed_tail;
#endif
    return it->second;
  }

  void Add(Key key, Result result) {
    const auto get_memory_bytes = [&](const Result& value) {
      return sizeof(Key) + sizeof(Result) + value.accepted_indices.size() * sizeof(int32_t) +
             value.rejected_indices.size() * sizeof(int32_t) +
             value.uncertain_indices.size() * sizeof(int32_t) +
             key.continuations.size() * sizeof(Grammar::Impl::FSMStateCacheKey);
    };
    const size_t new_memory_bytes = get_memory_bytes(result);
    std::lock_guard<std::mutex> lock(mutex_);
    const auto existing = cache_.find(key);
    if (existing != cache_.end()) {
      if (existing->second->covered_max_suffix_bytes >= result.covered_max_suffix_bytes) {
        return;
      }
      const size_t existing_memory_bytes = get_memory_bytes(*existing->second);
      const size_t memory_without_existing = memory_bytes_ - existing_memory_bytes;
      if (new_memory_bytes > max_memory_bytes_ - memory_without_existing) {
        return;
      }
      memory_bytes_ = memory_without_existing + new_memory_bytes;
      existing->second = std::make_shared<const Result>(std::move(result));
#ifdef XGRAMMAR_PROFILE_COMPILE
      ++profile_replacements_;
#endif
      return;
    }
    if (new_memory_bytes > max_memory_bytes_ - memory_bytes_) {
      return;
    }
    memory_bytes_ += new_memory_bytes;
    cache_.emplace(std::move(key), std::make_shared<const Result>(std::move(result)));
#ifdef XGRAMMAR_PROFILE_COMPILE
    ++profile_additions_;
#endif
  }

 private:
  std::mutex mutex_;
  const size_t max_memory_bytes_;
  size_t memory_bytes_ = 0;
  std::unordered_map<Key, std::shared_ptr<const Result>, KeyHash> cache_;
#ifdef XGRAMMAR_PROFILE_COMPILE
  uint64_t profile_hits_ = 0;
  uint64_t profile_misses_ = 0;
  uint64_t profile_additions_ = 0;
  uint64_t profile_replacements_ = 0;
  uint64_t profile_optional_chain_queries_ = 0;
  uint64_t profile_optional_chain_hits_ = 0;
  uint64_t profile_fixed_tail_queries_ = 0;
  uint64_t profile_fixed_tail_hits_ = 0;
#endif
};

/*!
 * \brief A compilation-local summary of how vocabulary tokens match one character class.
 *
 * Bounded repeats are lowered to one helper rule per remaining length. Without this cache, every
 * helper rule decodes every vocabulary token again even though only the length limit changes.
 * The first caller scans the vocabulary once; all remaining helper rules reuse the byte-decoding
 * result and only compare the recorded character count with their own limit.
 */
class OptionalCharacterClassTokenSummaryCache {
 public:
  struct Key {
    std::vector<int32_t> character_class;

    bool operator==(const Key& other) const { return character_class == other.character_class; }
  };

  struct KeyHash {
    size_t operator()(const Key& key) const {
      uint64_t result = 0x4f50545f53554d4dULL;
      for (int32_t value : key.character_class) {
        HashCombineBinary(result, static_cast<uint64_t>(value));
      }
      return result;
    }
  };

  struct TokenSummary {
    int32_t sorted_vocab_index;
    int32_t locally_consumed_characters;
    bool consumed_whole_token;
    bool has_completed_character_prefix;
  };

  using Result = std::vector<TokenSummary>;

  template <typename Builder>
  std::shared_ptr<const Result> GetOrCreate(const Key& key, Builder&& builder) {
    using SharedResult = std::shared_ptr<const Result>;
    std::shared_future<SharedResult> future;
    std::promise<SharedResult> producer;
    bool should_build = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto it = cache_.find(key);
      if (it != cache_.end()) {
        future = it->second;
#ifdef XGRAMMAR_PROFILE_COMPILE
        ++profile_hits_;
#endif
      } else {
        future = producer.get_future().share();
        cache_.emplace(key, future);
        should_build = true;
#ifdef XGRAMMAR_PROFILE_COMPILE
        ++profile_builds_;
#endif
      }
    }
    if (!should_build) {
      return future.get();
    }
    try {
      SharedResult result = std::make_shared<const Result>(builder());
#ifdef XGRAMMAR_PROFILE_COMPILE
      {
        std::lock_guard<std::mutex> lock(mutex_);
        profile_candidate_tokens_ += result->size();
      }
#endif
      producer.set_value(result);
      return result;
    } catch (...) {
      producer.set_exception(std::current_exception());
      throw;
    }
  }

#ifdef XGRAMMAR_PROFILE_COMPILE
  ~OptionalCharacterClassTokenSummaryCache() {
    XGRAMMAR_LOG(INFO) << "OptionalCharacterClassTokenSummaryProfile(builds=" << profile_builds_
                       << ", hits=" << profile_hits_
                       << ", candidate_tokens=" << profile_candidate_tokens_ << ")";
  }
#endif

 private:
  std::mutex mutex_;
  std::unordered_map<Key, std::shared_future<std::shared_ptr<const Result>>, KeyHash> cache_;
#ifdef XGRAMMAR_PROFILE_COMPILE
  uint64_t profile_builds_ = 0;
  uint64_t profile_hits_ = 0;
  uint64_t profile_candidate_tokens_ = 0;
#endif
};

struct OptionalCharacterClassChain {
  uint64_t character_class_hash;
  int32_t character_class_expr_id;
  int32_t remaining_count;
};

/*!
 * \brief Recognize the helper-rule chain emitted for a bounded optional character-class repeat.
 *
 * Each rule in the chain is `"" | character_class next_rule`, with the final rule omitting
 * `next_rule`. The result records the common character class and the maximum number of characters
 * that the current rule can still consume.
 */
std::optional<OptionalCharacterClassChain> RecognizeOptionalCharacterClassChain(
    const Grammar& grammar, int32_t initial_rule_id
) {
  using GrammarExprType = Grammar::Impl::GrammarExprType;

  std::optional<uint64_t> character_class_hash;
  std::optional<int32_t> character_class_expr_id;
  int32_t remaining_count = 0;
  int32_t rule_id = initial_rule_id;
  std::unordered_set<int32_t> visited_rules;
  while (visited_rules.insert(rule_id).second) {
    const auto& body = grammar->GetGrammarExpr(grammar->GetRule(rule_id).body_expr_id);
    if (body.type != GrammarExprType::kChoices || body.size() != 2) {
      return std::nullopt;
    }

    std::optional<int32_t> non_empty_sequence_id;
    bool has_empty_choice = false;
    for (int32_t choice_id : body) {
      const auto& choice = grammar->GetGrammarExpr(choice_id);
      if (choice.type == GrammarExprType::kEmptyStr) {
        has_empty_choice = true;
      } else if (choice.type == GrammarExprType::kSequence) {
        if (non_empty_sequence_id.has_value()) {
          return std::nullopt;
        }
        non_empty_sequence_id = choice_id;
      } else {
        return std::nullopt;
      }
    }
    if (!has_empty_choice || !non_empty_sequence_id.has_value()) {
      return std::nullopt;
    }

    const auto& sequence = grammar->GetGrammarExpr(*non_empty_sequence_id);
    if (sequence.size() < 1 || sequence.size() > 2) {
      return std::nullopt;
    }
    const auto& repeated_element = grammar->GetGrammarExpr(sequence[0]);
    if (repeated_element.type != GrammarExprType::kCharacterClass) {
      return std::nullopt;
    }
    uint64_t current_character_class_hash =
        HashCombine(uint64_t{0x4f50545f43484152ULL}, static_cast<int32_t>(repeated_element.type));
    for (int32_t value : repeated_element) {
      HashCombineBinary(current_character_class_hash, static_cast<uint64_t>(value));
    }
    if (character_class_hash.has_value() && *character_class_hash != current_character_class_hash) {
      return std::nullopt;
    }
    character_class_hash = current_character_class_hash;
    if (!character_class_expr_id.has_value()) {
      character_class_expr_id = sequence[0];
    }
    ++remaining_count;

    if (sequence.size() == 1) {
      return OptionalCharacterClassChain{
          *character_class_hash, *character_class_expr_id, remaining_count
      };
    }
    const auto& next = grammar->GetGrammarExpr(sequence[1]);
    if (next.type != GrammarExprType::kRuleRef || next.size() != 1) {
      return std::nullopt;
    }
    rule_id = next[0];
  }
  return std::nullopt;
}

OptionalCharacterClassTokenSummaryCache::Result BuildOptionalCharacterClassTokenSummaries(
    const Grammar::Impl::GrammarExpr& character_class,
    const std::vector<std::pair<int32_t, std::string>>& sorted_vocab
) {
  XGRAMMAR_DCHECK(character_class.type == Grammar::Impl::GrammarExprType::kCharacterClass);
  const bool is_negative = static_cast<bool>(character_class[0]);
  const auto codepoint_is_in_ranges = [&](TCodepoint codepoint) {
    for (int32_t range_index = 1; range_index < character_class.size(); range_index += 2) {
      if (codepoint >= character_class[range_index] &&
          codepoint <= character_class[range_index + 1]) {
        return true;
      }
    }
    return false;
  };
  const auto partial_codepoint_can_match =
      [&](TCodepoint partial, int32_t remaining_bytes, int32_t total_bytes) {
        if (is_negative) {
          return true;
        }
        static constexpr std::array<TCodepoint, 5> kMinCodepointByUtf8Length = {
            0, 0, 0x80, 0x800, 0x10000
        };
        const TCodepoint raw_min_codepoint = partial << (6 * remaining_bytes);
        const TCodepoint min_codepoint =
            std::max(raw_min_codepoint, kMinCodepointByUtf8Length[total_bytes]);
        const TCodepoint max_codepoint = std::min<TCodepoint>(
            raw_min_codepoint | ((TCodepoint{1} << (6 * remaining_bytes)) - 1), 0x10FFFF
        );
        if (min_codepoint > max_codepoint) {
          return false;
        }
        for (int32_t range_index = 1; range_index < character_class.size(); range_index += 2) {
          if (max_codepoint >= character_class[range_index] &&
              min_codepoint <= character_class[range_index + 1]) {
            return true;
          }
        }
        return false;
      };

  OptionalCharacterClassTokenSummaryCache::Result result;
  result.reserve(sorted_vocab.size());
  for (int32_t sorted_vocab_index = 0;
       sorted_vocab_index < static_cast<int32_t>(sorted_vocab.size());
       ++sorted_vocab_index) {
    const auto& [token_id, token] = sorted_vocab[sorted_vocab_index];
    static_cast<void>(token_id);
    int32_t byte_offset = 0;
    int32_t completed_characters = 0;
    bool incomplete_character = false;
    bool mismatch = false;
    while (byte_offset < static_cast<int32_t>(token.size())) {
      const uint8_t first_byte = static_cast<uint8_t>(token[byte_offset]);
      auto [valid_first_byte, total_bytes, partial_codepoint] = HandleUTF8FirstByte(first_byte);
      if (!valid_first_byte) {
        mismatch = true;
        break;
      }
      if (total_bytes == 1) {
        const bool in_ranges = codepoint_is_in_ranges(partial_codepoint);
        if (in_ranges == is_negative) {
          mismatch = true;
          break;
        }
        ++completed_characters;
        ++byte_offset;
      } else {
        int32_t consumed_bytes = 1;
        bool valid_continuations = true;
        while (consumed_bytes < total_bytes &&
               byte_offset + consumed_bytes < static_cast<int32_t>(token.size())) {
          const uint8_t continuation = static_cast<uint8_t>(token[byte_offset + consumed_bytes]);
          if ((continuation & 0xC0) != 0x80) {
            valid_continuations = false;
            break;
          }
          partial_codepoint = (partial_codepoint << 6) | (continuation & 0x3F);
          ++consumed_bytes;
        }
        if (!valid_continuations) {
          mismatch = true;
          break;
        }
        if (consumed_bytes < total_bytes) {
          incomplete_character = partial_codepoint_can_match(
              partial_codepoint, total_bytes - consumed_bytes, total_bytes
          );
          mismatch = !incomplete_character;
          byte_offset = token.size();
          break;
        }
        static constexpr std::array<TCodepoint, 5> kMinCodepointByUtf8Length = {
            0, 0, 0x80, 0x800, 0x10000
        };
        if (!is_negative && (partial_codepoint < kMinCodepointByUtf8Length[total_bytes] ||
                             partial_codepoint > 0x10FFFF)) {
          mismatch = true;
          break;
        }
        const bool in_ranges = codepoint_is_in_ranges(partial_codepoint);
        if (in_ranges == is_negative) {
          mismatch = true;
          break;
        }
        ++completed_characters;
        byte_offset += total_bytes;
      }
    }
    const bool consumed_whole_token =
        byte_offset == static_cast<int32_t>(token.size()) && !mismatch;
    const int32_t locally_consumed_characters =
        completed_characters + static_cast<int32_t>(incomplete_character);
    if (consumed_whole_token || completed_characters > 0) {
      result.push_back(OptionalCharacterClassTokenSummaryCache::TokenSummary{
          sorted_vocab_index,
          locally_consumed_characters,
          consumed_whole_token,
          completed_characters > 0
      });
    }
  }
  return result;
}

/*! \brief The concrete implementation of GrammarMatcherNode. */
class GrammarMatcherForTokenMaskCache : public EarleyParser {
 public:
#ifdef XGRAMMAR_PROFILE_COMPILE
  struct ProfileCounters {
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
    bool used_optional_character_class_direct_mask = false;
  };
#endif

  GrammarMatcherForTokenMaskCache(
      const Grammar& grammar,
      const EarleyParserGrammarMetadata& grammar_metadata,
      const ParserState& init_state,
      const std::unordered_map<int32_t, DynamicBitset>&
          tag_dispatch_rule_id_to_second_slicing_bitset,
      const TokenizerInfo& tokenizer_info,
      std::optional<RuleLevelCache>& rule_level_cache,
      const std::shared_ptr<FirstByteTokenMaskCache>& first_byte_cache,
      const std::shared_ptr<OptionalCharacterClassTokenSummaryCache>&
          optional_character_class_token_summary_cache,
      bool cache_direct_masks_across_grammars
  )
      : EarleyParser(grammar, grammar_metadata, init_state),
        init_rule_id_(init_state.rule_id),
        initial_state_(init_state),
        tag_dispatch_rule_id_to_second_slicing_bitset_(tag_dispatch_rule_id_to_second_slicing_bitset
        ),
        tokenizer_info_(tokenizer_info),
        rule_level_cache_(rule_level_cache),
        first_byte_cache_(first_byte_cache),
        optional_character_class_token_summary_cache_(optional_character_class_token_summary_cache),
        cache_direct_masks_across_grammars_(cache_direct_masks_across_grammars) {}
  /*!
   * \brief Get the adaptive token mask for the given ParserState.
   * \param is_root_rule Whether to consider the parent rule. If false, there will be
   * no uncertain tokens. Useful for the root rule.
   */
  AdaptiveTokenMask GetAdaptiveTokenMask(bool is_root_rule);

  std::optional<AdaptiveTokenMask> GetOptionalCharacterClassDirectMask(bool is_root_rule) const;

  /*!
   * \brief Get the token mask for the given ParserState.
   * \param first_char_mask The first character mask.
   * \param is_root_rule Whether to consider the parent rule. If false, there will be
   * no uncertain tokens. Useful for the root rule.
   * \returns True if the rejected indices are filled as usual, False otherwise.
   * It's used to determine which construction function will be used.
   */
  bool GetTokenMaskWithFirstCharacterCheck(
      const std::bitset<256>& first_char_mask,
      bool is_root_rule,
      const std::vector<int32_t>& token_edge_accepted
  );

  /*!
   * \brief Adapt the cache with lookahead assertion.
   * \param cache The adaptive token mask to be adapted.
   * \param is_root_rule Whether to consider the parent rule.
   */
  void AdaptCacheWithLookahead(AdaptiveTokenMask* cache, bool is_root_rule);

#ifdef XGRAMMAR_PROFILE_COMPILE
  const ProfileCounters& GetProfileCounters() const { return profile_counters_; }
#endif

 private:
  /*!
   * \brief Hash the sole direct character class repeated within the initial rule.
   *
   * Returns null when the rule contains fewer than two direct character-class elements or mixes
   * distinct classes.
   */
  std::optional<uint64_t> GetHomogeneousCharacterClassHash() const;

  /*!
   * \brief Count consecutive copies of the current character class along an ASCII transition.
   *
   * This recognizes the fixed unzip tail emitted in one FSM sequence. A null result means the
   * current state is not on such a homogeneous run.
   */
  std::optional<int32_t> GetRemainingAsciiCharacterClassRun(uint8_t first_byte) const;

  /*! \brief Get the rule-local canonical id of an FSM state. */
  std::optional<int32_t> GetCanonicalFsmStateId(int32_t state_id) const;

  /*! \brief Check if a token can pass the lookahead assertion. */
  std::pair</*acceptable*/ bool, /*can reach end*/ bool> IsTokenPassLookaheadAssertion(
      const std::string& token, const std::vector<uint8_t>& can_reach_end_stack
  );

  /*!
   * \brief Check if speculative calculation will be applied.
   * \return first: whether speculative calculation is applicable.
   * \return second: part of the first character mask,
   * which can be used in speculative calculation.
   */
  std::pair<bool, std::bitset<256>> GetSpeculativeCalculation();

  /*!
   * \brief Get the first character mask.
   * \param first_character_mask the bitset to store the first character mask.
   */
  void GetFirstCharacterMask(std::bitset<256>& first_character_mask);

  /*!
   * \brief Compute sorted vocab indices accepted by token edges at the current FSM state.
   * Token(ids) edges accept listed token IDs.
   * ExcludeToken(ids) edges accept all tokens except listed IDs.
   * \return Sorted, deduplicated vector of accepted sorted vocab indices.
   */
  const std::vector<int32_t>& GetTokenEdgeAcceptedIndices();

  // The id of the initial rule.
  int32_t init_rule_id_;

  // The initial state of the parser.
  ParserState initial_state_;

  /*!
   * \brief This is a mapping from TagDispatch rule id to the bitset used for second slicing.
   * \note If a rule is a TagDispatch rule, then there will be an AC automaton for its triggers.
   *  Which means that it can accept a lot of tokens. However, it will be slow to check a lot of
   *  tokens. The DynamicBitset here is used to do a second slicing: if a token's substr(1, n - 1)
   *  can be accepted by the start state of the AC automaton, then it will be True in the bitset.
   *  When we check a token, we first check if its first character can transit to the start state.
   *  If yes, then we check if it is in the bitset. If yes, then we accept it directly.
   */
  const std::unordered_map<int32_t, DynamicBitset>& tag_dispatch_rule_id_to_second_slicing_bitset_;

  const TokenizerInfo& tokenizer_info_;

  std::optional<RuleLevelCache> rule_level_cache_;

  std::shared_ptr<FirstByteTokenMaskCache> first_byte_cache_;

  std::shared_ptr<OptionalCharacterClassTokenSummaryCache>
      optional_character_class_token_summary_cache_;

  bool cache_direct_masks_across_grammars_;

  // Temporary data for GetAdaptiveTokenMask.
  std::vector<int32_t> tmp_accepted_indices_;
  std::vector<int32_t> tmp_rejected_indices_;
  std::vector<int32_t> tmp_uncertain_indices_;
  std::vector<int32_t> tmp_rejected_by_lookahead_indices_;
  std::vector<int32_t> tmp_accepted_by_lookahead_indices_;
  std::vector<uint8_t> tmp_can_reach_end_stack_;
  std::vector<uint8_t> tmp_can_reach_end_prefix_or_stack_;
  // Temporary data for GetTokenEdgeAcceptedIndices.
  std::vector<int32_t> tmp_token_edge_accepted_;
  std::vector<int32_t> tmp_token_edge_excluded_;
#ifdef XGRAMMAR_PROFILE_COMPILE
  ProfileCounters profile_counters_;
#endif
};

std::optional<AdaptiveTokenMask>
GrammarMatcherForTokenMaskCache::GetOptionalCharacterClassDirectMask(bool is_root_rule) const {
  if (is_root_rule || initial_state_.sub_element_id != 0 ||
      grammar_->GetRule(init_rule_id_).lookahead_assertion_id != -1 ||
      initial_state_.element_id != grammar_->per_rule_fsms[init_rule_id_]->GetFsm().GetStart()) {
    return std::nullopt;
  }
  const auto chain = RecognizeOptionalCharacterClassChain(grammar_, init_rule_id_);
  if (!chain.has_value()) {
    return std::nullopt;
  }
  const auto& character_class = grammar_->GetGrammarExpr(chain->character_class_expr_id);
  XGRAMMAR_DCHECK(character_class.type == Grammar::Impl::GrammarExprType::kCharacterClass);
  const auto& sorted_vocab = tokenizer_info_.GetSortedDecodedVocab();
  const size_t vocab_size = tokenizer_info_.GetVocabSize();
  OptionalCharacterClassTokenSummaryCache::Key summary_key;
  summary_key.character_class.assign(character_class.begin(), character_class.end());
  XGRAMMAR_DCHECK(optional_character_class_token_summary_cache_ != nullptr);
  const auto token_summaries =
      optional_character_class_token_summary_cache_->GetOrCreate(summary_key, [&]() {
        return BuildOptionalCharacterClassTokenSummaries(character_class, sorted_vocab);
      });

  std::vector<int32_t> accepted_indices;
  std::vector<int32_t> uncertain_indices;
  accepted_indices.reserve(token_summaries->size());
  uncertain_indices.reserve(token_summaries->size() / 4);
  for (const auto& summary : *token_summaries) {
    if (summary.consumed_whole_token &&
        summary.locally_consumed_characters <= chain->remaining_count) {
      accepted_indices.push_back(summary.sorted_vocab_index);
    } else if (summary.has_completed_character_prefix) {
      uncertain_indices.push_back(summary.sorted_vocab_index);
    }
  }
  return AdaptiveTokenMask(
      vocab_size, sorted_vocab, std::move(accepted_indices), std::move(uncertain_indices)
  );
}

std::optional<uint64_t> GrammarMatcherForTokenMaskCache::GetHomogeneousCharacterClassHash() const {
  using GrammarExprType = Grammar::Impl::GrammarExprType;

  const auto& body = grammar_->GetGrammarExpr(grammar_->GetRule(init_rule_id_).body_expr_id);
  if (body.type != GrammarExprType::kChoices) {
    return std::nullopt;
  }
  std::optional<uint64_t> character_class_hash;
  int32_t character_class_count = 0;
  for (int32_t choice_id : body) {
    const auto& choice = grammar_->GetGrammarExpr(choice_id);
    if (choice.type == GrammarExprType::kEmptyStr) {
      continue;
    }
    if (choice.type != GrammarExprType::kSequence) {
      return std::nullopt;
    }
    for (int32_t element_id : choice) {
      const auto& element = grammar_->GetGrammarExpr(element_id);
      if (element.type != GrammarExprType::kCharacterClass) {
        continue;
      }
      uint64_t current_hash =
          HashCombine(uint64_t{0x484f4d5f43484152ULL}, static_cast<int32_t>(element.type));
      for (int32_t value : element) {
        HashCombineBinary(current_hash, static_cast<uint64_t>(value));
      }
      if (character_class_hash.has_value() && *character_class_hash != current_hash) {
        return std::nullopt;
      }
      character_class_hash = current_hash;
      ++character_class_count;
    }
  }
  return character_class_count >= 2 ? character_class_hash : std::nullopt;
}

std::optional<int32_t> GrammarMatcherForTokenMaskCache::GetRemainingAsciiCharacterClassRun(
    uint8_t first_byte
) const {
  const auto& fsm = grammar_->per_rule_fsms[init_rule_id_]->GetFsm();
  auto get_char_edge_signature = [&](int32_t state_id) -> std::optional<uint64_t> {
    uint64_t signature = uint64_t{0x434841525f454447ULL};
    int32_t char_edge_count = 0;
    for (const auto& edge : fsm.GetFsm().GetEdges(state_id)) {
      if (!edge.IsCharRange()) {
        return std::nullopt;
      }
      signature = HashCombine(signature, edge.min, edge.max);
      ++char_edge_count;
    }
    return char_edge_count == 0 ? std::nullopt : std::optional<uint64_t>(signature);
  };

  const auto initial_signature = get_char_edge_signature(initial_state_.element_id);
  if (!initial_signature.has_value()) {
    return std::nullopt;
  }
  // A non-ASCII first byte enters the UTF-8 sub-FSM before completing the current character
  // class. Count the number of repeated class elements with an accepted ASCII probe instead. The
  // resulting byte-length saturation bound remains conservative for the original multi-byte
  // token, while allowing its bucket to share the same homogeneous-tail cache.
  uint8_t traversal_byte = first_byte;
  if (traversal_byte >= 0x80) {
    bool found_ascii_probe = false;
    for (const auto& edge : fsm.GetFsm().GetEdges(initial_state_.element_id)) {
      if (!edge.IsCharRange()) {
        return std::nullopt;
      }
      const int32_t probe_min = std::max<int32_t>(edge.min, 0);
      const int32_t probe_max = std::min<int32_t>(edge.max, 0x7f);
      if (probe_min <= probe_max) {
        traversal_byte = static_cast<uint8_t>(probe_min);
        found_ascii_probe = true;
        break;
      }
    }
    if (!found_ascii_probe) {
      return std::nullopt;
    }
  }
  int32_t state_id = initial_state_.element_id;
  int32_t run_length = 0;
  std::unordered_set<int32_t> visited_states;
  while (visited_states.insert(state_id).second) {
    const auto signature = get_char_edge_signature(state_id);
    if (!signature.has_value() || *signature != *initial_signature) {
      break;
    }
    std::optional<int32_t> next_state;
    for (const auto& edge : fsm.GetFsm().GetEdges(state_id)) {
      if (traversal_byte < edge.min || traversal_byte > edge.max) {
        continue;
      }
      if (next_state.has_value() && *next_state != edge.target) {
        return std::nullopt;
      }
      next_state = edge.target;
    }
    if (!next_state.has_value()) {
      break;
    }
    ++run_length;
    state_id = *next_state;
  }
  return run_length >= 2 ? std::optional<int32_t>(run_length) : std::nullopt;
}

std::optional<int32_t> GrammarMatcherForTokenMaskCache::GetCanonicalFsmStateId(int32_t state_id
) const {
  const auto& state_ids = grammar_->per_rule_fsm_new_state_ids[init_rule_id_];
  const auto it = std::find_if(state_ids.begin(), state_ids.end(), [&](const auto& item) {
    return item.first == state_id;
  });
  return it == state_ids.end() ? std::nullopt : std::optional<int32_t>(it->second);
}

void GrammarMatcherForTokenMaskCache::AdaptCacheWithLookahead(
    AdaptiveTokenMask* cache_ptr, bool is_root_rule
) {
  AdaptiveTokenMask& cache = *cache_ptr;
  const auto& sorted_decoded_vocab = tokenizer_info_.GetSortedDecodedVocab();
  const auto& subtree_nodes_range = tokenizer_info_.GetTrieSubtreeNodesRange();
  const std::string* prev_token = nullptr;
  bool is_exact_lookahead = grammar_->GetRule(init_rule_id_).is_exact_lookahead;
  int prev_matched_size = 0;
  int last_rejected_range = 0;
  int last_uncertain_range = 0;
  if (is_root_rule) {
    tmp_rejected_indices_ = cache.uncertain_indices;
  } else {
    const auto& lookahead_id = grammar_->GetRule(init_rule_id_).lookahead_assertion_id;
    if (lookahead_id == -1) {
      return;
    }
    for (const auto& uncertain_index : cache.uncertain_indices) {
      const auto& token = sorted_decoded_vocab[uncertain_index].second;
      // Many tokens may contain the same prefix, so we will avoid unnecessary matching
      // by finding the longest common prefix with the previous token.
      bool accepted = true;
      if (uncertain_index < last_rejected_range) {
        tmp_rejected_indices_.push_back(uncertain_index);
        continue;
      }
      if (uncertain_index < last_uncertain_range) {
        // This token is already marked as uncertain.
        continue;
      }
      if (prev_token != nullptr) {
        int lcp_len =
            std::mismatch(token.begin(), token.end(), prev_token->begin(), prev_token->end())
                .first -
            token.begin();
        if (lcp_len > prev_matched_size) {
          // Case 1. The common prefix is rejected by the matcher in the last token. Reject
          // directly.
          accepted = false;
        } else if (lcp_len < prev_matched_size) {
          // Case 2. The common prefix is shorter than the previous matched size. Rollback
          // the non-common part.
          PopLastStates(prev_matched_size - lcp_len);
          tmp_can_reach_end_stack_.resize(
              tmp_can_reach_end_stack_.size() - (prev_matched_size - lcp_len)
          );
          tmp_can_reach_end_prefix_or_stack_.resize(
              tmp_can_reach_end_prefix_or_stack_.size() - (prev_matched_size - lcp_len)
          );
        }
        prev_matched_size = std::min(prev_matched_size, lcp_len);
      }

      prev_token = &token;

      if (accepted) {
        // Accept the rest chars one by one.
        for (int j = prev_matched_size; j < static_cast<int>(token.size()); ++j) {
          if (!Advance(token[j])) {
            accepted = false;
            break;
          }
          tmp_can_reach_end_stack_.push_back(IsCompleted());
          tmp_can_reach_end_prefix_or_stack_.push_back(
              tmp_can_reach_end_stack_.back() || tmp_can_reach_end_prefix_or_stack_.back()
          );
          prev_matched_size = j + 1;
        }
      }

      XGRAMMAR_DCHECK(!tmp_can_reach_end_prefix_or_stack_.empty());
      bool can_reach_end = tmp_can_reach_end_prefix_or_stack_.back();

      XGRAMMAR_DCHECK(!accepted) << "All the tokens are at least uncertain!";
      if (can_reach_end && prev_matched_size > 0) {
        auto [lookahead_accepted, lookahead_completed] =
            IsTokenPassLookaheadAssertion(token, tmp_can_reach_end_stack_);
        if ((!is_root_rule) && lookahead_accepted) {
          if (lookahead_completed || !is_exact_lookahead) {
            tmp_uncertain_indices_.push_back(uncertain_index);
          } else {
            tmp_accepted_indices_.push_back(uncertain_index);
          }
        } else {
          tmp_rejected_indices_.push_back(uncertain_index);
          last_rejected_range = subtree_nodes_range[uncertain_index];
        }
      } else {
        tmp_rejected_indices_.push_back(uncertain_index);
        last_rejected_range = subtree_nodes_range[uncertain_index];
      }
    }
  }

  // This strategy ensures the consistency of the cache storage type in most cases.
  // However, in this case, the storage type is inconsistent:
  // 1. The original cache is accepted_indices, and rejected_indices is also small.
  // After adapting with lookahead, |accepted_indices| + |accepted_by_lookahead_indices| >
  // |rejected_indices| + |rejected_by_lookahead_indices|, and |rejected_indices| +
  // |rejected_by_lookahead_indices| < AdaptiveTokenMask::USE_BITSET_THRESHOLD. In this case, it
  // should be kRejected, but ignored.
  // 2. The original cache is rejected_indices, and accepted_indices is also small.
  // After adapting with lookahead, |accepted_indices| + |accepted_by_lookahead_indices| <
  // |rejected_indices| + |rejected_by_lookahead_indices|, and |accepted_indices| +
  // |accepted_by_lookahead_indices| < AdaptiveTokenMask::USE_BITSET_THRESHOLD. In this case, it
  // should be kAccepted, but ignored. These two cases are very rare in practice, and the impact is
  // very limited, so we ignore them for simplicity.
  cache.uncertain_indices = tmp_uncertain_indices_;
  switch (cache.store_type) {
    case AdaptiveTokenMask::StoreType::kAccepted: {
      if (cache.accepted_indices.size() + tmp_accepted_indices_.size() <
          AdaptiveTokenMask::USE_BITSET_THRESHOLD) {
        IntsetUnion(&cache.accepted_indices, tmp_accepted_indices_);
        break;
      }
      // Transform to bitset.
      cache.store_type = AdaptiveTokenMask::StoreType::kAcceptedBitset;
      cache.accepted_bitset = DynamicBitset(tokenizer_info_.GetVocabSize());
      for (const auto& accepted_index : cache.accepted_indices) {
        cache.accepted_bitset.Set(sorted_decoded_vocab[accepted_index].first);
      }
      for (const auto& accepted_index : tmp_accepted_indices_) {
        cache.accepted_bitset.Set(sorted_decoded_vocab[accepted_index].first);
      }
      cache.accepted_indices.clear();
      break;
    }
    case AdaptiveTokenMask::StoreType::kRejected: {
      if (cache.rejected_indices.size() + tmp_rejected_indices_.size() <
          AdaptiveTokenMask::USE_BITSET_THRESHOLD) {
        IntsetUnion(&cache.rejected_indices, tmp_rejected_indices_);
        break;
      }
      // Transform to bitset.
      cache.store_type = AdaptiveTokenMask::StoreType::kAcceptedBitset;
      cache.accepted_bitset = DynamicBitset(tokenizer_info_.GetVocabSize());
      cache.accepted_bitset.Set();
      for (const auto& special_index : tokenizer_info_.GetSpecialTokenIds()) {
        cache.accepted_bitset.Reset(special_index);
      }
      for (const auto& uncertain_index : cache.uncertain_indices) {
        cache.accepted_bitset.Reset(sorted_decoded_vocab[uncertain_index].first);
      }
      for (const auto& rejected_index : cache.rejected_indices) {
        cache.accepted_bitset.Reset(sorted_decoded_vocab[rejected_index].first);
      }
      for (const auto& rejected_index : tmp_rejected_indices_) {
        cache.accepted_bitset.Reset(sorted_decoded_vocab[rejected_index].first);
      }
      cache.rejected_indices.clear();
      break;
    }
    case AdaptiveTokenMask::StoreType::kAcceptedBitset: {
      for (const auto& accepted_index : tmp_accepted_indices_) {
        cache.accepted_bitset.Set(sorted_decoded_vocab[accepted_index].first);
      }
      break;
    }
  }
  cache.RebuildDerivedData(tokenizer_info_.GetVocabSize(), sorted_decoded_vocab);
}

std::pair<bool, bool> GrammarMatcherForTokenMaskCache::IsTokenPassLookaheadAssertion(
    const std::string& token, const std::vector<uint8_t>& can_reach_end_stack
) {
  bool accepted = true;
  bool can_reach_end = true;
  auto lookahead_assertion_id = grammar_->GetRule(init_rule_id_).lookahead_assertion_id;
  if (lookahead_assertion_id == -1) {
    return {accepted, can_reach_end};
  }
  auto lookahead_state =
      ParserState(/*rule_id*/ -1, lookahead_assertion_id, 0, ParserState::kNoPrevInputPos, 0);
  PushStateAndExpand(lookahead_state);
  int token_len = token.size();
  if (IsCompleted()) {
    // If the lookahead assertion is already completed, we can accept the token.
    PopLastStates(1);
    return {accepted, can_reach_end};
  }

  // Find all positions that can come to and end. Then check if the suffix from that position
  // can be accepted by the lookahead assertion.
  for (int i = static_cast<int>(can_reach_end_stack.size()) - 1; i >= 0; --i) {
    if (!can_reach_end_stack[i]) {
      continue;
    }
    int last_accept_pos = i - 1;
    for (int pos = i; pos < token_len; ++pos) {
      if (!Advance(token[pos])) {
        break;
      }
      last_accept_pos = pos;
      // Case 1. The whole rule is finished.
      if (IsCompleted()) {
        // accepted chars: pos - i + 1
        // we need to rollback the pushed initial state as well
        PopLastStates(pos - i + 2);
        return {accepted, can_reach_end};
      }
    }
    // Case 2. The whole token is accepted
    if (last_accept_pos == token_len - 1) {
      PopLastStates(last_accept_pos - i + 2);
      can_reach_end = false;
      return {accepted, can_reach_end};
    }
    // Case 3. The token is not accepted. Check the next position.
    PopLastStates(last_accept_pos - i + 1);
  }

  PopLastStates(1);
  can_reach_end = false;
  accepted = false;
  return {accepted, can_reach_end};
}

// Comparator for std::pair<int32_t, std::string> based on the string value.
class IntStringPairComparator {
 public:
  bool operator()(
      const std::pair<int32_t, std::string>& lhs, const std::pair<int32_t, std::string>& rhs
  ) const {
    return lhs.second < rhs.second;
  }
};

int GetPossibleTokenIntervals(
    const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
    const std::bitset<256>& first_char_mask,
    std::vector<std::pair<int32_t, int32_t>>& possible_intervals
) {
  int possible_token_num = 0;
  int matched_size = 0;
  int last_interval_end = -1;
  for (int32_t i = 0; i < 256; i++) {
    if (first_char_mask[i]) {
      if (last_interval_end == -1) {
        last_interval_end = i;
      }
    } else {
      if (last_interval_end != -1) {
        int32_t interval_left_end =
            std::lower_bound(
                sorted_decoded_vocab.begin() + matched_size,
                sorted_decoded_vocab.end(),
                std::make_pair(0, std::string(1, static_cast<uint8_t>(last_interval_end))),
                IntStringPairComparator()
            ) -
            sorted_decoded_vocab.begin();
        int32_t interval_right_end = std::lower_bound(
                                         sorted_decoded_vocab.begin() + interval_left_end,
                                         sorted_decoded_vocab.end(),
                                         std::make_pair(0, std::string(1, static_cast<uint8_t>(i))),
                                         IntStringPairComparator()
                                     ) -
                                     sorted_decoded_vocab.begin();
        possible_intervals.emplace_back(interval_left_end, interval_right_end);
        possible_token_num += interval_right_end - interval_left_end;
        last_interval_end = -1;
        matched_size = interval_right_end;
      }
    }
  }

  if (last_interval_end != -1) {
    // If the last interval is not closed, we need to close it.
    int32_t interval_left_end =
        std::lower_bound(
            sorted_decoded_vocab.begin() + matched_size,
            sorted_decoded_vocab.end(),
            std::make_pair(0, std::string(1, static_cast<uint8_t>(last_interval_end))),
            IntStringPairComparator()
        ) -
        sorted_decoded_vocab.begin();
    possible_intervals.emplace_back(interval_left_end, sorted_decoded_vocab.size());
    possible_token_num += sorted_decoded_vocab.size() - interval_left_end;
  }
  return possible_token_num;
}

std::pair<bool, std::bitset<256>> GrammarMatcherForTokenMaskCache::GetSpeculativeCalculation() {
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  // If the initial rule is a tag dispatch, we will check if it can achieve its initial state.
  const auto& rule = grammar_->GetRule(init_rule_id_);
  const auto& rule_body = grammar_->GetGrammarExpr(rule.body_expr_id);
  if (rule_body.type == GrammarExprType::kTagDispatch) {
    std::bitset<256> speculative_mask;
    XGRAMMAR_DCHECK(grammar_->per_rule_fsms[init_rule_id_].has_value());
    const auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
    for (const auto& edge : fsm.GetFsm().GetFsm().GetEdges(initial_state_.element_id)) {
      if (edge.target != fsm.GetFsm().GetStart()) {
        continue;
      }
      if (!edge.IsCharRange()) {
        continue;
      }
      for (int32_t ch = edge.min; ch <= edge.max; ++ch) {
        speculative_mask.set(ch);
      }
    }
    return {true, speculative_mask};
  }

  // Check if the initial state is self-recursive-like via FSM.
  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[init_rule_id_].has_value());
  bool can_be_applied = false;
  std::bitset<256> speculative_mask;
  const auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
  XGRAMMAR_DCHECK(initial_state_.element_id < fsm.GetFsm().NumStates())
      << "Initial State's element id cannot exceed the whole FSM's number of states.";
  for (const auto& edge : fsm.GetFsm().GetFsm().GetEdges(initial_state_.element_id)) {
    if (edge.IsCharRange()) {
      // Case A: The edge is towards itself.
      if (edge.target == initial_state_.element_id) {
        can_be_applied = true;
        for (int ch = edge.min; ch <= edge.max; ++ch) {
          speculative_mask.set(ch);
        }
        continue;
      }

      // Case B: The state is the start state, and there's an edge to another state,
      // which calls the fsm itself.
      if (fsm.GetFsm().GetStart() == initial_state_.element_id) {
        for (const auto& next_edge : fsm.GetFsm().GetFsm().GetEdges(edge.target)) {
          if ((next_edge.IsRuleRef() && next_edge.GetRefRuleId() == init_rule_id_) ||
              (next_edge.IsRepeatRef() &&
               fsm.GetFsm().GetFsm().GetRepeatEdgeInfo(next_edge.GetAuxIndex()).RuleId() ==
                   init_rule_id_)) {
            can_be_applied = true;
            for (int ch = edge.min; ch <= edge.max; ++ch) {
              speculative_mask.set(ch);
            }
            break;
          }
        }
      }
    }
  }
  return {can_be_applied, speculative_mask};
}

bool GrammarMatcherForTokenMaskCache::GetTokenMaskWithFirstCharacterCheck(
    const std::bitset<256>& first_char_mask,
    bool is_root_rule,
    const std::vector<int32_t>& token_edge_accepted
) {
  const auto& sorted_decoded_vocab = tokenizer_info_.GetSortedDecodedVocab();
  const auto& subtree_nodes_range = tokenizer_info_.GetTrieSubtreeNodesRange();
  // the pair (a, b) means [a, b). Intialize the possible intervals.
  std::vector<std::pair<int32_t, int32_t>> possible_intervals;
  int possible_token_num =
      GetPossibleTokenIntervals(sorted_decoded_vocab, first_char_mask, possible_intervals);
#ifdef XGRAMMAR_PROFILE_COMPILE
  profile_counters_.possible_tokens += possible_token_num;
#endif

  // Check if the type of the mask can be rejected.
  tmp_accepted_indices_.reserve(possible_token_num);
  bool fill_reject_indices =
      (sorted_decoded_vocab.size() - possible_token_num) < AdaptiveTokenMask::USE_BITSET_THRESHOLD;

  XGRAMMAR_DCHECK(possible_intervals.size() > 0)
      << "There should be at least one possible interval for the first character mask.";

  if (possible_intervals[0].first != 0 && fill_reject_indices) {
    for (int i = 0; i < possible_intervals[0].first; ++i) {
      tmp_rejected_indices_.push_back(i);
    }
  }

  XGRAMMAR_DCHECK(init_rule_id_ != -1 && grammar_->per_rule_fsms[init_rule_id_].has_value());
  auto [speculative_calculation, speculative_mask] = GetSpeculativeCalculation();

  int prev_matched_size = 0;
  int last_rejected_range = 0;
  const bool& is_exact_lookahead = grammar_->GetRule(init_rule_id_).is_exact_lookahead;
  std::optional<const DynamicBitset*> definite_accepted_bitset = std::nullopt;
  const bool is_tag_dispatch_rule =
      grammar_->GetGrammarExpr(grammar_->GetRule(init_rule_id_).body_expr_id).type ==
      Grammar::Impl::GrammarExprType::kTagDispatch;
  if (is_tag_dispatch_rule) {
    XGRAMMAR_DCHECK(tag_dispatch_rule_id_to_second_slicing_bitset_.count(init_rule_id_) > 0);
    definite_accepted_bitset = &tag_dispatch_rule_id_to_second_slicing_bitset_.at(init_rule_id_);
  }

  const bool can_use_first_byte_cache =
      first_byte_cache_ != nullptr && token_edge_accepted.empty() && !speculative_calculation &&
      !is_root_rule && grammar_->GetRule(init_rule_id_).lookahead_assertion_id == -1 &&
      initial_state_.sub_element_id == 0 && !is_tag_dispatch_rule &&
      !grammar_->per_rule_fsm_state_cache_keys[init_rule_id_].empty();
  const auto optional_character_class_chain =
      can_use_first_byte_cache ? RecognizeOptionalCharacterClassChain(grammar_, init_rule_id_)
                               : std::nullopt;
  const auto homogeneous_character_class_hash =
      can_use_first_byte_cache ? GetHomogeneousCharacterClassHash() : std::nullopt;
  const auto get_repeat_saturation_limit = [&](uint8_t first_byte) -> std::optional<size_t> {
    if (homogeneous_character_class_hash.has_value()) {
      const auto remaining_run = GetRemainingAsciiCharacterClassRun(first_byte);
      if (remaining_run.has_value()) {
        return static_cast<size_t>(*remaining_run - 1);
      }
    }
    if (optional_character_class_chain.has_value()) {
      return static_cast<size_t>(optional_character_class_chain->remaining_count - 1);
    }
    return std::nullopt;
  };
  const auto get_first_byte_cache_key =
      [&](uint8_t first_byte, bool collect_rejected, size_t max_suffix_bytes
      ) -> std::optional<FirstByteTokenMaskCache::Key> {
    if (!can_use_first_byte_cache) {
      return std::nullopt;
    }
    FirstByteTokenMaskCache::Key key{first_byte, collect_rejected, {}};
    const auto& fsm = grammar_->per_rule_fsms[init_rule_id_]->GetFsm();

    // The large-range expansion also emits one rule containing a fixed tail of 128 identical
    // character classes. If a token bucket is shorter than the remaining tail, it cannot observe
    // the exact position in that tail. Canonicalize those positions without changing the runtime
    // grammar or its precomputed masks.
    if (homogeneous_character_class_hash.has_value()) {
      const auto saturation_limit = get_repeat_saturation_limit(first_byte);
      if (saturation_limit.has_value() && max_suffix_bytes <= *saturation_limit) {
        key.continuations.push_back(Grammar::Impl::FSMStateCacheKey{
            HashCombine(uint64_t{0x5341545f46495844ULL}, *homogeneous_character_class_hash), -2, -2
        });
        return key;
      }
    }

    // A token whose suffix is shorter than the remaining optional character-class chain cannot
    // observe the chain's exact upper bound. For that vocabulary bucket, every longer chain has
    // identical behavior. Use a saturated, rule-local continuation key so the 128 helper states
    // emitted by a bounded string length share their already-computed bucket result. The byte
    // length is conservative for multi-byte codepoints: it can reduce sharing, never cross a
    // reachable repetition boundary.
    if (optional_character_class_chain.has_value() &&
        max_suffix_bytes <= *get_repeat_saturation_limit(first_byte)) {
      for (const auto& edge : fsm.GetFsm().GetEdges(initial_state_.element_id)) {
        if (!edge.IsCharRange() || first_byte < edge.min || first_byte > edge.max) {
          continue;
        }
        const auto canonical_target_id = GetCanonicalFsmStateId(edge.target);
        if (!canonical_target_id.has_value()) {
          key.continuations.clear();
          break;
        }
        key.continuations.push_back(Grammar::Impl::FSMStateCacheKey{
            HashCombine(
                uint64_t{0x5341545f4f505443ULL},
                optional_character_class_chain->character_class_hash,
                *canonical_target_id
            ),
            -1,
            -1
        });
      }
      if (!key.continuations.empty()) {
        std::sort(key.continuations.begin(), key.continuations.end());
        key.continuations.erase(
            std::unique(key.continuations.begin(), key.continuations.end()), key.continuations.end()
        );
        return key;
      }
    }

    const auto& state_cache_keys = grammar_->per_rule_fsm_state_cache_keys[init_rule_id_];
    for (const auto& edge : fsm.GetFsm().GetEdges(initial_state_.element_id)) {
      if (!edge.IsCharRange() || first_byte < edge.min || first_byte > edge.max) {
        continue;
      }
      const auto state_cache_key =
          std::find_if(state_cache_keys.begin(), state_cache_keys.end(), [&](const auto& item) {
            return item.first == edge.target;
          });
      if (state_cache_key == state_cache_keys.end()) {
        return std::nullopt;
      }
      key.continuations.push_back(state_cache_key->second);
    }
    if (key.continuations.empty()) {
      return std::nullopt;
    }
    std::sort(key.continuations.begin(), key.continuations.end());
    key.continuations.erase(
        std::unique(key.continuations.begin(), key.continuations.end()), key.continuations.end()
    );
    return key;
  };

  const std::string* prev_token = nullptr;
  int32_t skip_ptr = 0;
  const int32_t skip_size = static_cast<int32_t>(token_edge_accepted.size());
  for (size_t interval_idx = 0; interval_idx < possible_intervals.size(); ++interval_idx) {
    const auto& interval = possible_intervals[interval_idx];
    int group_begin = interval.first;
    while (group_begin < interval.second) {
      XGRAMMAR_DCHECK(!sorted_decoded_vocab[group_begin].second.empty());
      const uint8_t first_byte = static_cast<uint8_t>(sorted_decoded_vocab[group_begin].second[0]);
      size_t first_byte_group_max_suffix_bytes =
          sorted_decoded_vocab[group_begin].second.size() - 1;
      int first_byte_group_end = group_begin + 1;
      while (first_byte_group_end < interval.second &&
             !sorted_decoded_vocab[first_byte_group_end].second.empty() &&
             static_cast<uint8_t>(sorted_decoded_vocab[first_byte_group_end].second[0]) ==
                 first_byte) {
        first_byte_group_max_suffix_bytes = std::max(
            first_byte_group_max_suffix_bytes,
            sorted_decoded_vocab[first_byte_group_end].second.size() - 1
        );
        ++first_byte_group_end;
      }

      int group_end = first_byte_group_end;
      size_t group_max_suffix_bytes = first_byte_group_max_suffix_bytes;

      const bool collect_rejected = fill_reject_indices;
      auto first_byte_cache_key =
          get_first_byte_cache_key(first_byte, collect_rejected, group_max_suffix_bytes);
      std::shared_ptr<const FirstByteTokenMaskCache::Result> saturated_subset_cache;
      if (first_byte_cache_key.has_value()) {
        auto cached = first_byte_cache_->Get(*first_byte_cache_key);
        if (cached != nullptr && cached->covered_max_suffix_bytes >= group_max_suffix_bytes) {
#ifdef XGRAMMAR_PROFILE_COMPILE
          profile_counters_.first_byte_cache_reused_tokens += group_end - group_begin;
#endif
          tmp_accepted_indices_.insert(
              tmp_accepted_indices_.end(),
              cached->accepted_indices.begin(),
              cached->accepted_indices.end()
          );
          tmp_uncertain_indices_.insert(
              tmp_uncertain_indices_.end(),
              cached->uncertain_indices.begin(),
              cached->uncertain_indices.end()
          );
          if (collect_rejected) {
            tmp_rejected_indices_.insert(
                tmp_rejected_indices_.end(),
                cached->rejected_indices.begin(),
                cached->rejected_indices.end()
            );
            if (tmp_rejected_indices_.size() >= AdaptiveTokenMask::USE_BITSET_THRESHOLD) {
              fill_reject_indices = false;
            }
          }
          group_begin = group_end;
          continue;
        }
        saturated_subset_cache = std::move(cached);
      }

      // A long token in this first-byte bucket can observe the exact remaining repeat bound and
      // therefore prevents caching the whole bucket under the saturated key. The shorter tokens
      // are still indistinguishable from the same bucket at a longer repeat tail. Reuse their
      // classifications from that already-computed saturated bucket and simulate only the tokens
      // that can actually cross the current boundary.
      const auto repeat_saturation_limit = get_repeat_saturation_limit(first_byte);
      if (repeat_saturation_limit.has_value() &&
          group_max_suffix_bytes > *repeat_saturation_limit) {
        auto saturated_key =
            get_first_byte_cache_key(first_byte, collect_rejected, /*max_suffix_bytes=*/0);
        if (saturated_key.has_value() && saturated_subset_cache == nullptr) {
          saturated_subset_cache = first_byte_cache_->Get(*saturated_key);
        }
      }

      const size_t accepted_begin = tmp_accepted_indices_.size();
      const size_t rejected_begin = tmp_rejected_indices_.size();
      const size_t uncertain_begin = tmp_uncertain_indices_.size();
      size_t cached_accepted_pos = 0;
      size_t cached_rejected_pos = 0;
      size_t cached_uncertain_pos = 0;
      for (int i = group_begin; i < group_end; ++i) {
        if (saturated_subset_cache != nullptr && repeat_saturation_limit.has_value() &&
            sorted_decoded_vocab[i].second.size() - 1 <=
                std::min(
                    *repeat_saturation_limit, saturated_subset_cache->covered_max_suffix_bytes
                )) {
#ifdef XGRAMMAR_PROFILE_COMPILE
          ++profile_counters_.saturated_cache_reused_tokens;
#endif
          const auto advance_to_index = [&](const std::vector<int32_t>& indices, size_t* position) {
            while (*position < indices.size() && indices[*position] < i) {
              ++*position;
            }
            return *position < indices.size() && indices[*position] == i;
          };
          const bool is_accepted =
              advance_to_index(saturated_subset_cache->accepted_indices, &cached_accepted_pos);
          const bool is_rejected =
              collect_rejected &&
              advance_to_index(saturated_subset_cache->rejected_indices, &cached_rejected_pos);
          const bool is_uncertain =
              advance_to_index(saturated_subset_cache->uncertain_indices, &cached_uncertain_pos);
          const int cached_classification_count = static_cast<int>(is_accepted) +
                                                  static_cast<int>(is_rejected) +
                                                  static_cast<int>(is_uncertain);
          XGRAMMAR_DCHECK(
              cached_classification_count <= 1 &&
              (!collect_rejected || cached_classification_count == 1)
          );
          if (is_accepted) {
            tmp_accepted_indices_.push_back(i);
          } else if (is_uncertain) {
            tmp_uncertain_indices_.push_back(i);
          } else if (collect_rejected) {
            tmp_rejected_indices_.push_back(i);
            if (tmp_rejected_indices_.size() >= AdaptiveTokenMask::USE_BITSET_THRESHOLD) {
              fill_reject_indices = false;
            }
          }
          continue;
        }

        // Skip tokens already accepted by token edges (avoid expensive Earley simulation).
        while (skip_ptr < skip_size && token_edge_accepted[skip_ptr] < i) ++skip_ptr;
        if (skip_ptr < skip_size && token_edge_accepted[skip_ptr] == i) {
#ifdef XGRAMMAR_PROFILE_COMPILE
          ++profile_counters_.token_edge_skipped_tokens;
#endif
          continue;
        }

        // Check if the current token is in the rejected range. i.e. check if the current token
        // is on the subtree of the rejected token.
        if (i < last_rejected_range) {
          if (fill_reject_indices) {
#ifdef XGRAMMAR_PROFILE_COMPILE
            ++profile_counters_.subtree_pruned_tokens;
#endif
            tmp_rejected_indices_.push_back(i);
            fill_reject_indices =
                tmp_rejected_indices_.size() >= AdaptiveTokenMask::USE_BITSET_THRESHOLD
                    ? false
                    : fill_reject_indices;
          } else {
#ifdef XGRAMMAR_PROFILE_COMPILE
            profile_counters_.subtree_pruned_tokens += last_rejected_range - i;
#endif
            i = last_rejected_range - 1;
          }
          continue;
        }
        const auto& token = sorted_decoded_vocab[i].second;
        // This optimization is useful for simple self-recursive rules, like string content.
        if (speculative_calculation) {
          // Optimization for tag dispatch rules.
          if (definite_accepted_bitset.has_value()) {
            // If the token is empty, it must be accepted.
            if (token.empty()) {
#ifdef XGRAMMAR_PROFILE_COMPILE
              ++profile_counters_.speculative_accepted_tokens;
#endif
              tmp_accepted_indices_.push_back(i);
              continue;
            }
            // If the token doesn't contain tags or stop strings since the second character, and it
            // will transit to the start state after consuming the first character, it must be
            // accepted.
            if (speculative_mask[static_cast<uint8_t>(token[0])] &&
                (*definite_accepted_bitset.value())[i]) {
#ifdef XGRAMMAR_PROFILE_COMPILE
              ++profile_counters_.speculative_accepted_tokens;
#endif
              tmp_accepted_indices_.push_back(i);
              continue;
            }
          } else {
            bool all_accepted = true;
            for (char ch : token) {
              // If the first character is not the ascii character or can't be accepted by the
              // first character mask, we need to check them in the parser.
              if (isascii(ch) == 0 || !speculative_mask[static_cast<uint8_t>(ch)]) {
                all_accepted = false;
                break;
              }
            }
            if (all_accepted) {
#ifdef XGRAMMAR_PROFILE_COMPILE
              ++profile_counters_.speculative_accepted_tokens;
#endif
              tmp_accepted_indices_.push_back(i);
              continue;
            }
          }
        }
        // Many tokens may contain the same prefix, so we will avoid unnecessary matching
        // by finding the longest common prefix with the previous token.
#ifdef XGRAMMAR_PROFILE_COMPILE
        ++profile_counters_.parser_simulated_tokens;
        profile_counters_.parser_naive_token_bytes += token.size();
#endif
        bool accepted = true;
        if (prev_token != nullptr) {
          int lcp_len =
              std::mismatch(token.begin(), token.end(), prev_token->begin(), prev_token->end())
                  .first -
              token.begin();
          if (lcp_len > prev_matched_size) {
            // Case 1. The common prefix is rejected by the matcher in the last token. Reject
            // directly.
            accepted = false;
          } else if (lcp_len < prev_matched_size) {
            // Case 2. The common prefix is shorter than the previous matched size. Rollback
            // the non-common part.
            PopLastStates(prev_matched_size - lcp_len);
            tmp_can_reach_end_stack_.resize(
                tmp_can_reach_end_stack_.size() - (prev_matched_size - lcp_len)
            );
            tmp_can_reach_end_prefix_or_stack_.resize(
                tmp_can_reach_end_prefix_or_stack_.size() - (prev_matched_size - lcp_len)
            );
          }
          prev_matched_size = std::min(prev_matched_size, lcp_len);
        }

        prev_token = &token;

        if (accepted) {
          // Accept the rest chars one by one.
          for (int j = prev_matched_size; j < static_cast<int>(token.size()); ++j) {
#ifdef XGRAMMAR_PROFILE_COMPILE
            ++profile_counters_.parser_advance_calls;
#endif
            if (!Advance(token[j])) {
#ifdef XGRAMMAR_PROFILE_COMPILE
              ++profile_counters_.parser_failed_advance_calls;
#endif
              accepted = false;
              break;
            }
            tmp_can_reach_end_stack_.push_back(IsCompleted());
            tmp_can_reach_end_prefix_or_stack_.push_back(
                tmp_can_reach_end_stack_.back() || tmp_can_reach_end_prefix_or_stack_.back()
            );
            prev_matched_size = j + 1;
          }
        }

        bool can_reach_end = tmp_can_reach_end_prefix_or_stack_.back();

        if (accepted) {
          tmp_accepted_indices_.push_back(i);
        } else if (can_reach_end && prev_matched_size > 0) {
          auto [lookahead_accepted, lookahead_completed] =
              IsTokenPassLookaheadAssertion(token, tmp_can_reach_end_stack_);
          if ((!is_root_rule) && lookahead_accepted) {
            if (lookahead_completed || !is_exact_lookahead) {
              tmp_uncertain_indices_.push_back(i);
            } else {
              tmp_accepted_indices_.push_back(i);
              tmp_accepted_by_lookahead_indices_.push_back(i);
            }
          } else {
#ifdef XGRAMMAR_PROFILE_COMPILE
            profile_counters_.subtree_pruned_tokens += subtree_nodes_range[i] - i - 1;
#endif
            for (int j = i; j < subtree_nodes_range[i]; j++) {
              tmp_rejected_indices_.push_back(j);
              tmp_rejected_by_lookahead_indices_.push_back(j);
            }
            i = subtree_nodes_range[i] - 1;  // Skip the subtree nodes.
          }
        } else {
          tmp_rejected_indices_.push_back(i);
          last_rejected_range = subtree_nodes_range[i];
          fill_reject_indices =
              tmp_rejected_indices_.size() >= AdaptiveTokenMask::USE_BITSET_THRESHOLD
                  ? false
                  : fill_reject_indices;
        }
      }

      PopLastStates(prev_matched_size);
      tmp_can_reach_end_stack_.resize(tmp_can_reach_end_stack_.size() - prev_matched_size);
      tmp_can_reach_end_prefix_or_stack_.resize(
          tmp_can_reach_end_prefix_or_stack_.size() - prev_matched_size
      );
      prev_matched_size = 0;
      prev_token = nullptr;
      last_rejected_range = 0;

      if (first_byte_cache_key.has_value() && (!collect_rejected || fill_reject_indices)) {
        if (repeat_saturation_limit.has_value() &&
            group_max_suffix_bytes > *repeat_saturation_limit &&
            (saturated_subset_cache == nullptr ||
             saturated_subset_cache->covered_max_suffix_bytes < *repeat_saturation_limit)) {
          const auto filter_saturated_indices = [&](auto begin, auto end) {
            std::vector<int32_t> result;
            result.reserve(static_cast<size_t>(end - begin));
            for (auto it = begin; it != end; ++it) {
              if (sorted_decoded_vocab[*it].second.size() - 1 <= *repeat_saturation_limit) {
                result.push_back(*it);
              }
            }
            return result;
          };
          auto saturated_key =
              get_first_byte_cache_key(first_byte, collect_rejected, /*max_suffix_bytes=*/0);
          if (saturated_key.has_value()) {
            first_byte_cache_->Add(
                std::move(*saturated_key),
                FirstByteTokenMaskCache::Result{
                    filter_saturated_indices(
                        tmp_accepted_indices_.begin() + accepted_begin, tmp_accepted_indices_.end()
                    ),
                    filter_saturated_indices(
                        tmp_rejected_indices_.begin() + rejected_begin, tmp_rejected_indices_.end()
                    ),
                    filter_saturated_indices(
                        tmp_uncertain_indices_.begin() + uncertain_begin,
                        tmp_uncertain_indices_.end()
                    ),
                    *repeat_saturation_limit
                }
            );
          }
        }
        first_byte_cache_->Add(
            std::move(*first_byte_cache_key),
            FirstByteTokenMaskCache::Result{
                {tmp_accepted_indices_.begin() + accepted_begin, tmp_accepted_indices_.end()},
                {tmp_rejected_indices_.begin() + rejected_begin, tmp_rejected_indices_.end()},
                {tmp_uncertain_indices_.begin() + uncertain_begin, tmp_uncertain_indices_.end()},
                group_max_suffix_bytes
            }
        );
      }
      group_begin = group_end;
    }
    if (interval_idx != possible_intervals.size() - 1 && fill_reject_indices) {
      const auto& next_interval = possible_intervals[interval_idx + 1];
      for (int i = interval.second; i < next_interval.first; ++i) {
        tmp_rejected_indices_.push_back(i);
      }
      fill_reject_indices = tmp_rejected_indices_.size() >= AdaptiveTokenMask::USE_BITSET_THRESHOLD
                                ? false
                                : fill_reject_indices;
    }
  }

  XGRAMMAR_DCHECK(prev_matched_size == 0);

  if (possible_intervals.back().second != static_cast<int>(sorted_decoded_vocab.size()) &&
      fill_reject_indices) {
    // If the last interval is not closed, we need to reject the rest tokens.
    for (int i = possible_intervals.back().second;
         i < static_cast<int>(sorted_decoded_vocab.size());
         ++i) {
      tmp_rejected_indices_.push_back(i);
    }
  }

  return fill_reject_indices;
}

void GrammarMatcherForTokenMaskCache::GetFirstCharacterMask(std::bitset<256>& first_character_mask
) {
  first_character_mask.reset();
  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[init_rule_id_].has_value());
  const auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
  const auto& edges = fsm.GetFsm().GetFsm().GetEdges(initial_state_.element_id);
  for (const auto& edge : edges) {
    if (edge.IsCharRange()) {
      for (int c = edge.min; c <= edge.max; ++c) {
        first_character_mask[c] = true;
      }
    }
  }
}

const std::vector<int32_t>& GrammarMatcherForTokenMaskCache::GetTokenEdgeAcceptedIndices() {
  // Compute sorted vocab indices accepted by Token(ids) and ExcludeToken(ids) edges.
  // Result is stored in tmp_token_edge_accepted_.

  tmp_token_edge_accepted_.clear();
  tmp_token_edge_excluded_.clear();

  XGRAMMAR_DCHECK(grammar_->per_rule_fsms[init_rule_id_].has_value());
  const auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
  const auto& edges = fsm.GetFsm().GetFsm().GetEdges(initial_state_.element_id);

  const auto& sorted_decoded_vocab = tokenizer_info_.GetSortedDecodedVocab();
  int32_t sorted_size = static_cast<int32_t>(sorted_decoded_vocab.size());
  const auto& tid_to_sorted = tokenizer_info_.ImplPtr()->GetTokenIdToSortedVocabIndex();

  bool has_exclude_token = false;

  for (const auto& edge : edges) {
    if (edge.IsToken()) {
      auto info = fsm.GetFsm().GetFsm().GetTokenEdgeInfo(edge.GetAuxIndex());
      for (int32_t i = 0; i < info.Count(); ++i) {
        int32_t tid = info.TokenIds()[i];
        XGRAMMAR_DCHECK(tid >= 0 && tid < static_cast<int32_t>(tid_to_sorted.size()));
        if (tid_to_sorted[tid] >= 0) {
          tmp_token_edge_accepted_.push_back(tid_to_sorted[tid]);
        }
      }
    } else if (edge.IsExcludeToken()) {
      has_exclude_token = true;
      auto info = fsm.GetFsm().GetFsm().GetExcludeTokenEdgeInfo(edge.GetAuxIndex());
      for (int32_t i = 0; i < info.Count(); ++i) {
        int32_t tid = info.TokenIds()[i];
        XGRAMMAR_DCHECK(tid >= 0 && tid < static_cast<int32_t>(tid_to_sorted.size()));
        if (tid_to_sorted[tid] >= 0) {
          tmp_token_edge_excluded_.push_back(tid_to_sorted[tid]);
        }
      }
    }
  }

  // Token-only: result = token_accepted
  if (!has_exclude_token) {
    if (!tmp_token_edge_accepted_.empty()) {
      std::sort(tmp_token_edge_accepted_.begin(), tmp_token_edge_accepted_.end());
      tmp_token_edge_accepted_.erase(
          std::unique(tmp_token_edge_accepted_.begin(), tmp_token_edge_accepted_.end()),
          tmp_token_edge_accepted_.end()
      );
    }
    return tmp_token_edge_accepted_;
  }

  // ExcludeToken: result = [0, sorted_size) - (excluded - token_accepted)
  // Token(ids) overrides ExcludeToken(ids) when both present.
  if (!tmp_token_edge_accepted_.empty()) {
    std::sort(tmp_token_edge_accepted_.begin(), tmp_token_edge_accepted_.end());
    tmp_token_edge_accepted_.erase(
        std::unique(tmp_token_edge_accepted_.begin(), tmp_token_edge_accepted_.end()),
        tmp_token_edge_accepted_.end()
    );
  }
  std::sort(tmp_token_edge_excluded_.begin(), tmp_token_edge_excluded_.end());
  tmp_token_edge_excluded_.erase(
      std::unique(tmp_token_edge_excluded_.begin(), tmp_token_edge_excluded_.end()),
      tmp_token_edge_excluded_.end()
  );
  IntsetDifference(&tmp_token_edge_excluded_, tmp_token_edge_accepted_);
  IntsetComplement(&tmp_token_edge_accepted_, sorted_size, tmp_token_edge_excluded_);
  return tmp_token_edge_accepted_;
}

AdaptiveTokenMask GrammarMatcherForTokenMaskCache::GetAdaptiveTokenMask(bool is_root_rule) {
  tmp_accepted_indices_.clear();
  tmp_rejected_indices_.clear();
  tmp_uncertain_indices_.clear();
  tmp_rejected_by_lookahead_indices_.clear();
  tmp_accepted_by_lookahead_indices_.clear();
  tmp_can_reach_end_prefix_or_stack_.clear();
  tmp_can_reach_end_stack_.clear();
  // For every character in the current token, stores whether it is possible to reach the end of
  // the rule when matching until this character. Store it in a stack for later rollback.
  tmp_can_reach_end_stack_.push_back(false);
  tmp_can_reach_end_prefix_or_stack_.push_back(false);

  // Try to get the crossing cache.
  bool rule_level_cache_is_available = rule_level_cache_.has_value();
  std::optional<uint64_t> fsm_hash = std::nullopt;
  int32_t new_state_id = -1;
  int32_t cache_state_count = -1;
  int32_t cache_edge_count = -1;
  std::optional<AdaptiveTokenMask> crossing_cache = std::nullopt;
  int lookahead_id = grammar_->GetRule(initial_state_.rule_id).lookahead_assertion_id;
  bool is_exact_lookahead = grammar_->GetRule(initial_state_.rule_id).is_exact_lookahead;
  std::optional<uint64_t> lookahead_hash = std::nullopt;
  if (rule_level_cache_is_available) {
    const auto& state_cache_keys = grammar_->per_rule_fsm_state_cache_keys[init_rule_id_];
    const auto state_cache_key =
        std::find_if(state_cache_keys.begin(), state_cache_keys.end(), [&](const auto& item) {
          return item.first == initial_state_.element_id;
        });
    if (state_cache_key != state_cache_keys.end()) {
      // This domain-separated key describes the complete continuation reachable from the current
      // state. It permits exact reuse across larger FSMs that differ only in unreachable prefixes.
      fsm_hash = state_cache_key->second.hash;
      new_state_id = 0;
      cache_state_count = state_cache_key->second.state_count;
      cache_edge_count = state_cache_key->second.edge_count;
    } else if (grammar_->per_rule_fsm_hashes[init_rule_id_].has_value()) {
      const auto& original_to_new_id = grammar_->per_rule_fsm_new_state_ids[init_rule_id_];
      fsm_hash = grammar_->per_rule_fsm_hashes[init_rule_id_].value();
      const auto& fsm = grammar_->per_rule_fsms[init_rule_id_].value();
      cache_state_count = fsm.GetNodeNum();
      cache_edge_count = fsm.GetEdgeNum();
      for (const auto& original_new_pair : original_to_new_id) {
        if (original_new_pair.first == initial_state_.element_id) {
          new_state_id = original_new_pair.second;
          break;
        }
      }
      XGRAMMAR_DCHECK(new_state_id != -1);
    } else {
      rule_level_cache_is_available = false;
    }
  }
  if (rule_level_cache_is_available) {
    lookahead_hash = GrammarFSMHasher::HashSequence(grammar_, lookahead_id);
    if (lookahead_hash.has_value()) {
      crossing_cache = rule_level_cache_->GetCache(
          HashCombine(fsm_hash.value(), lookahead_hash.value(), is_exact_lookahead),
          new_state_id,
          cache_state_count,
          cache_edge_count
      );
      if (crossing_cache.has_value()) {
        // A perfect match.
#ifdef XGRAMMAR_PROFILE_COMPILE
        ++profile_counters_.rule_cache_hits;
#endif
        return crossing_cache.value();
      }
    }
    crossing_cache = rule_level_cache_->GetCache(
        fsm_hash.value(), new_state_id, cache_state_count, cache_edge_count
    );
    // If the rule doesn't have a lookahead, then it is exactly the same fsm.
    if (crossing_cache.has_value()) {
      AdaptCacheWithLookahead(&crossing_cache.value(), is_root_rule);
#ifdef XGRAMMAR_PROFILE_COMPILE
      ++profile_counters_.rule_cache_hits;
#endif
      return std::move(crossing_cache.value());
    }
  }

  if (auto direct_optional_chain_mask = GetOptionalCharacterClassDirectMask(is_root_rule)) {
#ifdef XGRAMMAR_PROFILE_COMPILE
    profile_counters_.used_optional_character_class_direct_mask = true;
#endif
    if (rule_level_cache_is_available && cache_direct_masks_across_grammars_) {
      rule_level_cache_->AddCache(
          fsm_hash.value(),
          new_state_id,
          cache_state_count,
          cache_edge_count,
          *direct_optional_chain_mask
      );
    }
    return std::move(*direct_optional_chain_mask);
  }

  std::bitset<256> first_character_mask;
  GetFirstCharacterMask(first_character_mask);

  // Token edge accepted indices (for byte path skip + merge).
  const auto& token_edge_accepted = GetTokenEdgeAcceptedIndices();

  // Byte path: skip tokens already accepted by token edges.
  bool rejected_filled;
  if (first_character_mask.none()) {
    rejected_filled = false;
  } else {
    rejected_filled = GetTokenMaskWithFirstCharacterCheck(
        first_character_mask, is_root_rule, token_edge_accepted
    );
  }

  // Merge: token edge accepted overrides byte path classification.
  // accepted  = accepted + token_edge_accepted
  // rejected  = rejected - token_edge_accepted
  // uncertain = uncertain - token_edge_accepted
  if (!token_edge_accepted.empty()) {
    IntsetUnion(&tmp_accepted_indices_, token_edge_accepted);
    IntsetDifference(&tmp_rejected_indices_, token_edge_accepted);
    IntsetDifference(&tmp_uncertain_indices_, token_edge_accepted);
  }
  if (rejected_filled) {
    auto return_value = AdaptiveTokenMask(
        tokenizer_info_.GetVocabSize(),
        tokenizer_info_.GetSortedDecodedVocab(),
        tmp_accepted_indices_,
        tmp_rejected_indices_,
        tmp_uncertain_indices_
    );
    if (rule_level_cache_is_available) {
      if (lookahead_id == -1 && !is_root_rule) {
        // If the rule doesn't have a lookahead, then it is exactly the same fsm.
        rule_level_cache_->AddCache(
            fsm_hash.value(), new_state_id, cache_state_count, cache_edge_count, return_value
        );
        return return_value;
      }

      // We can add a cache for basic fsm, and a better one for lookahead.
      // All the tokens rejected by lookahead should be uncertain.
      IntsetUnion(&tmp_uncertain_indices_, tmp_rejected_by_lookahead_indices_);
      IntsetUnion(&tmp_uncertain_indices_, tmp_accepted_by_lookahead_indices_);
      std::vector<int32_t> rejected_indices_without_lookahead;
      std::vector<int32_t> accepted_indices_without_lookahead;
      rejected_indices_without_lookahead.reserve(
          tmp_rejected_indices_.size() - tmp_rejected_by_lookahead_indices_.size()
      );
      accepted_indices_without_lookahead.reserve(
          tmp_accepted_indices_.size() - tmp_accepted_by_lookahead_indices_.size()
      );
      std::set_difference(
          tmp_rejected_indices_.begin(),
          tmp_rejected_indices_.end(),
          tmp_rejected_by_lookahead_indices_.begin(),
          tmp_rejected_by_lookahead_indices_.end(),
          std::back_inserter(rejected_indices_without_lookahead)
      );
      std::set_difference(
          tmp_accepted_indices_.begin(),
          tmp_accepted_indices_.end(),
          tmp_accepted_by_lookahead_indices_.begin(),
          tmp_accepted_by_lookahead_indices_.end(),
          std::back_inserter(accepted_indices_without_lookahead)
      );
      rule_level_cache_->AddCache(
          fsm_hash.value(),
          new_state_id,
          cache_state_count,
          cache_edge_count,
          AdaptiveTokenMask(
              tokenizer_info_.GetVocabSize(),
              tokenizer_info_.GetSortedDecodedVocab(),
              accepted_indices_without_lookahead,
              rejected_indices_without_lookahead,
              tmp_uncertain_indices_
          )
      );
      if (lookahead_hash.has_value()) {
        rule_level_cache_->AddCache(
            HashCombine(fsm_hash.value(), lookahead_hash.value(), is_exact_lookahead),
            new_state_id,
            cache_state_count,
            cache_edge_count,
            return_value
        );
      }
    }
    return return_value;
  } else {
    auto return_value = AdaptiveTokenMask(
        tokenizer_info_.GetVocabSize(),
        tokenizer_info_.GetSortedDecodedVocab(),
        tmp_accepted_indices_,
        tmp_uncertain_indices_
    );

    if (rule_level_cache_is_available) {
      // Prepare for cache.
      if (lookahead_id == -1 && !is_root_rule) {
        // If the rule doesn't have a lookahead, then it is exactly the same fsm.
        rule_level_cache_->AddCache(
            fsm_hash.value(), new_state_id, cache_state_count, cache_edge_count, return_value
        );
        return return_value;
      }

      // Add 2 caches.
      IntsetUnion(&tmp_uncertain_indices_, tmp_rejected_by_lookahead_indices_);
      IntsetUnion(&tmp_uncertain_indices_, tmp_accepted_by_lookahead_indices_);
      std::vector<int32_t> accepted_indices_without_lookahead;
      accepted_indices_without_lookahead.reserve(
          tmp_accepted_indices_.size() - tmp_accepted_by_lookahead_indices_.size()
      );
      std::set_difference(
          tmp_accepted_indices_.begin(),
          tmp_accepted_indices_.end(),
          tmp_accepted_by_lookahead_indices_.begin(),
          tmp_accepted_by_lookahead_indices_.end(),
          std::back_inserter(accepted_indices_without_lookahead)
      );
      rule_level_cache_->AddCache(
          fsm_hash.value(),
          new_state_id,
          cache_state_count,
          cache_edge_count,
          AdaptiveTokenMask(
              tokenizer_info_.GetVocabSize(),
              tokenizer_info_.GetSortedDecodedVocab(),
              accepted_indices_without_lookahead,
              tmp_uncertain_indices_
          )
      );

      if (lookahead_hash.has_value()) {
        rule_level_cache_->AddCache(
            HashCombine(fsm_hash.value(), lookahead_hash.value(), is_exact_lookahead),
            new_state_id,
            cache_state_count,
            cache_edge_count,
            return_value
        );
      }
    }
    return return_value;
  }
}

/******************* GrammarCompilerNoCache *******************/

/*!
 * \brief The base class for the grammar compiler. Handles the compilation logic without cache.
 */
class GrammarCompilerSub {
 public:
  GrammarCompilerSub(
      const TokenizerInfo& tokenizer_info,
      int max_threads,
      std::optional<RuleLevelCache> rule_level_cache
  )
      : tokenizer_info_(tokenizer_info),
        max_threads_(max_threads),
        thread_pool_(max_threads > 1 ? std::make_unique<ThreadPool>(max_threads) : nullptr),
        rule_level_cache_(rule_level_cache) {}

  CompiledGrammar CompileBuiltinJSONGrammar();

  CompiledGrammar CompileJSONSchema(
      const std::string& schema,
      bool any_whitespace,
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool strict_mode,
      std::optional<int> max_whitespace_cnt,
      bool any_order
  );

  CompiledGrammar CompileRegex(const std::string& regex);

  CompiledGrammar CompileStructuralTag(const std::string& structural_tag_json);

  CompiledGrammar CompileGrammar(const Grammar& grammar);

  CompiledGrammar CompileGrammar(const std::string& ebnf_str, std::string root_rule_name);

 private:
  /*! \brief The main logic. Compile the grammar with multi-threading. */
  CompiledGrammar MultiThreadCompileGrammar(Grammar grammar);
  /*! \brief Optimization for TagDispatch.
   *  \param compiled_grammar_impl the compiled_grammar to be optimized.
   *  \param tag_dispatch_rule_id_to_second_slicing_bitset Return value. Mapping from the rule_id to
   * the definite accepted token mask.
   */
  void TagDispatchOptimization(
      std::shared_ptr<CompiledGrammar::Impl> compiled_grammar_impl,
      std::unordered_map<int32_t, DynamicBitset>* tag_dispatch_rule_id_to_second_slicing_bitset
  );

  /*! \brief The vocabulary associated with this storage class. */
  const TokenizerInfo tokenizer_info_;
  /*! \brief The maximum number of threads to use. */
  const int max_threads_;
  /*! \brief Reused workers so each compile avoids thread startup and teardown. */
  std::unique_ptr<ThreadPool> thread_pool_;

  /*! \brief The manager of the rule level cache.*/
  std::optional<RuleLevelCache> rule_level_cache_;
};

struct AdaptiveMaskTaskKey {
  uint64_t exact_fsm_hash;
  int32_t canonical_state_id;
  int32_t state_count;
  int32_t edge_count;

  bool operator==(const AdaptiveMaskTaskKey& other) const {
    return exact_fsm_hash == other.exact_fsm_hash &&
           canonical_state_id == other.canonical_state_id && state_count == other.state_count &&
           edge_count == other.edge_count;
  }
};

struct AdaptiveMaskTaskKeyHash {
  size_t operator()(const AdaptiveMaskTaskKey& key) const {
    return HashCombine(key.exact_fsm_hash, key.canonical_state_id, key.state_count, key.edge_count);
  }
};

struct AdaptiveMaskTaskGroup {
  std::vector<ParserState> states;
  bool is_root_rule;
};

CompiledGrammar GrammarCompilerSub::MultiThreadCompileGrammar(Grammar grammar_unoptimized) {
#ifdef XGRAMMAR_PROFILE_COMPILE
  const auto compile_started_at = std::chrono::steady_clock::now();
  const auto optimizer_started_at = compile_started_at;
#endif
  auto compiled_grammar_impl = std::make_shared<CompiledGrammar::Impl>();
  compiled_grammar_impl->grammar = GrammarOptimizer::Apply(grammar_unoptimized);
#ifdef XGRAMMAR_PROFILE_COMPILE
  const auto optimizer_finished_at = std::chrono::steady_clock::now();
  const auto optimizer_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                        optimizer_finished_at - optimizer_started_at
  )
                                        .count();
  const auto earley_metadata_started_at = std::chrono::steady_clock::now();
#endif
  compiled_grammar_impl->earley_parser_metadata =
      EarleyParserGrammarMetadata(compiled_grammar_impl->grammar);
#ifdef XGRAMMAR_PROFILE_COMPILE
  const auto earley_metadata_elapsed_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - earley_metadata_started_at
      )
          .count();
#endif
  compiled_grammar_impl->tokenizer_info = tokenizer_info_;
  if (tokenizer_info_.GetVocabSize() == 0) {
    return CompiledGrammar(compiled_grammar_impl);
  }
#ifdef XGRAMMAR_PROFILE_COMPILE
  const auto tag_dispatch_started_at = std::chrono::steady_clock::now();
#endif
  std::unordered_map<int32_t, DynamicBitset> tag_dispatch_rule_id_to_second_slicing_bitset;
  TagDispatchOptimization(compiled_grammar_impl, &tag_dispatch_rule_id_to_second_slicing_bitset);
#ifdef XGRAMMAR_PROFILE_COMPILE
  const auto tag_dispatch_finished_at = std::chrono::steady_clock::now();
  const auto tag_dispatch_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                           tag_dispatch_finished_at - tag_dispatch_started_at
  )
                                           .count();
#endif

  // Reuse structurally identical rule FSMs within one compilation even when the persistent
  // cross-grammar cache is disabled. Large schemas contain many generated JSON rules with the
  // same automata; recomputing a full-vocabulary Earley mask for every copy dominates cold compile
  // latency. The local cache is discarded with this call and is bounded independently.
  constexpr size_t kLocalRuleCacheMaxBytes = 256 * 1024 * 1024;
  std::optional<RuleLevelCache> active_rule_level_cache = rule_level_cache_;
  const bool cache_direct_masks_across_grammars = rule_level_cache_.has_value();
  if (!active_rule_level_cache.has_value()) {
    active_rule_level_cache.emplace(kLocalRuleCacheMaxBytes);
  }
  constexpr size_t kFirstByteCacheMaxBytes = 64 * 1024 * 1024;
  auto first_byte_cache = std::make_shared<FirstByteTokenMaskCache>(kFirstByteCacheMaxBytes);
  auto optional_character_class_token_summary_cache =
      std::make_shared<OptionalCharacterClassTokenSummaryCache>();
#ifdef XGRAMMAR_PROFILE_COMPILE
  const auto fsm_hash_started_at = std::chrono::steady_clock::now();
#endif
  GrammarFSMHasher().Apply(&compiled_grammar_impl->grammar);
#ifdef XGRAMMAR_PROFILE_COMPILE
  const auto fsm_hash_finished_at = std::chrono::steady_clock::now();
  const auto fsm_hash_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                       fsm_hash_finished_at - fsm_hash_started_at
  )
                                       .count();
#endif
  // Step 3. Compute the adaptive token mask cache
  // The token mask cache is computed for these positions in the grammar:
  // 1. All character class or character class star (with last_utf8_bytes=0, 1, 2, 3)
  // 2. All byte strings (with element_in_string=0, 1, 2, ...)
  // since other positions will be expanded to the above positions

#ifdef XGRAMMAR_PROFILE_COMPILE
  struct AggregateMaskComputeProfile {
    std::atomic<uint64_t> mask_cpu_us{0};
    std::atomic<uint64_t> possible_tokens{0};
    std::atomic<uint64_t> rule_cache_hits{0};
    std::atomic<uint64_t> first_byte_cache_reused_tokens{0};
    std::atomic<uint64_t> saturated_cache_reused_tokens{0};
    std::atomic<uint64_t> token_edge_skipped_tokens{0};
    std::atomic<uint64_t> speculative_accepted_tokens{0};
    std::atomic<uint64_t> subtree_pruned_tokens{0};
    std::atomic<uint64_t> parser_simulated_tokens{0};
    std::atomic<uint64_t> parser_naive_token_bytes{0};
    std::atomic<uint64_t> parser_advance_calls{0};
    std::atomic<uint64_t> parser_failed_advance_calls{0};
    std::atomic<uint64_t> optional_character_class_direct_tasks{0};
    std::atomic<uint64_t> optional_character_class_direct_cpu_us{0};
  } aggregate_mask_compute_profile;
#endif

  std::mutex adaptive_token_mask_ids_mutex;
  auto add_adaptive_token_mask = [&](const AdaptiveMaskTaskGroup& task_group,
                                     size_t task_group_id) {
    XGRAMMAR_DCHECK(!task_group.states.empty());
    const ParserState& state = task_group.states.front();
#ifdef XGRAMMAR_PROFILE_COMPILE
    const auto mask_started_at = std::chrono::steady_clock::now();
#endif
    auto grammar_matcher = GrammarMatcherForTokenMaskCache(
        compiled_grammar_impl->grammar,
        compiled_grammar_impl->earley_parser_metadata,
        state,
        tag_dispatch_rule_id_to_second_slicing_bitset,
        tokenizer_info_,
        active_rule_level_cache,
        first_byte_cache,
        optional_character_class_token_summary_cache,
        cache_direct_masks_across_grammars
    );
    auto cur_adaptive_token_mask_cache =
        grammar_matcher.GetAdaptiveTokenMask(task_group.is_root_rule);
#ifdef XGRAMMAR_PROFILE_COMPILE
    const auto mask_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now() - mask_started_at
    )
                                     .count();
    aggregate_mask_compute_profile.mask_cpu_us.fetch_add(
        mask_elapsed_us, std::memory_order_relaxed
    );
    const auto& task_profile = grammar_matcher.GetProfileCounters();
    aggregate_mask_compute_profile.possible_tokens.fetch_add(
        task_profile.possible_tokens, std::memory_order_relaxed
    );
    aggregate_mask_compute_profile.rule_cache_hits.fetch_add(
        task_profile.rule_cache_hits, std::memory_order_relaxed
    );
    aggregate_mask_compute_profile.first_byte_cache_reused_tokens.fetch_add(
        task_profile.first_byte_cache_reused_tokens, std::memory_order_relaxed
    );
    aggregate_mask_compute_profile.saturated_cache_reused_tokens.fetch_add(
        task_profile.saturated_cache_reused_tokens, std::memory_order_relaxed
    );
    aggregate_mask_compute_profile.token_edge_skipped_tokens.fetch_add(
        task_profile.token_edge_skipped_tokens, std::memory_order_relaxed
    );
    aggregate_mask_compute_profile.speculative_accepted_tokens.fetch_add(
        task_profile.speculative_accepted_tokens, std::memory_order_relaxed
    );
    aggregate_mask_compute_profile.subtree_pruned_tokens.fetch_add(
        task_profile.subtree_pruned_tokens, std::memory_order_relaxed
    );
    aggregate_mask_compute_profile.parser_simulated_tokens.fetch_add(
        task_profile.parser_simulated_tokens, std::memory_order_relaxed
    );
    aggregate_mask_compute_profile.parser_naive_token_bytes.fetch_add(
        task_profile.parser_naive_token_bytes, std::memory_order_relaxed
    );
    aggregate_mask_compute_profile.parser_advance_calls.fetch_add(
        task_profile.parser_advance_calls, std::memory_order_relaxed
    );
    aggregate_mask_compute_profile.parser_failed_advance_calls.fetch_add(
        task_profile.parser_failed_advance_calls, std::memory_order_relaxed
    );
    if (task_profile.used_optional_character_class_direct_mask) {
      aggregate_mask_compute_profile.optional_character_class_direct_tasks.fetch_add(
          1, std::memory_order_relaxed
      );
      aggregate_mask_compute_profile.optional_character_class_direct_cpu_us.fetch_add(
          mask_elapsed_us, std::memory_order_relaxed
      );
    }
    AdaptiveMaskCompileProfile compile_profile;
    compile_profile.representative_state = state;
    compile_profile.mask_cpu_us = mask_elapsed_us;
    compile_profile.possible_tokens = task_profile.possible_tokens;
    compile_profile.rule_cache_hits = task_profile.rule_cache_hits;
    compile_profile.first_byte_cache_reused_tokens = task_profile.first_byte_cache_reused_tokens;
    compile_profile.saturated_cache_reused_tokens = task_profile.saturated_cache_reused_tokens;
    compile_profile.token_edge_skipped_tokens = task_profile.token_edge_skipped_tokens;
    compile_profile.speculative_accepted_tokens = task_profile.speculative_accepted_tokens;
    compile_profile.subtree_pruned_tokens = task_profile.subtree_pruned_tokens;
    compile_profile.parser_simulated_tokens = task_profile.parser_simulated_tokens;
    compile_profile.parser_naive_token_bytes = task_profile.parser_naive_token_bytes;
    compile_profile.parser_advance_calls = task_profile.parser_advance_calls;
    compile_profile.parser_failed_advance_calls = task_profile.parser_failed_advance_calls;
    compile_profile.mask_bytes = MemorySize(cur_adaptive_token_mask_cache);
    compile_profile.state_count = task_group.states.size();
    compiled_grammar_impl->adaptive_mask_compile_profiles[task_group_id] = compile_profile;
    if (mask_elapsed_us >= 2000) {
      uint64_t uncertain_hash = 0;
      for (int32_t index : cur_adaptive_token_mask_cache.uncertain_indices) {
        HashCombineBinary(uncertain_hash, static_cast<uint64_t>(index));
      }
      XGRAMMAR_LOG(INFO) << "SlowAdaptiveTokenMask(rule_id=" << state.rule_id << ", rule_name="
                         << compiled_grammar_impl->grammar->GetRule(state.rule_id).name
                         << ", state_id=" << state.element_id << ", elapsed_us=" << mask_elapsed_us
                         << ", accepted=" << cur_adaptive_token_mask_cache.accepted_indices.size()
                         << ", rejected=" << cur_adaptive_token_mask_cache.rejected_indices.size()
                         << ", uncertain=" << cur_adaptive_token_mask_cache.uncertain_indices.size()
                         << ", uncertain_hash=" << uncertain_hash << ")";
    }
#endif
    compiled_grammar_impl->adaptive_token_masks[task_group_id] =
        std::move(cur_adaptive_token_mask_cache);
    {
      std::lock_guard<std::mutex> lock(adaptive_token_mask_ids_mutex);
      for (const ParserState& grouped_state : task_group.states) {
        compiled_grammar_impl->adaptive_token_mask_ids.emplace(
            grouped_state, static_cast<uint32_t>(task_group_id)
        );
      }
    }
  };

  auto add_task_adaptive_token_mask = [&](const AdaptiveMaskTaskGroup& task_group,
                                          size_t task_group_id) {
    // Execute depending on whether we use thread_pool
    if (max_threads_ > 1) {
      thread_pool_->Execute([add_adaptive_token_mask, task_group, task_group_id]() {
        add_adaptive_token_mask(task_group, task_group_id);
      });
    } else {
      add_adaptive_token_mask(task_group, task_group_id);
    }
  };

  auto root_rule_id = compiled_grammar_impl->grammar->GetRootRuleId();
  std::vector<AdaptiveMaskTaskGroup> task_groups;
  std::unordered_map<AdaptiveMaskTaskKey, size_t, AdaptiveMaskTaskKeyHash> exact_task_group_indices;
  size_t scanable_state_count = 0;
#ifdef XGRAMMAR_PROFILE_COMPILE
  const auto task_discovery_started_at = std::chrono::steady_clock::now();
#endif

  for (int32_t rule_id = 0; rule_id < static_cast<int>(compiled_grammar_impl->grammar->NumRules());
       ++rule_id) {
    auto rule = compiled_grammar_impl->grammar->GetRule(rule_id);
    const auto& rule_fsm = compiled_grammar_impl->grammar->per_rule_fsms[rule_id];
    XGRAMMAR_DCHECK(rule_fsm.has_value());
    std::optional<uint64_t> lookahead_hash;
    const auto& fsm_hash = compiled_grammar_impl->grammar->per_rule_fsm_hashes[rule_id];
    const int32_t lookahead_id = rule.lookahead_assertion_id;
    if (lookahead_id != -1) {
      lookahead_hash = GrammarFSMHasher::HashSequence(compiled_grammar_impl->grammar, lookahead_id);
    }
    auto cur_stack_element =
        ParserState(rule_id, rule.body_expr_id, 0, ParserState::kNoPrevInputPos, 0);
    std::unordered_set<int> reachable_states;
    rule_fsm->GetFsm().GetReachableStates(&reachable_states);
    for (int i : reachable_states) {
      cur_stack_element.element_id = i;
      if (!rule_fsm->GetFsm().IsScanableState(i)) {
        continue;
      }
      ++scanable_state_count;
      std::optional<AdaptiveMaskTaskKey> exact_task_key;
      if (rule_id != root_rule_id && (lookahead_id == -1 || lookahead_hash.has_value())) {
        const auto& state_cache_keys =
            compiled_grammar_impl->grammar->per_rule_fsm_state_cache_keys[rule_id];
        const auto state_cache_key =
            std::find_if(state_cache_keys.begin(), state_cache_keys.end(), [i](const auto& item) {
              return item.first == i;
            });
        if (state_cache_key != state_cache_keys.end()) {
          const uint64_t exact_hash =
              lookahead_id == -1
                  ? state_cache_key->second.hash
                  : HashCombine(
                        state_cache_key->second.hash, *lookahead_hash, rule.is_exact_lookahead
                    );
          exact_task_key = AdaptiveMaskTaskKey{
              exact_hash, 0, state_cache_key->second.state_count, state_cache_key->second.edge_count
          };
        } else if (fsm_hash.has_value()) {
          const auto& original_to_new_id =
              compiled_grammar_impl->grammar->per_rule_fsm_new_state_ids[rule_id];
          const auto canonical_state = std::find_if(
              original_to_new_id.begin(),
              original_to_new_id.end(),
              [i](const auto& item) { return item.first == i; }
          );
          XGRAMMAR_DCHECK(canonical_state != original_to_new_id.end());
          const uint64_t exact_hash =
              lookahead_id == -1 ? *fsm_hash
                                 : HashCombine(*fsm_hash, *lookahead_hash, rule.is_exact_lookahead);
          exact_task_key = AdaptiveMaskTaskKey{
              exact_hash, canonical_state->second, rule_fsm->GetNodeNum(), rule_fsm->GetEdgeNum()
          };
        }
      }
      if (exact_task_key.has_value()) {
        const auto [group_it, inserted] =
            exact_task_group_indices.emplace(*exact_task_key, task_groups.size());
        if (inserted) {
          task_groups.push_back(AdaptiveMaskTaskGroup{{cur_stack_element}, /*is_root_rule=*/false});
        } else {
          task_groups[group_it->second].states.push_back(cur_stack_element);
        }
      } else {
        task_groups.push_back(AdaptiveMaskTaskGroup{{cur_stack_element}, rule_id == root_rule_id});
      }
    }
  }

#ifdef XGRAMMAR_PROFILE_COMPILE
  const auto task_discovery_finished_at = std::chrono::steady_clock::now();
  const auto task_discovery_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                             task_discovery_finished_at - task_discovery_started_at
  )
                                             .count();
#endif

  compiled_grammar_impl->adaptive_token_masks.resize(task_groups.size());
  compiled_grammar_impl->adaptive_token_mask_ids.reserve(scanable_state_count);
#ifdef XGRAMMAR_PROFILE_COMPILE
  compiled_grammar_impl->adaptive_mask_compile_profiles.resize(task_groups.size());
  const auto mask_wall_started_at = std::chrono::steady_clock::now();
#endif
  for (size_t task_group_id = 0; task_group_id < task_groups.size(); ++task_group_id) {
    add_task_adaptive_token_mask(task_groups[task_group_id], task_group_id);
  }

  if (max_threads_ > 1) {
    thread_pool_->Wait();
  }

#ifdef XGRAMMAR_PROFILE_COMPILE
  const auto mask_wall_finished_at = std::chrono::steady_clock::now();
  const auto mask_wall_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                        mask_wall_finished_at - mask_wall_started_at
  )
                                        .count();
  const auto compile_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                      mask_wall_finished_at - compile_started_at
  )
                                      .count();
  size_t hashable_rule_count = 0;
  size_t profiled_scanable_state_count = 0;
  for (int32_t rule_id = 0;
       rule_id < static_cast<int32_t>(compiled_grammar_impl->grammar->NumRules());
       ++rule_id) {
    hashable_rule_count += compiled_grammar_impl->grammar->per_rule_fsm_hashes[rule_id].has_value();
    std::unordered_set<int> reachable_states;
    compiled_grammar_impl->grammar->per_rule_fsms[rule_id]->GetFsm().GetReachableStates(
        &reachable_states
    );
    for (int state_id : reachable_states) {
      profiled_scanable_state_count +=
          compiled_grammar_impl->grammar->per_rule_fsms[rule_id]->GetFsm().IsScanableState(state_id
          );
    }
  }
  XGRAMMAR_LOG(INFO
  ) << "GrammarCompileProfile(rules="
    << compiled_grammar_impl->grammar->NumRules() << ", hashable_rules=" << hashable_rule_count
    << ", scanable_states=" << profiled_scanable_state_count
    << ", mask_task_groups=" << task_groups.size()
    << ", masks=" << compiled_grammar_impl->adaptive_token_mask_ids.size()
    << ", optimizer_us=" << optimizer_elapsed_us
    << ", earley_metadata_us=" << earley_metadata_elapsed_us
    << ", tag_dispatch_us=" << tag_dispatch_elapsed_us << ", fsm_hash_us=" << fsm_hash_elapsed_us
    << ", task_discovery_us=" << task_discovery_elapsed_us
    << ", mask_wall_us=" << mask_wall_elapsed_us << ", total_us=" << compile_elapsed_us
    << ", deterministic_byte_tables="
    << compiled_grammar_impl->earley_parser_metadata.deterministic_byte_transitions.size()
    << ", earley_metadata_bytes=" << MemorySize(compiled_grammar_impl->earley_parser_metadata)
    << ", mask_cpu_us=" << aggregate_mask_compute_profile.mask_cpu_us.load()
    << ", possible_tokens=" << aggregate_mask_compute_profile.possible_tokens.load()
    << ", rule_cache_hits=" << aggregate_mask_compute_profile.rule_cache_hits.load()
    << ", first_byte_cache_reused_tokens="
    << aggregate_mask_compute_profile.first_byte_cache_reused_tokens.load()
    << ", saturated_cache_reused_tokens="
    << aggregate_mask_compute_profile.saturated_cache_reused_tokens.load()
    << ", token_edge_skipped_tokens="
    << aggregate_mask_compute_profile.token_edge_skipped_tokens.load()
    << ", speculative_accepted_tokens="
    << aggregate_mask_compute_profile.speculative_accepted_tokens.load()
    << ", subtree_pruned_tokens=" << aggregate_mask_compute_profile.subtree_pruned_tokens.load()
    << ", parser_simulated_tokens=" << aggregate_mask_compute_profile.parser_simulated_tokens.load()
    << ", parser_naive_token_bytes="
    << aggregate_mask_compute_profile.parser_naive_token_bytes.load()
    << ", parser_advance_calls=" << aggregate_mask_compute_profile.parser_advance_calls.load()
    << ", parser_failed_advance_calls="
    << aggregate_mask_compute_profile.parser_failed_advance_calls.load()
    << ", optional_character_class_direct_tasks="
    << aggregate_mask_compute_profile.optional_character_class_direct_tasks.load()
    << ", optional_character_class_direct_cpu_us="
    << aggregate_mask_compute_profile.optional_character_class_direct_cpu_us.load()
    << ", compiled_bytes=" << MemorySize(*compiled_grammar_impl) << ")";
#endif

  return CompiledGrammar(compiled_grammar_impl);
}

CompiledGrammar GrammarCompilerSub::CompileBuiltinJSONGrammar() {
  return MultiThreadCompileGrammar(Grammar::BuiltinJSONGrammar());
}

CompiledGrammar GrammarCompilerSub::CompileJSONSchema(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    std::optional<int> max_whitespace_cnt,
    bool any_order
) {
#ifdef XGRAMMAR_PROFILE_COMPILE
  const auto frontend_started_at = std::chrono::steady_clock::now();
#endif
  auto grammar = Grammar::FromJSONSchema(
      schema,
      any_whitespace,
      indent,
      separators,
      strict_mode,
      max_whitespace_cnt,
      /*print_converted_ebnf=*/false,
      any_order
  );
#ifdef XGRAMMAR_PROFILE_COMPILE
  XGRAMMAR_LOG(INFO) << "JSONSchemaFrontendProfile(elapsed_us="
                     << std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - frontend_started_at
                        )
                            .count()
                     << ")";
#endif
  return MultiThreadCompileGrammar(std::move(grammar));
}

CompiledGrammar GrammarCompilerSub::CompileStructuralTag(const std::string& structural_tag_json) {
  auto result = Grammar::FromStructuralTag(structural_tag_json, tokenizer_info_);
  XGRAMMAR_CHECK(std::holds_alternative<Grammar>(result))
      << GetMessageFromVariantError(std::get<1>(result));
  return MultiThreadCompileGrammar(std::get<0>(result));
}

CompiledGrammar GrammarCompilerSub::CompileRegex(const std::string& regex) {
  return MultiThreadCompileGrammar(Grammar::FromRegex(regex));
}

CompiledGrammar GrammarCompilerSub::CompileGrammar(const Grammar& grammar) {
  return MultiThreadCompileGrammar(grammar);
}

CompiledGrammar GrammarCompilerSub::CompileGrammar(
    const std::string& ebnf_str, std::string root_rule_name
) {
  return MultiThreadCompileGrammar(Grammar::FromEBNF(ebnf_str, root_rule_name));
}

void GrammarCompilerSub::TagDispatchOptimization(
    std::shared_ptr<CompiledGrammar::Impl> compiled_grammar_impl,
    std::unordered_map<int32_t, DynamicBitset>* tag_dispatch_rule_id_to_second_slicing_bitset
) {
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  tag_dispatch_rule_id_to_second_slicing_bitset->clear();

  // Optimization for TagDispatch: Precompute the definitely accepted tokens.
  for (int i = 0; i < compiled_grammar_impl->grammar->NumRules(); i++) {
    const auto& rule = compiled_grammar_impl->grammar->GetRule(i);
    const auto& rule_body = compiled_grammar_impl->grammar->GetGrammarExpr(rule.body_expr_id);
    if (rule_body.type != GrammarExprType::kTagDispatch) {
      continue;
    }
    XGRAMMAR_DCHECK(rule_body.type == GrammarExprType::kTagDispatch);
    Grammar::Impl::TagDispatch tag_dispatch =
        compiled_grammar_impl->GetGrammar()->GetTagDispatch(rule.body_expr_id);
    const auto& sorted_decoded_vocab = tokenizer_info_.GetSortedDecodedVocab();
    DynamicBitset definite_accepted_tokens_since_second_char(sorted_decoded_vocab.size());
    for (int j = 0; j < static_cast<int32_t>(sorted_decoded_vocab.size()); j++) {
      bool definite_accept_since_second_char = true;
      const auto& token = sorted_decoded_vocab[j].second;
      if (token.empty()) {
        definite_accepted_tokens_since_second_char.Set(j);
        continue;
      }

      // Check if the token contains any string trigger or exclude string after first char.
      for (const auto& [trigger, rule_id] : tag_dispatch.tag_rule_pairs) {
        if (token.find(trigger, 1) != std::string::npos) {
          definite_accept_since_second_char = false;
          break;
        }
      }
      if (definite_accept_since_second_char) {
        for (const auto& excl : tag_dispatch.excludes) {
          if (token.find(excl, 1) != std::string::npos) {
            definite_accept_since_second_char = false;
            break;
          }
        }
      }

      if (definite_accept_since_second_char) {
        definite_accepted_tokens_since_second_char.Set(j);
      }
    }
    (*tag_dispatch_rule_id_to_second_slicing_bitset)[i] =
        definite_accepted_tokens_since_second_char;
  }
}

/******************* GrammarCompiler::Impl *******************/

/*!
 * \brief The keys for the cache. This is defined here instead of inside the GrammarCompiler::Impl
 * class due C++ template specialization and hash specialization rules.
 */
class GrammarCompilerCacheKeys {
 public:
  struct SchemaKey {
    std::string schema;
    bool any_whitespace;
    std::optional<int> indent;
    std::optional<std::pair<std::string, std::string>> separators;
    bool strict_mode;
    std::optional<int> max_whitespace_cnt;
    bool any_order;

    XGRAMMAR_EQUAL_BY_MEMBERS(
        SchemaKey,
        &SchemaKey::schema,
        &SchemaKey::any_whitespace,
        &SchemaKey::indent,
        &SchemaKey::separators,
        &SchemaKey::strict_mode,
        &SchemaKey::max_whitespace_cnt,
        &SchemaKey::any_order
    );
  };

  struct StructuralTagKey {
    std::string structural_tag_json;

    XGRAMMAR_EQUAL_BY_MEMBERS(StructuralTagKey, &StructuralTagKey::structural_tag_json);
  };

  struct GrammarKey {
    std::string ebnf_str;
    std::string root_rule_name;

    XGRAMMAR_EQUAL_BY_MEMBERS(GrammarKey, &GrammarKey::ebnf_str, &GrammarKey::root_rule_name);
  };

  struct RegexKey {
    std::string regex;

    XGRAMMAR_EQUAL_BY_MEMBERS(RegexKey, &RegexKey::regex);
  };

  struct BuiltinJSONGrammarKey {
    XGRAMMAR_EQUAL_BY_MEMBERS_EMPTY(BuiltinJSONGrammarKey);
  };

  using UnionKey =
      std::variant<SchemaKey, StructuralTagKey, GrammarKey, RegexKey, BuiltinJSONGrammarKey>;
};

}  // namespace xgrammar

XGRAMMAR_HASH_BY_MEMBERS(
    xgrammar::GrammarCompilerCacheKeys::SchemaKey,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::schema,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::any_whitespace,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::indent,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::separators,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::strict_mode,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::max_whitespace_cnt,
    &xgrammar::GrammarCompilerCacheKeys::SchemaKey::any_order
);

XGRAMMAR_HASH_BY_MEMBERS(
    xgrammar::GrammarCompilerCacheKeys::StructuralTagKey,
    &xgrammar::GrammarCompilerCacheKeys::StructuralTagKey::structural_tag_json
);

XGRAMMAR_HASH_BY_MEMBERS(
    xgrammar::GrammarCompilerCacheKeys::GrammarKey,
    &xgrammar::GrammarCompilerCacheKeys::GrammarKey::ebnf_str,
    &xgrammar::GrammarCompilerCacheKeys::GrammarKey::root_rule_name
);

XGRAMMAR_HASH_BY_MEMBERS(
    xgrammar::GrammarCompilerCacheKeys::RegexKey,
    &xgrammar::GrammarCompilerCacheKeys::RegexKey::regex
);

XGRAMMAR_HASH_BY_MEMBERS_EMPTY(xgrammar::GrammarCompilerCacheKeys::BuiltinJSONGrammarKey);

namespace xgrammar {

/*!
 * \brief The implementation of the grammar compiler with cache. It calls the no cache compiler
 * to compile the grammar, and implements the cache logic upon it.
 */
class GrammarCompiler::Impl {
 public:
  Impl(
      const TokenizerInfo& tokenizer_info,
      int max_threads,
      bool cache_enabled,
      int64_t max_memory_bytes
  )
      : cache_enabled_(cache_enabled),
        rule_level_cache_(
            cache_enabled
                ? std::optional<RuleLevelCache>(
                      max_memory_bytes == -1
                          ? static_cast<std::size_t>(-1)
                          : static_cast<std::size_t>(max_memory_bytes - max_memory_bytes / 3 * 2)
                  )
                : std::nullopt
        ),
        no_cache_compiler_(tokenizer_info, max_threads, rule_level_cache_),
        grammar_level_cache_(
            max_memory_bytes == -1 ? static_cast<std::size_t>(-1)
                                   : static_cast<std::size_t>(max_memory_bytes / 3 * 2),
            Computer(*this)
        ) {
    if (max_memory_bytes < -1) {
      XGRAMMAR_LOG(FATAL) << "Invalid max_memory_bytes: " << max_memory_bytes << ". "
                          << "It should be -1 (unlimited) or a non-negative integer.";
    }
  }

  CompiledGrammar CompileBuiltinJSONGrammar();

  CompiledGrammar CompileJSONSchema(
      const std::string& schema,
      bool any_whitespace,
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool strict_mode,
      std::optional<int> max_whitespace_cnt,
      bool any_order
  );

  CompiledGrammar CompileStructuralTag(const std::string& structural_tag_json);

  CompiledGrammar CompileRegex(const std::string& regex);

  CompiledGrammar CompileGrammar(const Grammar& grammar);

  CompiledGrammar CompileGrammar(const std::string& ebnf_str, std::string root_rule_name);

  void ClearCache();

  int64_t GetCacheSizeBytes() const;

  int64_t CacheLimitBytes() const;

 private:
  using SchemaKey = GrammarCompilerCacheKeys::SchemaKey;
  using StructuralTagKey = GrammarCompilerCacheKeys::StructuralTagKey;
  using GrammarKey = GrammarCompilerCacheKeys::GrammarKey;
  using RegexKey = GrammarCompilerCacheKeys::RegexKey;
  using BuiltinJSONGrammarKey = GrammarCompilerCacheKeys::BuiltinJSONGrammarKey;
  using UnionKey = GrammarCompilerCacheKeys::UnionKey;

  CompiledGrammar Compute(const UnionKey& key);

  struct Computer {
    Computer(Impl& compiler) : compiler(compiler) {}
    // Forward the key to GrammarCompiler::Impl::Compute(key)
    CompiledGrammar operator()(const UnionKey& key) const { return compiler.Compute(key); }
    GrammarCompiler::Impl& compiler;
  };

  struct SizeEstimator {
    std::size_t operator()(const CompiledGrammar& value) const { return value.MemorySizeBytes(); }
  };

  /*! \brief Whether the cache is enabled. */
  const bool cache_enabled_;

  /*! \brief The crossing cache manager for compiled grammars. */
  std::optional<RuleLevelCache> rule_level_cache_ = std::nullopt;

  /*! \brief The no cache compiler. */
  GrammarCompilerSub no_cache_compiler_;

  /*! \brief The cache for compiled grammars. */
  ThreadSafeLRUCache<UnionKey, CompiledGrammar, Computer, SizeEstimator> grammar_level_cache_;
};

CompiledGrammar GrammarCompiler::Impl::Compute(const UnionKey& key) {
  return std::visit(
      [this](const auto& key) -> CompiledGrammar {
        using KeyType = std::decay_t<decltype(key)>;
        if constexpr (std::is_same_v<KeyType, GrammarKey>) {
          const auto& [ebnf_str, root_rule_name] = key;
          return this->no_cache_compiler_.CompileGrammar(ebnf_str, root_rule_name);
        } else if constexpr (std::is_same_v<KeyType, SchemaKey>) {
          const auto& [schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, any_order] =
              key;
          return this->no_cache_compiler_.CompileJSONSchema(
              schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, any_order
          );
        } else if constexpr (std::is_same_v<KeyType, StructuralTagKey>) {
          const auto& [structural_tag_json] = key;
          return this->no_cache_compiler_.CompileStructuralTag(structural_tag_json);
        } else if constexpr (std::is_same_v<KeyType, RegexKey>) {
          const auto& [regex] = key;
          return this->no_cache_compiler_.CompileRegex(regex);
        } else if constexpr (std::is_same_v<KeyType, BuiltinJSONGrammarKey>) {
          return this->no_cache_compiler_.CompileBuiltinJSONGrammar();
        } else {
          XGRAMMAR_UNREACHABLE();
        }
      },
      key
  );
}

CompiledGrammar GrammarCompiler::Impl::CompileBuiltinJSONGrammar() {
  if (!cache_enabled_) {
    return no_cache_compiler_.CompileBuiltinJSONGrammar();
  }
  return grammar_level_cache_.Get(BuiltinJSONGrammarKey{});
}

CompiledGrammar GrammarCompiler::Impl::CompileJSONSchema(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    std::optional<int> max_whitespace_cnt,
    bool any_order
) {
  if (!cache_enabled_) {
    return no_cache_compiler_.CompileJSONSchema(
        schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, any_order
    );
  }
  return grammar_level_cache_.Get(SchemaKey{
      schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, any_order
  });
}

CompiledGrammar GrammarCompiler::Impl::CompileStructuralTag(const std::string& structural_tag_json
) {
  if (!cache_enabled_) {
    return no_cache_compiler_.CompileStructuralTag(structural_tag_json);
  }
  return grammar_level_cache_.Get(StructuralTagKey{structural_tag_json});
}

CompiledGrammar GrammarCompiler::Impl::CompileRegex(const std::string& regex) {
  if (!cache_enabled_) {
    return no_cache_compiler_.CompileRegex(regex);
  }
  return grammar_level_cache_.Get(RegexKey{regex});
}

CompiledGrammar GrammarCompiler::Impl::CompileGrammar(const Grammar& grammar) {
  if (!cache_enabled_) {
    return no_cache_compiler_.CompileGrammar(grammar);
  }
  return grammar_level_cache_.Get(GrammarKey{grammar.ToString(), grammar->GetRootRule().name});
}

CompiledGrammar GrammarCompiler::Impl::CompileGrammar(
    const std::string& ebnf_str, std::string root_rule_name
) {
  if (!cache_enabled_) {
    return no_cache_compiler_.CompileGrammar(ebnf_str, root_rule_name);
  }
  return grammar_level_cache_.Get(GrammarKey{ebnf_str, root_rule_name});
}

void GrammarCompiler::Impl::ClearCache() {
  grammar_level_cache_.Clear();
  if (rule_level_cache_.has_value()) {
    rule_level_cache_->ClearCache();
  }
}

int64_t GrammarCompiler::Impl::GetCacheSizeBytes() const {
  return static_cast<int64_t>(grammar_level_cache_.MemorySize()) +
         static_cast<int64_t>(MemorySize(rule_level_cache_));
}

int64_t GrammarCompiler::Impl::CacheLimitBytes() const {
  const auto size = grammar_level_cache_.MaxMemorySize();
  if (size == grammar_level_cache_.kUnlimitedSize) return -1;
  return static_cast<int64_t>(size) + (rule_level_cache_.has_value()
                                           ? static_cast<int64_t>(rule_level_cache_->GetMaxSize())
                                           : 0);
}

/******************* GrammarCompiler *******************/

GrammarCompiler::GrammarCompiler(
    const TokenizerInfo& tokenizer_info,
    int max_threads,
    bool cache_enabled,
    int64_t max_memory_bytes
)
    : pimpl_(std::make_shared<Impl>(tokenizer_info, max_threads, cache_enabled, max_memory_bytes)) {
}

CompiledGrammar GrammarCompiler::CompileJSONSchema(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    std::optional<int> max_whitespace_cnt,
    bool any_order
) {
  return pimpl_->CompileJSONSchema(
      schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, any_order
  );
}

CompiledGrammar GrammarCompiler::CompileBuiltinJSONGrammar() {
  return pimpl_->CompileBuiltinJSONGrammar();
}

CompiledGrammar GrammarCompiler::CompileStructuralTag(const std::string& structural_tag_json) {
  return pimpl_->CompileStructuralTag(structural_tag_json);
}

CompiledGrammar GrammarCompiler::CompileRegex(const std::string& regex) {
  return pimpl_->CompileRegex(regex);
}

CompiledGrammar GrammarCompiler::CompileGrammar(const Grammar& grammar) {
  return pimpl_->CompileGrammar(grammar);
}

CompiledGrammar GrammarCompiler::CompileGrammar(
    const std::string& ebnf_str, const std::string& root_rule_name
) {
  return pimpl_->CompileGrammar(ebnf_str, root_rule_name);
}

void GrammarCompiler::ClearCache() { pimpl_->ClearCache(); }

int64_t GrammarCompiler::GetCacheSizeBytes() const { return pimpl_->GetCacheSizeBytes(); }

int64_t GrammarCompiler::CacheLimitBytes() const { return pimpl_->CacheLimitBytes(); }

}  // namespace xgrammar
