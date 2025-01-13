/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/compiler.cc
 */

#include <xgrammar/compiler.h>

#include "compiled_grammar_data_structure.h"
#include "grammar_data_structure.h"
#include "grammar_functor.h"
#include "grammar_matcher_base.h"
#include "support/thread_pool.h"
#include "support/thread_safe_cache.h"

namespace xgrammar {

/******************* AdaptiveTokenMask and CompiledGrammar *******************/

AdaptiveTokenMask::AdaptiveTokenMask(
    size_t vocab_size,
    const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
    const std::vector<int32_t>& accepted_indices,
    const std::vector<int32_t>& rejected_indices,
    const std::vector<int32_t>& uncertain_indices
) {
  auto size_acc = accepted_indices.size();
  auto size_rej = rejected_indices.size();

  store_type = size_acc >= USE_BITSET_THRESHOLD && size_rej >= USE_BITSET_THRESHOLD
                   ? StoreType::kAcceptedBitset
               : size_acc < size_rej ? StoreType::kAccepted
                                     : StoreType::kRejected;

  if (store_type == StoreType::kAcceptedBitset) {
    accepted_bitset = DynamicBitset(vocab_size);
    for (auto idx : accepted_indices) {
      accepted_bitset.Set(sorted_decoded_vocab[idx].first, true);
    }
  } else if (store_type == StoreType::kAccepted) {
    this->accepted_indices = accepted_indices;
  } else {
    this->rejected_indices = rejected_indices;
  }

  this->uncertain_indices = uncertain_indices;
}

Grammar CompiledGrammar::GetGrammar() const { return pimpl_->GetGrammar(); }

TokenizerInfo CompiledGrammar::GetTokenizerInfo() const { return pimpl_->GetTokenizerInfo(); }

/******************* Use GrammarMatcher to generate the AdaptiveTokenMaskCache *******************/

/*! \brief The concrete implementation of GrammarMatcherNode. */
class GrammarMatcherForTokenMaskCache : public GrammarMatcherBase {
 public:
  // Do not expand the initial stack element: we want to find the accepted/rejected tokens
  // that exactly start from the initial stack element.
  GrammarMatcherForTokenMaskCache(const Grammar& grammar, StackElement init_stack_element)
      : GrammarMatcherBase(grammar, init_stack_element, false),
        init_rule_id(init_stack_element.rule_id) {}

  /*!
   * \brief Get the adaptive token mask for the given StackElement.
   * \param consider_parent_rule Whether to consider the parent rule. If false, there will be
   * no uncertain tokens. Useful for the root rule.
   */
  AdaptiveTokenMask GetAdaptiveTokenMask(
      size_t vocab_size,
      const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
      bool consider_parent_rule
  );

 private:
  /*! \brief Check if a token can pass the lookahead assertion. */
  bool IsTokenPassLookaheadAssertion(
      const std::string& token, const std::vector<bool>& can_reach_end_stack
  );

  // The id of the initial rule.
  int32_t init_rule_id;

  // Temporary data for GetAdaptiveTokenMask.
  std::vector<int32_t> tmp_accepted_indices_;
  std::vector<int32_t> tmp_rejected_indices_;
  std::vector<int32_t> tmp_uncertain_indices_;
  std::vector<bool> tmp_can_reach_end_stack_;
  std::vector<bool> tmp_can_reach_end_prefix_or_stack_;
};

bool GrammarMatcherForTokenMaskCache::IsTokenPassLookaheadAssertion(
    const std::string& token, const std::vector<bool>& can_reach_end_stack
) {
  auto lookahead_assertion_id = grammar_->GetRule(init_rule_id).lookahead_assertion_id;
  if (lookahead_assertion_id == -1) {
    return true;
  }
  auto lookahead_stack_element = StackElement(-1, lookahead_assertion_id, 0);
  PushInitialState(lookahead_stack_element, true);
  int token_len = token.size();

  // Find all positions that can come to and end. Then check if the suffix from that position
  // can be accepted by the lookahead assertion.
  for (int i = static_cast<int>(can_reach_end_stack.size()); i >= 0; --i) {
    if (!can_reach_end_stack[i]) {
      continue;
    }
    int last_accept_pos = i - 1;
    for (int pos = i; pos < token_len; ++pos) {
      if (!AcceptChar(token[pos])) {
        break;
      }
      last_accept_pos = pos;
      // Case 1. The whole rule is finished.
      if (CanReachEnd()) {
        // accepted chars: pos - i + 1
        // we need to rollback the pushed initial state as well
        RollbackChars(pos - i + 2);
        return true;
      }
    }
    // Case 2. The whole token is accepted
    if (last_accept_pos == token_len - 1) {
      RollbackChars(last_accept_pos - i + 2);
      return true;
    }
    // Case 3. The token is not accepted. Check the next position.
    RollbackChars(last_accept_pos - i + 1);
  }

  RollbackChars(1);
  return false;
}

AdaptiveTokenMask GrammarMatcherForTokenMaskCache::GetAdaptiveTokenMask(
    size_t vocab_size,
    const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
    bool consider_parent_rule
) {
  tmp_accepted_indices_.clear();
  tmp_rejected_indices_.clear();
  tmp_uncertain_indices_.clear();

  // For every character in the current token, stores whether it is possible to reach the end of
  // the rule when matching until this character. Store it in a stack for later rollback.
  tmp_can_reach_end_stack_.assign({CanReachEnd()});
  tmp_can_reach_end_prefix_or_stack_.assign({tmp_can_reach_end_stack_.back()});

  int prev_matched_size = 0;
  for (int i = 0; i < static_cast<int>(sorted_decoded_vocab.size()); ++i) {
    const auto& token = sorted_decoded_vocab[i].second;

    bool accepted = true;

    // Many tokens may contain the same prefix, so we will avoid unnecessary matching
    // by finding the longest common prefix with the previous token.
    if (i > 0) {
      const auto& prev_token = sorted_decoded_vocab[i - 1].second;
      int lcp_len =
          std::mismatch(token.begin(), token.end(), prev_token.begin(), prev_token.end()).first -
          token.begin();
      if (lcp_len > prev_matched_size) {
        // Case 1. The common prefix is rejected by the matcher in the last token. Reject directly.
        accepted = false;
      } else if (lcp_len < prev_matched_size) {
        // Case 2. The common prefix is shorter than the previous matched size. Rollback
        // the non-common part.
        RollbackChars(prev_matched_size - lcp_len);
        tmp_can_reach_end_stack_.erase(
            tmp_can_reach_end_stack_.end() - (prev_matched_size - lcp_len),
            tmp_can_reach_end_stack_.end()
        );
        tmp_can_reach_end_prefix_or_stack_.erase(
            tmp_can_reach_end_prefix_or_stack_.end() - (prev_matched_size - lcp_len),
            tmp_can_reach_end_prefix_or_stack_.end()
        );
      }
      prev_matched_size = std::min(prev_matched_size, lcp_len);
    }

    if (accepted) {
      // Accept the rest chars one by one
      for (int j = prev_matched_size; j < static_cast<int>(token.size()); ++j) {
        if (!AcceptChar(token[j], false)) {
          accepted = false;
          break;
        }
        tmp_can_reach_end_stack_.push_back(CanReachEnd());
        tmp_can_reach_end_prefix_or_stack_.push_back(
            tmp_can_reach_end_stack_.back() || tmp_can_reach_end_prefix_or_stack_.back()
        );
        prev_matched_size = j + 1;
      }
    }

    bool can_reach_end = tmp_can_reach_end_prefix_or_stack_.back();

    if (accepted) {
      tmp_accepted_indices_.push_back(i);
    } else if (can_reach_end && consider_parent_rule &&
               IsTokenPassLookaheadAssertion(token, tmp_can_reach_end_stack_)) {
      // 1. If the current rule is the root rule (consider_parent_rule=false), there are no
      // uncertain tokens. Not accepted tokens are just rejected.
      // 2. If a token cannot pass the lookahead assertion, it is rejected.
      tmp_uncertain_indices_.push_back(i);
    } else {
      tmp_rejected_indices_.push_back(i);
    }
  }
  // Rollback the last matched part
  RollbackChars(prev_matched_size);
  return AdaptiveTokenMask(
      vocab_size,
      sorted_decoded_vocab,
      tmp_accepted_indices_,
      tmp_rejected_indices_,
      tmp_uncertain_indices_
  );
}

/******************* GrammarCompiler::Impl *******************/

class GrammarCompiler::Impl {
 public:
  Impl(const TokenizerInfo& tokenizer_info, int max_threads, bool cache_enabled)
      : tokenizer_info_(tokenizer_info),
        max_threads_(max_threads),
        cache_enabled_(cache_enabled),
        compile_json_schema_cache_(GetCompileJSONSchemaCacheFunc(cache_enabled_)),
        compile_builtin_json_grammar_cache_(GetCompileBuiltinJSONGrammarCacheFunc(cache_enabled_)),
        compile_grammar_cache_(GetCompileGrammarCacheFunc(cache_enabled_)) {}

  CompiledGrammar CompileBuiltinJSONGrammar();

  CompiledGrammar CompileJSONSchema(
      const std::string& schema,
      bool any_whitespace,
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool strict_mode = true
  );

  CompiledGrammar CompileGrammar(const Grammar& grammar);

  void ClearCache();

 private:
  /*! \brief Multi-thread compile the grammar. */
  CompiledGrammar MultiThreadCompileGrammar(Grammar grammar, int max_threads);

  /*! \brief The cache for the compiled grammar of a JSON schema. */
  using SchemaKey =
      std::tuple<std::string, bool, std::optional<int>, std::pair<std::string, std::string>, bool>;

  /*! \brief The cache function for the compiled grammar of a JSON schema. */
  std::function<CompiledGrammar(const SchemaKey&)> GetCompileJSONSchemaCacheFunc(bool cache_enabled
  );

  /*! \brief The cache function for the compiled grammar for pure JSON. */
  std::function<CompiledGrammar()> GetCompileBuiltinJSONGrammarCacheFunc(bool cache_enabled);

  using GrammarKey = std::pair<std::string, std::string>;
  /*! \brief The cache function for the compiled grammar for a given grammar. */
  std::function<CompiledGrammar(const GrammarKey&)> GetCompileGrammarCacheFunc(bool cache_enabled);

  /*! \brief The vocabulary associated with this storage class. */
  const TokenizerInfo tokenizer_info_;
  /*! \brief The maximum number of threads to use. */
  const int max_threads_;
  /*! \brief Whether the cache is enabled. */
  const bool cache_enabled_;
  /*! \brief The cache for the compiled grammar of a JSON schema. */
  ThreadSafeCache<SchemaKey, CompiledGrammar> compile_json_schema_cache_;
  /*! \brief The cache for the compiled grammar for JSON. */
  ThreadSafeCache<CompiledGrammar> compile_builtin_json_grammar_cache_;
  /*! \brief The cache for the compiled grammar for bnf grammar. */
  ThreadSafeCache<GrammarKey, CompiledGrammar> compile_grammar_cache_;
};

CompiledGrammar GrammarCompiler::Impl::MultiThreadCompileGrammar(Grammar grammar, int max_threads) {
  using RuleExprType = Grammar::Impl::RuleExprType;

  auto compiled_grammar_impl = std::make_shared<CompiledGrammar::Impl>();

  compiled_grammar_impl->grammar = grammar;
  compiled_grammar_impl->tokenizer_info = tokenizer_info_;

  // Step 1. Compute the ids of rules that can be empty
  compiled_grammar_impl->grammar->allow_empty_rule_ids = AllowEmptyRuleAnalyzer::Apply(grammar);

  auto root_rule_id = grammar->GetRootRuleId();

  if (tokenizer_info_.GetVocabSize() == 0) {
    return CompiledGrammar(compiled_grammar_impl);
  }

  // Step 2. Compute the adaptive token mask cache
  // The token mask cache is computed for these positions in the grammar:
  // 1. All character class or character class star (with last_utf8_bytes=0, 1, 2, 3)
  // 2. All byte strings (with element_in_string=0, 1, 2, ...)
  // since other positions will be expanded to the above positions

  // TODO(Charlie): Figure out how to support ThreadPool and std::mutex in WebAssembly.
  // Only declare ThreadPool and mutex if max_threads > 1, so when max_threads = 1, we do
  // not need ThreadPool or std::mutex, which throws error in runtime in WebAssembly.
  std::optional<ThreadPool> thread_pool;
  std::optional<std::mutex> adaptive_token_mask_cache_mutex;

  if (max_threads > 1) {
    thread_pool.emplace(max_threads);
    adaptive_token_mask_cache_mutex.emplace();
  }

  for (int32_t rule_id = 0; rule_id < static_cast<int>(grammar->NumRules()); ++rule_id) {
    auto rule = grammar->GetRule(rule_id);
    auto rule_body = grammar->GetRuleExpr(rule.body_expr_id);
    XGRAMMAR_DCHECK(rule_body.type == RuleExprType::kChoices);
    for (auto sequence_id : rule_body) {
      auto sequence = grammar->GetRuleExpr(sequence_id);
      if (sequence.type == RuleExprType::kEmptyStr) {
        continue;
      }
      XGRAMMAR_DCHECK(sequence.type == RuleExprType::kSequence);
      for (int element_id = 0; element_id < sequence.size(); ++element_id) {
        auto element = grammar->GetRuleExpr(sequence[element_id]);
        if (element.type == RuleExprType::kRuleRef) {
          continue;
        }
        // Define the per-element processing logic for code reuse between
        // using thread_pool and not using thread_pool
        auto process_element = [&, rule_id, sequence_id, element_id, element]() {
          auto add_adaptive_token_mask = [&](const StackElement& stack_element) {
            auto grammar_matcher = GrammarMatcherForTokenMaskCache(grammar, stack_element);
            auto cur_adaptive_token_mask_cache = grammar_matcher.GetAdaptiveTokenMask(
                tokenizer_info_.GetVocabSize(),
                tokenizer_info_.GetSortedDecodedVocab(),
                rule_id != root_rule_id
            );
            if (max_threads > 1) {
              std::lock_guard<std::mutex> lock(adaptive_token_mask_cache_mutex.value());
              compiled_grammar_impl->adaptive_token_mask_cache[stack_element] =
                  cur_adaptive_token_mask_cache;
            } else {
              compiled_grammar_impl->adaptive_token_mask_cache[stack_element] =
                  cur_adaptive_token_mask_cache;
            }
          };

          auto cur_stack_element = StackElement(rule_id, sequence_id, element_id);
          if (element.type == RuleExprType::kByteString) {
            for (int idx = 0; idx < element.size(); ++idx) {
              cur_stack_element.element_in_string = idx;
              add_adaptive_token_mask(cur_stack_element);
            }
          } else {
            XGRAMMAR_DCHECK(
                element.type == RuleExprType::kCharacterClassStar ||
                element.type == RuleExprType::kCharacterClass
            );
            for (int left_utf8_bytes = 0; left_utf8_bytes <= 3; ++left_utf8_bytes) {
              cur_stack_element.left_utf8_bytes = left_utf8_bytes;
              add_adaptive_token_mask(cur_stack_element);
            }
          }
        };
        // Execute depending on whether we use thread_pool
        if (max_threads > 1) {
          thread_pool->Execute([process_element]() { process_element(); });
        } else {
          process_element();
        }
      }
    }
  }
  if (max_threads > 1) {
    thread_pool->Join();
  }

  return CompiledGrammar(compiled_grammar_impl);
}

std::function<CompiledGrammar(const GrammarCompiler::Impl::SchemaKey&)>
GrammarCompiler::Impl::GetCompileJSONSchemaCacheFunc(bool cache_enabled) {
  if (!cache_enabled) {
    return nullptr;
  }
  return [&](const SchemaKey& key) {
    auto [schema, any_whitespace, indent, separators, strict_mode] = key;
    auto grammar = Grammar::FromJSONSchema(schema, any_whitespace, indent, separators, strict_mode);
    return MultiThreadCompileGrammar(grammar, max_threads_);
  };
}

std::function<CompiledGrammar()> GrammarCompiler::Impl::GetCompileBuiltinJSONGrammarCacheFunc(
    bool cache_enabled
) {
  if (!cache_enabled) {
    return nullptr;
  }
  return [&]() { return MultiThreadCompileGrammar(Grammar::BuiltinJSONGrammar(), max_threads_); };
}

std::function<CompiledGrammar(const GrammarCompiler::Impl::GrammarKey&)>
GrammarCompiler::Impl::GetCompileGrammarCacheFunc(bool cache_enabled) {
  if (!cache_enabled) {
    return nullptr;
  }
  return [&](const GrammarKey& key) {
    auto [grammar_str, root_rule_name] = key;
    return MultiThreadCompileGrammar(Grammar::FromEBNF(grammar_str, root_rule_name), max_threads_);
  };
}

CompiledGrammar GrammarCompiler::Impl::CompileBuiltinJSONGrammar() {
  if (!cache_enabled_) {
    return MultiThreadCompileGrammar(Grammar::BuiltinJSONGrammar(), max_threads_);
  }
  return compile_builtin_json_grammar_cache_.Get();
}

CompiledGrammar GrammarCompiler::Impl::CompileJSONSchema(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode
) {
  if (!cache_enabled_) {
    return MultiThreadCompileGrammar(
        Grammar::FromJSONSchema(schema, any_whitespace, indent, separators, strict_mode),
        max_threads_
    );
  }
  auto separators_value = separators.value_or(
      (indent == std::nullopt) ? std::make_pair(", ", ": ") : std::make_pair(",", ": ")
  );
  auto key = std::make_tuple(schema, any_whitespace, indent, separators_value, strict_mode);
  return compile_json_schema_cache_.Get(key);
}

CompiledGrammar GrammarCompiler::Impl::CompileGrammar(const Grammar& grammar) {
  if (!cache_enabled_) {
    return MultiThreadCompileGrammar(grammar, max_threads_);
  }
  auto key = std::make_pair(grammar.ToString(), grammar->GetRootRule().name);
  return compile_grammar_cache_.Get(key);
}

void GrammarCompiler::Impl::ClearCache() {
  compile_builtin_json_grammar_cache_.Clear();
  compile_json_schema_cache_.Clear();
}

/******************* GrammarCompiler *******************/

GrammarCompiler::GrammarCompiler(
    const TokenizerInfo& tokenizer_info, int max_threads, bool cache_enabled
)
    : pimpl_(std::make_shared<Impl>(tokenizer_info, max_threads, cache_enabled)) {}

CompiledGrammar GrammarCompiler::CompileJSONSchema(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode
) {
  return pimpl_->CompileJSONSchema(schema, any_whitespace, indent, separators, strict_mode);
}

CompiledGrammar GrammarCompiler::CompileBuiltinJSONGrammar() {
  return pimpl_->CompileBuiltinJSONGrammar();
}

CompiledGrammar GrammarCompiler::CompileGrammar(const Grammar& grammar) {
  return pimpl_->CompileGrammar(grammar);
}

void GrammarCompiler::ClearCache() { pimpl_->ClearCache(); }

}  // namespace xgrammar
