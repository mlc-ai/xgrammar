/**
 * \file tests/cpp/test_regex_fsm_cache.cc
 * \brief Regression tests for regex FSM reuse across JSON Schema compilation stages.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>
#include <string>

#include "grammar_functor.h"
#include "json_schema_converter.h"
#include "regex_fsm_cache.h"

using namespace xgrammar;

TEST(XGrammarRegexFSMCacheTest, JSONSchemaConversionCacheSurvivesOptimization) {
  const std::string pattern = "[a-z]{1,3}";
  RegexFSMCache regex_fsm_cache;
  auto grammar = GrammarNormalizer::Apply(JSONSchemaToGrammar(
      R"({"type":"string","pattern":"[a-z]{1,3}"})",
      /*any_whitespace=*/false,
      /*indent=*/std::nullopt,
      /*separators=*/std::nullopt,
      /*strict_mode=*/true,
      /*max_whitespace_cnt=*/std::nullopt,
      /*any_order=*/false,
      JSONFormat::kJSON,
      &regex_fsm_cache
  ));

  // JSON Schema search semantics rewrite the source pattern before it becomes a cache key.
  // This schema contributes exactly one regex FSM, so retain that entry without depending on
  // the converter's private rewritten spelling.
  ASSERT_EQ(regex_fsm_cache.size(), 1);
  auto cached = regex_fsm_cache.begin();
  const std::size_t cache_size = regex_fsm_cache.size();
  const std::string cache_key = cached->first;
  const auto* cached_fsm_impl = cached->second.GetFsm().ImplPtr();

  auto optimized =
      GrammarOptimizer::Apply(grammar, /*expand_repetition_ranges=*/false, &regex_fsm_cache);

  EXPECT_TRUE(optimized->optimized);
  EXPECT_GT(optimized->complete_fsm.NumStates(), 0);
  EXPECT_EQ(regex_fsm_cache.size(), cache_size);
  cached = regex_fsm_cache.find(cache_key);
  ASSERT_NE(cached, regex_fsm_cache.end());
  EXPECT_EQ(cached->second.GetFsm().ImplPtr(), cached_fsm_impl);
}
