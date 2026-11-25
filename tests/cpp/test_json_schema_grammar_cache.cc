/*!
 *  Copyright (c) 2026 by Contributors
 * \file tests/cpp/test_json_schema_grammar_cache.cc
 * \brief Tests of the process-wide cache of grammars converted from JSON schemas.
 */

#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <atomic>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "grammar_functor.h"
#include "json_schema_converter.h"
#include "json_schema_grammar_cache.h"
#include "structural_tag.h"
#include "support/recursion_guard.h"

namespace xgrammar {
namespace {

constexpr const char* kSimpleSchema =
    R"({"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},)"
    R"("required":["city"]})";

// A tree whose children are trees, through $defs and $ref.
constexpr const char* kTreeSchema =
    R"({"type":"object","properties":{"root":{"$ref":"#/$defs/node"},)"
    R"("backup":{"$ref":"#/$defs/node"}},"required":["root"],)"
    R"("$defs":{"node":{"type":"object","properties":{"value":{"type":"string"},)"
    R"("children":{"type":"array","items":{"$ref":"#/$defs/node"}}},"required":["value"]}}})";

// Mutual recursion between two definitions, plus a definition used by both.
constexpr const char* kMutualSchema =
    R"({"type":"object","properties":{"expr":{"$ref":"#/$defs/expr"}},"required":["expr"],)"
    R"("$defs":{"expr":{"anyOf":[{"$ref":"#/$defs/leaf"},{"type":"object","properties":)"
    R"({"op":{"type":"string"},"args":{"$ref":"#/$defs/args"}},"required":["op","args"]}]},)"
    R"("args":{"type":"array","items":{"$ref":"#/$defs/expr"},"minItems":1},)"
    R"("leaf":{"type":"object","properties":{"num":{"type":"number"}},"required":["num"]}}})";

// Many keywords. minimax_m3_xml rejects format, length limits and free-form keys, so it uses
// kVariedSchemaPlain instead.
constexpr const char* kVariedSchema =
    R"({"type":"object","properties":{"name":{"type":"string","minLength":1,"maxLength":20},)"
    R"("tags":{"type":"array","items":{"type":"string"},"maxItems":3},)"
    R"("mode":{"enum":["fast","slow"]},"ratio":{"type":"number","minimum":0,"maximum":1},)"
    R"("when":{"type":"string","format":"date"},)"
    R"("extra":{"type":"object","additionalProperties":{"type":"integer"}},)"
    R"("flag":{"type":"boolean"},"nothing":{"type":"null"}},"required":["name","mode"]})";
constexpr const char* kVariedSchemaPlain =
    R"({"type":"object","properties":{"name":{"type":"string"},)"
    R"("tags":{"type":"array","items":{"type":"string"},"maxItems":3},)"
    R"("mode":{"enum":["fast","slow"]},"ratio":{"type":"number","minimum":0,"maximum":1},)"
    R"("flag":{"type":"boolean"},"nothing":{"type":"null"}},"required":["name","mode"]})";

const std::vector<JSONFormat> kAllFormats = {
    JSONFormat::kJSON,
    JSONFormat::kQwenXML,
    JSONFormat::kMiniMaxXML,
    JSONFormat::kDeepSeekXML,
    JSONFormat::kGlmXML,
    JSONFormat::kCohereXML,
    JSONFormat::kKimiK3XML,
    JSONFormat::kMiniMaxM3XML,
    JSONFormat::kDeepSeekV41XML,
    JSONFormat::kGemma,
    JSONFormat::kMuseXML,
};

/*! \brief One conversion's inputs. */
struct Options {
  bool any_whitespace = true;
  std::optional<int> indent = std::nullopt;
  std::optional<std::pair<std::string, std::string>> separators = std::nullopt;
  bool strict_mode = true;
  std::optional<int> max_whitespace_cnt = std::nullopt;
  bool any_order = false;
  JSONFormat format = JSONFormat::kJSON;
  std::vector<std::string> excludes = {};
};

/*! \brief Convert through the cache. */
std::string Cached(const std::string& schema, const Options& o = {}) {
  return JSONSchemaToNormalizedGrammar(
             schema,
             o.any_whitespace,
             o.indent,
             o.separators,
             o.strict_mode,
             o.max_whitespace_cnt,
             o.any_order,
             o.format,
             o.excludes
  )
      .ToString();
}

/*! \brief Convert without the cache: what JSONSchemaToNormalizedGrammar computes on a miss. */
std::string Uncached(const std::string& schema, const Options& o = {}) {
  return GrammarNormalizer::Apply(JSONSchemaToGrammar(
                                      schema,
                                      o.any_whitespace,
                                      o.indent,
                                      o.separators,
                                      o.strict_mode,
                                      o.max_whitespace_cnt,
                                      o.any_order,
                                      o.format,
                                      o.excludes
                                  ))
      .ToString();
}

std::string StructuralTagToString(const std::string& stag_json) {
  auto result = StructuralTagToGrammar(stag_json);
  EXPECT_TRUE(result.IsOk());
  return std::move(result).Unwrap().ToString();
}

/*! \brief A structural tag offering one tool per schema, all with the given style. */
std::string ToolsTag(const std::vector<std::string>& schemas, const std::string& style = "json") {
  std::string tags;
  for (size_t i = 0; i < schemas.size(); ++i) {
    if (i > 0) tags += ",";
    tags += R"({"type":"tag","begin":"<tool=t)" + std::to_string(i) +
            R"(>","content":{"type":"json_schema","json_schema":)" + schemas[i] + R"(,"style":")" +
            style + R"("},"end":"</tool>"})";
  }
  return R"({"type":"structural_tag","format":{"type":"triggered_tags","triggers":["<tool="],)"
         R"("tags":[)" +
         tags + "]}}";
}

/*! \brief Give the process-wide cache a budget and an empty state; restore both afterwards. */
class JSONSchemaGrammarCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    saved_enabled_ = Cache().GetEnabled();
    saved_budget_ = Cache().GetMaxMemoryBytes();
    Cache().SetEnabled(true);
    Cache().SetMaxMemoryBytes(JSONSchemaGrammarCache::kUnlimited);
    Cache().Clear();
  }
  void TearDown() override {
    Cache().SetEnabled(saved_enabled_);
    Cache().SetMaxMemoryBytes(saved_budget_);
    Cache().Clear();
  }
  static JSONSchemaGrammarCache& Cache() { return JSONSchemaGrammarCache::Global(); }
  static JSONSchemaGrammarCache::Stats Stats() { return Cache().GetStats(); }

 private:
  bool saved_enabled_ = true;
  int64_t saved_budget_ = 0;
};

/*! \brief A small, distinct grammar per index, for exercising the LRU on a private cache. */
Grammar SmallGrammar(int i) {
  return Grammar::FromEBNF("root ::= \"value_" + std::to_string(i) + "\"");
}

TEST_F(JSONSchemaGrammarCacheTest, DefaultBudget) {
  EXPECT_EQ(JSONSchemaGrammarCache().GetMaxMemoryBytes(), JSONSchemaGrammarCache::kUnlimited);
}

TEST_F(JSONSchemaGrammarCacheTest, HitAcrossConversions) {
  std::string first = Cached(kSimpleSchema);
  EXPECT_EQ(Stats().misses, 1);
  EXPECT_EQ(Stats().hits, 0);
  EXPECT_EQ(Stats().num_entries, 1);
  EXPECT_GT(Stats().memory_bytes, 0);

  std::string second = Cached(kSimpleSchema);
  EXPECT_EQ(Stats().misses, 1);
  EXPECT_EQ(Stats().hits, 1);
  EXPECT_EQ(first, second);
  EXPECT_EQ(first, Uncached(kSimpleSchema));
}

TEST_F(JSONSchemaGrammarCacheTest, IdenticalToUncachedForEveryStyle) {
  for (JSONFormat format : kAllFormats) {
    for (const char* schema : {kSimpleSchema, kTreeSchema, kMutualSchema, kVariedSchema}) {
      if (format == JSONFormat::kMiniMaxM3XML && schema == kVariedSchema) {
        schema = kVariedSchemaPlain;
      }
      Options options;
      options.format = format;
      SCOPED_TRACE(
          std::string("format=") + std::to_string(static_cast<int>(format)) + " " + schema
      );
      std::string expected = Uncached(schema, options);
      EXPECT_EQ(Cached(schema, options), expected);  // miss
      EXPECT_EQ(Cached(schema, options), expected);  // hit
    }
  }
  EXPECT_EQ(Stats().hits, Stats().misses);
}

TEST_F(JSONSchemaGrammarCacheTest, StructuralTagsShareToolEntries) {
  std::string first = ToolsTag({kSimpleSchema, kTreeSchema, kMutualSchema}, "muse_xml");
  // The next request adds a tool, as in deferred tool loading.
  std::string second =
      ToolsTag({kSimpleSchema, kTreeSchema, kMutualSchema, kVariedSchema}, "muse_xml");

  Cache().SetMaxMemoryBytes(0);
  std::string expected = StructuralTagToString(second);
  Cache().SetMaxMemoryBytes(JSONSchemaGrammarCache::kUnlimited);

  StructuralTagToString(first);
  EXPECT_EQ(Stats().misses, 3);
  EXPECT_EQ(Stats().hits, 0);

  EXPECT_EQ(StructuralTagToString(second), expected);
  EXPECT_EQ(Stats().misses, 4);
  EXPECT_EQ(Stats().hits, 3);
}

TEST_F(JSONSchemaGrammarCacheTest, DifferentOptionsNeverShareAnEntry) {
  std::vector<Options> variants(16);
  variants[1].any_whitespace = false;
  variants[2].any_whitespace = false;
  variants[2].indent = 2;
  variants[3].any_whitespace = false;
  variants[3].indent = 4;
  variants[4].any_whitespace = false;
  variants[4].separators = std::make_pair(",", ":");
  variants[5].any_whitespace = false;
  variants[5].separators = std::make_pair(", ", ": ");
  variants[6].strict_mode = false;
  variants[7].max_whitespace_cnt = 2;
  variants[8].max_whitespace_cnt = 3;
  variants[9].any_order = true;
  variants[10].format = JSONFormat::kQwenXML;
  variants[11].format = JSONFormat::kMuseXML;
  variants[12].format = JSONFormat::kMuseXML;
  variants[12].excludes = {"</tool>"};
  variants[13].format = JSONFormat::kMuseXML;
  variants[13].excludes = {"</tool>", "<tool="};
  variants[14].format = JSONFormat::kMuseXML;
  variants[14].excludes = {"<tool=", "</tool>"};
  variants[15].format = JSONFormat::kKimiK3XML;
  variants[15].any_order = true;

  for (int round = 0; round < 2; ++round) {
    for (size_t i = 0; i < variants.size(); ++i) {
      SCOPED_TRACE("variant " + std::to_string(i));
      EXPECT_EQ(Cached(kVariedSchema, variants[i]), Uncached(kVariedSchema, variants[i]));
    }
  }
  // Every variant has its own key, so each converted once and hit once.
  EXPECT_EQ(Stats().num_entries, static_cast<int64_t>(variants.size()));
  EXPECT_EQ(Stats().misses, static_cast<int64_t>(variants.size()));
  EXPECT_EQ(Stats().hits, static_cast<int64_t>(variants.size()));
}

TEST_F(JSONSchemaGrammarCacheTest, SchemaTextIsPartOfTheKey) {
  // Property order changes a fixed-order grammar.
  std::string a =
      R"({"type":"object","properties":{"x":{"type":"string"},"y":{"type":"integer"}}})";
  std::string b =
      R"({"type":"object","properties":{"y":{"type":"integer"},"x":{"type":"string"}}})";
  EXPECT_NE(Cached(a), Cached(b));
  EXPECT_EQ(Stats().num_entries, 2);
}

TEST_F(JSONSchemaGrammarCacheTest, RecursionLimitIsPartOfTheKey) {
  std::string schema = R"({"type":"string"})";
  for (int i = 0; i < 60; ++i) {
    schema = R"({"type":"object","properties":{"a":)" + schema + R"(},"required":["a"]})";
  }
  int saved_depth = RecursionGuard::GetMaxRecursionDepth();
  Cached(schema);
  RecursionGuard::SetMaxRecursionDepth(10);
  EXPECT_THROW(Cached(schema), std::exception);
  RecursionGuard::SetMaxRecursionDepth(saved_depth);
  Cached(schema);
  EXPECT_EQ(Stats().hits, 1);
}

TEST_F(JSONSchemaGrammarCacheTest, FailedConversionIsNotCached) {
  std::string bad = R"({"type":"object","properties":{"a":{"$ref":"#/$defs/missing"}}})";
  EXPECT_THROW(Cached(bad), std::exception);
  EXPECT_THROW(Cached(bad), std::exception);
  EXPECT_EQ(Stats().num_entries, 0);
}

TEST_F(JSONSchemaGrammarCacheTest, FromJSONSchemaReturnsAPrivateCopy) {
  Grammar returned = Grammar::FromJSONSchema(kTreeSchema);
  std::string expected = returned.ToString();
  // In-place passes on the returned grammar must not reach the cached one.
  RepetitionNormalizer::Apply(&returned);
  GrammarFSMBuilder::Apply(&returned);
  EXPECT_EQ(Grammar::FromJSONSchema(kTreeSchema).ToString(), expected);
  EXPECT_EQ(Stats().hits, 1);
}

TEST_F(JSONSchemaGrammarCacheTest, DisabledCacheKeepsNothing) {
  Cache().SetMaxMemoryBytes(0);
  EXPECT_FALSE(Cache().Enabled());
  Cached(kSimpleSchema);
  Cached(kSimpleSchema);
  EXPECT_EQ(Stats().num_entries, 0);
  EXPECT_EQ(Stats().memory_bytes, 0);
  EXPECT_EQ(Stats().hits, 0);
}

TEST_F(JSONSchemaGrammarCacheTest, TurningOffRemovesEntriesAndKeepsTheLimit) {
  SetJSONSchemaCacheLimitBytes(1 << 20);
  Cached(kSimpleSchema);
  EXPECT_EQ(Stats().num_entries, 1);

  SetJSONSchemaCacheEnabled(false);
  EXPECT_FALSE(GetJSONSchemaCacheEnabled());
  EXPECT_FALSE(Cache().Enabled());
  EXPECT_EQ(Stats().num_entries, 0);
  EXPECT_EQ(Stats().memory_bytes, 0);
  // Still converts correctly, without storing anything.
  EXPECT_EQ(Cached(kTreeSchema), Uncached(kTreeSchema));
  EXPECT_EQ(Stats().num_entries, 0);

  SetJSONSchemaCacheEnabled(true);
  EXPECT_EQ(GetJSONSchemaCacheLimitBytes(), 1 << 20);
  Cached(kTreeSchema);
  Cached(kTreeSchema);
  EXPECT_EQ(Stats().num_entries, 1);
  EXPECT_GE(Stats().hits, 1);
}

TEST_F(JSONSchemaGrammarCacheTest, LimitThroughTheConfigAPI) {
  SetJSONSchemaCacheLimitBytes(-1);
  EXPECT_EQ(GetJSONSchemaCacheLimitBytes(), -1);
  Cached(kSimpleSchema);
  Cached(kTreeSchema);
  EXPECT_EQ(Stats().num_entries, 2);

  // A limit too small for any entry empties the cache and keeps it empty.
  SetJSONSchemaCacheLimitBytes(64);
  EXPECT_EQ(Stats().num_entries, 0);
  EXPECT_EQ(Cached(kSimpleSchema), Uncached(kSimpleSchema));
  EXPECT_EQ(Stats().num_entries, 0);

  EXPECT_THROW(SetJSONSchemaCacheLimitBytes(-2), std::exception);
  EXPECT_EQ(GetJSONSchemaCacheLimitBytes(), 64);
}

TEST_F(JSONSchemaGrammarCacheTest, ConcurrentConversionsWhileTheBudgetChanges) {
  std::vector<std::string> schemas = {kSimpleSchema, kTreeSchema, kMutualSchema, kVariedSchema};
  for (int i = 0; i < 8; ++i) {
    schemas.push_back(
        R"({"type":"object","properties":{"field_)" + std::to_string(i) + R"(":{"type":"string"}}})"
    );
  }
  std::vector<Options> styles(2);
  styles[1].format = JSONFormat::kMuseXML;
  std::vector<std::string> expected;
  for (const auto& schema : schemas) {
    for (const auto& style : styles) expected.push_back(Uncached(schema, style));
  }

  std::atomic<int> mismatches{0};
  std::atomic<bool> failed{false};
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&, t] {
      try {
        for (int round = 0; round < 20; ++round) {
          for (size_t k = 0; k < expected.size(); ++k) {
            size_t i = (k + t + round) % expected.size();
            if (Cached(schemas[i / styles.size()], styles[i % styles.size()]) != expected[i]) {
              ++mismatches;
            }
          }
        }
      } catch (...) {
        failed = true;
      }
    });
  }
  threads.emplace_back([&] {
    const int64_t budgets[] = {JSONSchemaGrammarCache::kUnlimited, 200000, 50000, 0};
    for (int round = 0; round < 200; ++round) {
      Cache().SetMaxMemoryBytes(budgets[round % 4]);
      if (round % 7 == 0) Cache().Clear();
      Cache().GetStats();
    }
  });
  for (auto& thread : threads) thread.join();
  EXPECT_FALSE(failed);
  EXPECT_EQ(mismatches, 0);

  Cache().SetMaxMemoryBytes(JSONSchemaGrammarCache::kUnlimited);
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(Cached(schemas[i / styles.size()], styles[i % styles.size()]), expected[i]);
  }
}

// The LRU itself, on a private instance so the budget is exact.

TEST(JSONSchemaGrammarCacheLRU, EvictsLeastRecentlyUsedWithinBudget) {
  Grammar sample = SmallGrammar(0);
  // Keys of equal length, so every entry has the same estimated size.
  auto key = [](int i) { return "key_" + std::to_string(100 + i); };
  int64_t entry_size =
      static_cast<int64_t>(JSONSchemaGrammarCache::EstimateEntrySize(key(0), sample));
  JSONSchemaGrammarCache cache(entry_size * 5 + entry_size / 2);  // room for five entries

  for (int i = 0; i < 20; ++i) cache.Put(key(i), SmallGrammar(0));
  EXPECT_EQ(cache.GetStats().num_entries, 5);
  EXPECT_EQ(cache.GetStats().evictions, 15);
  EXPECT_LE(cache.GetStats().memory_bytes, cache.GetMaxMemoryBytes());

  // The five most recent entries (15..19) are kept.
  EXPECT_FALSE(cache.Get(key(14)).has_value());
  EXPECT_TRUE(cache.Get(key(15)).has_value());  // now the most recently used
  cache.Put(key(0), SmallGrammar(0));           // evicts the oldest, 16
  EXPECT_TRUE(cache.Get(key(15)).has_value());
  EXPECT_FALSE(cache.Get(key(16)).has_value());

  // Shrinking the budget evicts at once.
  cache.SetMaxMemoryBytes(entry_size * 2 + entry_size / 2);
  EXPECT_EQ(cache.GetStats().num_entries, 2);
}

TEST(JSONSchemaGrammarCacheLRU, EntryLargerThanTheBudgetIsSkipped) {
  JSONSchemaGrammarCache cache(64);
  cache.Put("key", SmallGrammar(0));
  EXPECT_EQ(cache.GetStats().num_entries, 0);
  EXPECT_EQ(cache.GetStats().memory_bytes, 0);
}

TEST(JSONSchemaGrammarCacheLRU, SecondPutOfAKeyKeepsTheFirstEntry) {
  JSONSchemaGrammarCache cache(JSONSchemaGrammarCache::kUnlimited);
  cache.Put("key", SmallGrammar(1));
  cache.Put("key", SmallGrammar(2));
  EXPECT_EQ(cache.GetStats().num_entries, 1);
  EXPECT_EQ(cache.Get("key")->ToString(), SmallGrammar(1).ToString());
}

TEST(JSONSchemaGrammarCacheLRU, ClearResetsEntriesAndCounters) {
  JSONSchemaGrammarCache cache(JSONSchemaGrammarCache::kUnlimited);
  cache.Put("key", SmallGrammar(0));
  cache.Get("key");
  cache.Get("other");
  cache.Clear();
  auto stats = cache.GetStats();
  EXPECT_EQ(stats.num_entries, 0);
  EXPECT_EQ(stats.memory_bytes, 0);
  EXPECT_EQ(stats.hits, 0);
  EXPECT_EQ(stats.misses, 0);
}

}  // namespace
}  // namespace xgrammar
