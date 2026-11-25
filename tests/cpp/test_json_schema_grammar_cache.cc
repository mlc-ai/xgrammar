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
Grammar CachedGrammar(const std::string& schema, const Options& o = {}) {
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
  );
}

std::string Cached(const std::string& schema, const Options& o = {}) {
  return CachedGrammar(schema, o).ToString();
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

std::string Key(const std::string& schema, const Options& o = {}) {
  return JSONSchemaCacheKey(
      schema,
      o.any_whitespace,
      o.indent,
      o.separators,
      o.strict_mode,
      o.max_whitespace_cnt,
      o.any_order,
      o.format,
      o.excludes
  );
}

/*! \brief A hit returns the stored grammar itself, so two conversions share one Impl. */
bool SameStoredGrammar(const Grammar& a, const Grammar& b) { return a.ImplPtr() == b.ImplPtr(); }

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
  static bool InCache(const std::string& schema, const Options& o = {}) {
    return Cache().Get(Key(schema, o)).has_value();
  }

 private:
  bool saved_enabled_ = true;
  int64_t saved_budget_ = 0;
};

/*! \brief A small, distinct grammar per index, for exercising the LRU on a private cache. */
Grammar SmallGrammar(int i) {
  return Grammar::FromEBNF("root ::= \"value_" + std::to_string(i) + "\"");
}

TEST_F(JSONSchemaGrammarCacheTest, DefaultHasNoLimit) {
  EXPECT_EQ(JSONSchemaGrammarCache().GetMaxMemoryBytes(), JSONSchemaGrammarCache::kUnlimited);
}

TEST_F(JSONSchemaGrammarCacheTest, HitAcrossConversions) {
  EXPECT_FALSE(InCache(kSimpleSchema));
  Grammar first = CachedGrammar(kSimpleSchema);
  EXPECT_TRUE(InCache(kSimpleSchema));
  EXPECT_GT(Cache().MemorySize(), 0u);

  Grammar second = CachedGrammar(kSimpleSchema);
  EXPECT_TRUE(SameStoredGrammar(first, second));
  EXPECT_EQ(second.ToString(), Uncached(kSimpleSchema));
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
      Grammar miss = CachedGrammar(schema, options);
      Grammar hit = CachedGrammar(schema, options);
      EXPECT_EQ(miss.ToString(), expected);
      EXPECT_EQ(hit.ToString(), expected);
      EXPECT_TRUE(SameStoredGrammar(miss, hit));
    }
  }
}

TEST_F(JSONSchemaGrammarCacheTest, StructuralTagsShareToolEntries) {
  std::vector<std::string> first_tools = {kSimpleSchema, kTreeSchema, kMutualSchema};
  std::string first = ToolsTag(first_tools, "muse_xml");
  // The next request adds a tool, as in deferred tool loading.
  std::string second =
      ToolsTag({kSimpleSchema, kTreeSchema, kMutualSchema, kVariedSchema}, "muse_xml");
  // The options a muse_xml json_schema node converts with.
  Options muse;
  muse.format = JSONFormat::kMuseXML;

  Cache().SetEnabled(false);
  std::string expected = StructuralTagToString(second);
  Cache().SetEnabled(true);

  StructuralTagToString(first);
  for (const auto& schema : first_tools) EXPECT_TRUE(InCache(schema, muse));
  EXPECT_FALSE(InCache(kVariedSchema, muse));

  // The first three tools come from the cache; their stored grammars are reused, not rebuilt.
  Grammar stored = CachedGrammar(kTreeSchema, muse);
  EXPECT_EQ(StructuralTagToString(second), expected);
  EXPECT_TRUE(InCache(kVariedSchema, muse));
  EXPECT_TRUE(SameStoredGrammar(stored, CachedGrammar(kTreeSchema, muse)));
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

  // Every variant has its own key.
  std::vector<std::string> keys;
  for (const auto& v : variants) keys.push_back(Key(kVariedSchema, v));
  for (size_t i = 0; i < keys.size(); ++i) {
    for (size_t j = 0; j < i; ++j) EXPECT_NE(keys[i], keys[j]) << i << " vs " << j;
  }

  // Each variant converts to its own grammar, before and after all of them are cached.
  for (int round = 0; round < 2; ++round) {
    for (size_t i = 0; i < variants.size(); ++i) {
      SCOPED_TRACE("round " + std::to_string(round) + " variant " + std::to_string(i));
      EXPECT_EQ(Cached(kVariedSchema, variants[i]), Uncached(kVariedSchema, variants[i]));
    }
  }
}

TEST_F(JSONSchemaGrammarCacheTest, SchemaTextIsPartOfTheKey) {
  // Property order changes a fixed-order grammar.
  std::string a =
      R"({"type":"object","properties":{"x":{"type":"string"},"y":{"type":"integer"}}})";
  std::string b =
      R"({"type":"object","properties":{"y":{"type":"integer"},"x":{"type":"string"}}})";
  EXPECT_NE(Key(a), Key(b));
  EXPECT_NE(Cached(a), Cached(b));
  EXPECT_TRUE(InCache(a));
  EXPECT_TRUE(InCache(b));
}

TEST_F(JSONSchemaGrammarCacheTest, RecursionLimitIsPartOfTheKey) {
  std::string schema = R"({"type":"string"})";
  for (int i = 0; i < 60; ++i) {
    schema = R"({"type":"object","properties":{"a":)" + schema + R"(},"required":["a"]})";
  }
  int saved_depth = RecursionGuard::GetMaxRecursionDepth();
  Grammar first = CachedGrammar(schema);
  RecursionGuard::SetMaxRecursionDepth(10);
  EXPECT_THROW(Cached(schema), std::exception);
  RecursionGuard::SetMaxRecursionDepth(saved_depth);
  EXPECT_TRUE(SameStoredGrammar(first, CachedGrammar(schema)));
}

TEST_F(JSONSchemaGrammarCacheTest, FailedConversionIsNotCached) {
  std::string bad = R"({"type":"object","properties":{"a":{"$ref":"#/$defs/missing"}}})";
  EXPECT_THROW(Cached(bad), std::exception);
  EXPECT_THROW(Cached(bad), std::exception);
  EXPECT_FALSE(InCache(bad));
  EXPECT_EQ(Cache().MemorySize(), 0u);
}

TEST_F(JSONSchemaGrammarCacheTest, FromJSONSchemaReturnsAPrivateCopy) {
  Grammar returned = Grammar::FromJSONSchema(kTreeSchema);
  std::string expected = returned.ToString();
  EXPECT_FALSE(SameStoredGrammar(returned, CachedGrammar(kTreeSchema)));
  // In-place passes on the returned grammar must not reach the cached one.
  RepetitionNormalizer::Apply(&returned);
  GrammarFSMBuilder::Apply(&returned);
  EXPECT_EQ(Grammar::FromJSONSchema(kTreeSchema).ToString(), expected);
}

TEST_F(JSONSchemaGrammarCacheTest, ZeroLimitKeepsNothing) {
  Cache().SetMaxMemoryBytes(0);
  EXPECT_FALSE(Cache().Enabled());
  Grammar first = CachedGrammar(kSimpleSchema);
  Grammar second = CachedGrammar(kSimpleSchema);
  EXPECT_FALSE(SameStoredGrammar(first, second));
  EXPECT_EQ(Cache().MemorySize(), 0u);
}

TEST_F(JSONSchemaGrammarCacheTest, TurningOffRemovesEntriesAndKeepsTheLimit) {
  SetJSONSchemaCacheLimitBytes(1 << 20);
  Cached(kSimpleSchema);
  EXPECT_TRUE(InCache(kSimpleSchema));

  SetJSONSchemaCacheEnabled(false);
  EXPECT_FALSE(GetJSONSchemaCacheEnabled());
  EXPECT_FALSE(Cache().Enabled());
  EXPECT_EQ(Cache().MemorySize(), 0u);
  // Still converts correctly, without storing anything.
  EXPECT_EQ(Cached(kTreeSchema), Uncached(kTreeSchema));
  EXPECT_EQ(Cache().MemorySize(), 0u);

  SetJSONSchemaCacheEnabled(true);
  EXPECT_EQ(GetJSONSchemaCacheLimitBytes(), 1 << 20);
  EXPECT_FALSE(InCache(kSimpleSchema));
  Grammar first = CachedGrammar(kTreeSchema);
  EXPECT_TRUE(SameStoredGrammar(first, CachedGrammar(kTreeSchema)));
}

TEST_F(JSONSchemaGrammarCacheTest, LimitThroughTheConfigAPI) {
  SetJSONSchemaCacheLimitBytes(-1);
  EXPECT_EQ(GetJSONSchemaCacheLimitBytes(), -1);
  Cached(kSimpleSchema);
  Cached(kTreeSchema);
  EXPECT_TRUE(InCache(kSimpleSchema));
  EXPECT_TRUE(InCache(kTreeSchema));

  // A limit too small for any entry empties the cache and keeps it empty.
  SetJSONSchemaCacheLimitBytes(64);
  EXPECT_EQ(Cache().MemorySize(), 0u);
  EXPECT_EQ(Cached(kSimpleSchema), Uncached(kSimpleSchema));
  EXPECT_EQ(Cache().MemorySize(), 0u);

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
      Cache().SetEnabled(round % 5 != 0);
      if (round % 7 == 0) Cache().Clear();
      Cache().MemorySize();
    }
  });
  for (auto& thread : threads) thread.join();
  EXPECT_FALSE(failed);
  EXPECT_EQ(mismatches, 0);

  Cache().SetEnabled(true);
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
  std::size_t entry_size = JSONSchemaGrammarCache::EstimateEntrySize(key(0), sample);
  JSONSchemaGrammarCache cache(static_cast<int64_t>(entry_size * 5 + entry_size / 2));

  // Room for five entries: the five most recent (15..19) are kept.
  for (int i = 0; i < 20; ++i) cache.Put(key(i), SmallGrammar(0));
  EXPECT_EQ(cache.MemorySize(), entry_size * 5);
  EXPECT_FALSE(cache.Get(key(14)).has_value());
  EXPECT_TRUE(cache.Get(key(15)).has_value());  // now the most recently used
  cache.Put(key(0), SmallGrammar(0));           // evicts the oldest, 16
  EXPECT_TRUE(cache.Get(key(15)).has_value());
  EXPECT_FALSE(cache.Get(key(16)).has_value());

  // Shrinking the budget evicts at once.
  cache.SetMaxMemoryBytes(static_cast<int64_t>(entry_size * 2 + entry_size / 2));
  EXPECT_EQ(cache.MemorySize(), entry_size * 2);
}

TEST(JSONSchemaGrammarCacheLRU, EntryLargerThanTheBudgetIsSkipped) {
  JSONSchemaGrammarCache cache(64);
  cache.Put("key", SmallGrammar(0));
  EXPECT_FALSE(cache.Get("key").has_value());
  EXPECT_EQ(cache.MemorySize(), 0u);
}

TEST(JSONSchemaGrammarCacheLRU, SecondPutOfAKeyKeepsTheFirstEntry) {
  JSONSchemaGrammarCache cache(JSONSchemaGrammarCache::kUnlimited);
  cache.Put("key", SmallGrammar(1));
  std::size_t one_entry = cache.MemorySize();
  cache.Put("key", SmallGrammar(2));
  EXPECT_EQ(cache.MemorySize(), one_entry);
  EXPECT_EQ(cache.Get("key")->ToString(), SmallGrammar(1).ToString());
}

TEST(JSONSchemaGrammarCacheLRU, ClearRemovesEverything) {
  JSONSchemaGrammarCache cache(JSONSchemaGrammarCache::kUnlimited);
  cache.Put("key", SmallGrammar(0));
  cache.Clear();
  EXPECT_FALSE(cache.Get("key").has_value());
  EXPECT_EQ(cache.MemorySize(), 0u);
}

}  // namespace
}  // namespace xgrammar
