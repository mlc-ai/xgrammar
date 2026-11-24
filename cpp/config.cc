/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/config.cc
 */

#include <xgrammar/config.h>

#include "cross_grammar_cache.h"
#include "support/json_serializer.h"
#include "support/recursion_guard.h"

namespace xgrammar {

void SetMaxRecursionDepth(int max_recursion_depth) {
  RecursionGuard::SetMaxRecursionDepth(max_recursion_depth);
}

int GetMaxRecursionDepth() { return RecursionGuard::GetMaxRecursionDepth(); }

std::string GetSerializationVersion() { return std::string(SerializeVersion::GetVersion()); }

void SetCrossGrammarCacheMaxMemory(int64_t max_memory_bytes) {
  CrossGrammarCache::Global().SetMaxMemoryBytes(max_memory_bytes);
}

int64_t GetCrossGrammarCacheMaxMemory() { return CrossGrammarCache::Global().GetMaxMemoryBytes(); }

void ClearCrossGrammarCache() { CrossGrammarCache::Global().Clear(); }

CrossGrammarCacheStats GetCrossGrammarCacheStats() {
  auto stats = CrossGrammarCache::Global().GetStats();
  return CrossGrammarCacheStats{
      stats.hits,
      stats.misses,
      stats.evictions,
      stats.num_entries,
      stats.memory_bytes,
      stats.max_memory_bytes
  };
}

}  // namespace xgrammar
