/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/config.cc
 */

#include <xgrammar/config.h>

#include "json_schema_grammar_cache.h"
#include "support/json_serializer.h"
#include "support/logging.h"
#include "support/recursion_guard.h"

namespace xgrammar {

void SetMaxRecursionDepth(int max_recursion_depth) {
  RecursionGuard::SetMaxRecursionDepth(max_recursion_depth);
}

int GetMaxRecursionDepth() { return RecursionGuard::GetMaxRecursionDepth(); }

std::string GetSerializationVersion() { return std::string(SerializeVersion::GetVersion()); }

void SetJSONSchemaCacheEnabled(bool enabled) {
  JSONSchemaGrammarCache::Global().SetEnabled(enabled);
}

bool GetJSONSchemaCacheEnabled() { return JSONSchemaGrammarCache::Global().GetEnabled(); }

void SetJSONSchemaCacheLimitBytes(int64_t limit_bytes) {
  XGRAMMAR_CHECK(limit_bytes >= -1) << "Invalid JSON schema cache limit: " << limit_bytes
                                    << ". It should be -1 (no limit) or a non-negative integer.";
  JSONSchemaGrammarCache::Global().SetMaxMemoryBytes(limit_bytes);
}

int64_t GetJSONSchemaCacheLimitBytes() {
  return JSONSchemaGrammarCache::Global().GetMaxMemoryBytes();
}

}  // namespace xgrammar
