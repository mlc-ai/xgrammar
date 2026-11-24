/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/config.h
 * \brief Global configuration for XGrammar.
 */

#ifndef XGRAMMAR_CONFIG_H_
#define XGRAMMAR_CONFIG_H_

#include <cstdint>
#include <string>

namespace xgrammar {

/*!
 * \brief Set the maximum recursion depth for the grammar.
 * \param max_recursion_depth The maximum recursion depth.
 */
void SetMaxRecursionDepth(int max_recursion_depth);

/*!
 * \brief Get the maximum recursion depth for the grammar.
 * \return The maximum recursion depth.
 */
int GetMaxRecursionDepth();

/*!
 * \brief Get the serialization version for the grammar.
 * \return The serialization version.
 * \note This is used to check the compatibility of the serialized grammar.
 */
std::string GetSerializationVersion();

/*!
 * \brief Set the memory budget of the process-wide cross-grammar cache, which keeps converted JSON
 * schema grammars so that later conversions of the same schema (e.g. a tool repeated across
 * structural tags) reuse them.
 * \param max_memory_bytes The budget in bytes. 0 disables the cache; -1 removes the bound.
 */
void SetCrossGrammarCacheMaxMemory(int64_t max_memory_bytes);

/*! \brief Get the memory budget of the cross-grammar cache in bytes (-1 means unbounded). */
int64_t GetCrossGrammarCacheMaxMemory();

/*! \brief Remove all entries of the cross-grammar cache and reset its counters. */
void ClearCrossGrammarCache();

/*! \brief Counters of the cross-grammar cache. */
struct CrossGrammarCacheStats {
  int64_t hits;
  int64_t misses;
  int64_t evictions;
  int64_t num_entries;
  int64_t memory_bytes;
  int64_t max_memory_bytes;
};

/*! \brief Get the counters of the cross-grammar cache. */
CrossGrammarCacheStats GetCrossGrammarCacheStats();

}  // namespace xgrammar

#endif  // XGRAMMAR_CONFIG_H_
