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
 * \brief Turn the JSON schema grammar cache on or off.
 * \details The cache keeps the grammar converted from each JSON schema, together with its
 * conversion options, so a later conversion of the same schema reuses it. Grammar::FromJSONSchema,
 * the json_schema contents of Grammar::FromStructuralTag, and the GrammarCompiler methods that use
 * them all go through it. Converted grammars are identical with and without the cache. There is
 * one cache per process, separate from the GrammarCompiler caches. Turning it off removes all its
 * entries. It is on by default. This function is thread-safe.
 * \param enabled Whether to use the cache.
 */
void SetJSONSchemaCacheEnabled(bool enabled);

/*! \brief Whether the JSON schema grammar cache is on. */
bool GetJSONSchemaCacheEnabled();

/*!
 * \brief Set the memory limit of the JSON schema grammar cache. Least recently used entries are
 * evicted to stay within it. By default there is no limit. This function is thread-safe.
 * \param limit_bytes The limit in bytes, or -1 for no limit.
 */
void SetJSONSchemaCacheLimitBytes(int64_t limit_bytes);

/*! \brief Get the memory limit of the JSON schema grammar cache in bytes (-1 means no limit). */
int64_t GetJSONSchemaCacheLimitBytes();

}  // namespace xgrammar

#endif  // XGRAMMAR_CONFIG_H_
