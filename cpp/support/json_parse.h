/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/support/json_parse.h
 * \brief picojson::parse wrappers that reject inputs nested deeper than the recursion limit.
 * picojson recurses once per nesting level and has no depth limit of its own, so a deeply nested
 * input would overflow the stack before any of the recursion guards in xgrammar run.
 */
#ifndef XGRAMMAR_SUPPORT_JSON_PARSE_H_
#define XGRAMMAR_SUPPORT_JSON_PARSE_H_

#include <picojson.h>

#include <optional>
#include <string>

#include "recursion_guard.h"

namespace xgrammar {

namespace detail {

/*!
 * \brief Check that the JSON value starting at begin is not nested deeper than the maximum
 * recursion depth. Scanning stops at the end of the first top-level array or object.
 * \return The error message if the nesting is too deep.
 */
template <typename Iter>
inline std::optional<std::string> CheckJSONNestingDepth(Iter begin, Iter end) {
  const int max_depth = RecursionGuard::GetMaxRecursionDepth();
  int depth = 0;
  bool in_string = false;
  for (Iter it = begin; it != end; ++it) {
    const char c = *it;
    if (in_string) {
      if (c == '\\') {
        if (++it == end) break;
      } else if (c == '"') {
        in_string = false;
      }
    } else if (c == '"') {
      in_string = true;
    } else if (c == '[' || c == '{') {
      if (++depth > max_depth) {
        return "JSON is nested deeper than the maximum recursion depth " +
               std::to_string(max_depth);
      }
    } else if ((c == ']' || c == '}') && --depth <= 0) {
      break;
    }
  }
  return std::nullopt;
}

}  // namespace detail

/*!
 * \brief Parse a JSON string. Same as picojson::parse(out, json), but rejects inputs nested
 * deeper than the maximum recursion depth.
 * \return The error message, empty on success.
 */
inline std::string ParseJSON(picojson::value& out, const std::string& json) {
  if (auto error = detail::CheckJSONNestingDepth(json.begin(), json.end())) {
    return *error;
  }
  return picojson::parse(out, json);
}

/*!
 * \brief Parse the JSON value at the start of [begin, end). Same as
 * picojson::parse(out, begin, end, err), but rejects inputs nested deeper than the maximum
 * recursion depth.
 * \return The iterator past the parsed value; begin if the nesting check failed.
 */
template <typename Iter>
inline Iter ParseJSON(picojson::value& out, Iter begin, Iter end, std::string* err) {
  if (auto error = detail::CheckJSONNestingDepth(begin, end)) {
    *err = *error;
    return begin;
  }
  return picojson::parse(out, begin, end, err);
}

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_JSON_PARSE_H_
