/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/regex_fsm_cache.h
 * \brief Cache type shared by schema conversion and grammar FSM construction.
 */
#ifndef XGRAMMAR_REGEX_FSM_CACHE_H_
#define XGRAMMAR_REGEX_FSM_CACHE_H_

#include <string>
#include <unordered_map>

#include "fsm.h"

namespace xgrammar {

using RegexFSMCache = std::unordered_map<std::string, FSMWithStartEnd>;

inline std::string MakeRegexFSMCacheKey(
    const std::string& regex, bool json_string, bool byte_mode = false
) {
  std::string result;
  result.reserve(regex.size() + 2);
  result.push_back(static_cast<char>(json_string));
  result.push_back(static_cast<char>(byte_mode));
  result.append(regex);
  return result;
}

}  // namespace xgrammar

#endif  // XGRAMMAR_REGEX_FSM_CACHE_H_
