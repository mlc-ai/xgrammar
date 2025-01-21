/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/structural_tag.cc
 */
#include "structural_tag.h"

#include <algorithm>
#include <string>
#include <string_view>

#include "support/logging.h"

namespace xgrammar {

std::string StructuralTagToEBNF(
    const std::vector<StructuralTagItem>& tags, const std::vector<std::string>& triggers
) {
  // Step 1: handle triggers. Triggers should not be mutually inclusive
  std::vector<std::string> sorted_triggers(triggers.begin(), triggers.end());
  std::sort(sorted_triggers.begin(), sorted_triggers.end());
  for (int i = 0; i < static_cast<int>(sorted_triggers.size()) - 1; ++i) {
    XGRAMMAR_CHECK(
        std::string_view(sorted_triggers[i + 1]).substr(0, sorted_triggers[i].size()) ==
        sorted_triggers[i]
    ) << "Triggers should not be mutually inclusive, but "
      << sorted_triggers[i] << " is a prefix of " << sorted_triggers[i + 1];
  }

  // Step 2: For each tag, find the trigger that is a prefix of the tag.begin

  std::vector<std::vector<StructuralTagItem>> sorted_tags_groups;
  return "";
}

}  // namespace xgrammar
