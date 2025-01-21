/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/structural_tag.cc
 */
#include "structural_tag.h"

#include <algorithm>
#include <string>
#include <string_view>

#include "grammar_functor.h"
#include "support/logging.h"

namespace xgrammar {

Grammar StructuralTagToGrammar(
    const std::vector<StructuralTagItem>& tags, const std::vector<std::string>& triggers
) {
  // Step 1: handle triggers. Triggers should not be mutually inclusive
  std::vector<std::string> sorted_triggers(triggers.begin(), triggers.end());
  std::sort(sorted_triggers.begin(), sorted_triggers.end());
  for (int i = 0; i < static_cast<int>(sorted_triggers.size()) - 1; ++i) {
    XGRAMMAR_CHECK(
        sorted_triggers[i + 1].size() < sorted_triggers[i].size() ||
        std::string_view(sorted_triggers[i + 1]).substr(0, sorted_triggers[i].size()) !=
            sorted_triggers[i]
    ) << "Triggers should not be mutually inclusive, but "
      << sorted_triggers[i] << " is a prefix of " << sorted_triggers[i + 1];
  }

  // Step 2: For each tag, find the trigger that is a prefix of the tag.begin
  // Convert the schema to grammar at the same time
  std::vector<std::vector<std::pair<StructuralTagItem, Grammar>>> tag_groups(triggers.size());
  for (const auto& tag : tags) {
    bool found = false;
    for (int i = 0; i < static_cast<int>(sorted_triggers.size()); ++i) {
      const auto& trigger = sorted_triggers[i];
      if (trigger.size() <= tag.start.size() &&
          std::string_view(tag.start).substr(0, trigger.size()) == trigger) {
        auto schema_grammar = Grammar::FromJSONSchema(tag.schema, true);
        tag_groups[i].push_back(std::make_pair(tag, schema_grammar));
        found = true;
        break;
      }
    }
    XGRAMMAR_CHECK(found) << "Tag " << tag.start << " does not match any trigger";
  }

  // Step 3: Combine the tags to form a grammar
  // root ::= TagDispatch((trigger1, rule1), (trigger2, rule2), ...)
  // Suppose tag1 and tag2 matches trigger1, then
  // rule1 ::= (tag1.start[trigger1.size():] + ToEBNF(tag1.schema) + tag1.end) |
  //            (tag2.start[trigger1.size():] + ToEBNF(tag2.schema) + tag2.end) | ...
  //
  // Suppose tag3 matches trigger2, then
  // rule2 ::= (tag3.start[trigger2.size():] + ToEBNF(tag3.schema) + tag3.end)
  //
  // ...
  return StructuralTagGrammarCreator::Apply(sorted_triggers, tag_groups);
}

}  // namespace xgrammar
