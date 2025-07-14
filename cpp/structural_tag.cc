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
  std::vector<Grammar> schema_grammars;
  schema_grammars.reserve(tags.size());
  for (const auto& tag : tags) {
    auto schema_grammar = Grammar::FromJSONSchema(tag.schema, true);
    schema_grammars.push_back(schema_grammar);
  }

  std::vector<std::vector<std::pair<StructuralTagItem, Grammar>>> tag_groups(triggers.size());
  for (int it_tag = 0; it_tag < static_cast<int>(tags.size()); ++it_tag) {
    const auto& tag = tags[it_tag];
    bool found = false;
    for (int it_trigger = 0; it_trigger < static_cast<int>(sorted_triggers.size()); ++it_trigger) {
      const auto& trigger = sorted_triggers[it_trigger];
      if (trigger.size() <= tag.begin.size() &&
          std::string_view(tag.begin).substr(0, trigger.size()) == trigger) {
        tag_groups[it_trigger].push_back(std::make_pair(tag, schema_grammars[it_tag]));
        found = true;
        break;
      }
    }
    XGRAMMAR_CHECK(found) << "Tag " << tag.begin << " does not match any trigger";
  }

  // Step 3: Combine the tags to form a grammar
  // root ::= TagDispatch((trigger1, rule1), (trigger2, rule2), ...)
  // Suppose tag1 and tag2 matches trigger1, then
  // rule1 ::= (tag1.begin[trigger1.size():] + ToEBNF(tag1.schema) + tag1.end) |
  //            (tag2.begin[trigger1.size():] + ToEBNF(tag2.schema) + tag2.end) | ...
  //
  // Suppose tag3 matches trigger2, then
  // rule2 ::= (tag3.begin[trigger2.size():] + ToEBNF(tag3.schema) + tag3.end)
  //
  // ...
  return StructuralTagGrammarCreator::Apply(sorted_triggers, tag_groups);
}

}  // namespace xgrammar
