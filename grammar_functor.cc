class StructuralTagGrammarCreatorImpl : public SubGrammarAdder {
 public:
  Grammar Apply(
      const std::vector<Grammar>& schema_grammars,
      const std::vector<std::string>& triggers,
      const std::vector<std::vector<StructuralTagItem>>& tag_groups
  ) {
    XGRAMMAR_CHECK(triggers.size() == tag_groups.size())
        << "Number of triggers must match number of tag groups";

    builder_ = GrammarBuilder();
    auto root_rule_id = builder_.AddEmptyRule("root");

    // Create rules for each trigger group
    std::vector<std::pair<int32_t, int32_t>> trigger_rule_pairs;
    for (size_t i = 0; i < triggers.size(); i++) {
      auto rule_name = "trigger_rule" + std::to_string(i);
      auto rule_id = builder_.AddEmptyRule(rule_name);

      // Convert trigger string to byte string expr
      std::vector<int32_t> trigger_bytes;
      trigger_bytes.reserve(triggers[i].size());
      for (char c : triggers[i]) {
        trigger_bytes.push_back(static_cast<int32_t>(c));
      }
      auto trigger_expr_id = builder_.AddByteString(trigger_bytes);

      // Create choices for each tag in this trigger group
      std::vector<int32_t> choices;
      for (const auto& tag : tag_groups[i]) {
        XGRAMMAR_CHECK(
            tag.schema_idx >= 0 && tag.schema_idx < static_cast<int>(schema_grammars.size())
        ) << "Invalid schema index: "
          << tag.schema_idx;
        XGRAMMAR_CHECK(tag.start.substr(0, triggers[i].size()) == triggers[i])
            << "Tag start must begin with trigger";

        // Visit the schema grammar for this tag
        auto schema_rule_id = VisitSubGrammar(schema_grammars[tag.schema_idx]);

        // Create sequence: start_suffix + schema + end
        std::vector<int32_t> seq_elements;

        // Add start suffix (everything after trigger)
        if (tag.start.size() > triggers[i].size()) {
          std::string suffix = tag.start.substr(triggers[i].size());
          std::vector<int32_t> suffix_bytes;
          suffix_bytes.reserve(suffix.size());
          for (char c : suffix) {
            suffix_bytes.push_back(static_cast<int32_t>(c));
          }
          seq_elements.push_back(builder_.AddByteString(suffix_bytes));
        }

        // Add schema reference
        seq_elements.push_back(builder_.AddRuleRef(schema_rule_id));

        // Add end string
        if (!tag.end.empty()) {
          std::vector<int32_t> end_bytes;
          end_bytes.reserve(tag.end.size());
          for (char c : tag.end) {
            end_bytes.push_back(static_cast<int32_t>(c));
          }
          seq_elements.push_back(builder_.AddByteString(end_bytes));
        }

        choices.push_back(builder_.AddSequence(seq_elements));
      }

      builder_.UpdateRuleBody(rule_id, builder_.AddChoices(choices));
      trigger_rule_pairs.emplace_back(trigger_expr_id, rule_id);
    }

    // Create root TagDispatch rule
    std::vector<int32_t> tag_dispatch_data;
    tag_dispatch_data.reserve(trigger_rule_pairs.size() * 2);
    for (const auto& [trigger_id, rule_id] : trigger_rule_pairs) {
      tag_dispatch_data.push_back(trigger_id);
      tag_dispatch_data.push_back(rule_id);
    }

    builder_.UpdateRuleBody(root_rule_id, builder_.AddTagDispatch(tag_dispatch_data));
    return builder_.Get(root_rule_id);
  }

  // Avoid hiding the original Apply(const Grammar&)
  Grammar Apply(const Grammar& grammar) override { XGRAMMAR_LOG(FATAL) << "Should not be called"; }
};
