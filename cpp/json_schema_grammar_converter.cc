/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/json_schema_grammar_converter.cc
 */

#include "json_schema_grammar_converter.h"

#include <picojson.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "grammar_builder.h"
#include "grammar_functor.h"
#include "regex_converter.h"
#include "support/logging.h"

namespace xgrammar {
namespace {

using CharacterClassElement = GrammarBuilder::CharacterClassElement;

class ASTCreator {
 public:
  std::string AllocateRuleName(const std::string& name_hint) {
    std::string name = builder_.GetNewRuleName(name_hint);
    builder_.AddEmptyRule(name);
    return name;
  }

  void ReserveRule(const std::string& name) {
    XGRAMMAR_CHECK(builder_.GetRuleId(name) == -1) << "Rule " << name << " already exists";
    builder_.AddEmptyRule(name);
  }

  std::string AddRule(const std::string& name_hint, int32_t body_expr_id) {
    std::string name = AllocateRuleName(name_hint);
    AddRuleWithAllocatedName(name, body_expr_id);
    return name;
  }

  void AddRuleWithAllocatedName(const std::string& name, int32_t body_expr_id) {
    int32_t rule_id = builder_.GetRuleId(name);
    XGRAMMAR_CHECK(rule_id != -1) << "Rule " << name << " is not allocated";
    builder_.UpdateRuleBody(rule_id, body_expr_id);
  }

  int32_t Empty() {
    if (!empty_expr_id_.has_value()) {
      empty_expr_id_ = builder_.AddEmptyStr();
    }
    return *empty_expr_id_;
  }

  int32_t ByteString(const std::string& value) {
    auto it = byte_string_expr_ids_.find(value);
    if (it != byte_string_expr_ids_.end()) {
      return it->second;
    }
    int32_t expr_id = value.empty() ? Empty() : builder_.AddByteString(value);
    byte_string_expr_ids_[value] = expr_id;
    return expr_id;
  }

  int32_t CharacterClass(
      const std::vector<CharacterClassElement>& elements, bool is_negative = false
  ) {
    return builder_.AddCharacterClass(elements, is_negative);
  }

  int32_t CharacterClassStar(
      const std::vector<CharacterClassElement>& elements, bool is_negative = false
  ) {
    return builder_.AddCharacterClassStar(elements, is_negative);
  }

  int32_t Regex(const std::string& regex, bool json_string = false) {
    return builder_.AddRegex(regex, json_string);
  }

  int32_t RuleRef(int32_t rule_id) {
    auto it = rule_ref_expr_ids_.find(rule_id);
    if (it != rule_ref_expr_ids_.end()) {
      return it->second;
    }
    int32_t expr_id = builder_.AddRuleRef(rule_id);
    rule_ref_expr_ids_[rule_id] = expr_id;
    return expr_id;
  }

  int32_t RuleRef(const std::string& rule_name) {
    int32_t rule_id = builder_.GetRuleId(rule_name);
    XGRAMMAR_CHECK(rule_id != -1) << "Rule " << rule_name << " is not allocated";
    return RuleRef(rule_id);
  }

  int32_t Sequence(const std::vector<int32_t>& elements) {
    if (elements.empty()) {
      return Empty();
    }
    if (elements.size() == 1) {
      return elements[0];
    }
    return builder_.AddSequence(elements);
  }

  int32_t Choice(const std::vector<int32_t>& choices) {
    if (choices.empty()) {
      return Empty();
    }
    if (choices.size() == 1) {
      return choices[0];
    }
    return builder_.AddChoices(choices);
  }

  int32_t Repeat(
      const std::string& rule_name_hint, int32_t expr_id, int32_t min_count, int32_t max_count
  ) {
    if (min_count == 0 && max_count == 0) {
      return Empty();
    }
    if (min_count == 1 && max_count == 1) {
      return expr_id;
    }
    if (min_count == 0 && max_count == 1) {
      return Choice({Empty(), expr_id});
    }
    if (min_count == 0 && max_count == -1) {
      auto expr = builder_.GetGrammarExpr(expr_id);
      if (expr.type == GrammarBuilder::GrammarExprType::kCharacterClass) {
        std::vector<int32_t> data(expr.begin(), expr.end());
        return builder_.AddGrammarExpr(
            {GrammarBuilder::GrammarExprType::kCharacterClassStar,
             data.data(),
             static_cast<int32_t>(data.size())}
        );
      }
    }
    return builder_.AddRepeatFromExpr(rule_name_hint, expr_id, min_count, max_count);
  }

  void SetLookahead(const std::string& rule_name, int32_t lookahead_expr_id) {
    builder_.UpdateLookaheadAssertion(rule_name, lookahead_expr_id);
  }

  int32_t AddSubGrammar(const Grammar& grammar) {
    int32_t rule_id = SubGrammarAdder::Apply(&builder_, grammar);
    return RuleRef(rule_id);
  }

  Grammar Get(const std::string& root_rule_name) { return builder_.Get(root_rule_name); }

 private:
  GrammarBuilder builder_;
  std::optional<int32_t> empty_expr_id_;
  std::unordered_map<std::string, int32_t> byte_string_expr_ids_;
  std::unordered_map<int32_t, int32_t> rule_ref_expr_ids_;
};

class ASTIndentManager {
 public:
  ASTIndentManager(
      ASTCreator* creator,
      std::optional<int> indent,
      std::string separator,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt
  )
      : creator_(creator),
        any_whitespace_(any_whitespace),
        enable_newline_(indent.has_value()),
        indent_(indent.value_or(0)),
        separator_(std::move(separator)),
        total_indent_(0),
        is_first_({true}),
        max_whitespace_cnt_(max_whitespace_cnt) {
    XGRAMMAR_CHECK(!max_whitespace_cnt_.has_value() || *max_whitespace_cnt_ > 0)
        << "max_whitespace_cnt must be positive.";
  }

  void StartIndent() {
    total_indent_ += indent_;
    is_first_.push_back(true);
  }

  void EndIndent() {
    total_indent_ -= indent_;
    is_first_.pop_back();
  }

  int32_t Whitespace() {
    std::vector<CharacterClassElement> elements = {{' ', ' '}, {'\n', '\n'}, {'\t', '\t'}};
    if (!max_whitespace_cnt_.has_value()) {
      if (!whitespace_expr_id_.has_value()) {
        whitespace_expr_id_ = creator_->CharacterClassStar(elements);
      }
      return *whitespace_expr_id_;
    }
    // Keep bounded repetitions distinct. The EBNF parser historically creates one helper rule
    // per occurrence, and preserving that shape keeps optimized grammar printing stable.
    return creator_->Repeat(
        "whitespace",
        creator_->CharacterClass(elements),
        0,
        static_cast<int32_t>(*max_whitespace_cnt_)
    );
  }

  int32_t StartSeparator() {
    if (any_whitespace_) {
      return Whitespace();
    }
    if (!enable_newline_) {
      return creator_->Empty();
    }
    return creator_->ByteString("\n" + std::string(total_indent_, ' '));
  }

  int32_t MiddleSeparator() {
    if (any_whitespace_) {
      return creator_->Sequence({Whitespace(), creator_->ByteString(separator_), Whitespace()});
    }
    if (!enable_newline_) {
      return creator_->ByteString(separator_);
    }
    return creator_->ByteString(separator_ + "\n" + std::string(total_indent_, ' '));
  }

  int32_t EndSeparator() {
    if (any_whitespace_) {
      return Whitespace();
    }
    if (!enable_newline_) {
      return creator_->Empty();
    }
    return creator_->ByteString("\n" + std::string(total_indent_ - indent_, ' '));
  }

  int32_t EmptySeparator() { return any_whitespace_ ? Whitespace() : creator_->Empty(); }

  int32_t NextSeparator(bool is_end = false) {
    if (any_whitespace_) {
      if (is_first_.back() || is_end) {
        is_first_.back() = false;
        return Whitespace();
      }
      return creator_->Sequence({Whitespace(), creator_->ByteString(separator_), Whitespace()});
    }

    std::string separator;
    if (!is_first_.back() && !is_end) {
      separator += separator_;
    }
    is_first_.back() = false;
    if (enable_newline_) {
      separator += '\n';
    }
    separator += std::string(is_end ? total_indent_ - indent_ : total_indent_, ' ');
    return creator_->ByteString(separator);
  }

 private:
  ASTCreator* creator_;
  bool any_whitespace_;
  bool enable_newline_;
  int64_t indent_;
  std::string separator_;
  int64_t total_indent_;
  std::vector<bool> is_first_;
  std::optional<int> max_whitespace_cnt_;
  std::optional<int32_t> whitespace_expr_id_;
};

bool IsMultipleOf(int64_t value, int64_t multiple_of) { return value % multiple_of == 0; }

struct DirectTrieNode {
  bool is_terminal = false;
  std::map<uint8_t, DirectTrieNode> children;
};

}  // namespace

class JSONSchemaGrammarConverter::Impl {
 public:
  Impl(
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt,
      RefResolver ref_resolver,
      bool any_order
  )
      : indent_manager_(
            &creator_,
            indent,
            separators.has_value() ? separators->first
                                   : (any_whitespace ? "," : (indent.has_value() ? "," : ", ")),
            any_whitespace,
            max_whitespace_cnt
        ),
        any_whitespace_(any_whitespace),
        max_whitespace_cnt_(max_whitespace_cnt),
        any_order_(any_order),
        ref_resolver_(std::move(ref_resolver)) {
    std::string colon_separator =
        separators.has_value() ? separators->second : (any_whitespace ? ":" : ": ");
    colon_expr_id_ = any_whitespace ? creator_.Sequence(
                                          {indent_manager_.Whitespace(),
                                           creator_.ByteString(colon_separator),
                                           indent_manager_.Whitespace()}
                                      )
                                    : creator_.ByteString(colon_separator);
  }

  Grammar Convert(const SchemaSpecPtr& spec) {
    AddBasicRules();

    std::string root_rule_name = creator_.AllocateRuleName("root");
    uri_to_rule_name_["#"] = root_rule_name;

    auto cached_rule = GetCache(spec->cache_key);
    if (cached_rule.has_value()) {
      creator_.AddRuleWithAllocatedName(root_rule_name, creator_.RuleRef(*cached_rule));
    } else {
      if (!spec->cache_key.empty()) {
        AddCache(spec->cache_key, root_rule_name);
      }
      creator_.AddRuleWithAllocatedName(root_rule_name, GenerateFromSpec(spec, root_rule_name));
    }
    return creator_.Get(root_rule_name);
  }

 private:
  int32_t GenerateFromSpec(const SchemaSpecPtr& spec, const std::string& rule_name_hint) {
    return std::visit(
        [this, &rule_name_hint](const auto& value) -> int32_t {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, IntegerSpec>) {
            return GenerateInteger(value, rule_name_hint);
          } else if constexpr (std::is_same_v<T, NumberSpec>) {
            return GenerateNumber(value, rule_name_hint);
          } else if constexpr (std::is_same_v<T, StringSpec>) {
            return GenerateString(value, rule_name_hint);
          } else if constexpr (std::is_same_v<T, BooleanSpec>) {
            return GenerateBoolean(value, rule_name_hint);
          } else if constexpr (std::is_same_v<T, NullSpec>) {
            return GenerateNull(value, rule_name_hint);
          } else if constexpr (std::is_same_v<T, ArraySpec>) {
            return GenerateArray(value, rule_name_hint);
          } else if constexpr (std::is_same_v<T, ObjectSpec>) {
            return GenerateObject(value, rule_name_hint);
          } else if constexpr (std::is_same_v<T, AnySpec>) {
            return GenerateAny(value, rule_name_hint);
          } else if constexpr (std::is_same_v<T, ConstSpec>) {
            return GenerateConst(value, rule_name_hint);
          } else if constexpr (std::is_same_v<T, EnumSpec>) {
            return GenerateEnum(value, rule_name_hint);
          } else if constexpr (std::is_same_v<T, RefSpec>) {
            return GenerateRef(value, rule_name_hint);
          } else if constexpr (std::is_same_v<T, AnyOfSpec>) {
            return GenerateAnyOf(value, rule_name_hint);
          } else if constexpr (std::is_same_v<T, OneOfSpec>) {
            return GenerateOneOf(value, rule_name_hint);
          } else if constexpr (std::is_same_v<T, AllOfSpec>) {
            return GenerateAllOf(value, rule_name_hint);
          } else if constexpr (std::is_same_v<T, TypeArraySpec>) {
            return GenerateTypeArray(value, rule_name_hint);
          } else {
            XGRAMMAR_LOG(FATAL) << "Unknown JSON schema specification type";
          }
        },
        spec->spec
    );
  }

  std::string CreateRule(const SchemaSpecPtr& spec, const std::string& rule_name_hint) {
    auto cached = GetCache(spec->cache_key);
    if (cached.has_value()) {
      return *cached;
    }
    std::string rule_name = creator_.AllocateRuleName(rule_name_hint);
    creator_.AddRuleWithAllocatedName(rule_name, GenerateFromSpec(spec, rule_name));
    return rule_name;
  }

  void AddCache(const std::string& key, const std::string& rule_name) {
    if (!key.empty()) {
      rule_cache_[key] = rule_name;
    }
  }

  std::optional<std::string> GetCache(const std::string& key) const {
    if (key.empty()) {
      return std::nullopt;
    }
    auto it = rule_cache_.find(key);
    return it == rule_cache_.end() ? std::nullopt : std::optional<std::string>(it->second);
  }

  int32_t RegexExpression(
      const std::string& regex, bool json_string = false, bool force_cfg_expansion = false
  ) {
    bool can_use_fsm = !force_cfg_expansion;
    if (json_string) {
      can_use_fsm =
          can_use_fsm && std::all_of(regex.begin(), regex.end(), [](unsigned char character) {
            return character >= 0x20 && character <= 0x7e;
          });
    }
    if (can_use_fsm) {
      auto fsm_result = GrammarFSMBuilder::Regex(regex, json_string);
      if (fsm_result.IsOk()) {
        auto fsm = std::move(fsm_result).Unwrap();
        std::unordered_set<int> reachable_states;
        fsm.GetReachableStates(&reachable_states);
        bool language_is_empty =
            std::none_of(reachable_states.begin(), reachable_states.end(), [&](int state) {
              return fsm.IsEndState(state);
            });
        if (!language_is_empty) {
          return creator_.Regex(regex, json_string);
        }
      }
    }

    // Regex conversion has its own syntax and compatibility behavior. Build that syntax directly
    // as a subgrammar when the finite-state regex node cannot represent the expression.
    return creator_.AddSubGrammar(RegexToGrammar(regex));
  }

  void AddBasicRules() {
    const std::vector<std::string> basic_rule_names = {
        JSONSchemaConverter::kBasicEscape,
        JSONSchemaConverter::kBasicStringSub,
        JSONSchemaConverter::kBasicAny,
        JSONSchemaConverter::kBasicInteger,
        JSONSchemaConverter::kBasicNumber,
        JSONSchemaConverter::kBasicString,
        JSONSchemaConverter::kBasicBoolean,
        JSONSchemaConverter::kBasicNull,
        JSONSchemaConverter::kBasicArray,
        JSONSchemaConverter::kBasicObject,
    };
    for (const auto& name : basic_rule_names) {
      creator_.ReserveRule(name);
    }
    AddHelperRules();

    auto saved_indent_manager = indent_manager_;
    indent_manager_ = ASTIndentManager(
        &creator_,
        std::nullopt,
        any_whitespace_ ? "," : ", ",
        any_whitespace_,
        any_whitespace_ ? max_whitespace_cnt_ : std::nullopt
    );

    auto any_spec = SchemaSpec::Make(AnySpec{}, "{}", JSONSchemaConverter::kBasicAny);
    creator_.AddRuleWithAllocatedName(
        JSONSchemaConverter::kBasicAny,
        GenerateAny(std::get<AnySpec>(any_spec->spec), JSONSchemaConverter::kBasicAny)
    );
    AddCache("{}", JSONSchemaConverter::kBasicAny);

    constexpr const char* kIntegerCacheKey = "{\"type\":\"integer\"}";
    creator_.AddRuleWithAllocatedName(
        JSONSchemaConverter::kBasicInteger,
        GenerateInteger(IntegerSpec{}, JSONSchemaConverter::kBasicInteger)
    );
    AddCache(kIntegerCacheKey, JSONSchemaConverter::kBasicInteger);

    constexpr const char* kNumberCacheKey = "{\"type\":\"number\"}";
    creator_.AddRuleWithAllocatedName(
        JSONSchemaConverter::kBasicNumber,
        GenerateNumber(NumberSpec{}, JSONSchemaConverter::kBasicNumber)
    );
    AddCache(kNumberCacheKey, JSONSchemaConverter::kBasicNumber);

    constexpr const char* kStringCacheKey = "{\"type\":\"string\"}";
    creator_.AddRuleWithAllocatedName(
        JSONSchemaConverter::kBasicString,
        creator_.Sequence(
            {creator_.ByteString("\""), creator_.RuleRef(JSONSchemaConverter::kBasicStringSub)}
        )
    );
    AddCache(kStringCacheKey, JSONSchemaConverter::kBasicString);

    constexpr const char* kBooleanCacheKey = "{\"type\":\"boolean\"}";
    creator_.AddRuleWithAllocatedName(
        JSONSchemaConverter::kBasicBoolean,
        GenerateBoolean(BooleanSpec{}, JSONSchemaConverter::kBasicBoolean)
    );
    AddCache(kBooleanCacheKey, JSONSchemaConverter::kBasicBoolean);

    constexpr const char* kNullCacheKey = "{\"type\":\"null\"}";
    creator_.AddRuleWithAllocatedName(
        JSONSchemaConverter::kBasicNull, GenerateNull(NullSpec{}, JSONSchemaConverter::kBasicNull)
    );
    AddCache(kNullCacheKey, JSONSchemaConverter::kBasicNull);

    constexpr const char* kArrayCacheKey = "{\"type\":\"array\"}";
    ArraySpec array_spec;
    array_spec.allow_additional_items = true;
    array_spec.additional_items = any_spec;
    creator_.AddRuleWithAllocatedName(
        JSONSchemaConverter::kBasicArray,
        GenerateArray(array_spec, JSONSchemaConverter::kBasicArray)
    );
    AddCache(kArrayCacheKey, JSONSchemaConverter::kBasicArray);

    constexpr const char* kObjectCacheKey = "{\"type\":\"object\"}";
    ObjectSpec object_spec;
    object_spec.allow_additional_properties = true;
    object_spec.additional_properties_schema = any_spec;
    creator_.AddRuleWithAllocatedName(
        JSONSchemaConverter::kBasicObject,
        GenerateObject(object_spec, JSONSchemaConverter::kBasicObject)
    );
    AddCache(kObjectCacheKey, JSONSchemaConverter::kBasicObject);

    indent_manager_ = saved_indent_manager;
  }

  void AddHelperRules() {
    if (max_whitespace_cnt_.has_value()) {
      // Preserve historical helper-rule numbering after grammar optimization. The text parser
      // allocated one initial bounded-repetition helper that dead-code elimination later removed.
      creator_.AddRule(JSONSchemaConverter::kBasicStringSub, creator_.Empty());
    }
    int32_t escaped_character = creator_.CharacterClass(
        {{'"', '"'},
         {'\\', '\\'},
         {'/', '/'},
         {'b', 'b'},
         {'f', 'f'},
         {'n', 'n'},
         {'r', 'r'},
         {'t', 't'}}
    );
    int32_t hexadecimal_character = creator_.CharacterClass({{'A', 'F'}, {'a', 'f'}, {'0', '9'}});
    int32_t unicode_escape = creator_.Sequence(
        {creator_.ByteString("u"),
         hexadecimal_character,
         hexadecimal_character,
         hexadecimal_character,
         hexadecimal_character}
    );
    creator_.AddRuleWithAllocatedName(
        JSONSchemaConverter::kBasicEscape, creator_.Choice({escaped_character, unicode_escape})
    );

    int32_t normal_character = creator_.CharacterClass(
        {{0, 0x1f}, {'"', '"'}, {'\\', '\\'}, {'\r', '\r'}, {'\n', '\n'}}, true
    );
    int32_t string_sub_ref = creator_.RuleRef(JSONSchemaConverter::kBasicStringSub);
    int32_t string_sub_body = creator_.Choice(
        {creator_.ByteString("\""),
         creator_.Sequence({normal_character, string_sub_ref}),
         creator_.Sequence(
             {creator_.ByteString("\\"),
              creator_.RuleRef(JSONSchemaConverter::kBasicEscape),
              string_sub_ref}
         )}
    );
    creator_.AddRuleWithAllocatedName(JSONSchemaConverter::kBasicStringSub, string_sub_body);
    int32_t closing_context =
        creator_.CharacterClass({{',', ','}, {'}', '}'}, {']', ']'}, {':', ':'}});
    creator_.SetLookahead(
        JSONSchemaConverter::kBasicStringSub,
        creator_.Sequence({indent_manager_.Whitespace(), closing_context})
    );
  }

  int32_t GenerateInteger(const IntegerSpec& spec, const std::string& rule_name) {
    std::optional<int64_t> start = spec.minimum;
    std::optional<int64_t> end = spec.maximum;
    if (spec.exclusive_minimum.has_value() &&
        (!start.has_value() || *spec.exclusive_minimum >= *start)) {
      XGRAMMAR_CHECK(*spec.exclusive_minimum != std::numeric_limits<int64_t>::max());
      start = *spec.exclusive_minimum + 1;
    }
    if (spec.exclusive_maximum.has_value() &&
        (!end.has_value() || *spec.exclusive_maximum <= *end)) {
      XGRAMMAR_CHECK(*spec.exclusive_maximum != std::numeric_limits<int64_t>::min());
      end = *spec.exclusive_maximum - 1;
    }

    if (spec.multiple_of.has_value()) {
      if (start.has_value() && end.has_value()) {
        std::vector<int32_t> multiples;
        for (int64_t value = *start; value <= *end; ++value) {
          if (IsMultipleOf(value, *spec.multiple_of)) {
            multiples.push_back(creator_.ByteString(std::to_string(value)));
          }
          if (value == std::numeric_limits<int64_t>::max()) {
            break;
          }
        }
        return creator_.Choice(multiples);
      }
      return GenerateIntegerMultipleOfDFA(*spec.multiple_of, rule_name);
    }
    if (start.has_value() || end.has_value()) {
      return RegexExpression(GenerateRangeRegex(start, end), false, /*force_cfg_expansion=*/true);
    }
    int32_t optional_minus = creator_.Choice({creator_.Empty(), creator_.ByteString("-")});
    return creator_.Choice(
        {creator_.ByteString("0"),
         creator_.Sequence(
             {optional_minus,
              creator_.CharacterClass({{'1', '9'}}),
              creator_.CharacterClassStar({{'0', '9'}})}
         )}
    );
  }

  int32_t GenerateIntegerMultipleOfDFA(int64_t multiple_of, const std::string& rule_name) {
    std::vector<std::string> states(multiple_of);
    for (int64_t state = 0; state < multiple_of; ++state) {
      states[state] = creator_.AllocateRuleName(
          rule_name + "_multiple_of_" + std::to_string(multiple_of) + "_mod_" +
          std::to_string(state)
      );
    }
    for (int64_t state = 0; state < multiple_of; ++state) {
      std::vector<int32_t> transitions;
      if (state == 0) {
        transitions.push_back(creator_.Empty());
      }
      for (int64_t digit = 0; digit <= 9; ++digit) {
        int64_t next_state = (state * 10 + digit) % multiple_of;
        transitions.push_back(creator_.Sequence(
            {creator_.ByteString(std::to_string(digit)), creator_.RuleRef(states[next_state])}
        ));
      }
      creator_.AddRuleWithAllocatedName(states[state], creator_.Choice(transitions));
    }

    std::vector<int32_t> non_zero_starts;
    for (int64_t digit = 1; digit <= 9; ++digit) {
      non_zero_starts.push_back(creator_.Sequence(
          {creator_.ByteString(std::to_string(digit)), creator_.RuleRef(states[digit % multiple_of])
          }
      ));
    }
    return creator_.Choice(
        {creator_.ByteString("0"),
         creator_.Sequence(
             {creator_.Choice({creator_.Empty(), creator_.ByteString("-")}),
              creator_.Choice(non_zero_starts)}
         )}
    );
  }

  int32_t GenerateNumber(const NumberSpec& spec, const std::string& rule_name) {
    std::optional<double> start = spec.minimum;
    std::optional<double> end = spec.maximum;
    bool exclusive_start = false;
    bool exclusive_end = false;
    if (spec.exclusive_minimum.has_value() &&
        (!start.has_value() || *spec.exclusive_minimum >= *start)) {
      start = spec.exclusive_minimum;
      exclusive_start = true;
    }
    if (spec.exclusive_maximum.has_value() &&
        (!end.has_value() || *spec.exclusive_maximum <= *end)) {
      end = spec.exclusive_maximum;
      exclusive_end = true;
    }
    if (start.has_value() || end.has_value()) {
      return RegexExpression(
          GenerateFloatRangeRegex(start, end, exclusive_start, exclusive_end),
          false,
          /*force_cfg_expansion=*/true
      );
    }

    int32_t optional_minus = creator_.Choice({creator_.Empty(), creator_.ByteString("-")});
    int32_t integer_part = creator_.Choice(
        {creator_.ByteString("0"),
         creator_.Sequence(
             {creator_.CharacterClass({{'1', '9'}}), creator_.CharacterClassStar({{'0', '9'}})}
         )}
    );
    int32_t one_or_more_digits =
        creator_.Repeat(rule_name + "_digits", creator_.CharacterClass({{'0', '9'}}), 1, -1);
    int32_t fraction = creator_.Choice(
        {creator_.Empty(), creator_.Sequence({creator_.ByteString("."), one_or_more_digits})}
    );
    int32_t exponent = creator_.Choice(
        {creator_.Empty(),
         creator_.Sequence(
             {creator_.CharacterClass({{'e', 'e'}, {'E', 'E'}}),
              creator_.Choice({creator_.Empty(), creator_.CharacterClass({{'+', '+'}, {'-', '-'}})}
              ),
              one_or_more_digits}
         )}
    );
    return creator_.Sequence({optional_minus, integer_part, fraction, exponent});
  }

  int32_t GenerateString(const StringSpec& spec, const std::string& rule_name) {
    if (spec.format.has_value()) {
      auto regex = JSONSchemaConverter::JSONFormatToRegexPattern(*spec.format);
      if (regex.has_value()) {
        return creator_.Sequence(
            {creator_.ByteString("\""),
             RegexExpression(*regex, false, true),
             creator_.ByteString("\"")}
        );
      }
    }
    if (spec.pattern.has_value()) {
      return creator_.Sequence(
          {creator_.ByteString("\""),
           RegexExpression(*spec.pattern, true),
           creator_.ByteString("\"")}
      );
    }
    if (spec.min_length != 0 || spec.max_length != -1) {
      int32_t character =
          creator_.CharacterClass({{'"', '"'}, {'\\', '\\'}, {'\r', '\r'}, {'\n', '\n'}}, true);
      int32_t body =
          creator_.Repeat(rule_name + "_characters", character, spec.min_length, spec.max_length);
      return creator_.Sequence({creator_.ByteString("\""), body, creator_.ByteString("\"")});
    }
    return creator_.Sequence(
        {creator_.ByteString("\""), creator_.RuleRef(JSONSchemaConverter::kBasicStringSub)}
    );
  }

  int32_t GenerateBoolean(const BooleanSpec&, const std::string&) {
    return creator_.Choice({creator_.ByteString("true"), creator_.ByteString("false")});
  }

  int32_t GenerateNull(const NullSpec&, const std::string&) { return creator_.ByteString("null"); }

  int32_t GenerateArray(const ArraySpec& spec, const std::string& rule_name) {
    indent_manager_.StartIndent();
    int32_t start_separator = indent_manager_.StartSeparator();
    int32_t middle_separator = indent_manager_.MiddleSeparator();
    int32_t end_separator = indent_manager_.EndSeparator();
    int32_t empty_separator = indent_manager_.EmptySeparator();

    std::vector<std::string> item_rules;
    for (size_t index = 0; index < spec.prefix_items.size(); ++index) {
      item_rules.push_back(
          CreateRule(spec.prefix_items[index], rule_name + "_item_" + std::to_string(index))
      );
    }
    std::string additional_rule;
    if (spec.allow_additional_items && spec.additional_items) {
      additional_rule = CreateRule(spec.additional_items, rule_name + "_additional");
    }
    indent_manager_.EndIndent();

    int32_t left_bracket = creator_.ByteString("[");
    int32_t right_bracket = creator_.ByteString("]");
    int32_t empty_array = creator_.Sequence({left_bracket, empty_separator, right_bracket});

    if (item_rules.empty()) {
      if (!spec.allow_additional_items || spec.max_items == 0) {
        return empty_array;
      }
      int32_t additional = creator_.RuleRef(additional_rule);
      int32_t tail = creator_.Repeat(
          rule_name + "_items",
          creator_.Sequence({middle_separator, additional}),
          spec.min_items == 0 ? 0 : static_cast<int32_t>(spec.min_items - 1),
          spec.max_items == -1 ? -1 : static_cast<int32_t>(spec.max_items - 1)
      );
      int32_t nonempty = creator_.Sequence(
          {left_bracket, start_separator, additional, tail, end_separator, right_bracket}
      );
      return spec.min_items == 0 ? creator_.Choice({nonempty, empty_array}) : nonempty;
    }

    std::vector<int32_t> prefix_elements;
    for (size_t index = 0; index < item_rules.size(); ++index) {
      if (index != 0) {
        prefix_elements.push_back(middle_separator);
      }
      prefix_elements.push_back(creator_.RuleRef(item_rules[index]));
    }
    int32_t prefix = creator_.Sequence(prefix_elements);
    if (!spec.allow_additional_items) {
      return creator_.Sequence({left_bracket, start_separator, prefix, end_separator, right_bracket}
      );
    }

    int64_t minimum_additional =
        std::max(int64_t{0}, spec.min_items - static_cast<int64_t>(item_rules.size()));
    int32_t additional_tail = creator_.Repeat(
        rule_name + "_additional_items",
        creator_.Sequence({middle_separator, creator_.RuleRef(additional_rule)}),
        static_cast<int32_t>(minimum_additional),
        spec.max_items == -1
            ? -1
            : static_cast<int32_t>(spec.max_items - static_cast<int64_t>(item_rules.size()))
    );
    return creator_.Sequence(
        {left_bracket, start_separator, prefix, additional_tail, end_separator, right_bracket}
    );
  }

  int32_t FormatPropertyKey(const std::string& key) {
    return creator_.ByteString(picojson::value(key).serialize());
  }

  int32_t FormatProperty(const std::string& key, const std::string& value_rule) {
    return creator_.Sequence({FormatPropertyKey(key), colon_expr_id_, creator_.RuleRef(value_rule)}
    );
  }

  int32_t FormatOtherProperty(int32_t key_pattern, const std::string& value_rule) {
    return creator_.Sequence({key_pattern, colon_expr_id_, creator_.RuleRef(value_rule)});
  }

  int32_t BuildTrieBody(const DirectTrieNode& node, const std::string& rule_name) {
    std::vector<int32_t> choices;
    if (!node.is_terminal) {
      choices.push_back(creator_.ByteString("\""));
    }

    std::vector<CharacterClassElement> excluded = {
        {0, 0x1f}, {'"', '"'}, {'\\', '\\'}, {'\r', '\r'}, {'\n', '\n'}
    };
    for (const auto& [character, child] : node.children) {
      static_cast<void>(child);
      excluded.push_back({character, character});
    }
    choices.push_back(creator_.Sequence(
        {creator_.CharacterClass(excluded, true),
         creator_.RuleRef(JSONSchemaConverter::kBasicStringSub)}
    ));
    choices.push_back(creator_.Sequence(
        {creator_.ByteString("\\"),
         creator_.RuleRef(JSONSchemaConverter::kBasicEscape),
         creator_.RuleRef(JSONSchemaConverter::kBasicStringSub)}
    ));
    for (const auto& [character, child] : node.children) {
      choices.push_back(creator_.Sequence(
          {creator_.ByteString(std::string(1, static_cast<char>(character))),
           BuildTrieBody(child, rule_name)}
      ));
    }
    return creator_.Choice(choices);
  }

  int32_t GetKeyPatternExcluding(
      const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
  ) {
    if (properties.empty()) {
      return creator_.RuleRef(JSONSchemaConverter::kBasicString);
    }

    DirectTrieNode root;
    for (const auto& property : properties) {
      DirectTrieNode* current = &root;
      for (unsigned char character : property.name) {
        current = &current->children[character];
      }
      current->is_terminal = true;
    }

    std::string key_rule_name = creator_.AllocateRuleName(rule_name + "_addl_key");
    creator_.AddRuleWithAllocatedName(
        key_rule_name,
        creator_.Sequence({creator_.ByteString("\""), BuildTrieBody(root, key_rule_name)})
    );
    creator_.SetLookahead(
        key_rule_name,
        creator_.Sequence(
            {indent_manager_.Whitespace(),
             creator_.CharacterClass({{',', ','}, {'}', '}'}, {']', ']'}, {':', ':'}})}
        )
    );
    return creator_.RuleRef(key_rule_name);
  }

  int32_t GetPropertyWithNumberConstraints(
      int32_t pattern,
      int min_properties,
      int max_properties,
      int already_repeated_times,
      const std::string& rule_name
  ) {
    if (max_properties != -1 && max_properties == already_repeated_times) {
      return creator_.Empty();
    }
    int lower = std::max(0, min_properties - already_repeated_times);
    int upper = max_properties == -1 ? -1 : std::max(-1, max_properties - already_repeated_times);
    return creator_.Repeat(rule_name + "_properties", pattern, lower, upper);
  }

  int32_t GetAnyOrderRuleForProperties(
      const std::vector<ObjectSpec::Property>& properties,
      const std::unordered_set<std::string>& required,
      const SchemaSpecPtr& additional,
      const std::string& rule_name,
      const std::string& additional_suffix,
      int min_properties,
      int max_properties,
      const std::optional<int32_t>& additional_property_override
  ) {
    int32_t first_separator = indent_manager_.NextSeparator();
    int32_t middle_separator = indent_manager_.NextSeparator();
    int32_t last_separator = indent_manager_.NextSeparator(true);

    std::vector<int32_t> items;
    for (size_t index = 0; index < properties.size(); ++index) {
      const auto& property = properties[index];
      std::string value_rule =
          CreateRule(property.schema, rule_name + "_prop_" + std::to_string(index));
      items.push_back(FormatProperty(property.name, value_rule));
    }
    if (additional != nullptr) {
      if (additional_property_override.has_value()) {
        items.push_back(*additional_property_override);
      } else {
        std::string value_rule = CreateRule(additional, rule_name + "_" + additional_suffix);
        items.push_back(
            FormatOtherProperty(GetKeyPatternExcluding(properties, rule_name), value_rule)
        );
      }
    }

    std::string item_rule = creator_.AddRule(rule_name + "_item", creator_.Choice(items));
    int minimum_count = std::max(min_properties, static_cast<int>(required.size()));
    int32_t repeated_items = GetPropertyWithNumberConstraints(
        creator_.Sequence({middle_separator, creator_.RuleRef(item_rule)}),
        minimum_count,
        max_properties,
        1,
        rule_name
    );
    return creator_.Sequence(
        {first_separator, creator_.RuleRef(item_rule), repeated_items, last_separator}
    );
  }

  int32_t GetPartialRuleForProperties(
      const std::vector<ObjectSpec::Property>& properties,
      const std::unordered_set<std::string>& required,
      const SchemaSpecPtr& additional,
      const std::string& rule_name,
      const std::string& additional_suffix,
      int min_properties,
      int max_properties,
      const std::optional<int32_t>& additional_property_override = std::nullopt
  ) {
    if (max_properties == 0) {
      return creator_.Empty();
    }
    if (any_order_) {
      return GetAnyOrderRuleForProperties(
          properties,
          required,
          additional,
          rule_name,
          additional_suffix,
          min_properties,
          max_properties,
          additional_property_override
      );
    }

    int32_t first_separator = indent_manager_.NextSeparator();
    int32_t middle_separator = indent_manager_.NextSeparator();
    int32_t last_separator = indent_manager_.NextSeparator(true);

    std::vector<int32_t> property_patterns;
    for (size_t index = 0; index < properties.size(); ++index) {
      std::string value_rule =
          CreateRule(properties[index].schema, rule_name + "_prop_" + std::to_string(index));
      property_patterns.push_back(FormatProperty(properties[index].name, value_rule));
    }

    bool allow_additional = additional != nullptr;
    std::optional<int32_t> additional_pattern;
    auto get_additional_pattern = [&]() -> int32_t {
      if (!additional_pattern.has_value()) {
        if (additional_property_override.has_value()) {
          additional_pattern = *additional_property_override;
        } else {
          std::string value_rule = CreateRule(additional, rule_name + "_" + additional_suffix);
          additional_pattern =
              FormatOtherProperty(GetKeyPatternExcluding(properties, rule_name), value_rule);
        }
      }
      return *additional_pattern;
    };

    if (min_properties == 0 && max_properties == -1) {
      std::vector<int32_t> tails(properties.size(), creator_.Empty());
      std::vector<uint8_t> is_required(properties.size(), false);

      if (allow_additional) {
        int32_t repeated_additional = creator_.Repeat(
            rule_name + "_additional_properties",
            creator_.Sequence({middle_separator, get_additional_pattern()}),
            0,
            -1
        );
        std::string tail_rule = creator_.AddRule(
            rule_name + "_part_" + std::to_string(static_cast<int>(properties.size()) - 1),
            repeated_additional
        );
        tails.back() = creator_.RuleRef(tail_rule);
      }

      for (int index = static_cast<int>(properties.size()) - 2; index >= 0; --index) {
        int32_t with_property =
            creator_.Sequence({middle_separator, property_patterns[index + 1], tails[index + 1]});
        int32_t body = with_property;
        if (!required.count(properties[index + 1].name)) {
          body = creator_.Choice({tails[index + 1], with_property});
        } else {
          is_required[index + 1] = true;
        }
        std::string tail_rule =
            creator_.AddRule(rule_name + "_part_" + std::to_string(index), body);
        tails[index] = creator_.RuleRef(tail_rule);
      }
      if (required.count(properties[0].name)) {
        is_required[0] = true;
      }

      std::vector<int32_t> choices;
      for (size_t index = 0; index < properties.size(); ++index) {
        choices.push_back(creator_.Sequence({property_patterns[index], tails[index]}));
        if (is_required[index]) {
          break;
        }
      }
      if (allow_additional && required.empty()) {
        choices.push_back(creator_.Sequence({get_additional_pattern(), tails.back()}));
      }
      return creator_.Sequence({first_separator, creator_.Choice(choices), last_separator});
    }

    const int property_count = static_cast<int>(properties.size());
    std::vector<uint8_t> is_required(property_count, false);
    std::vector<int> matched_min(property_count, 0);
    bool found_required = required.count(properties[0].name);
    matched_min[0] = 1;
    for (int index = 1; index < property_count; ++index) {
      if (required.count(properties[index].name)) {
        is_required[index] = true;
        matched_min[index] = matched_min[index - 1] + 1;
      } else {
        matched_min[index] = matched_min[index - 1];
      }
      if (!found_required) {
        matched_min[index] = 1;
      }
      if (is_required[index]) {
        found_required = true;
      }
    }
    if (required.count(properties[0].name)) {
      is_required[0] = true;
    }

    if (max_properties == -1) {
      std::vector<std::vector<int32_t>> tails(property_count);
      matched_min.back() = allow_additional ? std::max(1, matched_min.back())
                                            : std::max(min_properties, matched_min.back());
      for (int index = property_count - 2; index >= 0; --index) {
        matched_min[index] = std::max(matched_min[index], matched_min[index + 1] - 1);
      }

      for (int matched = matched_min.back(); matched <= property_count; ++matched) {
        int32_t body = allow_additional
                           ? GetPropertyWithNumberConstraints(
                                 creator_.Sequence({middle_separator, get_additional_pattern()}),
                                 min_properties,
                                 max_properties,
                                 matched,
                                 rule_name
                             )
                           : creator_.Empty();
        if (allow_additional) {
          std::string tail_rule = creator_.AddRule(
              rule_name + "_part_" + std::to_string(property_count - 1) + "_" +
                  std::to_string(matched),
              body
          );
          tails.back().push_back(creator_.RuleRef(tail_rule));
        } else {
          tails.back().push_back(body);
        }
      }

      for (int index = property_count - 2; index >= 0; --index) {
        for (int matched = matched_min[index]; matched <= index + 1; ++matched) {
          int32_t with_property = creator_.Sequence(
              {middle_separator,
               property_patterns[index + 1],
               tails[index + 1][matched + 1 - matched_min[index + 1]]}
          );
          int32_t body =
              (is_required[index + 1] || matched == matched_min[index + 1] - 1)
                  ? with_property
                  : creator_.Choice(
                        {tails[index + 1][matched - matched_min[index + 1]], with_property}
                    );
          std::string tail_rule = creator_.AddRule(
              rule_name + "_part_" + std::to_string(index) + "_" + std::to_string(matched), body
          );
          tails[index].push_back(creator_.RuleRef(tail_rule));
        }
      }

      std::vector<int32_t> choices;
      for (int index = 0; index < property_count; ++index) {
        if (matched_min[index] > 1) {
          break;
        }
        choices.push_back(
            creator_.Sequence({property_patterns[index], tails[index][1 - matched_min[index]]})
        );
        if (is_required[index]) {
          break;
        }
      }
      if (allow_additional && required.empty()) {
        choices.push_back(creator_.Sequence(
            {get_additional_pattern(),
             GetPropertyWithNumberConstraints(
                 creator_.Sequence({middle_separator, get_additional_pattern()}),
                 min_properties,
                 max_properties,
                 1,
                 rule_name
             )}
        ));
      }
      return creator_.Sequence({first_separator, creator_.Choice(choices), last_separator});
    }

    std::vector<std::vector<int32_t>> tails(property_count);
    std::vector<int> matched_max(property_count, property_count);
    matched_max[0] = 1;
    for (int index = 1; index < property_count; ++index) {
      matched_max[index] = matched_max[index - 1] + 1;
    }
    matched_min.back() = allow_additional ? std::max(1, matched_min.back())
                                          : std::max(min_properties, matched_min.back());
    matched_max.back() = std::min(max_properties, matched_max.back());
    for (int index = property_count - 2; index >= 0; --index) {
      matched_min[index] = std::max(matched_min[index], matched_min[index + 1] - 1);
      matched_max[index] = is_required[index + 1]
                               ? std::min(matched_max[index], matched_max[index + 1] - 1)
                               : std::min(matched_max[index], matched_max[index + 1]);
    }

    for (int matched = matched_min.back(); matched <= matched_max.back(); ++matched) {
      int32_t body = allow_additional
                         ? GetPropertyWithNumberConstraints(
                               creator_.Sequence({middle_separator, get_additional_pattern()}),
                               min_properties,
                               max_properties,
                               matched,
                               rule_name
                           )
                         : creator_.Empty();
      if (allow_additional) {
        std::string tail_rule = creator_.AddRule(
            rule_name + "_part_" + std::to_string(property_count - 1) + "_" +
                std::to_string(matched),
            body
        );
        tails.back().push_back(creator_.RuleRef(tail_rule));
      } else {
        tails.back().push_back(body);
      }
    }

    for (int index = property_count - 2; index >= 0; --index) {
      for (int matched = matched_min[index]; matched <= matched_max[index]; ++matched) {
        int32_t body;
        if (matched == matched_max[index + 1]) {
          body = tails[index + 1][matched - matched_min[index + 1]];
        } else {
          int32_t with_property = creator_.Sequence(
              {middle_separator,
               property_patterns[index + 1],
               tails[index + 1][matched + 1 - matched_min[index + 1]]}
          );
          body = (is_required[index + 1] || matched == matched_min[index + 1] - 1)
                     ? with_property
                     : creator_.Choice(
                           {tails[index + 1][matched - matched_min[index + 1]], with_property}
                       );
        }
        std::string tail_rule = creator_.AddRule(
            rule_name + "_part_" + std::to_string(index) + "_" + std::to_string(matched), body
        );
        tails[index].push_back(creator_.RuleRef(tail_rule));
      }
    }

    std::vector<int32_t> choices;
    for (int index = 0; index < property_count; ++index) {
      if (matched_max[index] < matched_min[index]) {
        continue;
      }
      if (matched_min[index] > 1) {
        break;
      }
      choices.push_back(
          creator_.Sequence({property_patterns[index], tails[index][1 - matched_min[index]]})
      );
      if (is_required[index]) {
        break;
      }
    }
    if (allow_additional && required.empty()) {
      choices.push_back(creator_.Sequence(
          {get_additional_pattern(),
           GetPropertyWithNumberConstraints(
               creator_.Sequence({middle_separator, get_additional_pattern()}),
               min_properties,
               max_properties,
               1,
               rule_name
           )}
      ));
    }
    return creator_.Sequence({first_separator, creator_.Choice(choices), last_separator});
  }

  int32_t GenerateObject(
      const ObjectSpec& spec, const std::string& rule_name, bool need_braces = true
  ) {
    std::string additional_suffix;
    SchemaSpecPtr additional_property;
    if (spec.allow_additional_properties && spec.additional_properties_schema) {
      additional_suffix = "addl";
      additional_property = spec.additional_properties_schema;
    } else if (spec.allow_unevaluated_properties && spec.unevaluated_properties_schema) {
      additional_suffix = "uneval";
      additional_property = spec.unevaluated_properties_schema;
    } else if (spec.allow_additional_properties || spec.allow_unevaluated_properties) {
      additional_suffix = "addl";
      additional_property = SchemaSpec::Make(AnySpec{}, "", "any");
    }

    indent_manager_.StartIndent();
    bool has_content = false;
    bool could_be_empty = false;
    int32_t content = creator_.Empty();

    if (!spec.properties.empty() && (!spec.pattern_properties.empty() || spec.property_names)) {
      SchemaSpecPtr effective_additional = additional_property;
      std::string effective_suffix = additional_suffix;
      std::optional<int32_t> additional_override;

      if (!spec.pattern_properties.empty()) {
        std::vector<int32_t> patterns;
        for (size_t index = 0; index < spec.pattern_properties.size(); ++index) {
          const auto& pattern_property = spec.pattern_properties[index];
          std::string value_rule =
              CreateRule(pattern_property.schema, rule_name + "_pp_" + std::to_string(index));
          patterns.push_back(creator_.Sequence(
              {creator_.ByteString("\""),
               RegexExpression(pattern_property.pattern, true),
               creator_.ByteString("\""),
               colon_expr_id_,
               creator_.RuleRef(value_rule)}
          ));
        }
        if (effective_additional) {
          std::string value_rule =
              CreateRule(effective_additional, rule_name + "_" + effective_suffix);
          patterns.push_back(
              FormatOtherProperty(creator_.RuleRef(JSONSchemaConverter::kBasicString), value_rule)
          );
        }
        additional_override = creator_.Choice(patterns);
        if (!effective_additional) {
          effective_additional = SchemaSpec::Make(AnySpec{}, "", "any");
        }
        effective_suffix = "pp";
      } else if (spec.property_names && effective_additional) {
        std::string key_rule = CreateRule(spec.property_names, rule_name + "_name");
        std::string value_rule =
            CreateRule(effective_additional, rule_name + "_" + effective_suffix);
        additional_override = creator_.Sequence(
            {creator_.RuleRef(key_rule), colon_expr_id_, creator_.RuleRef(value_rule)}
        );
        effective_suffix = "pn";
      }

      content = GetPartialRuleForProperties(
          spec.properties,
          spec.required,
          effective_additional,
          rule_name,
          effective_suffix,
          spec.min_properties,
          spec.max_properties,
          additional_override
      );
      has_content = spec.max_properties != 0;
      could_be_empty = spec.required.empty() && spec.min_properties == 0;
    } else if (!spec.pattern_properties.empty() || spec.property_names) {
      if (spec.max_properties != 0) {
        int32_t beginning_separator = indent_manager_.NextSeparator();
        std::vector<int32_t> property_choices;
        if (!spec.pattern_properties.empty()) {
          for (size_t index = 0; index < spec.pattern_properties.size(); ++index) {
            const auto& pattern_property = spec.pattern_properties[index];
            std::string value_rule =
                CreateRule(pattern_property.schema, rule_name + "_prop_" + std::to_string(index));
            property_choices.push_back(creator_.Sequence(
                {beginning_separator,
                 creator_.ByteString("\""),
                 RegexExpression(pattern_property.pattern, true),
                 creator_.ByteString("\""),
                 colon_expr_id_,
                 creator_.RuleRef(value_rule)}
            ));
          }
        } else {
          std::string key_rule = CreateRule(spec.property_names, rule_name + "_name");
          property_choices.push_back(creator_.Sequence(
              {beginning_separator,
               creator_.RuleRef(key_rule),
               colon_expr_id_,
               creator_.RuleRef(JSONSchemaConverter::kBasicAny)}
          ));
        }

        std::string property_rule =
            creator_.AddRule(rule_name + "_prop", creator_.Choice(property_choices));
        int32_t subsequent_property =
            creator_.Sequence({indent_manager_.NextSeparator(), creator_.RuleRef(property_rule)});
        content = creator_.Sequence(
            {creator_.RuleRef(property_rule),
             GetPropertyWithNumberConstraints(
                 subsequent_property, spec.min_properties, spec.max_properties, 1, rule_name
             ),
             indent_manager_.NextSeparator(true)}
        );
        has_content = true;
        could_be_empty = spec.min_properties == 0;
      } else {
        could_be_empty = true;
      }
    } else if (!spec.properties.empty()) {
      content = GetPartialRuleForProperties(
          spec.properties,
          spec.required,
          additional_property,
          rule_name,
          additional_suffix,
          spec.min_properties,
          spec.max_properties
      );
      has_content = spec.max_properties != 0;
      could_be_empty = spec.required.empty() && spec.min_properties == 0;
    } else if (additional_property) {
      if (spec.max_properties != 0) {
        std::string value_rule =
            CreateRule(additional_property, rule_name + "_" + additional_suffix);
        int32_t property =
            FormatOtherProperty(creator_.RuleRef(JSONSchemaConverter::kBasicString), value_rule);
        content = creator_.Sequence(
            {indent_manager_.NextSeparator(),
             property,
             GetPropertyWithNumberConstraints(
                 creator_.Sequence({indent_manager_.NextSeparator(), property}),
                 spec.min_properties,
                 spec.max_properties,
                 1,
                 rule_name
             ),
             indent_manager_.NextSeparator(true)}
        );
        has_content = true;
      }
      could_be_empty = spec.min_properties == 0;
    } else {
      could_be_empty = true;
    }

    indent_manager_.EndIndent();

    int32_t result =
        need_braces
            ? creator_.Sequence({creator_.ByteString("{"), content, creator_.ByteString("}")})
            : content;
    if (could_be_empty) {
      int32_t empty_content = any_whitespace_ ? indent_manager_.Whitespace() : creator_.Empty();
      int32_t empty_result =
          need_braces ? creator_.Sequence(
                            {creator_.ByteString("{"), empty_content, creator_.ByteString("}")}
                        )
                      : empty_content;
      return has_content ? creator_.Choice({result, empty_result}) : empty_result;
    }
    return result;
  }

  int32_t GenerateAny(const AnySpec&, const std::string&) {
    return creator_.Choice(
        {creator_.RuleRef(JSONSchemaConverter::kBasicNumber),
         creator_.RuleRef(JSONSchemaConverter::kBasicString),
         creator_.RuleRef(JSONSchemaConverter::kBasicBoolean),
         creator_.RuleRef(JSONSchemaConverter::kBasicNull),
         creator_.RuleRef(JSONSchemaConverter::kBasicArray),
         creator_.RuleRef(JSONSchemaConverter::kBasicObject)}
    );
  }

  int32_t GenerateConst(const ConstSpec& spec, const std::string&) {
    return creator_.ByteString(spec.json_value);
  }

  int32_t GenerateEnum(const EnumSpec& spec, const std::string& rule_name) {
    XGRAMMAR_DCHECK(!spec.json_values.empty())
        << "GenerateEnum called with empty enum spec for rule: " << rule_name;
    std::vector<int32_t> values;
    values.reserve(spec.json_values.size());
    for (const auto& value : spec.json_values) {
      values.push_back(creator_.ByteString(value));
    }
    return creator_.Choice(values);
  }

  int32_t GenerateRef(const RefSpec& spec, const std::string&) {
    auto mapped = uri_to_rule_name_.find(spec.uri);
    if (mapped != uri_to_rule_name_.end()) {
      return creator_.RuleRef(mapped->second);
    }
    XGRAMMAR_CHECK(ref_resolver_) << "Ref resolver not set; cannot resolve $ref: " << spec.uri;

    std::string rule_name_hint = "ref";
    if (spec.uri.size() >= 2 && spec.uri[0] == '#' && spec.uri[1] == '/') {
      std::string prefix;
      std::stringstream stream(spec.uri.substr(2));
      std::string part;
      while (std::getline(stream, part, '/')) {
        if (part.empty()) {
          continue;
        }
        if (!prefix.empty()) {
          prefix += '_';
        }
        for (char character : part) {
          if (std::isalpha(static_cast<unsigned char>(character)) || character == '_' ||
              character == '-' || character == '.') {
            prefix += character;
          }
        }
      }
      if (!prefix.empty()) {
        rule_name_hint = std::move(prefix);
      }
    }

    std::string allocated_rule = creator_.AllocateRuleName(rule_name_hint);
    uri_to_rule_name_[spec.uri] = allocated_rule;
    SchemaSpecPtr resolved = ref_resolver_(spec.uri, allocated_rule);
    creator_.AddRuleWithAllocatedName(allocated_rule, GenerateFromSpec(resolved, allocated_rule));
    if (!resolved->cache_key.empty()) {
      AddCache(resolved->cache_key, allocated_rule);
    }
    return creator_.RuleRef(allocated_rule);
  }

  int32_t GenerateAnyOf(const AnyOfSpec& spec, const std::string& rule_name) {
    std::vector<int32_t> choices;
    for (size_t index = 0; index < spec.options.size(); ++index) {
      choices.push_back(creator_.RuleRef(
          CreateRule(spec.options[index], rule_name + "_case_" + std::to_string(index))
      ));
    }
    return creator_.Choice(choices);
  }

  int32_t GenerateOneOf(const OneOfSpec& spec, const std::string& rule_name) {
    std::vector<int32_t> choices;
    for (size_t index = 0; index < spec.options.size(); ++index) {
      choices.push_back(creator_.RuleRef(
          CreateRule(spec.options[index], rule_name + "_case_" + std::to_string(index))
      ));
    }
    return creator_.Choice(choices);
  }

  int32_t GenerateAllOf(const AllOfSpec& spec, const std::string& rule_name) {
    if (spec.schemas.size() == 1) {
      return GenerateFromSpec(spec.schemas[0], rule_name + "_case_0");
    }
    XGRAMMAR_LOG(WARNING) << "Support for allOf with multiple options is still ongoing";
    return GenerateAny(AnySpec{}, rule_name);
  }

  int32_t GenerateTypeArray(const TypeArraySpec& spec, const std::string& rule_name) {
    std::vector<int32_t> choices;
    for (size_t index = 0; index < spec.type_schemas.size(); ++index) {
      choices.push_back(creator_.RuleRef(
          CreateRule(spec.type_schemas[index], rule_name + "_type_" + std::to_string(index))
      ));
    }
    return creator_.Choice(choices);
  }

  ASTCreator creator_;
  ASTIndentManager indent_manager_;
  int32_t colon_expr_id_;
  bool any_whitespace_;
  std::optional<int> max_whitespace_cnt_;
  bool any_order_;
  RefResolver ref_resolver_;
  std::unordered_map<std::string, std::string> rule_cache_;
  std::unordered_map<std::string, std::string> uri_to_rule_name_;
};

JSONSchemaGrammarConverter::JSONSchemaGrammarConverter(
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt,
    RefResolver ref_resolver,
    bool any_order
)
    : impl_(std::make_shared<Impl>(
          indent,
          std::move(separators),
          any_whitespace,
          max_whitespace_cnt,
          std::move(ref_resolver),
          any_order
      )) {}

Grammar JSONSchemaGrammarConverter::Convert(const SchemaSpecPtr& spec) {
  return impl_->Convert(spec);
}

}  // namespace xgrammar
