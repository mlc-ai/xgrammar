/*!
 * Copyright (c) 2026 by Contributors
 * \file xgrammar/json_schema_converter_ast.cc
 * \brief Direct JSON Schema to grammar AST conversion.
 */

#include "json_schema_converter_ast.h"

#include <picojson.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "fsm_builder.h"
#include "grammar_builder.h"
#include "grammar_functor.h"
#include "json_schema_converter.h"
#include "support/logging.h"

namespace xgrammar {

namespace gs = grammar_spec;

namespace json_schema_ast_detail {

using Expr = GrammarExprSpec;
using CharacterClassElement = gs::CharacterClassElement;

Expr Empty() { return gs::EmptyStr(); }
Expr Str(std::string value) { return gs::ByteString(std::move(value)); }
Expr Ref(int32_t rule_id) { return gs::RuleRef(rule_id); }

Expr Seq(std::vector<Expr> elements) {
  if (elements.empty()) return Empty();
  if (elements.size() == 1) return std::move(elements[0]);
  return gs::Sequence(std::move(elements));
}

template <typename... Args>
Expr Seq(Args&&... args) {
  std::vector<Expr> elements;
  elements.reserve(sizeof...(Args));
  (elements.emplace_back(std::forward<Args>(args)), ...);
  return Seq(std::move(elements));
}

Expr Or(std::vector<Expr> choices) {
  XGRAMMAR_DCHECK(!choices.empty());
  if (choices.size() == 1) return std::move(choices[0]);
  return gs::Choices(std::move(choices));
}

template <typename... Args>
Expr Or(Args&&... args) {
  std::vector<Expr> choices;
  choices.reserve(sizeof...(Args));
  (choices.emplace_back(std::forward<Args>(args)), ...);
  return Or(std::move(choices));
}

Expr Repeat(Expr element, int32_t min_count, int32_t max_count) {
  if (min_count == 0 && max_count == 0) return Empty();
  if (min_count == 1 && max_count == 1) return element;
  return gs::Repeat(std::move(element), min_count, max_count);
}

Expr CharClass(std::vector<CharacterClassElement> elements, bool is_negative = false) {
  return gs::CharacterClass(std::move(elements), is_negative);
}

Expr CharClassStar(std::vector<CharacterClassElement> elements, bool is_negative = false) {
  return gs::CharacterClassStar(std::move(elements), is_negative);
}

std::vector<CharacterClassElement> WhitespaceRanges() {
  return {{' ', ' '}, {'\n', '\n'}, {'\t', '\t'}};
}

struct SeparatorExpr {
  bool is_whitespace = false;
  std::string literal;
};

class ASTIndentManager {
 public:
  ASTIndentManager(
      std::optional<int> indent,
      std::string separator,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt
  )
      : any_whitespace_(any_whitespace),
        enable_newline_(indent.has_value()),
        indent_(indent.value_or(0)),
        separator_(std::move(separator)),
        max_whitespace_cnt_(max_whitespace_cnt) {}

  void StartIndent() {
    total_indent_ += indent_;
    is_first_.push_back(true);
  }

  void EndIndent() {
    total_indent_ -= indent_;
    is_first_.pop_back();
  }

  SeparatorExpr StartSeparator() const {
    if (any_whitespace_) return {true, {}};
    if (!enable_newline_) return {false, {}};
    return {false, "\n" + std::string(total_indent_, ' ')};
  }

  SeparatorExpr MiddleSeparator() const {
    if (any_whitespace_) return {true, separator_};
    if (!enable_newline_) return {false, separator_};
    return {false, separator_ + "\n" + std::string(total_indent_, ' ')};
  }

  SeparatorExpr EndSeparator() const {
    if (any_whitespace_) return {true, {}};
    if (!enable_newline_) return {false, {}};
    return {false, "\n" + std::string(total_indent_ - indent_, ' ')};
  }

  SeparatorExpr EmptySeparator() const {
    if (any_whitespace_) return {true, {}};
    return {false, {}};
  }

  SeparatorExpr NextSeparator(bool is_end = false) {
    XGRAMMAR_DCHECK(!is_first_.empty());
    if (any_whitespace_) {
      if (is_first_.back() || is_end) {
        is_first_.back() = false;
        return {true, {}};
      }
      return {true, separator_};
    }

    std::string result;
    if (!is_first_.back() && !is_end) result += separator_;
    is_first_.back() = false;
    if (enable_newline_) result += "\n";
    result += std::string(is_end ? total_indent_ - indent_ : total_indent_, ' ');
    return {false, std::move(result)};
  }

  std::optional<int> max_whitespace_cnt() const { return max_whitespace_cnt_; }

 private:
  bool any_whitespace_;
  bool enable_newline_;
  int64_t indent_;
  std::string separator_;
  int64_t total_indent_ = 0;
  std::vector<bool> is_first_;
  std::optional<int> max_whitespace_cnt_;
};

struct ASTTrieNode {
  bool is_terminal = false;
  std::map<unsigned char, ASTTrieNode> children;
};

struct ASTEffectiveIntegerRange {
  std::optional<int64_t> start;
  std::optional<int64_t> end;
};

ASTEffectiveIntegerRange GetASTEffectiveIntegerRange(const IntegerSpec& spec) {
  ASTEffectiveIntegerRange result;
  if (spec.minimum) result.start = spec.minimum;
  if (spec.exclusive_minimum) {
    int64_t exclusive_start = *spec.exclusive_minimum + 1;
    result.start = result.start ? std::max(*result.start, exclusive_start) : exclusive_start;
  }
  if (spec.maximum) result.end = spec.maximum;
  if (spec.exclusive_maximum) {
    int64_t exclusive_end = *spec.exclusive_maximum - 1;
    result.end = result.end ? std::min(*result.end, exclusive_end) : exclusive_end;
  }
  return result;
}

std::string JoinASTRegexAlternatives(const std::vector<std::string>& alternatives) {
  std::string result;
  for (size_t i = 0; i < alternatives.size(); ++i) {
    if (i != 0) result += '|';
    result += alternatives[i];
  }
  return result;
}

}  // namespace json_schema_ast_detail

using namespace json_schema_ast_detail;

class JSONSchemaASTConverter {
 public:
  using RefResolver = JSONSchemaConverter::RefResolver;

  JSONSchemaASTConverter(
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt,
      RefResolver ref_resolver,
      bool any_order
  )
      : indent_manager_(
            indent,
            separators ? separators->first
                       : (any_whitespace ? "," : (indent.has_value() ? "," : ", ")),
            any_whitespace,
            max_whitespace_cnt
        ),
        any_whitespace_(any_whitespace),
        max_whitespace_cnt_(max_whitespace_cnt),
        any_order_(any_order),
        ref_resolver_(std::move(ref_resolver)) {
    colon_separator_ = separators ? separators->second : (any_whitespace ? ":" : ": ");
  }

  Grammar Convert(const SchemaSpecPtr& spec) {
    AddBasicRules();
    int32_t root_rule_id = builder_.AddEmptyRule("root");
    uri_to_rule_id_["#"] = root_rule_id;

    auto cached_rule = GetCache(spec->cache_key);
    if (cached_rule) {
      builder_.UpdateRuleBody(
          root_rule_id, builder_.AddExpr(Ref(*cached_rule), root_rule_id, "root")
      );
    } else {
      if (!spec->cache_key.empty()) AddCache(spec->cache_key, root_rule_id);
      builder_.UpdateRuleBody(
          root_rule_id, builder_.AddExpr(GenerateFromSpec(spec, "root"), root_rule_id, "root")
      );
    }
    return GrammarNormalizer::Apply(builder_.Get(root_rule_id));
  }

 private:
  Expr Whitespace() const {
    if (!max_whitespace_cnt_) return CharClassStar(WhitespaceRanges());
    return Repeat(CharClass(WhitespaceRanges()), 0, *max_whitespace_cnt_);
  }

  Expr Separator(const SeparatorExpr& separator) const {
    if (!separator.is_whitespace) return Str(separator.literal);
    if (separator.literal.empty()) return Whitespace();
    return Seq(Whitespace(), Str(separator.literal), Whitespace());
  }

  Expr Colon() const {
    if (!any_whitespace_) return Str(colon_separator_);
    return Seq(Whitespace(), Str(colon_separator_), Whitespace());
  }

  int32_t AddRule(const std::string& name_hint, Expr body) {
    std::string name = builder_.GetNewRuleName(name_hint);
    return builder_.AddRule(name, body);
  }

  int32_t AllocateRule(const std::string& name_hint) {
    return builder_.AddEmptyRule(builder_.GetNewRuleName(name_hint));
  }

  void AddCache(const std::string& key, int32_t rule_id) {
    if (!key.empty()) cache_[key] = rule_id;
  }

  std::optional<int32_t> GetCache(const std::string& key) const {
    if (key.empty()) return std::nullopt;
    auto it = cache_.find(key);
    return it == cache_.end() ? std::nullopt : std::optional<int32_t>(it->second);
  }

  Expr RegexFallback(const std::string& regex) {
    int32_t root = SubGrammarAdder::Apply(&builder_, Grammar::FromRegex(regex));
    return Ref(root);
  }

  Expr JSONStringRegex(const std::string& regex) {
    bool printable_ascii = std::all_of(regex.begin(), regex.end(), [](unsigned char c) {
      return c >= 0x20 && c <= 0x7e;
    });
    if (printable_ascii) {
      auto fsm_result = GrammarFSMBuilder::Regex(regex, /*json_string=*/true);
      if (fsm_result.IsOk()) {
        auto fsm = std::move(fsm_result).Unwrap();
        std::unordered_set<int> reachable_states;
        fsm.GetReachableStates(&reachable_states);
        bool language_is_empty =
            std::none_of(reachable_states.begin(), reachable_states.end(), [&](int state) {
              return fsm.IsEndState(state);
            });
        if (!language_is_empty) return gs::Regex(regex, /*json_string=*/true);
      }
    }
    return RegexFallback(regex);
  }

  int32_t CreateRule(const SchemaSpecPtr& spec, const std::string& name_hint) {
    if (auto cached = GetCache(spec->cache_key)) return *cached;
    int32_t rule_id = AllocateRule(name_hint);
    const std::string rule_name = builder_.GetRule(rule_id).name;
    builder_.UpdateRuleBody(
        rule_id, builder_.AddExpr(GenerateFromSpec(spec, rule_name), rule_id, rule_name)
    );
    return rule_id;
  }

  Expr GenerateFromSpec(const SchemaSpecPtr& spec, const std::string& rule_name) {
    return std::visit(
        [&](const auto& value) -> Expr {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, IntegerSpec>) return GenerateInteger(value, rule_name);
          if constexpr (std::is_same_v<T, NumberSpec>) return GenerateNumber(value, rule_name);
          if constexpr (std::is_same_v<T, StringSpec>) return GenerateString(value, rule_name);
          if constexpr (std::is_same_v<T, BooleanSpec>) return GenerateBoolean(value, rule_name);
          if constexpr (std::is_same_v<T, NullSpec>) return GenerateNull(value, rule_name);
          if constexpr (std::is_same_v<T, ArraySpec>) return GenerateArray(value, rule_name);
          if constexpr (std::is_same_v<T, ObjectSpec>) return GenerateObject(value, rule_name);
          if constexpr (std::is_same_v<T, AnySpec>) return GenerateAny(value, rule_name);
          if constexpr (std::is_same_v<T, ConstSpec>) return GenerateConst(value, rule_name);
          if constexpr (std::is_same_v<T, EnumSpec>) return GenerateEnum(value, rule_name);
          if constexpr (std::is_same_v<T, RefSpec>) return GenerateRef(value, rule_name);
          if constexpr (std::is_same_v<T, AnyOfSpec>) return GenerateAnyOf(value, rule_name);
          if constexpr (std::is_same_v<T, OneOfSpec>) return GenerateOneOf(value, rule_name);
          if constexpr (std::is_same_v<T, AllOfSpec>) return GenerateAllOf(value, rule_name);
          if constexpr (std::is_same_v<T, TypeArraySpec>)
            return GenerateTypeArray(value, rule_name);
          XGRAMMAR_UNREACHABLE();
        },
        spec->spec
    );
  }

  void AddBasicRules() {
    basic_escape_ = AllocateRule(JSONSchemaConverter::kBasicEscape);
    basic_string_sub_ = AllocateRule(JSONSchemaConverter::kBasicStringSub);
    basic_any_ = AllocateRule(JSONSchemaConverter::kBasicAny);
    basic_integer_ = AllocateRule(JSONSchemaConverter::kBasicInteger);
    basic_number_ = AllocateRule(JSONSchemaConverter::kBasicNumber);
    basic_string_ = AllocateRule(JSONSchemaConverter::kBasicString);
    basic_boolean_ = AllocateRule(JSONSchemaConverter::kBasicBoolean);
    basic_null_ = AllocateRule(JSONSchemaConverter::kBasicNull);
    basic_array_ = AllocateRule(JSONSchemaConverter::kBasicArray);
    basic_object_ = AllocateRule(JSONSchemaConverter::kBasicObject);

    Expr escape_body =
        Or(CharClass(
               {{'"', '"'},
                {'\\', '\\'},
                {'/', '/'},
                {'b', 'b'},
                {'f', 'f'},
                {'n', 'n'},
                {'r', 'r'},
                {'t', 't'}}
           ),
           Seq(Str("u"),
               CharClass({{'A', 'F'}, {'a', 'f'}, {'0', '9'}}),
               CharClass({{'A', 'F'}, {'a', 'f'}, {'0', '9'}}),
               CharClass({{'A', 'F'}, {'a', 'f'}, {'0', '9'}}),
               CharClass({{'A', 'F'}, {'a', 'f'}, {'0', '9'}})));
    builder_.UpdateRuleBody(
        basic_escape_,
        builder_.AddExpr(escape_body, basic_escape_, JSONSchemaConverter::kBasicEscape)
    );

    Expr string_sub_body =
        Or(Str("\""),
           Seq(CharClass({{0, 0x1f}, {'"', '"'}, {'\\', '\\'}, {'\r', '\r'}, {'\n', '\n'}}, true),
               Ref(basic_string_sub_)),
           Seq(Str("\\"), Ref(basic_escape_), Ref(basic_string_sub_)));
    builder_.UpdateRuleBody(
        basic_string_sub_,
        builder_.AddExpr(string_sub_body, basic_string_sub_, JSONSchemaConverter::kBasicStringSub)
    );
    builder_.UpdateLookaheadAssertion(
        basic_string_sub_,
        builder_.AddExpr(
            Seq(Whitespace(), CharClass({{',', ','}, {'}', '}'}, {']', ']'}, {':', ':'}})),
            basic_string_sub_,
            JSONSchemaConverter::kBasicStringSub
        )
    );

    auto any_spec = SchemaSpec::Make(AnySpec{}, "{}", JSONSchemaConverter::kBasicAny);

    builder_.UpdateRuleBody(
        basic_any_,
        builder_.AddExpr(
            GenerateAny(AnySpec{}, JSONSchemaConverter::kBasicAny),
            basic_any_,
            JSONSchemaConverter::kBasicAny
        )
    );
    AddCache("{}", basic_any_);

    builder_.UpdateRuleBody(
        basic_integer_,
        builder_.AddExpr(
            GenerateInteger(IntegerSpec{}, JSONSchemaConverter::kBasicInteger),
            basic_integer_,
            JSONSchemaConverter::kBasicInteger
        )
    );
    AddCache("{\"type\":\"integer\"}", basic_integer_);

    builder_.UpdateRuleBody(
        basic_number_,
        builder_.AddExpr(
            GenerateNumber(NumberSpec{}, JSONSchemaConverter::kBasicNumber),
            basic_number_,
            JSONSchemaConverter::kBasicNumber
        )
    );
    AddCache("{\"type\":\"number\"}", basic_number_);

    builder_.UpdateRuleBody(
        basic_string_,
        builder_.AddExpr(
            Seq(Str("\""), Ref(basic_string_sub_)), basic_string_, JSONSchemaConverter::kBasicString
        )
    );
    AddCache("{\"type\":\"string\"}", basic_string_);

    builder_.UpdateRuleBody(
        basic_boolean_,
        builder_.AddExpr(
            GenerateBoolean(BooleanSpec{}, JSONSchemaConverter::kBasicBoolean),
            basic_boolean_,
            JSONSchemaConverter::kBasicBoolean
        )
    );
    AddCache("{\"type\":\"boolean\"}", basic_boolean_);

    builder_.UpdateRuleBody(
        basic_null_,
        builder_.AddExpr(
            GenerateNull(NullSpec{}, JSONSchemaConverter::kBasicNull),
            basic_null_,
            JSONSchemaConverter::kBasicNull
        )
    );
    AddCache("{\"type\":\"null\"}", basic_null_);

    ArraySpec array_spec;
    array_spec.allow_additional_items = true;
    array_spec.additional_items = any_spec;
    builder_.UpdateRuleBody(
        basic_array_,
        builder_.AddExpr(
            GenerateArray(array_spec, JSONSchemaConverter::kBasicArray),
            basic_array_,
            JSONSchemaConverter::kBasicArray
        )
    );
    AddCache("{\"type\":\"array\"}", basic_array_);

    ObjectSpec object_spec;
    object_spec.allow_additional_properties = true;
    object_spec.additional_properties_schema = any_spec;
    builder_.UpdateRuleBody(
        basic_object_,
        builder_.AddExpr(
            GenerateObject(object_spec, JSONSchemaConverter::kBasicObject),
            basic_object_,
            JSONSchemaConverter::kBasicObject
        )
    );
    AddCache("{\"type\":\"object\"}", basic_object_);
  }

  Expr GenerateInteger(const IntegerSpec& spec, const std::string& rule_name) {
    ASTEffectiveIntegerRange range = GetASTEffectiveIntegerRange(spec);
    if (spec.multiple_of) {
      if (range.start && range.end) {
        std::vector<std::string> multiples;
        for (int64_t value = *range.start; value <= *range.end; ++value) {
          if (value % *spec.multiple_of == 0) multiples.push_back(std::to_string(value));
          if (value == std::numeric_limits<int64_t>::max()) break;
        }
        return gs::Regex("^(" + JoinASTRegexAlternatives(multiples) + ")$");
      }
      return GenerateIntegerMultipleOfDFA(*spec.multiple_of, rule_name);
    }
    if (range.start || range.end) return gs::Regex(GenerateRangeRegex(range.start, range.end));
    return Or(
        Str("0"), Seq(Repeat(Str("-"), 0, 1), CharClass({{'1', '9'}}), CharClassStar({{'0', '9'}}))
    );
  }

  Expr GenerateIntegerMultipleOfDFA(int64_t multiple_of, const std::string& rule_name) {
    std::vector<int32_t> state_rules(multiple_of);
    for (int64_t state = 0; state < multiple_of; ++state) {
      state_rules[state] = AllocateRule(
          rule_name + "_multiple_of_" + std::to_string(multiple_of) + "_mod_" +
          std::to_string(state)
      );
    }
    for (int64_t state = 0; state < multiple_of; ++state) {
      std::vector<Expr> transitions;
      if (state == 0) transitions.push_back(Empty());
      for (int64_t digit = 0; digit <= 9; ++digit) {
        transitions.push_back(
            Seq(Str(std::string(1, static_cast<char>('0' + digit))),
                Ref(state_rules[(state * 10 + digit) % multiple_of]))
        );
      }
      const auto& state_rule = builder_.GetRule(state_rules[state]);
      builder_.UpdateRuleBody(
          state_rules[state],
          builder_.AddExpr(Or(std::move(transitions)), state_rules[state], state_rule.name)
      );
    }
    std::vector<Expr> starts;
    for (int64_t digit = 1; digit <= 9; ++digit) {
      starts.push_back(Seq(
          Str(std::string(1, static_cast<char>('0' + digit))), Ref(state_rules[digit % multiple_of])
      ));
    }
    return Or(Str("0"), Seq(Repeat(Str("-"), 0, 1), Or(std::move(starts))));
  }

  Expr GenerateNumber(const NumberSpec& spec, const std::string& rule_name) {
    std::optional<double> start;
    std::optional<double> end;
    bool exclusive_start = false;
    bool exclusive_end = false;
    if (spec.minimum) start = spec.minimum;
    if (spec.exclusive_minimum && (!start || *spec.exclusive_minimum >= *start)) {
      start = spec.exclusive_minimum;
      exclusive_start = true;
    }
    if (spec.maximum) end = spec.maximum;
    if (spec.exclusive_maximum && (!end || *spec.exclusive_maximum <= *end)) {
      end = spec.exclusive_maximum;
      exclusive_end = true;
    }
    if (start || end) {
      // Generated float-range regexes contain constructs that the FSM regex backend does not
      // yet preserve precisely. Convert only this fragment through the mature regex frontend;
      // the surrounding JSON schema grammar is still built directly as an AST.
      return RegexFallback(GenerateFloatRangeRegex(start, end, exclusive_start, exclusive_end));
    }
    return Seq(
        Repeat(Str("-"), 0, 1),
        Or(Str("0"), Seq(CharClass({{'1', '9'}}), CharClassStar({{'0', '9'}}))),
        Repeat(Seq(Str("."), Repeat(CharClass({{'0', '9'}}), 1, -1)), 0, 1),
        Repeat(
            Seq(CharClass({{'e', 'e'}, {'E', 'E'}}),
                Repeat(CharClass({{'+', '+'}, {'-', '-'}}), 0, 1),
                Repeat(CharClass({{'0', '9'}}), 1, -1)),
            0,
            1
        )
    );
  }

  Expr GenerateString(const StringSpec& spec, const std::string& rule_name) {
    if (spec.format) {
      auto regex = JSONSchemaConverter::JSONFormatToRegexPattern(*spec.format);
      if (regex) return Seq(Str("\""), RegexFallback(*regex), Str("\""));
    }
    if (spec.pattern) return Seq(Str("\""), JSONStringRegex(*spec.pattern), Str("\""));
    if (spec.min_length != 0 || spec.max_length != -1) {
      return Seq(
          Str("\""),
          Repeat(
              CharClass({{'"', '"'}, {'\\', '\\'}, {'\r', '\r'}, {'\n', '\n'}}, true),
              spec.min_length,
              spec.max_length
          ),
          Str("\"")
      );
    }
    return Seq(Str("\""), Ref(basic_string_sub_));
  }

  Expr GenerateBoolean(const BooleanSpec&, const std::string&) {
    return Or(Str("true"), Str("false"));
  }

  Expr GenerateNull(const NullSpec&, const std::string&) { return Str("null"); }

  Expr GenerateArray(const ArraySpec& spec, const std::string& rule_name) {
    indent_manager_.StartIndent();
    Expr start = Separator(indent_manager_.StartSeparator());
    Expr middle = Separator(indent_manager_.MiddleSeparator());
    Expr end = Separator(indent_manager_.EndSeparator());
    Expr empty_separator = Separator(indent_manager_.EmptySeparator());

    std::vector<int32_t> item_rules;
    for (size_t i = 0; i < spec.prefix_items.size(); ++i) {
      item_rules.push_back(
          CreateRule(spec.prefix_items[i], rule_name + "_item_" + std::to_string(i))
      );
    }
    int32_t additional_rule = -1;
    if (spec.allow_additional_items && spec.additional_items) {
      additional_rule = CreateRule(spec.additional_items, rule_name + "_additional");
    }
    indent_manager_.EndIndent();

    Expr empty_array = Seq(Str("["), empty_separator, Str("]"));
    if (item_rules.empty()) {
      if (!spec.allow_additional_items || (spec.min_items == 0 && spec.max_items == 0)) {
        return empty_array;
      }
      Expr items =
          Seq(Ref(additional_rule),
              Repeat(
                  Seq(middle, Ref(additional_rule)),
                  spec.min_items == 0 ? 0 : static_cast<int>(spec.min_items - 1),
                  spec.max_items == -1 ? -1 : static_cast<int>(spec.max_items - 1)
              ));
      Expr non_empty = Seq(Str("["), start, items, end, Str("]"));
      return spec.min_items == 0 ? Or(non_empty, empty_array) : non_empty;
    }

    std::vector<Expr> prefix;
    for (size_t i = 0; i < item_rules.size(); ++i) {
      if (i != 0) prefix.push_back(middle);
      prefix.push_back(Ref(item_rules[i]));
    }
    if (!spec.allow_additional_items) {
      return Seq(Str("["), start, Seq(std::move(prefix)), end, Str("]"));
    }
    int64_t min_additional = std::max<int64_t>(0, spec.min_items - item_rules.size());
    int64_t max_additional =
        spec.max_items == -1 ? -1 : spec.max_items - static_cast<int64_t>(item_rules.size());
    return Seq(
        Str("["),
        start,
        Seq(std::move(prefix)),
        Repeat(Seq(middle, Ref(additional_rule)), min_additional, max_additional),
        end,
        Str("]")
    );
  }

  // Object conversion helpers are defined below.
  Expr GenerateObject(
      const ObjectSpec& spec, const std::string& rule_name, bool need_braces = true
  );
  Expr GetPartialRuleForProperties(
      const std::vector<ObjectSpec::Property>& properties,
      const std::unordered_set<std::string>& required,
      const SchemaSpecPtr& additional,
      const std::string& rule_name,
      const std::string& additional_suffix,
      int min_properties,
      int max_properties,
      const std::optional<Expr>& additional_override = std::nullopt
  );
  Expr GetAnyOrderRuleForProperties(
      const std::vector<ObjectSpec::Property>& properties,
      const std::unordered_set<std::string>& required,
      const SchemaSpecPtr& additional,
      const std::string& rule_name,
      const std::string& additional_suffix,
      int min_properties,
      int max_properties,
      const std::optional<Expr>& additional_override
  );
  Expr GetKeyPatternExcluding(
      const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
  );
  Expr BuildTrieBody(const ASTTrieNode& node);

  Expr FormatProperty(const std::string& key, int32_t value_rule) {
    return Seq(Str(picojson::value(key).serialize()), Colon(), Ref(value_rule));
  }

  Expr FormatOtherProperty(Expr key_pattern, int32_t value_rule) {
    return Seq(std::move(key_pattern), Colon(), Ref(value_rule));
  }

  Expr PropertyRepeat(Expr pattern, int min_properties, int max_properties, int already = 0) {
    if (max_properties != -1 && max_properties == already) return Empty();
    int lower = std::max(0, min_properties - already);
    int upper = max_properties == -1 ? -1 : std::max(-1, max_properties - already);
    return Repeat(std::move(pattern), lower, upper);
  }

  Expr GenerateAny(const AnySpec&, const std::string&) {
    return Or(
        Ref(basic_number_),
        Ref(basic_string_),
        Ref(basic_boolean_),
        Ref(basic_null_),
        Ref(basic_array_),
        Ref(basic_object_)
    );
  }

  Expr GenerateConst(const ConstSpec& spec, const std::string&) { return Str(spec.json_value); }

  Expr GenerateEnum(const EnumSpec& spec, const std::string&) {
    std::vector<Expr> choices;
    choices.reserve(spec.json_values.size());
    for (const auto& value : spec.json_values) choices.push_back(Str(value));
    return Or(std::move(choices));
  }

  Expr GenerateRef(const RefSpec& spec, const std::string&) {
    if (auto it = uri_to_rule_id_.find(spec.uri); it != uri_to_rule_id_.end())
      return Ref(it->second);
    XGRAMMAR_CHECK(ref_resolver_) << "Ref resolver not set; cannot resolve $ref: " << spec.uri;

    std::string hint = "ref";
    if (spec.uri.size() >= 2 && spec.uri[0] == '#' && spec.uri[1] == '/') {
      std::stringstream stream(spec.uri.substr(2));
      std::string part;
      std::string derived;
      while (std::getline(stream, part, '/')) {
        if (!part.empty() && !derived.empty()) derived += '_';
        for (char c : part) {
          if (std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.') {
            derived += c;
          }
        }
      }
      if (!derived.empty()) hint = std::move(derived);
    }

    int32_t rule_id = AllocateRule(hint);
    uri_to_rule_id_[spec.uri] = rule_id;
    const std::string rule_name = builder_.GetRule(rule_id).name;
    SchemaSpecPtr resolved = ref_resolver_(spec.uri, rule_name);
    builder_.UpdateRuleBody(
        rule_id, builder_.AddExpr(GenerateFromSpec(resolved, rule_name), rule_id, rule_name)
    );
    if (!resolved->cache_key.empty()) AddCache(resolved->cache_key, rule_id);
    return Ref(rule_id);
  }

  template <typename OptionsSpec>
  Expr GenerateOptions(const OptionsSpec& spec, const std::string& rule_name) {
    std::vector<Expr> choices;
    choices.reserve(spec.options.size());
    for (size_t i = 0; i < spec.options.size(); ++i) {
      choices.push_back(Ref(CreateRule(spec.options[i], rule_name + "_case_" + std::to_string(i))));
    }
    return Or(std::move(choices));
  }

  Expr GenerateAnyOf(const AnyOfSpec& spec, const std::string& rule_name) {
    return GenerateOptions(spec, rule_name);
  }
  Expr GenerateOneOf(const OneOfSpec& spec, const std::string& rule_name) {
    return GenerateOptions(spec, rule_name);
  }
  Expr GenerateAllOf(const AllOfSpec& spec, const std::string& rule_name) {
    if (spec.schemas.size() == 1) return GenerateFromSpec(spec.schemas[0], rule_name + "_case_0");
    XGRAMMAR_LOG(WARNING) << "Support for allOf with multiple options is still ongoing";
    return GenerateAny(AnySpec{}, rule_name);
  }
  Expr GenerateTypeArray(const TypeArraySpec& spec, const std::string& rule_name) {
    std::vector<Expr> choices;
    choices.reserve(spec.type_schemas.size());
    for (size_t i = 0; i < spec.type_schemas.size(); ++i) {
      choices.push_back(
          Ref(CreateRule(spec.type_schemas[i], rule_name + "_type_" + std::to_string(i)))
      );
    }
    return Or(std::move(choices));
  }

  GrammarBuilder builder_;
  ASTIndentManager indent_manager_;
  bool any_whitespace_;
  std::optional<int> max_whitespace_cnt_;
  bool any_order_;
  std::string colon_separator_;
  RefResolver ref_resolver_;
  std::unordered_map<std::string, int32_t> cache_;
  std::unordered_map<std::string, int32_t> uri_to_rule_id_;
  int32_t basic_any_ = -1;
  int32_t basic_integer_ = -1;
  int32_t basic_number_ = -1;
  int32_t basic_string_ = -1;
  int32_t basic_boolean_ = -1;
  int32_t basic_null_ = -1;
  int32_t basic_array_ = -1;
  int32_t basic_object_ = -1;
  int32_t basic_escape_ = -1;
  int32_t basic_string_sub_ = -1;
};

Expr JSONSchemaASTConverter::BuildTrieBody(const ASTTrieNode& node) {
  std::vector<Expr> choices;
  if (!node.is_terminal) choices.push_back(Str("\""));

  std::vector<CharacterClassElement> excluded;
  excluded.reserve(node.children.size() + 5);
  for (const auto& [character, _] : node.children) {
    excluded.push_back({character, character});
  }
  excluded.push_back({0, 0x1f});
  excluded.push_back({'"', '"'});
  excluded.push_back({'\\', '\\'});
  excluded.push_back({'\r', '\r'});
  excluded.push_back({'\n', '\n'});
  choices.push_back(Seq(CharClass(std::move(excluded), true), Ref(basic_string_sub_)));
  choices.push_back(Seq(Str("\\"), Ref(basic_escape_), Ref(basic_string_sub_)));

  for (const auto& [character, child] : node.children) {
    choices.push_back(Seq(Str(std::string(1, static_cast<char>(character))), BuildTrieBody(child)));
  }
  return Or(std::move(choices));
}

Expr JSONSchemaASTConverter::GetKeyPatternExcluding(
    const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
) {
  if (properties.empty()) return Ref(basic_string_);

  ASTTrieNode root;
  for (const auto& property : properties) {
    ASTTrieNode* node = &root;
    for (unsigned char character : property.name) node = &node->children[character];
    node->is_terminal = true;
  }

  int32_t rule_id = AllocateRule(rule_name + "_addl_key");
  const std::string actual_name = builder_.GetRule(rule_id).name;
  builder_.UpdateRuleBody(
      rule_id, builder_.AddExpr(Seq(Str("\""), BuildTrieBody(root)), rule_id, actual_name)
  );
  builder_.UpdateLookaheadAssertion(
      rule_id,
      builder_.AddExpr(
          Seq(Whitespace(), CharClass({{',', ','}, {'}', '}'}, {']', ']'}, {':', ':'}})),
          rule_id,
          actual_name
      )
  );
  return Ref(rule_id);
}

Expr JSONSchemaASTConverter::GetAnyOrderRuleForProperties(
    const std::vector<ObjectSpec::Property>& properties,
    const std::unordered_set<std::string>& required,
    const SchemaSpecPtr& additional,
    const std::string& rule_name,
    const std::string& additional_suffix,
    int min_properties,
    int max_properties,
    const std::optional<Expr>& additional_override
) {
  Expr first_separator = Separator(indent_manager_.NextSeparator());
  Expr middle_separator = Separator(indent_manager_.NextSeparator());
  Expr last_separator = Separator(indent_manager_.NextSeparator(true));

  std::vector<Expr> item_patterns;
  item_patterns.reserve(properties.size() + (additional ? 1 : 0));
  for (size_t i = 0; i < properties.size(); ++i) {
    int32_t value_rule = CreateRule(properties[i].schema, rule_name + "_prop_" + std::to_string(i));
    item_patterns.push_back(FormatProperty(properties[i].name, value_rule));
  }
  if (additional) {
    if (additional_override) {
      item_patterns.push_back(*additional_override);
    } else {
      int32_t value_rule = CreateRule(additional, rule_name + "_" + additional_suffix);
      item_patterns.push_back(
          FormatOtherProperty(GetKeyPatternExcluding(properties, rule_name), value_rule)
      );
    }
  }

  int32_t item_rule = AddRule(rule_name + "_item", Or(std::move(item_patterns)));
  int min_count = std::max(min_properties, static_cast<int>(required.size()));
  Expr content =
      Seq(Ref(item_rule),
          PropertyRepeat(
              Seq(middle_separator, Ref(item_rule)), min_count, max_properties, /*already=*/1
          ));
  return Seq(first_separator, content, last_separator);
}

Expr JSONSchemaASTConverter::GetPartialRuleForProperties(
    const std::vector<ObjectSpec::Property>& properties,
    const std::unordered_set<std::string>& required,
    const SchemaSpecPtr& additional,
    const std::string& rule_name,
    const std::string& additional_suffix,
    int min_properties,
    int max_properties,
    const std::optional<Expr>& additional_override
) {
  if (max_properties == 0) return Empty();
  if (any_order_) {
    return GetAnyOrderRuleForProperties(
        properties,
        required,
        additional,
        rule_name,
        additional_suffix,
        min_properties,
        max_properties,
        additional_override
    );
  }

  Expr first_separator = Separator(indent_manager_.NextSeparator());
  Expr middle_separator = Separator(indent_manager_.NextSeparator());
  Expr last_separator = Separator(indent_manager_.NextSeparator(true));

  std::vector<Expr> property_patterns;
  property_patterns.reserve(properties.size());
  for (size_t i = 0; i < properties.size(); ++i) {
    int32_t value_rule = CreateRule(properties[i].schema, rule_name + "_prop_" + std::to_string(i));
    property_patterns.push_back(FormatProperty(properties[i].name, value_rule));
  }

  const bool allow_additional = additional != nullptr;
  std::optional<Expr> additional_pattern;
  auto ensure_additional_pattern = [&]() -> Expr {
    if (!additional_pattern) {
      if (additional_override) {
        additional_pattern = *additional_override;
      } else {
        int32_t value_rule = CreateRule(additional, rule_name + "_" + additional_suffix);
        additional_pattern =
            FormatOtherProperty(GetKeyPatternExcluding(properties, rule_name), value_rule);
      }
    }
    return *additional_pattern;
  };
  auto rule_or_empty = [](int32_t rule_id) -> Expr {
    return rule_id == -1 ? Empty() : Ref(rule_id);
  };

  if (min_properties == 0 && max_properties == -1) {
    std::vector<int32_t> part_rules(properties.size(), -1);
    std::vector<uint8_t> is_required(properties.size(), false);
    if (allow_additional) {
      Expr additional_property = ensure_additional_pattern();
      part_rules.back() = AddRule(
          rule_name + "_part_" + std::to_string(properties.size() - 1),
          Repeat(Seq(middle_separator, additional_property), 0, -1)
      );
    }

    for (int i = static_cast<int>(properties.size()) - 2; i >= 0; --i) {
      Expr suffix = rule_or_empty(part_rules[i + 1]);
      Expr with_property = Seq(middle_separator, property_patterns[i + 1], suffix);
      Expr body = with_property;
      if (!required.count(properties[i + 1].name)) {
        body = Or(suffix, with_property);
      } else {
        is_required[i + 1] = true;
      }
      part_rules[i] = AddRule(rule_name + "_part_" + std::to_string(i), body);
    }
    if (required.count(properties[0].name)) is_required[0] = true;

    std::vector<Expr> choices;
    for (size_t i = 0; i < properties.size(); ++i) {
      choices.push_back(Seq(property_patterns[i], rule_or_empty(part_rules[i])));
      if (is_required[i]) break;
    }
    if (allow_additional && required.empty()) {
      choices.push_back(Seq(ensure_additional_pattern(), Ref(part_rules.back())));
    }
    return Seq(first_separator, Or(std::move(choices)), last_separator);
  }

  const int property_count = static_cast<int>(properties.size());
  std::vector<int> matched_min(property_count, 0);
  std::vector<uint8_t> is_required(property_count, false);
  bool found_required = required.count(properties[0].name);
  matched_min[0] = 1;
  for (int i = 1; i < property_count; ++i) {
    if (required.count(properties[i].name)) {
      is_required[i] = true;
      matched_min[i] = matched_min[i - 1] + 1;
    } else {
      matched_min[i] = matched_min[i - 1];
    }
    if (!found_required) matched_min[i] = 1;
    if (is_required[i]) found_required = true;
  }
  if (required.count(properties[0].name)) is_required[0] = true;

  if (max_properties == -1) {
    std::vector<std::vector<int32_t>> part_rules(property_count);
    matched_min.back() = allow_additional ? std::max(1, matched_min.back())
                                          : std::max(min_properties, matched_min.back());
    for (int i = property_count - 2; i >= 0; --i) {
      matched_min[i] = std::max(matched_min[i], matched_min[i + 1] - 1);
    }

    if (allow_additional) {
      Expr additional_property = ensure_additional_pattern();
      for (int matched = matched_min.back(); matched <= property_count; ++matched) {
        part_rules.back().push_back(AddRule(
            rule_name + "_part_" + std::to_string(property_count - 1) + "_" +
                std::to_string(matched),
            PropertyRepeat(
                Seq(middle_separator, additional_property), min_properties, max_properties, matched
            )
        ));
      }
    } else {
      part_rules.back().resize(property_count - matched_min.back() + 1, -1);
    }

    for (int i = property_count - 2; i >= 0; --i) {
      for (int matched = matched_min[i]; matched <= i + 1; ++matched) {
        Expr body = Empty();
        if (is_required[i + 1] || matched == matched_min[i + 1] - 1) {
          body =
              Seq(middle_separator,
                  property_patterns[i + 1],
                  rule_or_empty(part_rules[i + 1][matched + 1 - matched_min[i + 1]]));
        } else {
          body =
              Or(rule_or_empty(part_rules[i + 1][matched - matched_min[i + 1]]),
                 Seq(middle_separator,
                     property_patterns[i + 1],
                     rule_or_empty(part_rules[i + 1][matched - matched_min[i + 1] + 1])));
        }
        part_rules[i].push_back(
            AddRule(rule_name + "_part_" + std::to_string(i) + "_" + std::to_string(matched), body)
        );
      }
    }

    std::vector<Expr> choices;
    for (int i = 0; i < property_count && matched_min[i] <= 1; ++i) {
      choices.push_back(Seq(property_patterns[i], rule_or_empty(part_rules[i][1 - matched_min[i]]))
      );
      if (is_required[i]) break;
    }
    if (allow_additional && required.empty()) {
      Expr additional_property = ensure_additional_pattern();
      choices.push_back(
          Seq(additional_property,
              PropertyRepeat(
                  Seq(middle_separator, additional_property), min_properties, max_properties, 1
              ))
      );
    }
    return Seq(first_separator, Or(std::move(choices)), last_separator);
  }

  std::vector<std::vector<int32_t>> part_rules(property_count);
  std::vector<int> matched_max(property_count, property_count);
  matched_max[0] = 1;
  for (int i = 1; i < property_count; ++i) matched_max[i] = matched_max[i - 1] + 1;
  if (allow_additional) {
    matched_min.back() = std::max(1, matched_min.back());
    matched_max.back() = std::min(max_properties, matched_max.back());
  } else {
    matched_min.back() = std::max(min_properties, matched_min.back());
    matched_max.back() = std::min(max_properties, matched_max.back());
  }
  for (int i = property_count - 2; i >= 0; --i) {
    matched_min[i] = std::max(matched_min[i], matched_min[i + 1] - 1);
    matched_max[i] = is_required[i + 1] ? std::min(matched_max[i], matched_max[i + 1] - 1)
                                        : std::min(matched_max[i], matched_max[i + 1]);
  }

  if (allow_additional) {
    Expr additional_property = ensure_additional_pattern();
    for (int matched = matched_min.back(); matched <= matched_max.back(); ++matched) {
      part_rules.back().push_back(AddRule(
          rule_name + "_part_" + std::to_string(property_count - 1) + "_" + std::to_string(matched),
          PropertyRepeat(
              Seq(middle_separator, additional_property), min_properties, max_properties, matched
          )
      ));
    }
  } else {
    part_rules.back().resize(std::max(0, matched_max.back() - matched_min.back() + 1), -1);
  }

  for (int i = property_count - 2; i >= 0; --i) {
    for (int matched = matched_min[i]; matched <= matched_max[i]; ++matched) {
      Expr body = Empty();
      if (matched == matched_max[i + 1]) {
        body = rule_or_empty(part_rules[i + 1][matched - matched_min[i + 1]]);
      } else if (is_required[i + 1] || matched == matched_min[i + 1] - 1) {
        body =
            Seq(middle_separator,
                property_patterns[i + 1],
                rule_or_empty(part_rules[i + 1][matched + 1 - matched_min[i + 1]]));
      } else {
        body =
            Or(rule_or_empty(part_rules[i + 1][matched - matched_min[i + 1]]),
               Seq(middle_separator,
                   property_patterns[i + 1],
                   rule_or_empty(part_rules[i + 1][matched - matched_min[i + 1] + 1])));
      }
      part_rules[i].push_back(
          AddRule(rule_name + "_part_" + std::to_string(i) + "_" + std::to_string(matched), body)
      );
    }
  }

  std::vector<Expr> choices;
  for (int i = 0; i < property_count; ++i) {
    if (matched_max[i] < matched_min[i]) continue;
    if (matched_min[i] > 1) break;
    choices.push_back(Seq(property_patterns[i], rule_or_empty(part_rules[i][1 - matched_min[i]])));
    if (is_required[i]) break;
  }
  if (allow_additional && required.empty()) {
    Expr additional_property = ensure_additional_pattern();
    choices.push_back(
        Seq(additional_property,
            PropertyRepeat(
                Seq(middle_separator, additional_property), min_properties, max_properties, 1
            ))
    );
  }
  return Seq(first_separator, Or(std::move(choices)), last_separator);
}

Expr JSONSchemaASTConverter::GenerateObject(
    const ObjectSpec& spec, const std::string& rule_name, bool need_braces
) {
  std::vector<Expr> result;
  if (need_braces) result.push_back(Str("{"));
  bool has_content = false;
  bool could_be_empty = false;

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
  if (!spec.properties.empty() && (!spec.pattern_properties.empty() || spec.property_names)) {
    SchemaSpecPtr effective_additional = additional_property;
    std::string effective_suffix = additional_suffix;
    std::optional<Expr> override_pattern;

    if (!spec.pattern_properties.empty()) {
      std::vector<Expr> pattern_choices;
      for (size_t i = 0; i < spec.pattern_properties.size(); ++i) {
        const auto& pattern_property = spec.pattern_properties[i];
        int32_t value_rule =
            CreateRule(pattern_property.schema, rule_name + "_pp_" + std::to_string(i));
        pattern_choices.push_back(
            Seq(Str("\""),
                JSONStringRegex(pattern_property.pattern),
                Str("\""),
                Colon(),
                Ref(value_rule))
        );
      }
      if (effective_additional) {
        int32_t value_rule = CreateRule(effective_additional, rule_name + "_" + effective_suffix);
        pattern_choices.push_back(FormatOtherProperty(Ref(basic_string_), value_rule));
      }
      override_pattern = Or(std::move(pattern_choices));
      if (!effective_additional) effective_additional = SchemaSpec::Make(AnySpec{}, "", "any");
      effective_suffix = "pp";
    } else if (spec.property_names && effective_additional) {
      int32_t key_rule = CreateRule(spec.property_names, rule_name + "_name");
      int32_t value_rule = CreateRule(effective_additional, rule_name + "_" + effective_suffix);
      override_pattern = Seq(Ref(key_rule), Colon(), Ref(value_rule));
      effective_suffix = "pn";
    }

    result.push_back(GetPartialRuleForProperties(
        spec.properties,
        spec.required,
        effective_additional,
        rule_name,
        effective_suffix,
        spec.min_properties,
        spec.max_properties,
        override_pattern
    ));
    has_content = spec.max_properties != 0;
    could_be_empty = spec.required.empty() && spec.min_properties == 0;
  } else if (!spec.pattern_properties.empty() || spec.property_names) {
    Expr beginning = Separator(indent_manager_.NextSeparator());
    if (spec.max_properties != 0) {
      std::vector<Expr> property_choices;
      if (!spec.pattern_properties.empty()) {
        for (size_t i = 0; i < spec.pattern_properties.size(); ++i) {
          const auto& pattern_property = spec.pattern_properties[i];
          int32_t value_rule =
              CreateRule(pattern_property.schema, rule_name + "_prop_" + std::to_string(i));
          property_choices.push_back(
              Seq(beginning,
                  Str("\""),
                  JSONStringRegex(pattern_property.pattern),
                  Str("\""),
                  Colon(),
                  Ref(value_rule))
          );
        }
      } else {
        int32_t key_rule = CreateRule(spec.property_names, rule_name + "_name");
        property_choices.push_back(Seq(beginning, Ref(key_rule), Colon(), Ref(basic_any_)));
      }
      int32_t property_rule = AddRule(rule_name + "_prop", Or(std::move(property_choices)));
      result.push_back(Ref(property_rule));
      result.push_back(PropertyRepeat(
          Seq(Separator(indent_manager_.NextSeparator()), Ref(property_rule)),
          spec.min_properties,
          spec.max_properties,
          1
      ));
      result.push_back(Separator(indent_manager_.NextSeparator(true)));
      has_content = true;
      could_be_empty = spec.min_properties == 0;
    } else {
      could_be_empty = true;
    }
  } else if (!spec.properties.empty()) {
    result.push_back(GetPartialRuleForProperties(
        spec.properties,
        spec.required,
        additional_property,
        rule_name,
        additional_suffix,
        spec.min_properties,
        spec.max_properties
    ));
    has_content = spec.max_properties != 0;
    could_be_empty = spec.required.empty() && spec.min_properties == 0;
  } else if (additional_property) {
    if (spec.max_properties != 0) {
      int32_t value_rule = CreateRule(additional_property, rule_name + "_" + additional_suffix);
      Expr property = FormatOtherProperty(Ref(basic_string_), value_rule);
      result.push_back(Separator(indent_manager_.NextSeparator()));
      result.push_back(property);
      result.push_back(PropertyRepeat(
          Seq(Separator(indent_manager_.NextSeparator()), property),
          spec.min_properties,
          spec.max_properties,
          1
      ));
      result.push_back(Separator(indent_manager_.NextSeparator(true)));
      has_content = true;
    }
    could_be_empty = spec.min_properties == 0;
  } else {
    could_be_empty = true;
  }
  indent_manager_.EndIndent();

  if (need_braces) result.push_back(Str("}"));
  Expr non_empty = Seq(std::move(result));
  if (!could_be_empty) return non_empty;

  Expr empty_object = need_braces
                          ? Seq(Str("{"), any_whitespace_ ? Whitespace() : Empty(), Str("}"))
                          : (any_whitespace_ ? Whitespace() : Empty());
  if (!has_content) return empty_object;
  return Or(non_empty, empty_object);
}

Grammar JSONSchemaSpecToGrammar(
    const SchemaSpecPtr& spec,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    std::optional<int> max_whitespace_cnt,
    JSONSchemaConverter::RefResolver ref_resolver,
    bool any_order
) {
  return JSONSchemaASTConverter(
             indent,
             separators,
             any_whitespace,
             max_whitespace_cnt,
             std::move(ref_resolver),
             any_order
  )
      .Convert(spec);
}

}  // namespace xgrammar
