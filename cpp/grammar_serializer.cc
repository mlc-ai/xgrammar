/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_serializer.cc
 */

#include "grammar_serializer.h"

#include <picojson.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "compiled_grammar_data_structure.h"
#include "persistent_stack.h"
#include "support/dynamic_bitset.h"
#include "support/encoding.h"
#include "support/logging.h"
#include "support/utils.h"
#include "tokenizer_internal.h"
#include "xgrammar/compiler.h"
#include "xgrammar/grammar.h"
#include "xgrammar/tokenizer_info.h"

namespace xgrammar {

std::string GrammarPrinter::PrintRule(const Rule& rule) {
  std::string res = rule.name + " ::= " + PrintRuleExpr(rule.body_expr_id);
  if (rule.lookahead_assertion_id != -1) {
    res += " (=" + PrintRuleExpr(rule.lookahead_assertion_id) + ")";
  }
  return res;
}

std::string GrammarPrinter::PrintRule(int32_t rule_id) {
  return PrintRule(grammar_->GetRule(rule_id));
}

std::string GrammarPrinter::PrintRuleExpr(const RuleExpr& rule_expr) {
  std::string result;
  switch (rule_expr.type) {
    case RuleExprType::kByteString:
      return PrintByteString(rule_expr);
    case RuleExprType::kCharacterClass:
      return PrintCharacterClass(rule_expr);
    case RuleExprType::kCharacterClassStar:
      return PrintCharacterClassStar(rule_expr);
    case RuleExprType::kEmptyStr:
      return PrintEmptyStr(rule_expr);
    case RuleExprType::kRuleRef:
      return PrintRuleRef(rule_expr);
    case RuleExprType::kSequence:
      return PrintSequence(rule_expr);
    case RuleExprType::kChoices:
      return PrintChoices(rule_expr);
    case RuleExprType::kTagDispatch:
      return PrintTagDispatch(rule_expr);
    default:
      XGRAMMAR_LOG(FATAL) << "Unexpected RuleExpr type: " << static_cast<int>(rule_expr.type);
  }
}

std::string GrammarPrinter::PrintRuleExpr(int32_t rule_expr_id) {
  return PrintRuleExpr(grammar_->GetRuleExpr(rule_expr_id));
}

std::string GrammarPrinter::PrintByteString(const RuleExpr& rule_expr) {
  std::string internal_str;
  internal_str.reserve(rule_expr.data_len);
  for (int i = 0; i < rule_expr.data_len; ++i) {
    internal_str += static_cast<char>(rule_expr[i]);
  }
  auto codepoints = ParseUTF8(internal_str.c_str(), true);
  std::string result;
  for (auto codepoint : codepoints) {
    result += PrintAsEscapedUTF8(codepoint);
  }
  return "\"" + result + "\"";
}

std::string GrammarPrinter::PrintCharacterClass(const RuleExpr& rule_expr) {
  static const std::unordered_map<TCodepoint, std::string> kCustomEscapeMap = {
      {'-', "\\-"}, {']', "\\]"}
  };
  std::string result = "[";
  bool is_negative = static_cast<bool>(rule_expr[0]);
  if (is_negative) {
    result += "^";
  }
  for (auto i = 1; i < rule_expr.data_len; i += 2) {
    result += PrintAsEscapedUTF8(rule_expr[i], kCustomEscapeMap);
    if (rule_expr[i] == rule_expr[i + 1]) {
      continue;
    }
    result += "-";
    result += PrintAsEscapedUTF8(rule_expr[i + 1], kCustomEscapeMap);
  }
  result += "]";
  return result;
}

std::string GrammarPrinter::PrintCharacterClassStar(const RuleExpr& rule_expr) {
  return PrintCharacterClass(rule_expr) + "*";
}

std::string GrammarPrinter::PrintEmptyStr(const RuleExpr& rule_expr) { return "\"\""; }

std::string GrammarPrinter::PrintRuleRef(const RuleExpr& rule_expr) {
  return grammar_->GetRule(rule_expr[0]).name;
}

std::string GrammarPrinter::PrintSequence(const RuleExpr& rule_expr) {
  std::string result;
  result += "(";
  for (int i = 0; i < rule_expr.data_len; ++i) {
    result += PrintRuleExpr(rule_expr[i]);
    if (i + 1 != rule_expr.data_len) {
      result += " ";
    }
  }
  result += ")";
  return result;
}

std::string GrammarPrinter::PrintChoices(const RuleExpr& rule_expr) {
  std::string result;

  result += "(";
  for (int i = 0; i < rule_expr.data_len; ++i) {
    result += PrintRuleExpr(rule_expr[i]);
    if (i + 1 != rule_expr.data_len) {
      result += " | ";
    }
  }
  result += ")";
  return result;
}

std::string GrammarPrinter::PrintTagDispatch(const RuleExpr& rule_expr) {
  std::string result = "TagDispatch(";
  for (int i = 0; i < rule_expr.data_len; i += 2) {
    result +=
        "(" + PrintRuleExpr(rule_expr[i]) + ", " + grammar_->GetRule(rule_expr[i + 1]).name + ")";
    if (i + 2 != rule_expr.data_len) {
      result += ", ";
    }
  }
  result += ")";
  return result;
}

std::string GrammarPrinter::ToString() {
  std::string result;
  int num_rules = grammar_->NumRules();
  for (auto i = 0; i < num_rules; ++i) {
    result += PrintRule(grammar_->GetRule(i)) + "\n";
  }
  return result;
}

picojson::value Grammar::Impl::SerializeToJSON() const {
  picojson::object grammar_json_obj;

  picojson::array rules_json;
  for (const auto& rule : rules_) {
    picojson::object rule_json;
    rule_json["name"] = picojson::value(rule.name);
    rule_json["body_expr_id"] = picojson::value(static_cast<int64_t>(rule.body_expr_id));
    rules_json.emplace_back(std::move(rule_json));
  }
  grammar_json_obj["rules"] = picojson::value(std::move(rules_json));

  picojson::array rule_expr_data_json;
  for (const auto& data : rule_expr_data_) {
    rule_expr_data_json.emplace_back(static_cast<int64_t>(data));
  }
  grammar_json_obj["rule_expr_data"] = picojson::value(std::move(rule_expr_data_json));

  picojson::array rule_expr_indptr_json;
  for (const auto& index_ptr : rule_expr_indptr_) {
    rule_expr_indptr_json.emplace_back(static_cast<int64_t>(index_ptr));
  }
  grammar_json_obj["rule_expr_indptr"] = picojson::value(std::move(rule_expr_indptr_json));

  AddSerializeVersion(grammar_json_obj);
  return picojson::value(std::move(grammar_json_obj));
}

Grammar Grammar::Impl::DeserializeFromJSON(const picojson::value& serialized_value) {
  auto node = std::make_shared<Grammar::Impl>();

  auto checker = [&](bool condition) {
    XGRAMMAR_CHECK(condition) << "Failed to deserialize XGrammar object: "
                              << serialized_value.serialize();
  };

  auto get_key = [&](const picojson::object& obj, const std::string& key) -> const auto& {
    auto iter = obj.find(key);
    checker(iter != obj.end());
    return iter->second;
  };

  auto as_type = [&](const picojson::value& val, auto type_obj) -> const auto& {
    using type = decltype(type_obj);
    checker(val.is<type>());
    return val.get<type>();
  };

  const auto& serialized_obj = as_type(serialized_value, picojson::object{});
  CheckSerializeVersion(serialized_obj);

  // rules
  const auto& rules_array = as_type(get_key(serialized_obj, "rules"), picojson::array{});
  checker(rules_array.size() > 0);
  for (const auto& rule_value : rules_array) {
    const auto& rule_obj = as_type(rule_value, picojson::object{});
    const auto& name = as_type(get_key(rule_obj, "name"), std::string{});
    const auto& rule_expr = as_type(get_key(rule_obj, "body_expr_id"), int64_t{});
    node->rules_.push_back(Grammar::Impl::Rule({name, static_cast<int32_t>(rule_expr)}));
  }

  // rule_expr_data
  const auto& rule_expr_data_array =
      as_type(get_key(serialized_obj, "rule_expr_data"), picojson::array{});
  for (const auto& data_json : rule_expr_data_array) {
    node->rule_expr_data_.push_back(static_cast<int32_t>(data_json.get<int64_t>()));
  }

  // rule_expr_indptr
  const auto& rule_expr_indptr_array =
      as_type(get_key(serialized_obj, "rule_expr_indptr"), picojson::array{});
  for (const auto& index_ptr_json : rule_expr_indptr_array) {
    node->rule_expr_indptr_.push_back(static_cast<int32_t>(index_ptr_json.get<int64_t>()));
  }

  return Grammar(node);
}

enum class SerializeMaskStoreType {
  Unknown = 0,  // to prevent uninitialized value
  AcceptedIndices = 1,
  RejectedIndices = 2,
  AcceptedBitsetIndices = 3,
  RejectedBitsetIndices = 4,
};

static constexpr std::size_t ENTRY_DATA_OFFSET = 10;

picojson::value CompiledGrammar::Impl::SerializeToJSON() const {
  static constexpr auto serialized_entry =
      [](const std::pair<const StackElement, AdaptiveTokenMask>& entry) -> picojson::value {
    picojson::array entry_json;
    entry_json.reserve(ENTRY_DATA_OFFSET);

    const auto& [elem, mask] = entry;
    // serialize stack element, except for ref count field
    entry_json.emplace_back(static_cast<int64_t>(elem.rule_id));
    entry_json.emplace_back(static_cast<int64_t>(elem.sequence_id));
    entry_json.emplace_back(static_cast<int64_t>(elem.element_id));
    entry_json.emplace_back(static_cast<int64_t>(elem.left_utf8_bytes));
    entry_json.emplace_back(static_cast<int64_t>(elem.element_in_string));
    entry_json.emplace_back(static_cast<int64_t>(elem.parent_id));

    // serialize adaptive token mask
    std::vector<int32_t> indices;
    SerializeMaskStoreType store_type;
    const std::vector<int32_t>* indice_ptr = nullptr;
    const auto mask_bitset_size = mask.accepted_bitset.Size();
    switch (mask.store_type) {
      case AdaptiveTokenMask::StoreType::kAccepted:
        indice_ptr = &mask.accepted_indices;
        store_type = SerializeMaskStoreType::AcceptedIndices;
        break;
      case AdaptiveTokenMask::StoreType::kRejected:
        indice_ptr = &mask.rejected_indices;
        store_type = SerializeMaskStoreType::RejectedIndices;
        break;
      case AdaptiveTokenMask::StoreType::kAcceptedBitset:
        if (const auto count_one = mask.accepted_bitset.Count(),
            count_zero = mask_bitset_size - count_one;
            count_one < count_zero) {
          store_type = SerializeMaskStoreType::AcceptedBitsetIndices;
          indices = mask.accepted_bitset.ToIndices(1, count_one);
        } else {
          store_type = SerializeMaskStoreType::RejectedBitsetIndices;
          indices = mask.accepted_bitset.ToIndices(0, count_zero);
        }
        indice_ptr = &indices;
        break;
      default:
        XGRAMMAR_UNREACHABLE();
    }
    entry_json.emplace_back(static_cast<int64_t>(store_type));
    entry_json.emplace_back(static_cast<int64_t>(indice_ptr->size()));
    entry_json.emplace_back(static_cast<int64_t>(mask_bitset_size));
    entry_json.emplace_back(static_cast<int64_t>(mask.uncertain_indices.size()));

    XGRAMMAR_CHECK(indice_ptr != nullptr && entry_json.size() == ENTRY_DATA_OFFSET);
    entry_json.reserve(ENTRY_DATA_OFFSET + indice_ptr->size() + mask.uncertain_indices.size());

    for (const auto& index_value : *indice_ptr)
      entry_json.emplace_back(static_cast<int64_t>(index_value));
    for (const auto& uncertain_index : mask.uncertain_indices)
      entry_json.emplace_back(static_cast<int64_t>(uncertain_index));

    return picojson::value(std::move(entry_json));
  };

  picojson::object compiled_grammar_json_obj;

  compiled_grammar_json_obj["grammar"] = grammar->SerializeToJSON();
  compiled_grammar_json_obj["tokenizer_info"] = tokenizer_info->SerializeToJSON();

  picojson::array adaptive_token_mask_json;
  for (const auto& entry : adaptive_token_mask_cache)
    adaptive_token_mask_json.push_back(serialized_entry(entry));

  compiled_grammar_json_obj["adaptive_token_mask_cache"] =
      picojson::value(std::move(adaptive_token_mask_json));

  AddSerializeVersion(compiled_grammar_json_obj);
  return picojson::value(std::move(compiled_grammar_json_obj));
}

static CompiledGrammar DeserializeFromJSONImpl(
    const picojson::value& value,
    const std::vector<std::string>& encoded_vocab,
    const TokenizerInfo* tokenizer_info_ptr  // nullptr if not provided
) {
  static constexpr auto deserialized_entry = [](auto& map,
                                                const picojson::value& entry_value,
                                                const TokenizerInfo& tokenizer_info) {
    const auto& entry_array = entry_value.get<picojson::array>();
    XGRAMMAR_CHECK(entry_array.size() >= ENTRY_DATA_OFFSET)
        << "Invalid AdaptiveTokenMask entry: " << entry_value.serialize();

    StackElement elem;
    elem.rule_id = static_cast<int32_t>(entry_array[0].get<int64_t>());
    elem.sequence_id = static_cast<int32_t>(entry_array[1].get<int64_t>());
    elem.element_id = static_cast<int32_t>(entry_array[2].get<int64_t>());
    elem.left_utf8_bytes = static_cast<int32_t>(entry_array[3].get<int64_t>());
    elem.element_in_string = static_cast<int32_t>(entry_array[4].get<int64_t>());
    elem.parent_id = static_cast<int32_t>(entry_array[5].get<int64_t>());

    const auto store_type = static_cast<SerializeMaskStoreType>(entry_array[6].get<int64_t>());
    const auto storage_indices_size = static_cast<std::size_t>(entry_array[7].get<int64_t>());
    const auto mask_bitset_size = static_cast<std::size_t>(entry_array[8].get<int64_t>());
    const auto uncertain_indices_size = static_cast<std::size_t>(entry_array[9].get<int64_t>());

    XGRAMMAR_CHECK(
        entry_array.size() == ENTRY_DATA_OFFSET + storage_indices_size + uncertain_indices_size
    ) << "Invalid AdaptiveTokenMask entry: "
      << entry_value.serialize();

    std::vector<int32_t> storage_indices;
    std::vector<int32_t> uncertain_indices;

    storage_indices.reserve(storage_indices_size);
    const auto start_s = ENTRY_DATA_OFFSET;
    for (std::size_t i = 0; i < storage_indices_size; ++i)
      storage_indices.push_back(static_cast<int32_t>(entry_array[start_s + i].get<int64_t>()));

    uncertain_indices.reserve(uncertain_indices_size);
    const auto start_u = start_s + storage_indices_size;
    for (std::size_t i = 0; i < uncertain_indices_size; ++i)
      uncertain_indices.push_back(static_cast<int32_t>(entry_array[start_u + i].get<int64_t>()));

    auto [iter, success] = map.try_emplace(elem);
    XGRAMMAR_CHECK(success) << "Duplicated StackElement in AdaptiveTokenMask entry: "
                            << entry_value.serialize();

    auto& mask = iter->second;
    mask.uncertain_indices = std::move(uncertain_indices);

    // extract indices data
    switch (store_type) {
      case SerializeMaskStoreType::AcceptedIndices:
        mask.store_type = AdaptiveTokenMask::StoreType::kAccepted;
        mask.accepted_indices = std::move(storage_indices);
        break;
      case SerializeMaskStoreType::RejectedIndices:
        mask.store_type = AdaptiveTokenMask::StoreType::kRejected;
        mask.rejected_indices = std::move(storage_indices);
        break;
      case SerializeMaskStoreType::AcceptedBitsetIndices:
        mask.store_type = AdaptiveTokenMask::StoreType::kAcceptedBitset;
        mask.accepted_bitset.FromIndices(storage_indices, mask_bitset_size, 1);
        break;
      case SerializeMaskStoreType::RejectedBitsetIndices:
        mask.store_type = AdaptiveTokenMask::StoreType::kAcceptedBitset;
        mask.accepted_bitset.FromIndices(storage_indices, mask_bitset_size, 0);
        break;
      default:
        XGRAMMAR_LOG(FATAL) << "Unexpected AdaptiveTokenMask::StoreType in deserialization: "
                            << static_cast<int>(store_type);
    }
  };

  auto node = std::make_shared<CompiledGrammar::Impl>();

  auto checker = [&](bool condition) {
    XGRAMMAR_CHECK(condition) << "Failed to deserialize CompiledGrammar object: "
                              << value.serialize();
  };

  auto get_key = [&](const picojson::object& obj, const std::string& key) -> const auto& {
    auto iter = obj.find(key);
    checker(iter != obj.end());
    return iter->second;
  };

  auto as_type = [&](const picojson::value& val, auto type_obj) -> const auto& {
    using type = decltype(type_obj);
    checker(val.is<type>());
    return val.get<type>();
  };

  const auto& serialized_obj = as_type(value, picojson::object{});
  CheckSerializeVersion(serialized_obj);

  const auto& serialized_grammar = get_key(serialized_obj, "grammar");
  node->grammar = Grammar::Impl::DeserializeFromJSON(serialized_grammar);

  const auto& serialized_tokenizer_info = get_key(serialized_obj, "tokenizer_info");
  if (tokenizer_info_ptr == nullptr) {
    node->tokenizer_info =
        TokenizerInfo::Impl::DeserializeFromJSON(serialized_tokenizer_info, encoded_vocab);
  } else {
    // verify the tokenizer information
    const auto& tokenizer_info = *tokenizer_info_ptr;
    checker(tokenizer_info->SerializeToJSON() == serialized_tokenizer_info);
    node->tokenizer_info = tokenizer_info;
  }

  const auto& serialized_adaptive_token_mask_cache_array =
      as_type(get_key(serialized_obj, "adaptive_token_mask_cache"), picojson::array{});

  for (const auto& entry_value : serialized_adaptive_token_mask_cache_array) {
    deserialized_entry(node->adaptive_token_mask_cache, entry_value, node->tokenizer_info);
  }

  return CompiledGrammar(node);
}

CompiledGrammar CompiledGrammar::Impl::DeserializeFromJSON(
    const picojson::value& value, const std::vector<std::string>& encoded_vocab
) {
  return DeserializeFromJSONImpl(value, encoded_vocab, nullptr);
}

CompiledGrammar CompiledGrammar::Impl::DeserializeFromJSON(
    const picojson::value& value, const TokenizerInfo& tokenizer_info
) {
  return DeserializeFromJSONImpl(value, {}, &tokenizer_info);
}

}  // namespace xgrammar
