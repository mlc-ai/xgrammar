/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/compiled_grammar.cc
 */

#include <xgrammar/compiler.h>

#include <algorithm>

#include "compiled_grammar_impl.h"
#include "support/json_serializer.h"
#include "testing.h"
#include "tokenizer_info_impl.h"
#include "xgrammar/exception.h"

namespace xgrammar {

/******************* AdaptiveTokenMask *******************/

AdaptiveTokenMask::AdaptiveTokenMask(
    size_t vocab_size,
    const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
    const std::vector<int32_t>& accepted_indices,
    const std::vector<int32_t>& rejected_indices,
    const std::vector<int32_t>& uncertain_indices
) {
  auto size_acc = accepted_indices.size();
  auto size_rej = rejected_indices.size();

  store_type = size_acc >= USE_BITSET_THRESHOLD && size_rej >= USE_BITSET_THRESHOLD
                   ? StoreType::kAcceptedBitset
               : size_acc < size_rej ? StoreType::kAccepted
                                     : StoreType::kRejected;

  if (store_type == StoreType::kAcceptedBitset) {
    accepted_bitset = DynamicBitset(vocab_size);
    for (auto idx : accepted_indices) {
      accepted_bitset.Set(sorted_decoded_vocab[idx].first, true);
    }
  } else if (store_type == StoreType::kAccepted) {
    this->accepted_indices = accepted_indices;
  } else {
    this->rejected_indices = rejected_indices;
  }

  this->uncertain_indices = uncertain_indices;
}

AdaptiveTokenMask::AdaptiveTokenMask(
    size_t vocab_size,
    const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
    const std::vector<int32_t>& accepted_indices,
    const std::vector<int32_t>& uncertain_indices
) {
  auto size_acc = accepted_indices.size();

  store_type = size_acc >= USE_BITSET_THRESHOLD ? StoreType::kAcceptedBitset : StoreType::kAccepted;

  if (store_type == StoreType::kAcceptedBitset) {
    accepted_bitset = DynamicBitset(vocab_size);
    for (auto idx : accepted_indices) {
      accepted_bitset.Set(sorted_decoded_vocab[idx].first, true);
    }
  } else {
    XGRAMMAR_DCHECK(store_type == StoreType::kAccepted);
    this->accepted_indices = accepted_indices;
  }
  this->uncertain_indices = uncertain_indices;
}

std::string AdaptiveTokenMask::Print(const TokenizerInfo& tokenizer_info) const {
  constexpr int kMaxPrintTokens = 100;
  std::stringstream ss;
  const auto& sorted_decoded_vocab = tokenizer_info.GetSortedDecodedVocab();
  std::vector<int32_t> accepted_indices;
  std::vector<int32_t> rejected_indices;
  std::unordered_set<int32_t> uncertain_indices_set(
      uncertain_indices.begin(), uncertain_indices.end()
  );

  accepted_indices.reserve(sorted_decoded_vocab.size());
  rejected_indices.reserve(sorted_decoded_vocab.size());

  if (store_type == StoreType::kAcceptedBitset) {
    for (int i = 0; i < static_cast<int>(sorted_decoded_vocab.size()); ++i) {
      if (uncertain_indices_set.count(i)) {
        continue;
      }
      if (accepted_bitset[sorted_decoded_vocab[i].first]) {
        accepted_indices.push_back(i);
      } else {
        rejected_indices.push_back(i);
      }
    }
  } else if (store_type == StoreType::kAccepted) {
    accepted_indices = this->accepted_indices;
    // Reject indices = [0, sorted_decoded_vocab.size()) \ accepted_indices \ uncertain_indices
    int acc_ptr = 0;
    for (int i = 0; i < static_cast<int>(sorted_decoded_vocab.size()); ++i) {
      while (acc_ptr < static_cast<int>(accepted_indices.size()) && accepted_indices[acc_ptr] < i) {
        ++acc_ptr;
      }
      if (acc_ptr < static_cast<int>(accepted_indices.size()) && accepted_indices[acc_ptr] == i) {
        continue;
      }
      if (uncertain_indices_set.count(i)) {
        continue;
      }
      rejected_indices.push_back(i);
    }
  } else {
    XGRAMMAR_DCHECK(store_type == StoreType::kRejected);
    rejected_indices = this->rejected_indices;
    // Accepted indices = [0, sorted_decoded_vocab.size()) \ rejected_indices \ uncertain_indices
    int rej_ptr = 0;
    for (int i = 0; i < static_cast<int>(sorted_decoded_vocab.size()); ++i) {
      while (rej_ptr < static_cast<int>(rejected_indices.size()) && rejected_indices[rej_ptr] < i) {
        ++rej_ptr;
      }
      if (rej_ptr < static_cast<int>(rejected_indices.size()) && rejected_indices[rej_ptr] == i) {
        continue;
      }
      if (uncertain_indices_set.count(i)) {
        continue;
      }
      accepted_indices.push_back(i);
    }
  }

  std::string storage_type_str = store_type == StoreType::kAcceptedBitset ? "AcceptedBitset"
                                 : store_type == StoreType::kAccepted     ? "Accepted"
                                                                          : "Rejected";

  ss << "AdaptiveTokenMask(num_tokens=" << sorted_decoded_vocab.size()
     << ", accepted_num=" << accepted_indices.size() << ", rejected_num=" << rejected_indices.size()
     << ", uncertain_num=" << uncertain_indices.size() << ", storage_type=" << storage_type_str
     << ",\n";

  // Convert indices to token ids for printing
  std::vector<int32_t> accepted_token_ids;
  std::vector<int32_t> rejected_token_ids;
  std::vector<int32_t> uncertain_token_ids;
  accepted_token_ids.reserve(accepted_indices.size());
  rejected_token_ids.reserve(rejected_indices.size());
  uncertain_token_ids.reserve(uncertain_indices.size());

  for (auto idx : accepted_indices) {
    accepted_token_ids.push_back(sorted_decoded_vocab[idx].first);
  }
  std::sort(accepted_token_ids.begin(), accepted_token_ids.end());
  for (auto idx : rejected_indices) {
    rejected_token_ids.push_back(sorted_decoded_vocab[idx].first);
  }
  std::sort(rejected_token_ids.begin(), rejected_token_ids.end());
  for (auto idx : uncertain_indices) {
    uncertain_token_ids.push_back(sorted_decoded_vocab[idx].first);
  }
  std::sort(uncertain_token_ids.begin(), uncertain_token_ids.end());

  ss << "accepted=" << PrintTokenByIds(accepted_token_ids, tokenizer_info, kMaxPrintTokens)
     << ",\nrejected=" << PrintTokenByIds(rejected_token_ids, tokenizer_info, kMaxPrintTokens)
     << ",\nuncertain=" << PrintTokenByIds(uncertain_token_ids, tokenizer_info, kMaxPrintTokens)
     << "\n)";
  return ss.str();
}

/************** CompiledGrammar::Impl **************/

void CompiledGrammar::Impl::InitializeDynamicTagTokenIndexes() {
  dynamic_tag_token_indexes.reset();
  if (!dynamic_tag_matcher_config.has_value()) {
    return;
  }
  auto indexes = std::make_shared<DynamicTagTokenIndexes>(
      *dynamic_tag_matcher_config, tokenizer_info.GetVocabSize()
  );
  indexes->first_byte_ranges.fill({-1, -1});
  const char prefix_completion_byte = dynamic_tag_matcher_config->element_prefix.back();
  const char suffix_byte = dynamic_tag_matcher_config->tag_suffix.front();
  const auto& sorted_decoded_vocab = tokenizer_info.GetSortedDecodedVocab();
  for (int32_t index = 0; index < static_cast<int32_t>(sorted_decoded_vocab.size()); ++index) {
    const auto& [token_id, token] = sorted_decoded_vocab[index];
    XGRAMMAR_DCHECK(!token.empty());
    if (token.find(prefix_completion_byte) != std::string::npos) {
      indexes->prefix_completion_token_ids.push_back(token_id);
    }
    if (token.find(dynamic_tag_matcher_config->element_prefix) != std::string::npos) {
      indexes->full_prefix_token_ids.push_back(token_id);
    }
    if (token.find(suffix_byte) != std::string::npos) {
      indexes->suffix_token_ids.push_back(token_id);
    }
    const bool contains_prefix_start =
        token.find(dynamic_tag_matcher_config->element_prefix.front()) != std::string::npos;
    if (contains_prefix_start) {
      indexes->content_boundary_token_ids.push_back(token_id);
    }
    if (contains_prefix_start || std::all_of(token.begin(), token.end(), [](char byte) {
          const uint8_t value = static_cast<uint8_t>(byte);
          return value == ' ' || value == '\t' || value == '\n' || value == '\r' || value == '\f' ||
                 value == '\v';
        })) {
      indexes->content_boundary_candidate_bitset.Set(token_id);
    }
    auto& range = indexes->first_byte_ranges[static_cast<uint8_t>(token.front())];
    if (range.first == -1) {
      range.first = index;
    }
    range.second = index + 1;
  }
  dynamic_tag_token_indexes = std::move(indexes);
}

picojson::value SerializeJSONValue(const CompiledGrammar::Impl& impl) {
  auto result = picojson::object{};
  result["grammar"] = AutoSerializeJSONValue(impl.grammar);
  result["tokenizer_metadata"] = impl.tokenizer_info->DumpMetadataValue();
  result["adaptive_token_mask_cache"] = AutoSerializeJSONValue(impl.adaptive_token_mask_cache);
  if (impl.dynamic_tag_matcher_config.has_value()) {
    result["dynamic_tag_matcher_config"] = AutoSerializeJSONValue(*impl.dynamic_tag_matcher_config);
  }
  return picojson::value(result);
}

std::optional<SerializationError> DeserializeJSONValue(
    CompiledGrammar::Impl* impl,
    const picojson::value& json_value,
    const TokenizerInfo& tokenizer_info
) {
  const auto& type_name = "CompiledGrammar";
  if (!json_value.is<picojson::object>()) {
    return ConstructDeserializeError("Expect an object", type_name);
  }
  const auto& object = json_value.get<picojson::object>();
  if (object.find("grammar") == object.end()) {
    return ConstructDeserializeError("Expect a 'grammar' field", type_name);
  }
  AutoDeserializeJSONValue(&(impl->grammar), object["grammar"], type_name);
  if (object.find("tokenizer_metadata") == object.end()) {
    return ConstructDeserializeError("Expect a 'tokenizer_metadata' field", type_name);
  }
  const auto& tokenizer_metadata = object["tokenizer_metadata"];
  if (auto error = tokenizer_info->CheckMetadataMatch(tokenizer_metadata)) {
    return ConstructDeserializeError(
        std::string("Tokenizer metadata mismatch: ") + error->what(), type_name
    );
  }
  impl->tokenizer_info = tokenizer_info;
  if (object.find("adaptive_token_mask_cache") == object.end()) {
    return ConstructDeserializeError("Expect a 'adaptive_token_mask_cache' field", type_name);
  }
  AutoDeserializeJSONValue(&(impl->adaptive_token_mask_cache), object["adaptive_token_mask_cache"]);
  if (object.find("dynamic_tag_matcher_config") != object.end()) {
    if (auto error = AutoDeserializeJSONValue(
            &(impl->dynamic_tag_matcher_config.emplace()),
            object["dynamic_tag_matcher_config"],
            type_name
        )) {
      return error;
    }
    if (auto error = ValidateDynamicTagMatcherConfig(*impl->dynamic_tag_matcher_config)) {
      return ConstructDeserializeError(
          *error, std::string(type_name) + ".dynamic_tag_matcher_config"
      );
    }
  }
  const auto& grammar_config = impl->grammar->GetDynamicTagMatcherConfig();
  if (impl->dynamic_tag_matcher_config.has_value() && grammar_config.has_value() &&
      !(*impl->dynamic_tag_matcher_config == *grammar_config)) {
    return ConstructDeserializeError(
        "The grammar and compiled grammar contain different dynamic-tag matcher configs", type_name
    );
  }
  if (!impl->dynamic_tag_matcher_config.has_value() && grammar_config.has_value()) {
    impl->dynamic_tag_matcher_config = *grammar_config;
  }
  impl->grammar->dynamic_tag_matcher_config = impl->dynamic_tag_matcher_config;
  impl->InitializeDynamicTagTokenIndexes();
  return std::nullopt;
}

/************** CompiledGrammar **************/

std::size_t MemorySize(const CompiledGrammar::Impl& impl) {
  std::size_t dynamic_tag_index_size = 0;
  if (impl.dynamic_tag_token_indexes) {
    dynamic_tag_index_size =
        sizeof(CompiledGrammar::Impl::DynamicTagTokenIndexes) +
        impl.dynamic_tag_token_indexes->initial_matcher.SharedDefinitionMemorySize() +
        MemorySize(impl.dynamic_tag_token_indexes->prefix_completion_token_ids) +
        MemorySize(impl.dynamic_tag_token_indexes->full_prefix_token_ids) +
        MemorySize(impl.dynamic_tag_token_indexes->suffix_token_ids) +
        MemorySize(impl.dynamic_tag_token_indexes->content_boundary_token_ids) +
        MemorySize(impl.dynamic_tag_token_indexes->content_boundary_candidate_bitset);
  }
  return MemorySize(impl.grammar) + MemorySize(impl.adaptive_token_mask_cache) +
         dynamic_tag_index_size +
         (impl.dynamic_tag_matcher_config.has_value() ? MemorySize(*impl.dynamic_tag_matcher_config)
                                                      : 0);
}

std::size_t CompiledGrammar::MemorySizeBytes() const { return MemorySize(*pimpl_); }

Grammar CompiledGrammar::GetGrammar() const { return pimpl_->GetGrammar(); }

TokenizerInfo CompiledGrammar::GetTokenizerInfo() const { return pimpl_->GetTokenizerInfo(); }

/*! \brief Return the serialized JSON string of the compiled grammar. */
std::string CompiledGrammar::SerializeJSON() const { return AutoSerializeJSON(*this, true); }

/*! \brief Deserialize a compiled grammar from a JSON string and tokenizer info. */
std::variant<CompiledGrammar, SerializationError> CompiledGrammar::DeserializeJSON(
    const std::string& json_string, const TokenizerInfo& tokenizer_info
) {
  picojson::value json_value;
  if (auto error = picojson::parse(json_value, json_string); !error.empty()) {
    return InvalidJSONError("Failed to parse JSON: " + error);
  }
  if (!json_value.is<picojson::object>()) {
    return DeserializeFormatError("Expect an object");
  }
  const auto& object = json_value.get<picojson::object>();
  if (auto error = SerializeVersion::Check(object)) {
    return error.value();
  }
  auto impl = std::make_shared<CompiledGrammar::Impl>();
  if (auto error = DeserializeJSONValue(impl.get(), json_value, tokenizer_info)) {
    return error.value();
  }
  return CompiledGrammar(std::move(impl));
}

}  // namespace xgrammar
