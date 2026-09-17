/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/compiled_grammar.cc
 */

#include <xgrammar/compiler.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "compiled_grammar_impl.h"
#include "support/encoding.h"
#include "support/int_set.h"
#include "support/json_parse.h"
#include "support/json_serializer.h"
#include "testing.h"
#include "tokenizer_info_impl.h"
#include "xgrammar/exception.h"

#include <map>

namespace xgrammar {

namespace {

/*!
 * \brief Follow single-element choice/sequence wrappers down to the element that is actually
 * matched; the optimizer may or may not leave those wrappers in place.
 */
Grammar::Impl::GrammarExpr UnwrapSingleElement(
    const Grammar& grammar, Grammar::Impl::GrammarExpr expr
) {
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  for (int depth = 0; depth < 4; ++depth) {
    const bool wrapper =
        expr.type == GrammarExprType::kChoices || expr.type == GrammarExprType::kSequence;
    if (!wrapper || expr.size() != 1) {
      break;
    }
    expr = grammar->GetGrammarExpr(expr[0]);
  }
  return expr;
}

/*!
 * \brief The character class matched by a rule body that is exactly one character class, which is
 * the shape of a repetition body that consumes one codepoint per repetition.
 * \return False for any other body, and for FSM states that do not start a fresh codepoint:
 * continuation states of a multi-byte class expect trailing bytes, for which the per-token
 * codepoint count used by the fast path does not apply.
 */
bool GetBodyCharacterClass(
    const Grammar& grammar,
    int32_t rule_id,
    int32_t element_id,
    Grammar::Impl::GrammarExpr* class_expr
) {
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  if (rule_id < 0) {
    return false;
  }
  const auto& rule_fsm = grammar->per_rule_fsms[rule_id];
  if (!rule_fsm.has_value() || rule_fsm->GetFsm().GetStart() != element_id) {
    return false;
  }
  const auto body = UnwrapSingleElement(
      grammar, grammar->GetGrammarExpr(grammar->GetRule(rule_id).body_expr_id)
  );
  if (body.type != GrammarExprType::kCharacterClass) {
    return false;
  }
  *class_expr = body;
  return true;
}

/*! \brief Whether `codepoint` is accepted by the character class expression. */
bool CharacterClassAccepts(const Grammar::Impl::GrammarExpr& class_expr, int32_t codepoint) {
  bool in_ranges = false;
  for (int i = 1; i + 1 < class_expr.size(); i += 2) {
    if (codepoint >= class_expr[i] && codepoint <= class_expr[i + 1]) {
      in_ranges = true;
      break;
    }
  }
  return class_expr[0] != 0 ? !in_ranges : in_ranges;
}

}  // namespace

void PopulateRepeatInteriorBitsets(
    const Grammar& grammar,
    const TokenizerInfo& tokenizer_info,
    std::unordered_map<ParserState, AdaptiveTokenMask, StateHashForCache, StateEqualForCache>*
        cache
) {
  const auto& sorted_vocab = tokenizer_info.GetSortedDecodedVocab();
  for (auto& [state, mask] : *cache) {
    mask.repeat_interior_char_counts.clear();
    mask.repeat_interior_bitsets.clear();
    Grammar::Impl::GrammarExpr class_expr;
    if (!GetBodyCharacterClass(grammar, state.rule_id, state.element_id, &class_expr)) {
      continue;
    }
    // Bucket the interior tokens by codepoint count, keeping the sorted-vocabulary order.
    std::map<int32_t, std::vector<int32_t>> interior_by_count;
    for (int32_t index = 0; index < static_cast<int32_t>(sorted_vocab.size()); ++index) {
      const auto& token = sorted_vocab[index].second;
      if (token.empty()) {
        continue;
      }
      int32_t num_chars = 0;
      bool inside = true;
      for (size_t offset = 0; offset < token.size();) {
        auto [codepoint, num_bytes] = ParseNextUTF8(token.data() + offset);
        if (codepoint == CharHandlingError::kInvalidUTF8 || num_bytes <= 0 ||
            !CharacterClassAccepts(class_expr, codepoint)) {
          inside = false;
          break;
        }
        offset += static_cast<size_t>(num_bytes);
        ++num_chars;
      }
      if (inside && num_chars > 0) {
        interior_by_count[num_chars].push_back(index);
      }
    }
    if (interior_by_count.empty()) {
      continue;
    }

    std::vector<int32_t> interior_indices;
    for (auto& [char_count, indices] : interior_by_count) {
      DynamicBitset cumulative =
          mask.repeat_interior_bitsets.empty() ? DynamicBitset(tokenizer_info.GetVocabSize())
                                               : mask.repeat_interior_bitsets.back();
      for (int32_t index : indices) {
        cumulative.Set(sorted_vocab[index].first, true);
        interior_indices.push_back(index);
      }
      mask.repeat_interior_char_counts.push_back(char_count);
      mask.repeat_interior_bitsets.push_back(std::move(cumulative));
    }
    std::sort(interior_indices.begin(), interior_indices.end());

    // Hand these tokens over to the fast path: they leave every static class, and the matcher
    // accepts the ones that fit the remaining repetition budget and rejects the rest with a bitset.
    // Nothing is added to `rejected_indices`, which would otherwise carry the whole in-class
    // vocabulary and be merged and intersected on every fill.
    IntsetDifference(&mask.uncertain_indices, interior_indices);
    if (mask.store_type == AdaptiveTokenMask::StoreType::kAcceptedBitset) {
      for (int32_t index : interior_indices) {
        mask.accepted_bitset.Set(sorted_vocab[index].first, false);
      }
    }
    IntsetDifference(&mask.accepted_indices, interior_indices);
  }
}

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

picojson::value SerializeJSONValue(const CompiledGrammar::Impl& impl) {
  auto result = picojson::object{};
  result["grammar"] = AutoSerializeJSONValue(impl.grammar);
  result["tokenizer_metadata"] = impl.tokenizer_info->DumpMetadataValue();
  result["adaptive_token_mask_cache"] = AutoSerializeJSONValue(impl.adaptive_token_mask_cache);
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
  if (auto error = AutoDeserializeJSONValue(&(impl->grammar), object["grammar"], type_name)) {
    return error;
  }
  if (impl->grammar.IsNull()) {
    return ConstructDeserializeError("Expect a non-null grammar", type_name);
  }
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
  if (auto error = AutoDeserializeJSONValue(
          &(impl->adaptive_token_mask_cache), object["adaptive_token_mask_cache"], type_name
      )) {
    return error;
  }
  // The masks index sorted_decoded_vocab and are OR-ed into a vocab_size-bit bitset, so their
  // contents must match the tokenizer they are deserialized with.
  const int64_t num_sorted_tokens = tokenizer_info.GetSortedDecodedVocab().size();
  auto indices_ok = [&](const std::vector<int32_t>& indices) {
    return std::all_of(indices.begin(), indices.end(), [&](int32_t index) {
      return index >= 0 && index < num_sorted_tokens;
    });
  };
  for (const auto& [state, mask] : impl->adaptive_token_mask_cache) {
    using StoreType = AdaptiveTokenMask::StoreType;
    const bool store_type_ok = mask.store_type == StoreType::kAccepted ||
                               mask.store_type == StoreType::kRejected ||
                               mask.store_type == StoreType::kAcceptedBitset;
    const bool bitset_ok = mask.store_type != StoreType::kAcceptedBitset ||
                           mask.accepted_bitset.Size() == tokenizer_info.GetVocabSize();
    if (!store_type_ok || !bitset_ok || !indices_ok(mask.accepted_indices) ||
        !indices_ok(mask.rejected_indices) || !indices_ok(mask.uncertain_indices)) {
      return ConstructDeserializeError(
          "adaptive_token_mask_cache contains a mask that does not match the tokenizer", type_name
      );
    }
  }
  // The repeat-interior fast path is derived from the grammar, so it is rebuilt here instead of
  // being part of the serialized form.
  PopulateRepeatInteriorBitsets(impl->grammar, tokenizer_info, &impl->adaptive_token_mask_cache);
  return std::nullopt;
}

/************** CompiledGrammar **************/

std::size_t MemorySize(const CompiledGrammar::Impl& impl) {
  return MemorySize(impl.grammar) + MemorySize(impl.adaptive_token_mask_cache);
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
  if (auto error = ParseJSON(json_value, json_string); !error.empty()) {
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
