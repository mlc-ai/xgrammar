#include "compiled_grammar_data_structure.h"
#include "grammar_data_structure.h"  // IWYU pragma: keep
#include "picojson.h"
#include "support/json_serialize.h"
#include "tokenizer_internal.h"  // IWYU pragma: keep

namespace xgrammar {

picojson::value CompiledGrammar::Impl::JSONSerialize() const {
  auto result = picojson::object{};
  result["grammar"] = AutoJSONSerialize(*grammar);
  result["tokenizer_metadata"] = AutoJSONSerialize(*tokenizer_info);
  result["adaptive_token_mask_cache"] = AutoJSONSerialize(adaptive_token_mask_cache);
  return picojson::value(std::move(result));
}

}  // namespace xgrammar
