/*
 * \file xgrammar_binding.cc
 * \brief XGrammar wasm runtime library pack.
 */

// Configuration for XGRAMMAR_LOG()
#define XGRAMMAR_LOG_CUSTOMIZE 1

#include <emscripten.h>
#include <emscripten/bind.h>
#include <xgrammar/xgrammar.h>

#include <iostream>
#include <memory>

// #include "../../cpp/support/logging.h"

namespace xgrammar {
// Override logging mechanism
[[noreturn]] void LogFatalImpl(const std::string& file, int lineno, const std::string& message) {
  std::cerr << "[FATAL] " << file << ":" << lineno << ": " << message << std::endl;
  abort();
}

void LogMessageImpl(const std::string& file, int lineno, int level, const std::string& message) {
  static const char* level_strings_[] = {
      "[DEBUG] ",
      "[INFO] ",
      "[WARNING] ",
      "[ERROR] ",
  };
  std::cout << level_strings_[level] << file << ":" << lineno << ": " << message << std::endl;
}

}  // namespace xgrammar

using namespace emscripten;
using namespace xgrammar;

TokenizerInfo TokenizerInfo_Init(
    const std::vector<std::string>& encoded_vocab,
    std::string vocab_type,
    bool prepend_space_in_tokenization
) {
  static const std::unordered_map<std::string, VocabType> VOCAB_TYPE_MAP = {
      {"RAW", VocabType::RAW},
      {"BYTE_FALLBACK", VocabType::BYTE_FALLBACK},
      {"BYTE_LEVEL", VocabType::BYTE_LEVEL},
  };
  return TokenizerInfo(encoded_vocab, VOCAB_TYPE_MAP.at(vocab_type), prepend_space_in_tokenization);
}

GrammarMatcher GrammarMatcher_Init(
    const BNFGrammar& grammar,
    const TokenizerInfo& tokenizer_info,
    std::optional<std::vector<int>> override_stop_tokens,
    bool terminate_without_stop_token,
    std::optional<int> mask_vocab_size,
    int max_rollback_tokens
) {
  return GrammarMatcher(
      CompiledGrammar(grammar, tokenizer_info),
      override_stop_tokens,
      terminate_without_stop_token,
      mask_vocab_size,
      max_rollback_tokens
  );
}

/*!
 * \brief Finds the next token bitmask of the matcher.
 */
std::vector<int32_t> GrammarMatcher_GetNextTokenBitmask(GrammarMatcher& matcher) {
  // 1. Initialize std::vector result
  auto buffer_size = GrammarMatcher::GetBufferSize(matcher.GetMaskVocabSize());
  std::vector<int32_t> result(buffer_size);
  // 2. Initialize DLTensor with the data pointer of the std vector.
  DLTensor tensor;
  tensor.data = result.data();
  tensor.device = DLDevice{kDLCPU, 0};
  tensor.ndim = 1;
  tensor.dtype = DLDataType{kDLInt, 32, 1};  // int32
  std::vector<int64_t> shape = {buffer_size};
  tensor.shape = &shape[0];
  std::vector<int64_t> strides = {1};
  tensor.strides = &strides[0];
  tensor.byte_offset = 0;
  // 3. Populate tensor, hence result
  matcher.GetNextTokenBitmask(&tensor);
  return result;
}

/*!
 * \brief Return the list of rejected token IDs based on the bit mask.
 * \note This method is mainly used in testing, so performance is not as important.
 */
std::vector<int> GrammarMatcher_GetRejectedTokensFromBitMask(
    std::vector<int32_t> token_bitmask, size_t mask_vocab_size
) {
  // 1. Convert token_bitmask into DLTensor
  DLTensor tensor;
  tensor.data = token_bitmask.data();
  tensor.device = DLDevice{kDLCPU, 0};
  tensor.ndim = 1;
  tensor.dtype = DLDataType{kDLInt, 32, 1};  // int32
  std::vector<int64_t> shape = {token_bitmask.size()};
  tensor.shape = &shape[0];
  std::vector<int64_t> strides = {1};
  tensor.strides = &strides[0];
  tensor.byte_offset = 0;
  // 2. Get rejected token IDs
  std::vector<int> result;
  GrammarMatcher::GetRejectedTokensFromBitMask(tensor, mask_vocab_size, &result);
  return result;
}

/*!
 * \brief Helps view an std::vector handle as Int32Array in JS without copying.
 */
emscripten::val vecIntToView(const std::vector<int>& vec) {
  return emscripten::val(typed_memory_view(vec.size(), vec.data()));
}

EMSCRIPTEN_BINDINGS(xgrammar) {
  // Register std::optional used in BuiltinGrammar::JSONSchema
  register_optional<int>();
  register_optional<std::pair<std::string, std::string>>();

  // Register std::vector<std::string> for TokenizerInfo.GetDecodedVocab()
  register_vector<std::string>("VectorString");
  function(
      "vecStringFromJSArray",
      select_overload<std::vector<std::string>(const emscripten::val&)>(&vecFromJSArray)
  );

  // Register std::optional<std::vector<int>> for GrammarMatcher_Init
  register_vector<int>("VectorInt");
  register_optional<std::vector<int>>();
  function(
      "vecIntFromJSArray",
      select_overload<std::vector<int>(const emscripten::val&)>(&vecFromJSArray)
  );

  // Register view so we can read std::vector<int32_t> as Int32Array in JS without copying
  function("vecIntToView", &vecIntToView);

  class_<BNFGrammar>("BNFGrammar")
      .constructor<std::string, std::string>()
      .smart_ptr<std::shared_ptr<BNFGrammar>>("BNFGrammar")
      .class_function("Deserialize", &BNFGrammar::Deserialize)
      .function("ToString", &BNFGrammar::ToString)
      .function("Serialize", &BNFGrammar::Serialize);

  class_<BuiltinGrammar>("BuiltinGrammar")
      .class_function("JSON", &BuiltinGrammar::JSON)
      .class_function("JSONSchema", &BuiltinGrammar::JSONSchema)
      .class_function("_JSONSchemaToEBNF", &BuiltinGrammar::_JSONSchemaToEBNF);

  class_<TokenizerInfo>("TokenizerInfo")
      .constructor(&TokenizerInfo_Init)
      .function("GetVocabSize", &TokenizerInfo::GetVocabSize)
      .function("GetDecodedVocab", &TokenizerInfo::GetDecodedVocab);

  class_<GrammarMatcher>("GrammarMatcher")
      .constructor(&GrammarMatcher_Init)
      .smart_ptr<std::shared_ptr<GrammarMatcher>>("GrammarMatcher")
      .function("GetMaskVocabSize", &GrammarMatcher::GetMaskVocabSize)
      .function("GetMaxRollbackTokens", &GrammarMatcher::GetMaxRollbackTokens)
      .function("AcceptToken", &GrammarMatcher::AcceptToken)
      .function("GetNextTokenBitmask", &GrammarMatcher_GetNextTokenBitmask)
      .class_function("GetRejectedTokensFromBitMask", &GrammarMatcher_GetRejectedTokensFromBitMask)
      .function("IsTerminated", &GrammarMatcher::IsTerminated)
      .function("Reset", &GrammarMatcher::Reset)
      .function("FindJumpForwardString", &GrammarMatcher::FindJumpForwardString)
      .function("Rollback", &GrammarMatcher::Rollback)
      .function("_AcceptString", &GrammarMatcher::AcceptString);
}
