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

#include "../../cpp/json_schema_converter.h"
#include "../../cpp/testing.h"

// #include "../../cpp/support/logging.h"

namespace xgrammar {
// Override logging mechanism
[[noreturn]] void LogFatalImpl(const std::string& file, int lineno, const std::string& message) {
  std::cerr << "[FATAL] " << file << ":" << lineno << ": " << message << std::endl;
  abort();
}

void LogMessageImpl(const std::string& file, int lineno, int level, const std::string& message) {
  static const char* level_strings_[] = {
      "[INFO] ",
      "[DEBUG] ",
      "[WARNING] ",
  };
  std::cout << level_strings_[level] << file << ":" << lineno << ": " << message << std::endl;
}

}  // namespace xgrammar

using namespace emscripten;
using namespace xgrammar;

TokenizerInfo TokenizerInfo_Init(
    const std::vector<std::string>& encoded_vocab,
    std::string vocab_type,
    std::optional<int> vocab_size,
    std::optional<std::vector<int>> stop_token_ids,
    bool add_prefix_space
) {
  static const std::unordered_map<std::string, VocabType> VOCAB_TYPE_MAP = {
      {"RAW", VocabType::RAW},
      {"BYTE_FALLBACK", VocabType::BYTE_FALLBACK},
      {"BYTE_LEVEL", VocabType::BYTE_LEVEL},
  };
  return TokenizerInfo(
      encoded_vocab, VOCAB_TYPE_MAP.at(vocab_type), vocab_size, stop_token_ids, add_prefix_space
  );
}

GrammarMatcher GrammarMatcher_Init(
    const CompiledGrammar& grammar,
    std::optional<std::vector<int>> override_stop_tokens,
    bool terminate_without_stop_token,
    int max_rollback_tokens
) {
  return GrammarMatcher(
      grammar, override_stop_tokens, terminate_without_stop_token, max_rollback_tokens
  );
}

/*!
 * \brief Finds the next token bitmask of the matcher.
 */
std::vector<int32_t> GrammarMatcher_GetNextTokenBitmask(GrammarMatcher& matcher, int vocab_size) {
  // 1. Initialize std::vector result
  auto buffer_size = GetBitmaskSize(vocab_size);
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
  matcher.FillNextTokenBitmask(&tensor);
  return result;
}

/*!
 * \brief Return the list of rejected token IDs based on the bit mask.
 * \note This method is mainly used in testing, so performance is not as important.
 */
std::vector<int> Testing_DebugGetMaskedTokensFromBitmask(
    std::vector<int32_t> token_bitmask, size_t vocab_size, int index
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
  _DebugGetMaskedTokensFromBitmask(&result, tensor, vocab_size, index);
  return result;
}

/*!
 * \brief Helps view an std::vector handle as Int32Array in JS without copying.
 */
emscripten::val vecIntToView(const std::vector<int>& vec) {
  return emscripten::val(typed_memory_view(vec.size(), vec.data()));
}

EMSCRIPTEN_BINDINGS(xgrammar) {
  // Register std::optional used in Grammar::FromJSONSchema
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

  // Testing methods
  function(
      "_JSONSchemaToEBNF",
      select_overload<std::string(
          const std::string&,
          bool,
          std::optional<int>,
          std::optional<std::pair<std::string, std::string>>,
          bool,
          std::optional<int>,
          JSONFormat
      )>(&JSONSchemaToEBNF)
  );
  function("DebugGetMaskedTokensFromBitmask", &Testing_DebugGetMaskedTokensFromBitmask);

  class_<Grammar>("Grammar")
      .function("ToString", &Grammar::ToString)
      .class_function("FromEBNF", &Grammar::FromEBNF)
      .class_function("FromJSONSchema", &Grammar::FromJSONSchema)
      .class_function("BuiltinJSONGrammar", &Grammar::BuiltinJSONGrammar);

  class_<TokenizerInfo>("TokenizerInfo")
      .constructor(&TokenizerInfo_Init)
      .function("GetVocabSize", &TokenizerInfo::GetVocabSize)
      .function("GetDecodedVocab", &TokenizerInfo::GetDecodedVocab);

  class_<CompiledGrammar>("CompiledGrammar")
      .function("GetGrammar", &CompiledGrammar::GetGrammar)
      .function("GetTokenizerInfo", &CompiledGrammar::GetTokenizerInfo);

  class_<GrammarCompiler>("GrammarCompiler")
      .constructor<const TokenizerInfo&, int, bool>()
      .function("CompileJSONSchema", &GrammarCompiler::CompileJSONSchema)
      .function("CompileBuiltinJSONGrammar", &GrammarCompiler::CompileBuiltinJSONGrammar)
      .function(
          "CompileGrammar",
          select_overload<CompiledGrammar(const Grammar&)>(&GrammarCompiler::CompileGrammar)
      )
      .function("ClearCache", &GrammarCompiler::ClearCache);

  class_<GrammarMatcher>("GrammarMatcher")
      .constructor(&GrammarMatcher_Init)
      .smart_ptr<std::shared_ptr<GrammarMatcher>>("GrammarMatcher")
      .function("GetMaxRollbackTokens", &GrammarMatcher::GetMaxRollbackTokens)
      .function("AcceptToken", &GrammarMatcher::AcceptToken)
      .function("GetNextTokenBitmask", &GrammarMatcher_GetNextTokenBitmask)
      .function("IsTerminated", &GrammarMatcher::IsTerminated)
      .function("Reset", &GrammarMatcher::Reset)
      .function("FindJumpForwardString", &GrammarMatcher::FindJumpForwardString)
      .function("Rollback", &GrammarMatcher::Rollback)
      .function("AcceptString", &GrammarMatcher::AcceptString);
}
