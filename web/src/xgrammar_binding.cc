#include <emscripten.h>
#include <emscripten/bind.h>
#include <xgrammar/xgrammar.h>

#include <memory>

// #include "../../cpp/support/logging.h"

using namespace emscripten;
using namespace xgrammar;

/*!
 * \brief Decode an entire token table with the provided decoder type. Used for instantiating
 * XGTokenTable in JS side.
 */
std::vector<std::string> DecodeTokenTable(
    const std::vector<std::string>& rawTokenTable, const std::string& decoderType
) {
  std::vector<std::string> decodedTokenTable;
  decodedTokenTable.reserve(rawTokenTable.size());
  for (const std::string& rawToken : rawTokenTable) {
    decodedTokenTable.push_back(XGTokenizer::DecodeToken(rawToken, decoderType));
  }
  return decodedTokenTable;
}

// TODO(Charlie): Should this return pointer, and use `allow_raw_pointers()`?
/*!
 * \brief Constructor for grammar state matcher in JS.
 */
GrammarStateMatcher GrammarStateMatcher_Init(
    const BNFGrammar& grammar,
    const std::vector<std::string>& vocab,
    std::optional<std::vector<int>> stop_token_ids,
    bool terminate_without_stop_token,
    int max_rollback_steps
) {
  return GrammarStateMatcher(
      GrammarStateMatcher::CreateInitContext(grammar, vocab),
      stop_token_ids,
      terminate_without_stop_token,
      max_rollback_steps
  );
}

/*!
 * \brief Finds the next token bitmask of the matcher.
 */
std::vector<int32_t> GrammarStateMatcher_FindNextTokenBitmask(GrammarStateMatcher& matcher) {
  // 1. Initialize std::vector result
  auto buffer_size = GrammarStateMatcher::GetBufferSize(matcher.GetVocabSize());
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
  matcher.FindNextTokenBitmask(&tensor);
  return result;
}

/*!
 * \brief Return the list of rejected token IDs based on the bit mask.
 * \note This method is mainly used in testing, so performance is not as important.
 */
std::vector<int> GrammarStateMatcher_GetRejectedTokensFromBitMask(
    std::vector<int32_t> token_bitmask, size_t vocab_size
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
  GrammarStateMatcher::GetRejectedTokensFromBitMask(tensor, vocab_size, &result);
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

  // Register std::vector<std::string> for DecodeTokenTable
  register_vector<std::string>("VectorString");
  function(
      "vecStringFromJSArray",
      select_overload<std::vector<std::string>(const emscripten::val&)>(&vecFromJSArray)
  );

  // Register std::optional<std::vector<int>> for GrammarStateMatcher_Init
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

  function("DecodeTokenTable", &DecodeTokenTable);

  class_<GrammarStateMatcher>("GrammarStateMatcher")
      .constructor(&GrammarStateMatcher_Init)
      .smart_ptr<std::shared_ptr<GrammarStateMatcher>>("GrammarStateMatcher")
      .function("GetVocabSize", &GrammarStateMatcher::GetVocabSize)
      .function("GetMaxRollbackSteps", &GrammarStateMatcher::GetMaxRollbackSteps)
      .function("AcceptToken", &GrammarStateMatcher::AcceptToken)
      .function("FindNextTokenBitmask", &GrammarStateMatcher_FindNextTokenBitmask)
      .class_function(
          "GetRejectedTokensFromBitMask", &GrammarStateMatcher_GetRejectedTokensFromBitMask
      )
      .function("IsTerminated", &GrammarStateMatcher::IsTerminated)
      .function("Reset", &GrammarStateMatcher::Reset)
      .function("FindJumpForwardString", &GrammarStateMatcher::FindJumpForwardString)
      .function("Rollback", &GrammarStateMatcher::Rollback)
      .function("_AcceptString", &GrammarStateMatcher::_AcceptString);
}
