/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/nanobind/nanobind.cc
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <xgrammar/xgrammar.h>

#include <cstdint>
#include <optional>
#include <thread>
#include <variant>
#include <vector>

#include "../grammar_functor.h"
#include "../json_schema_converter.h"
#include "../regex_converter.h"
#include "../testing.h"
#include "python_methods.h"
#include "xgrammar/exception.h"
#include "xgrammar/matcher.h"

namespace nb = nanobind;

namespace xgrammar {

std::vector<std::string> CommonEncodedVocabType(
    const nb::typed<nb::list, std::variant<std::string, nb::bytes>> encoded_vocab
) {
  std::vector<std::string> encoded_vocab_strs;
  encoded_vocab_strs.reserve(encoded_vocab.size());
  for (const auto& token : encoded_vocab) {
    if (nb::bytes result; nb::try_cast(token, result)) {
      encoded_vocab_strs.emplace_back(result.c_str());
    } else if (nb::str result; nb::try_cast(token, result)) {
      encoded_vocab_strs.emplace_back(result.c_str());
    } else {
      throw nb::type_error("Expected str or bytes for encoded_vocab");
    }
  }
  return encoded_vocab_strs;
}

bool GrammarMatcher_FillNextTokenBitmask(
    GrammarMatcher& matcher, nb::ndarray<> arr, int32_t index, bool debug_print
) {
  if (arr.ndim() != 1 && arr.ndim() != 2) {
    throw std::runtime_error("token_bitmask tensor must be 1D or 2D");
  }

  // 2. Device: ensure the tensor is on CPU
  if (arr.device_type() != nb::device::cpu::value) {
    throw std::runtime_error("token_bitmask array must be on CPU");
  }

  // 3. Data type: ensure 32-bit integers
  if (arr.dtype() != nb::dtype<int32_t>()) {
    throw std::runtime_error("token_bitmask array must be int32");
  }

  // Under the hood these are stored with the same standard (DLPack), but nanobind
  // defines its own types, and doesn't expose a way to just get the object directly.
  // We'll just do some pointer hackery to get there, rather than build the type back up manually:

  // The data in an ndarray is defined as:
  // detail::ndarray_handle* m_handle = nullptr;
  // dlpack::dltensor m_dltensor;
  // Assert this, then skip over m_handle and reinterpret m_dltensor.
  static_assert(sizeof(arr) == sizeof(void*) + sizeof(nb::dlpack::dltensor));

  DLTensor* bitmask_dltensor_ptr =
      reinterpret_cast<::DLTensor*>(reinterpret_cast<char*>(&arr) + sizeof(void*));

  return matcher.FillNextTokenBitmask(bitmask_dltensor_ptr, index, debug_print);
}

void GrammarMatcher_BatchFillNextTokenMask(
    BatchGrammarMatcher& batch_matcher,
    std::vector<GrammarMatcher>* matchers,
    nb::ndarray<> arr,
    const std::optional<std::vector<int32_t>>& indices,
    std::variant<int32_t, std::string> max_threads,
    bool debug_print
) {
  if (arr.ndim() != 2) {
    throw std::runtime_error("batch_token_bitmask tensor must be 2D");
  }
  if (arr.device_type() != nb::device::cpu::value) {
    throw std::runtime_error("token_bitmask array must be on CPU");
  }
  if (arr.dtype() != nb::dtype<int32_t>()) {
    throw std::runtime_error("token_bitmask array must be int32");
  }
  static_assert(sizeof(arr) == sizeof(void*) + sizeof(nb::dlpack::dltensor));

  DLTensor* bitmask_dltensor_ptr =
      reinterpret_cast<::DLTensor*>(reinterpret_cast<char*>(&arr) + sizeof(void*));

  batch_matcher.BatchFillNextTokenBitmask(matchers, bitmask_dltensor_ptr, indices, debug_print);
}

std::vector<uint8_t> GrammarMatcher_BatchAcceptString(
    std::vector<GrammarMatcher>* matchers,
    const std::vector<std::variant<nb::bytes, std::string>>& input_strs,
    bool debug_print
) {
  std::vector<std::string> input_strs_converted;
  input_strs_converted.reserve(input_strs.size());
  for (const auto& str : input_strs) {
    if (std::holds_alternative<std::string>(str)) {
      input_strs_converted.emplace_back(std::get<std::string>(str));
    } else {
      input_strs_converted.emplace_back(std::get<nb::bytes>(str).c_str());
    }
  }
  return BatchGrammarMatcher::BatchAcceptString(matchers, input_strs_converted);
}

std::vector<nanobind::bytes> TokenizerInfo_GetDecodedVocab(const TokenizerInfo& tokenizer) {
  const auto& decoded_vocab = tokenizer.GetDecodedVocab();
  std::vector<nanobind::bytes> py_result;
  py_result.reserve(decoded_vocab.size());
  for (const auto& item : decoded_vocab) {
    py_result.emplace_back(nanobind::bytes(item.c_str()));
  }
  return py_result;
}

}  // namespace xgrammar

using namespace xgrammar;

NB_MODULE(xgrammar_bindings, m) {
  auto pyTokenizerInfo = nb::class_<TokenizerInfo>(m, "TokenizerInfo");
  pyTokenizerInfo
      .def(
          "__init__",
          [](TokenizerInfo* out,
             const nb::typed<nb::list, std::variant<std::string, nb::bytes>> encoded_vocab,
             int vocab_type,
             std::optional<int> vocab_size,
             std::optional<std::vector<int32_t>> stop_token_ids,
             bool add_prefix_space) {
            new (out) TokenizerInfo{TokenizerInfo_Init(
                CommonEncodedVocabType(encoded_vocab),
                vocab_type,
                vocab_size,
                std::move(stop_token_ids),
                add_prefix_space
            )};
          },
          nb::arg("encoded_vocab"),
          nb::arg("vocab_type"),
          nb::arg("vocab_size").none(),
          nb::arg("stop_token_ids").none(),
          nb::arg("add_prefix_space")
      )
      .def_prop_ro("vocab_type", &TokenizerInfo_GetVocabType)
      .def_prop_ro("vocab_size", &TokenizerInfo::GetVocabSize)
      .def_prop_ro("add_prefix_space", &TokenizerInfo::GetAddPrefixSpace)
      .def_prop_ro("decoded_vocab", &TokenizerInfo_GetDecodedVocab)
      .def_prop_ro("stop_token_ids", &TokenizerInfo::GetStopTokenIds)
      .def_prop_ro("special_token_ids", &TokenizerInfo::GetSpecialTokenIds)
      .def("dump_metadata", &TokenizerInfo::DumpMetadata)
      .def_static(
          "from_vocab_and_metadata",
          [](const nb::typed<nb::list, std::variant<std::string, nb::bytes>> encoded_vocab,
             const std::string& metadata) {
            return TokenizerInfo::FromVocabAndMetadata(
                CommonEncodedVocabType(encoded_vocab), metadata
            );
          }
      )
      .def_static("_detect_metadata_from_hf", &TokenizerInfo::DetectMetadataFromHF)
      .def("serialize_json", &TokenizerInfo::SerializeJSON)
      .def_static("deserialize_json", &TokenizerInfo_DeserializeJSON);

  auto pyGrammar = nb::class_<Grammar>(m, "Grammar");
  pyGrammar.def("to_string", &Grammar::ToString)
      .def_static("from_ebnf", &Grammar::FromEBNF)
      .def_static(
          "from_json_schema",
          &Grammar::FromJSONSchema,
          nb::arg("schema"),
          nb::arg("any_whitespace"),
          nb::arg("indent").none(),
          nb::arg("separators").none(),
          nb::arg("strict_mode"),
          nb::arg("max_whitespace_cnt").none(),
          nb::arg("print_converted_ebnf"),
          nb::call_guard<nb::gil_scoped_release>()
      )
      .def_static("from_regex", &Grammar::FromRegex, nb::call_guard<nb::gil_scoped_release>())
      .def_static(
          "from_structural_tag",
          &Grammar_FromStructuralTag,
          nb::call_guard<nb::gil_scoped_release>()
      )
      .def_static("builtin_json_grammar", &Grammar::BuiltinJSONGrammar)
      .def_static("union", &Grammar::Union, nb::call_guard<nb::gil_scoped_release>())
      .def_static("concat", &Grammar::Concat, nb::call_guard<nb::gil_scoped_release>())
      .def("serialize_json", &Grammar::SerializeJSON)
      .def_static("deserialize_json", &Grammar_DeserializeJSON);

  auto pyCompiledGrammar = nb::class_<CompiledGrammar>(m, "CompiledGrammar");
  pyCompiledGrammar.def_prop_ro("grammar", &CompiledGrammar::GetGrammar)
      .def_prop_ro("tokenizer_info", &CompiledGrammar::GetTokenizerInfo)
      .def_prop_ro("memory_size_bytes", &CompiledGrammar::MemorySizeBytes)
      .def("serialize_json", &CompiledGrammar::SerializeJSON)
      .def_static("deserialize_json", &CompiledGrammar_DeserializeJSON);

  auto pyGrammarCompiler = nb::class_<GrammarCompiler>(m, "GrammarCompiler");
  pyGrammarCompiler.def(nb::init<const TokenizerInfo&, int, bool, int64_t>())
      .def(
          "compile_json_schema",
          &GrammarCompiler::CompileJSONSchema,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("schema"),
          nb::arg("any_whitespace"),
          nb::arg("indent").none(),
          nb::arg("separators").none(),
          nb::arg("strict_mode"),
          nb::arg("max_whitespace_cnt").none()
      )
      .def(
          "compile_builtin_json_grammar",
          &GrammarCompiler::CompileBuiltinJSONGrammar,
          nb::call_guard<nb::gil_scoped_release>()
      )
      .def(
          "compile_structural_tag",
          &GrammarCompiler::CompileStructuralTag,
          nb::call_guard<nb::gil_scoped_release>()
      )
      .def(
          "compile_regex", &GrammarCompiler::CompileRegex, nb::call_guard<nb::gil_scoped_release>()
      )
      .def(
          "compile_grammar",
          [](GrammarCompiler& self, const Grammar& grammar) {
            return self.CompileGrammar(grammar);
          },
          nb::call_guard<nb::gil_scoped_release>()
      )
      .def(
          "compile_grammar",
          [](GrammarCompiler& self, const std::string& ebnf_str, const std::string& root_rule_name
          ) { return self.CompileGrammar(ebnf_str, root_rule_name); },
          nb::call_guard<nb::gil_scoped_release>()
      )
      .def("clear_cache", &GrammarCompiler::ClearCache)
      .def("get_cache_size_bytes", &GrammarCompiler::GetCacheSizeBytes)
      .def_prop_ro("cache_limit_bytes", &GrammarCompiler::CacheLimitBytes);
  auto pyBatchGrammarMatcher = nb::class_<BatchGrammarMatcher>(m, "BatchGrammarMatcher");
  pyBatchGrammarMatcher
      .def(nb::init<std::variant<std::string, int32_t>>(), nb::arg("max_threads") = "auto")
      .def(
          "batch_fill_next_token_bitmask",
          &GrammarMatcher_BatchFillNextTokenMask,
          nb::arg("matchers"),
          nb::arg("batch_token_bitmask"),
          nb::arg("indices").none(),
          nb::arg("max_thread") = "auto",
          nb::arg("debug_print") = false,
          nb::call_guard<nb::gil_scoped_release>()
      )
      .def_static(
          "batch_accept_string",
          &GrammarMatcher_BatchAcceptString,
          nb::call_guard<nb::gil_scoped_release>()
      )
      .def_static(
          "batch_accept_token",
          &BatchGrammarMatcher::BatchAcceptToken,
          nb::call_guard<nb::gil_scoped_release>()
      );
  auto pyGrammarMatcher = nb::class_<GrammarMatcher>(m, "GrammarMatcher");
  pyGrammarMatcher
      .def(
          nb::init<const CompiledGrammar&, std::optional<std::vector<int>>, bool, int>(),
          nb::arg("compiled_grammar"),
          nb::arg("override_stop_tokens").none(),
          nb::arg("terminate_without_stop_token"),
          nb::arg("max_rollback_tokens")
      )
      .def("accept_token", &GrammarMatcher::AcceptToken, nb::call_guard<nb::gil_scoped_release>())
      .def("accept_string", &GrammarMatcher::AcceptString, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "accept_string",
          [](GrammarMatcher& self, const nb::bytes& input_str, bool debug_print) {
            return self.AcceptString(input_str.c_str(), debug_print);
          },
          nb::call_guard<nb::gil_scoped_release>()
      )
      .def(
          "fill_next_token_bitmask",
          &GrammarMatcher_FillNextTokenBitmask,
          nb::call_guard<nb::gil_scoped_release>()
      )
      .def(
          "find_jump_forward_string",
          &GrammarMatcher::FindJumpForwardString,
          nb::call_guard<nb::gil_scoped_release>()
      )
      .def("rollback", &GrammarMatcher::Rollback, nb::call_guard<nb::gil_scoped_release>())
      .def("is_terminated", &GrammarMatcher::IsTerminated)
      .def("reset", &GrammarMatcher::Reset, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("max_rollback_tokens", &GrammarMatcher::GetMaxRollbackTokens)
      .def_prop_ro("stop_token_ids", &GrammarMatcher::GetStopTokenIds)
      .def("_debug_print_internal_state", &GrammarMatcher::_DebugPrintInternalState);

  auto pyTestingModule = m.def_submodule("testing");
  pyTestingModule
      .def(
          "_json_schema_to_ebnf",
          [](const std::string& schema,
             bool any_whitespace,
             std::optional<int> indent,
             std::optional<std::pair<std::string, std::string>> separators,
             bool strict_mode,
             std::optional<int> max_whitespace_cnt) {
            return JSONSchemaToEBNF(
                schema,
                any_whitespace,
                indent,
                separators,
                strict_mode,
                max_whitespace_cnt,
                JSONFormat::kJSON
            );
          },
          nb::arg("schema"),
          nb::arg("any_whitespace"),
          nb::arg("indent").none(),
          nb::arg("separators").none(),
          nb::arg("strict_mode"),
          nb::arg("max_whitespace_cnt").none()
      )
      .def("_regex_to_ebnf", &RegexToEBNF)
      .def("_ebnf_to_grammar_no_normalization", &_EBNFToGrammarNoNormalization)
      .def("_get_masked_tokens_from_bitmask", &Testing_DebugGetMaskedTokensFromBitmask)
      .def("_is_single_token_bitmask", &Testing_IsSingleTokenBitmask)
      .def("_get_allow_empty_rule_ids", &GetAllowEmptyRuleIds)
      .def(
          "_generate_range_regex",
          [](std::optional<int> start, std::optional<int> end) {
            std::string result = GenerateRangeRegex(start, end);
            result.erase(std::remove(result.begin(), result.end(), '\0'), result.end());
            return result;
          },
          nb::arg("start").none(),
          nb::arg("end").none()
      )
      .def("_qwen_xml_tool_calling_to_ebnf", &QwenXMLToolCallingToEBNF, nb::arg("schema"))
      .def(
          "_generate_float_regex",
          [](std::optional<double> start, std::optional<double> end) {
            std::string result = GenerateFloatRangeRegex(start, end);
            result.erase(std::remove(result.begin(), result.end(), '\0'), result.end());
            return result;
          },
          nb::arg("start").none(),
          nb::arg("end").none()
      )
      .def("_print_grammar_fsms", &_PrintGrammarFSMs);

  auto pyGrammarFunctorModule = pyTestingModule.def_submodule("grammar_functor");
  pyGrammarFunctorModule.def("structure_normalizer", &StructureNormalizer::Apply)
      .def("byte_string_fuser", &ByteStringFuser::Apply)
      .def("rule_inliner", &RuleInliner::Apply)
      .def("dead_code_eliminator", &DeadCodeEliminator::Apply)
      .def("lookahead_assertion_analyzer", &LookaheadAssertionAnalyzer::Apply)
      .def("grammar_optimizer", &GrammarOptimizer::Apply)
      .def("repetition_normalizer", &RepetitionNormalizer::Apply);

  auto pyKernelsModule = m.def_submodule("kernels");
  pyKernelsModule.def(
      "apply_token_bitmask_inplace_cpu",
      &Kernels_ApplyTokenBitmaskInplaceCPU,
      nb::arg("logits_ptr"),
      nb::arg("logits_shape"),
      nb::arg("logits_strides"),
      nb::arg("bitmask_ptr"),
      nb::arg("bitmask_shape"),
      nb::arg("bitmask_strides"),
      nb::arg("vocab_size"),
      nb::arg("indices").none(),
      nb::arg("logit_type"),
      nb::call_guard<nb::gil_scoped_release>()
  );

  auto pyConfigModule = m.def_submodule("config");
  pyConfigModule
      .def(
          "set_max_recursion_depth", &SetMaxRecursionDepth, nb::call_guard<nb::gil_scoped_release>()
      )
      .def(
          "get_max_recursion_depth", &GetMaxRecursionDepth, nb::call_guard<nb::gil_scoped_release>()
      )
      .def(
          "get_serialization_version",
          &GetSerializationVersion,
          nb::call_guard<nb::gil_scoped_release>()
      );

  auto pyExceptionModule = m.def_submodule("exception");
  nb::exception<DeserializeFormatError>{
      pyExceptionModule, "DeserializeFormatError", PyExc_RuntimeError
  };
  nb::exception<DeserializeVersionError>{
      pyExceptionModule, "DeserializeVersionError", PyExc_RuntimeError
  };
  nb::exception<InvalidJSONError>{pyExceptionModule, "InvalidJSONError", PyExc_RuntimeError};
  nb::exception<InvalidStructuralTagError>{
      pyExceptionModule, "InvalidStructuralTagError", PyExc_RuntimeError
  };
}
