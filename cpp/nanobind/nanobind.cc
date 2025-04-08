/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/nanobind/nanobind.cc
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <xgrammar/xgrammar.h>

#include "../grammar_functor.h"
#include "../json_schema_converter.h"
#include "../regex_converter.h"
#include "../testing.h"
#include "python_methods.h"

namespace nb = nanobind;
using namespace xgrammar;

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

std::vector<nanobind::bytes> TokenizerInfo_GetDecodedVocab(const TokenizerInfo& tokenizer) {
  const auto& decoded_vocab = tokenizer.GetDecodedVocab();
  std::vector<nanobind::bytes> py_result;
  py_result.reserve(decoded_vocab.size());
  for (const auto& item : decoded_vocab) {
    py_result.emplace_back(nanobind::bytes(item.c_str()));
  }
  return py_result;
}

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
      .def_static("_detect_metadata_from_hf", &TokenizerInfo::DetectMetadataFromHF);

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
      .def_static("concat", &Grammar::Concat, nb::call_guard<nb::gil_scoped_release>());

  auto pyCompiledGrammar = nb::class_<CompiledGrammar>(m, "CompiledGrammar");
  pyCompiledGrammar.def_prop_ro("grammar", &CompiledGrammar::GetGrammar)
      .def_prop_ro("tokenizer_info", &CompiledGrammar::GetTokenizerInfo)
      .def_prop_ro("memory_size_bytes", &CompiledGrammar::MemorySizeBytes);

  auto pyGrammarCompiler = nb::class_<GrammarCompiler>(m, "GrammarCompiler");
  pyGrammarCompiler.def(nb::init<const TokenizerInfo&, int, bool, long long>())
      .def(
          "compile_json_schema",
          &GrammarCompiler::CompileJSONSchema,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("schema"),
          nb::arg("any_whitespace"),
          nb::arg("indent").none(),
          nb::arg("separators").none(),
          nb::arg("strict_mode")
      )
      .def(
          "compile_builtin_json_grammar",
          &GrammarCompiler::CompileBuiltinJSONGrammar,
          nb::call_guard<nb::gil_scoped_release>()
      )
      .def(
          "compile_structural_tag",
          &GrammarCompiler_CompileStructuralTag,
          nb::call_guard<nb::gil_scoped_release>()
      )
      .def(
          "compile_regex", &GrammarCompiler::CompileRegex, nb::call_guard<nb::gil_scoped_release>()
      )
      .def(
          "compile_grammar",
          &GrammarCompiler::CompileGrammar,
          nb::call_guard<nb::gil_scoped_release>()
      )
      .def("clear_cache", &GrammarCompiler::ClearCache)
      .def("get_cache_size_bytes", &GrammarCompiler::GetCacheSizeBytes)
      .def_prop_ro("cache_limit_bytes", &GrammarCompiler::CacheLimitBytes);

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
      .def(
          "_debug_accept_string",
          &GrammarMatcher::_DebugAcceptString,
          nb::call_guard<nb::gil_scoped_release>()
      )
      .def(
          "_debug_accept_string",
          [](GrammarMatcher& self, const nb::bytes& input_str, bool debug_print) {
            return self._DebugAcceptString(input_str.c_str(), debug_print);
          },
          nb::call_guard<nb::gil_scoped_release>()
      );

  auto pyTestingModule = m.def_submodule("testing");
  pyTestingModule
      .def(
          "_json_schema_to_ebnf",
          nb::overload_cast<
              const std::string&,
              bool,
              std::optional<int>,
              std::optional<std::pair<std::string, std::string>>,
              bool>(&JSONSchemaToEBNF),
          nb::arg("schema"),
          nb::arg("any_whitespace"),
          nb::arg("indent").none(),
          nb::arg("separators").none(),
          nb::arg("strict_mode")
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
      .def(
          "_generate_float_regex",
          [](std::optional<double> start, std::optional<double> end) {
            std::string result = GenerateFloatRangeRegex(start, end);
            result.erase(std::remove(result.begin(), result.end(), '\0'), result.end());
            return result;
          },
          nb::arg("start").none(),
          nb::arg("end").none()
      );

  auto pyGrammarFunctorModule = pyTestingModule.def_submodule("grammar_functor");
  pyGrammarFunctorModule.def("structure_normalizer", &StructureNormalizer::Apply)
      .def("byte_string_fuser", &ByteStringFuser::Apply)
      .def("rule_inliner", &RuleInliner::Apply)
      .def("dead_code_eliminator", &DeadCodeEliminator::Apply)
      .def("lookahead_assertion_analyzer", &LookaheadAssertionAnalyzer::Apply);

  auto pyKernelsModule = m.def_submodule("kernels");
  pyKernelsModule.def(
      "apply_token_bitmask_inplace_cpu",
      &Kernels_ApplyTokenBitmaskInplaceCPU,
      nb::arg("logits_ptr"),
      nb::arg("logits_shape"),
      nb::arg("bitmask_ptr"),
      nb::arg("bitmask_shape"),
      nb::arg("vocab_size"),
      nb::arg("indices").none(),
      nb::call_guard<nb::gil_scoped_release>()
  );
}
