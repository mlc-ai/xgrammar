/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/pybind.cc
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xgrammar/xgrammar.h>

#include "python_methods.h"

namespace py = pybind11;
using namespace xgrammar;

PYBIND11_MODULE(xgrammar_bindings, m) {
  auto pyBNFGrammar = py::class_<BNFGrammar>(m, "BNFGrammar");
  pyBNFGrammar.def(py::init<const std::string&, const std::string&>())
      .def("to_string", &BNFGrammar::ToString)
      .def("serialize", &BNFGrammar::Serialize)
      .def_static("deserialize", &BNFGrammar::Deserialize)
      .def_static("_init_no_normalization", &BNFGrammar_InitNoNormalization);

  auto pyBuiltinGrammar = py::class_<BuiltinGrammar>(m, "BuiltinGrammar");
  pyBuiltinGrammar.def_static("json", &BuiltinGrammar::JSON)
      .def_static("json_schema", &BuiltinGrammar::JSONSchema)
      .def_static("_json_schema_to_ebnf", &BuiltinGrammar::_JSONSchemaToEBNF)
      .def_static("_regex_to_ebnf", &BuiltinGrammar::_RegexToEBNF);

  auto pyTokenizerInfo = py::class_<TokenizerInfo>(m, "TokenizerInfo");
  pyTokenizerInfo.def(py::init(&TokenizerInfo_Init))
      .def_property_readonly("vocab_size", &TokenizerInfo::GetVocabSize)
      .def_property_readonly("vocab_type", &TokenizerInfo_GetVocabType)
      .def_property_readonly(
          "prepend_space_in_tokenization", &TokenizerInfo::GetPrependSpaceInTokenization
      )
      .def_property_readonly("decoded_vocab", &TokenizerInfo_GetDecodedVocab)
      .def("dump_metadata", &TokenizerInfo::DumpMetadata)
      .def_static("from_huggingface", &TokenizerInfo::FromHuggingFace)
      .def_static("from_vocab_and_metadata", &TokenizerInfo::FromVocabAndMetadata);

  auto pyCompiledGrammar = py::class_<CompiledGrammar>(m, "CompiledGrammar");
  pyCompiledGrammar.def(py::init<const BNFGrammar&, const TokenizerInfo&, int>());

  auto pyCachedGrammarCompiler = py::class_<CachedGrammarCompiler>(m, "CachedGrammarCompiler");
  pyCachedGrammarCompiler.def(py::init<const TokenizerInfo&, int>())
      .def(
          "compile_json_grammar",
          &CachedGrammarCompiler::CompileJSONGrammar,
          py::call_guard<py::gil_scoped_release>()
      )
      .def(
          "compile_json_schema_grammar",
          &CachedGrammarCompiler::CompileJSONSchemaGrammar,
          py::call_guard<py::gil_scoped_release>()
      )
      .def("clear", &CachedGrammarCompiler::Clear);

  auto pyGrammarMatcher = py::class_<GrammarMatcher>(m, "GrammarMatcher");
  pyGrammarMatcher
      .def(py::init<
           const CompiledGrammar&,
           std::optional<std::vector<int>>,
           bool,
           std::optional<int>,
           int>())
      .def("accept_token", &GrammarMatcher::AcceptToken)
      .def("accept_string", &GrammarMatcher::AcceptString)
      .def("fill_next_token_bitmask", &GrammarMatcher_FillNextTokenBitmask)
      .def("debug_get_masked_tokens_from_bitmask", &GrammarMatcher_DebugGetMaskedTokensFromBitmask)
      .def("is_terminated", &GrammarMatcher::IsTerminated)
      .def("reset", &GrammarMatcher::Reset)
      .def("find_jump_forward_string", &GrammarMatcher::FindJumpForwardString)
      .def("rollback", &GrammarMatcher::Rollback)
      .def_property_readonly("vocab_size", &GrammarMatcher::GetVocabSize)
      .def_property_readonly("max_rollback_tokens", &GrammarMatcher::GetMaxRollbackTokens)
      .def_property_readonly("stop_token_ids", &GrammarMatcher::GetStopTokenIds);
#ifdef XGRAMMAR_BUILD_KERNELS
  pyGrammarMatcher.def_static(
      "apply_token_bitmask_inplace", &GrammarMatcher_ApplyTokenBitmaskInplace
  );
#endif
}
