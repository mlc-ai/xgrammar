/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/pybind.cc
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xgrammar/xgrammar.h>

#include "../testing.h"
#include "python_methods.h"

namespace py = pybind11;
using namespace xgrammar;

PYBIND11_MODULE(xgrammar_bindings, m) {
  auto pyTokenizerInfo = py::class_<TokenizerInfo>(m, "TokenizerInfo");
  pyTokenizerInfo.def(py::init(&TokenizerInfo_Init))
      .def_property_readonly("vocab_type", &TokenizerInfo_GetVocabType)
      .def_property_readonly("vocab_size", &TokenizerInfo::GetVocabSize)
      .def_property_readonly(
          "prepend_space_in_tokenization", &TokenizerInfo::GetPrependSpaceInTokenization
      )
      .def_property_readonly("decoded_vocab", &TokenizerInfo_GetDecodedVocab)
      .def_property_readonly("stop_token_ids", &TokenizerInfo::GetStopTokenIds)
      .def_property_readonly("special_token_ids", &TokenizerInfo::GetSpecialTokenIds)
      .def("dump_metadata", &TokenizerInfo::DumpMetadata)
      .def_static("from_huggingface", &TokenizerInfo::FromHuggingFace)
      .def_static("from_vocab_and_metadata", &TokenizerInfo::FromVocabAndMetadata);

  auto pyBNFGrammar = py::class_<BNFGrammar>(m, "BNFGrammar");
  pyBNFGrammar.def("to_string", &BNFGrammar::ToString)
      .def_static("from_ebnf", &BNFGrammar::FromEBNF)
      .def_static("from_json_schema", &BNFGrammar::FromJSONSchema)
      .def_static("builtin_json_grammar", &BNFGrammar::BuiltinJSONGrammar);

  auto pyCompiledGrammar = py::class_<CompiledGrammar>(m, "CompiledGrammar");
  pyCompiledGrammar.def_property_readonly("grammar", &CompiledGrammar::GetGrammar)
      .def_property_readonly("tokenizer_info", &CompiledGrammar::GetTokenizerInfo);

  auto pyGrammarCompiler = py::class_<GrammarCompiler>(m, "GrammarCompiler");
  pyGrammarCompiler.def(py::init<const TokenizerInfo&, int, bool>())
      .def(
          "compile_json_schema",
          &GrammarCompiler::CompileJSONSchema,
          py::call_guard<py::gil_scoped_release>()
      )
      .def(
          "compile_builtin_json_grammar",
          &GrammarCompiler::CompileBuiltinJSONGrammar,
          py::call_guard<py::gil_scoped_release>()
      )
      .def(
          "compile_bnf_grammar",
          &GrammarCompiler::CompileBNFGrammar,
          py::call_guard<py::gil_scoped_release>()
      )
      .def("clear_cache", &GrammarCompiler::ClearCache);

  auto pyGrammarMatcher = py::class_<GrammarMatcher>(m, "GrammarMatcher");
  pyGrammarMatcher
      .def(py::init<const CompiledGrammar&, std::optional<std::vector<int>>, bool, int>())
      .def("accept_token", &GrammarMatcher::AcceptToken)
      .def("fill_next_token_bitmask", &GrammarMatcher_FillNextTokenBitmask)
      .def("find_jump_forward_string", &GrammarMatcher::FindJumpForwardString)
      .def("rollback", &GrammarMatcher::Rollback)
      .def("is_terminated", &GrammarMatcher::IsTerminated)
      .def("reset", &GrammarMatcher::Reset)
      .def_property_readonly("max_rollback_tokens", &GrammarMatcher::GetMaxRollbackTokens)
      .def_property_readonly("stop_token_ids", &GrammarMatcher::GetStopTokenIds)
      .def("_debug_accept_string", &GrammarMatcher::_DebugAcceptString);

  auto pyMatcherModule = m.def_submodule("matcher");
#ifdef XGRAMMAR_BUILD_KERNELS
  pyMatcherModule.def("apply_token_bitmask_inplace", &Matcher_ApplyTokenBitmaskInplace);
#endif

  auto pyTestingModule = m.def_submodule("testing");
  pyTestingModule.def("_json_schema_to_ebnf", &_JSONSchemaToEBNF)
      .def("_regex_to_ebnf", &_RegexToEBNF)
      .def("_get_masked_tokens_from_bitmask", &Matcher_DebugGetMaskedTokensFromBitmask);
}
