/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/pybind.cc
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xgrammar/xgrammar.h>

#include "../json_schema_converter.h"
#include "../regex_converter.h"
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

  auto pyGrammar = py::class_<Grammar>(m, "Grammar");
  pyGrammar.def("to_string", &Grammar::ToString)
      .def_static("from_ebnf", &Grammar::FromEBNF)
      .def_static("from_json_schema", &Grammar::FromJSONSchema)
      .def_static("builtin_json_grammar", &Grammar::BuiltinJSONGrammar);

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
          "compile_grammar",
          &GrammarCompiler::CompileGrammar,
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

  auto pyTestingModule = m.def_submodule("testing");
  pyTestingModule
      .def(
          "_json_schema_to_ebnf",
          py::overload_cast<
              const std::string&,
              bool,
              std::optional<int>,
              std::optional<std::pair<std::string, std::string>>,
              bool>(&JSONSchemaToEBNF)
      )
      .def("_regex_to_ebnf", &RegexToEBNF)
      .def("_get_masked_tokens_from_bitmask", &Matcher_DebugGetMaskedTokensFromBitmask);
}
