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
      .def_static("_json_schema_to_ebnf", &BuiltinGrammar::_JSONSchemaToEBNF);

  auto pyTokenizerInfo = py::class_<TokenizerInfo>(m, "TokenizerInfo");
  pyTokenizerInfo.def(py::init(&TokenizerInfo_Init))
      .def_property_readonly("vocab_size", &TokenizerInfo::GetVocabSize)
      .def_property_readonly("vocab_type", &TokenizerInfo_GetVocabType)
      .def_property_readonly(
          "prepend_space_in_tokenization", &TokenizerInfo::GetPrependSpaceInTokenization
      )
      .def_property_readonly("raw_vocab", &TokenizerInfo_GetRawVocab)
      .def("dump_metadata", &TokenizerInfo::DumpMetadata)
      .def_static("from_huggingface", &TokenizerInfo::FromHuggingFace)
      .def_static("from_vocab_and_metadata", &TokenizerInfo::FromVocabAndMetadata);

  auto pyGrammarMatcherInitContext =
      py::class_<GrammarMatcherInitContext>(m, "GrammarMatcherInitContext");
  pyGrammarMatcherInitContext.def(py::init<const BNFGrammar&, const std::vector<std::string>&>())
      .def(py::init<const BNFGrammar&, const TokenizerInfo&>());

  auto pyGrammarMatcherInitContextCache =
      py::class_<GrammarMatcherInitContextCache>(m, "GrammarMatcherInitContextCache");
  pyGrammarMatcherInitContextCache.def(py::init<const TokenizerInfo&>())
      .def("get_init_context_for_json", &GrammarMatcherInitContextCache::GetInitContextForJSON)
      .def(
          "get_init_context_for_json_schema",
          &GrammarMatcherInitContextCache::GetInitContextForJSONSchema
      );

  auto pyGrammarMatcher = py::class_<GrammarMatcher>(m, "GrammarMatcher");
  pyGrammarMatcher
      .def(py::init<
           const GrammarMatcherInitContext&,
           std::optional<std::vector<int>>,
           bool,
           std::optional<int>,
           int>())
      .def("accept_token", &GrammarMatcher::AcceptToken)
      .def("accept_string", &GrammarMatcher::AcceptString)
      .def("find_next_token_bitmask", &GrammarMatcher_FindNextTokenBitmask)
      .def_static("get_rejected_tokens_from_bitmask", &GrammarMatcher_GetRejectedTokensFromBitMask)
      .def("is_terminated", &GrammarMatcher::IsTerminated)
      .def("reset", &GrammarMatcher::Reset)
      .def_property_readonly("mask_vocab_size", &GrammarMatcher::GetMaskVocabSize)
      .def("find_jump_forward_string", &GrammarMatcher::FindJumpForwardString)
      .def("rollback", &GrammarMatcher::Rollback)
      .def_property_readonly("max_rollback_tokens", &GrammarMatcher::GetMaxRollbackTokens);
}
