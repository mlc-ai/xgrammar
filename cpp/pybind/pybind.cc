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
  pyTokenizerInfo.def(py::init<const std::string&>())
      .def("to_string", &TokenizerInfo::ToString)
      .def("get_decoded_vocab", &TokenizerInfo_GetDecodedVocab);

  auto pyGrammarStateMatcher = py::class_<GrammarStateMatcher>(m, "GrammarStateMatcher");
  pyGrammarStateMatcher
      .def(py::init(py::overload_cast<
                    const BNFGrammar&,
                    const std::vector<std::string>&,
                    std::optional<std::vector<int>>,
                    bool,
                    int>(&GrammarStateMatcher_Init)))
      .def(py::init(py::overload_cast<
                    const BNFGrammar&,
                    std::nullptr_t,
                    std::optional<std::vector<int>>,
                    bool,
                    int>(&GrammarStateMatcher_Init)))
      .def("accept_token", &GrammarStateMatcher::AcceptToken)
      .def("_accept_string", &GrammarStateMatcher::_AcceptString)
      .def("find_next_token_bitmask", &GrammarStateMatcher_FindNextTokenBitmask)
      .def_static(
          "get_rejected_tokens_from_bitmask", &GrammarStateMatcher_GetRejectedTokensFromBitMask
      )
      .def("is_terminated", &GrammarStateMatcher::IsTerminated)
      .def("reset", &GrammarStateMatcher::Reset)
      .def("get_vocab_size", &GrammarStateMatcher::GetVocabSize)
      .def("find_jump_forward_string", &GrammarStateMatcher::FindJumpForwardString)
      .def("rollback", &GrammarStateMatcher::Rollback)
      .def("get_max_rollback_steps", &GrammarStateMatcher::GetMaxRollbackSteps);
}
