/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/compiled_grammar.cc
 */

#include <xgrammar/compiler.h>

#include "compiled_grammar_impl.h"

namespace xgrammar {

std::string CompiledGrammar::SerializeJSON() const {
  return AutoSerializeJSON(*this->ImplPtr(), true);
}

std::variant<CompiledGrammar, std::runtime_error> CompiledGrammar::DeserializeJSON(
    const std::string& json_string
) {
  CompiledGrammar compiled_grammar(std::make_shared<CompiledGrammar::Impl>());
  auto err = AutoDeserializeJSON(compiled_grammar, json_string, true);
  if (err) {
    return err.value();
  }
  return compiled_grammar;
}

}  // namespace xgrammar
