/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_builder.h
 * \brief The header for the building the BNF AST.
 */
#include <xgrammar/grammar.h>

namespace xgrammar {
Grammar Grammar::Empty() { return Grammar::FromEBNF("root ::= \"\""); }
}  // namespace xgrammar
