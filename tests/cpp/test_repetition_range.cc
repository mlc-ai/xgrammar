/*!
 *  Copyright (c) 2026 by Contributors
 * \file tests/cpp/test_repetition_range.cc
 * \brief Regression test for RepetitionRangeExpander segfault when
 *        grammar_expr_id (a builder ID) is looked up in base_grammar_.
 */

#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include "grammar_functor.h"

using namespace xgrammar;

// HandleRepetitionRange looks up grammar_expr_id (a builder ID) in
// base_grammar_ instead of builder_. The first expansion inflates the
// builder's ID space; the second expansion's grammar_expr_id then
// exceeds base_grammar_'s expression count, causing an OOB / segfault.
TEST(RepetitionRangeExpanderTest, UnboundedRepetitionAboveThresholdDoesNotCrash) {
  // Two unbounded repeats with lower > kUnzipThreshold (128) in the
  // same rule. The first expand creates many builder expressions; the
  // second's grammar_expr_id lands out of bounds in base_grammar_.
  auto grammar = Grammar::FromEBNF(R"(root ::= [a-z]{129,} [0-9]{129,})");
  EXPECT_NO_THROW(RepetitionRangeExpander::Apply(grammar));
}
