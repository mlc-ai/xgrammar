#include <dlpack/dlpack.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <future>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "xgrammar/xgrammar.h"

namespace xgrammar {
namespace {

constexpr int kNumThreads = 8;
constexpr int kThreadIterations = 100;

Grammar BasicDynamicTagGrammar() {
  return Grammar::FromEBNF(R"(
    root ::= DynamicTag("<", name, ">", content, "</", ">")
    name ::= [a-z_][a-z0-9_]*
    content ::= [A-Z]*
  )");
}

Grammar UniqueKeyDynamicTagGrammar() {
  return Grammar::FromEBNF(R"(
    root ::= scope
    scope ::= element+
    element ::= DynamicTag(
      "<", name, ">", content, "</", ">",
      unique_key_scope=scope,
      reserved_names=("fixed",)
    )
    name ::= [a-z_][a-z0-9_]*
    content ::= [A-Z]* | scope
  )");
}

CompiledGrammar Compile(
    const Grammar& grammar,
    const TokenizerInfo& tokenizer_info = TokenizerInfo(std::vector<std::string>{})
) {
  return GrammarCompiler(tokenizer_info, 4, false).CompileGrammar(grammar);
}

bool Accepts(const CompiledGrammar& compiled, const std::string& input) {
  GrammarMatcher matcher(compiled, std::nullopt, true);
  return matcher.AcceptString(input) && matcher.IsCompleted();
}

std::vector<int> GetMaskedTokens(
    GrammarMatcher* matcher, const TokenizerInfo& tokenizer_info, int32_t batch_size = 1
) {
  const int32_t bitmask_size = GetBitmaskSize(tokenizer_info.GetVocabSize());
  std::vector<int32_t> bitmask_data(batch_size * bitmask_size);
  int64_t shape[2] = {batch_size, bitmask_size};
  DLTensor bitmask{};
  bitmask.data = bitmask_data.data();
  bitmask.device = DLDevice{kDLCPU, 0};
  bitmask.ndim = 2;
  bitmask.dtype = GetBitmaskDLType();
  bitmask.shape = shape;
  matcher->FillNextTokenBitmask(&bitmask);

  std::vector<int> masked;
  _DebugGetMaskedTokensFromBitmask(&masked, bitmask, tokenizer_info.GetVocabSize());
  return masked;
}

TEST(DynamicTagGrammarTest, MatchesOpeningNameAtTheClosingTag) {
  const auto compiled = Compile(BasicDynamicTagGrammar());

  EXPECT_TRUE(Accepts(compiled, "<key>VALUE</key>"));
  EXPECT_TRUE(Accepts(compiled, "<other_2></other_2>"));
  EXPECT_FALSE(Accepts(compiled, "<key>VALUE</other>"));
  EXPECT_FALSE(Accepts(compiled, "<key>lowercase</key>"));
}

TEST(DynamicTagGrammarTest, AcceptStringPreservesCaptureAcrossChunks) {
  const auto compiled = Compile(BasicDynamicTagGrammar());
  GrammarMatcher matcher(compiled, std::nullopt, true);

  ASSERT_TRUE(matcher.AcceptString("<run"));
  ASSERT_TRUE(matcher.AcceptString("time_1>VALUE</run"));
  EXPECT_FALSE(matcher.AcceptString("wrong>"));
  ASSERT_TRUE(matcher.AcceptString("time_1>"));
  EXPECT_TRUE(matcher.IsCompleted());
}

TEST(DynamicTagGrammarTest, ConstraintIsScopedToTheSelectedUnionBranch) {
  const auto dynamic = BasicDynamicTagGrammar();
  const auto plain = Grammar::FromEBNF(R"(root ::= "plain")");
  const auto compiled = Compile(Grammar::Union({plain, dynamic}));

  EXPECT_TRUE(Accepts(compiled, "plain"));
  EXPECT_TRUE(Accepts(compiled, "<key>VALUE</key>"));
  EXPECT_FALSE(Accepts(compiled, "<key>VALUE</other>"));
}

TEST(DynamicTagGrammarTest, SurvivesGrammarConcatenation) {
  const auto prefix = Grammar::FromEBNF(R"(root ::= "prefix:")");
  const auto suffix = Grammar::FromEBNF(R"(root ::= ":suffix")");
  const auto grammar = Grammar::Concat({prefix, BasicDynamicTagGrammar(), suffix});
  const auto compiled = Compile(grammar);

  EXPECT_TRUE(Accepts(compiled, "prefix:<key>VALUE</key>:suffix"));
  EXPECT_FALSE(Accepts(compiled, "prefix:<key>VALUE</wrong>:suffix"));
}

TEST(DynamicTagGrammarTest, SupportsNestedAndSiblingOccurrences) {
  const auto grammar = Grammar::FromEBNF(R"(
    root ::= element
    element ::= DynamicTag("<", name, ">", content, "</", ">")
    name ::= [a-z]+
    content ::= [A-Z]* | element+
  )");
  const auto compiled = Compile(grammar);

  EXPECT_TRUE(Accepts(compiled, "<outer><left>A</left><right>B</right></outer>"));
  EXPECT_TRUE(Accepts(compiled, "<a><b><c>X</c></b></a>"));
  EXPECT_FALSE(Accepts(compiled, "<outer><inner>X</outer></inner>"));
  EXPECT_FALSE(Accepts(compiled, "<outer><inner>X</wrong></outer>"));
}

TEST(DynamicTagGrammarTest, UniqueKeyScopeRejectsSiblingAndReservedNames) {
  const auto compiled = Compile(UniqueKeyDynamicTagGrammar());

  EXPECT_TRUE(Accepts(compiled, "<a>A</a><b>B</b>"));
  EXPECT_FALSE(Accepts(compiled, "<a>A</a><a>B</a>"));
  EXPECT_FALSE(Accepts(compiled, "<fixed>A</fixed>"));
}

TEST(DynamicTagGrammarTest, UniqueKeyScopeIsOccurrenceLocalForNestedObjects) {
  const auto compiled = Compile(UniqueKeyDynamicTagGrammar());

  EXPECT_TRUE(Accepts(compiled, "<left><x>A</x></left><right><x>B</x></right>"));
  EXPECT_FALSE(Accepts(compiled, "<outer><x>A</x><x>B</x></outer>"));
}

TEST(DynamicTagGrammarTest, UniqueKeyTokenMaskAndRollbackUseMatcherLocalState) {
  const std::vector<std::string> vocab = {"a>", "b>", "fixed>"};
  const TokenizerInfo tokenizer_info(vocab, VocabType::RAW, std::nullopt, std::vector<int32_t>{});
  const auto compiled = Compile(UniqueKeyDynamicTagGrammar(), tokenizer_info);
  GrammarMatcher matcher(compiled, std::nullopt, true);
  ASSERT_TRUE(matcher.AcceptString("<a>A</a><"));

  auto masked = GetMaskedTokens(&matcher, tokenizer_info);
  EXPECT_NE(std::find(masked.begin(), masked.end(), 0), masked.end());
  EXPECT_EQ(std::find(masked.begin(), masked.end(), 1), masked.end());
  EXPECT_NE(std::find(masked.begin(), masked.end(), 2), masked.end());
  EXPECT_FALSE(matcher.AcceptToken(0));

  ASSERT_TRUE(matcher.AcceptToken(1));
  matcher.Rollback();
  masked = GetMaskedTokens(&matcher, tokenizer_info);
  EXPECT_EQ(std::find(masked.begin(), masked.end(), 1), masked.end());
  EXPECT_TRUE(matcher.AcceptToken(1));
}

TEST(DynamicTagGrammarTest, UniqueKeyStateSurvivesAtomicAndBytePathMerge) {
  const auto grammar = Grammar::FromEBNF(R"(
    root ::= "<" Token(0) "atomic" | scope
    scope ::= element+
    element ::= DynamicTag(
      "<", name, ">", content, "</", ">",
      unique_key_scope=scope,
      reserved_names=()
    )
    name ::= [a-z]+
    content ::= [A-Z]*
  )");
  const TokenizerInfo tokenizer_info({"a>"}, VocabType::RAW, std::nullopt, std::vector<int32_t>{});
  const auto compiled = Compile(grammar, tokenizer_info);
  GrammarMatcher matcher(compiled, std::nullopt, true);

  ASSERT_TRUE(matcher.AcceptString("<"));
  ASSERT_TRUE(matcher.AcceptToken(0));
  EXPECT_FALSE(matcher.AcceptString("A</a><a>B</a>"));
}

TEST(DynamicTagGrammarTest, CapturesUtf8NamesByteForByte) {
  const auto grammar = Grammar::FromEBNF(R"(
    root ::= DynamicTag("]<]minimax[>[<", name, ">", content, "]<]minimax[>[</", ">")
    name ::= [^>/]+
    content ::= [^<]*
  )");
  const auto compiled = Compile(grammar);
  const std::string ns = "]<]minimax[>[";

  EXPECT_TRUE(Accepts(compiled, ns + "<城市>HANGZHOU" + ns + "</城市>"));
  EXPECT_FALSE(Accepts(compiled, ns + "<城市>HANGZHOU" + ns + "</天气>"));
}

TEST(DynamicTagGrammarTest, SupportsEmptyFixedDelimiters) {
  const auto grammar = Grammar::FromEBNF(R"(
    root ::= DynamicTag("", name, ":", content, "/", "")
    name ::= [a-z]+
    content ::= [A-Z]*
  )");
  const auto compiled = Compile(grammar);

  EXPECT_TRUE(Accepts(compiled, "key:VALUE/key"));
  EXPECT_FALSE(Accepts(compiled, "key:VALUE/other"));
}

TEST(DynamicTagGrammarTest, RejectsAmbiguousOrEmptyRuntimeNames) {
  const auto grammar = Grammar::FromEBNF(R"(
    root ::= DynamicTag("<", name, ">", content, "</", ">")
    name ::= [^]*
    content ::= [A-Z]*
  )");
  const auto compiled = Compile(grammar);

  EXPECT_TRUE(Accepts(compiled, "<key with space>VALUE</key with space>"));
  EXPECT_FALSE(Accepts(compiled, "<>VALUE</>"));
  EXPECT_FALSE(Accepts(compiled, "< \t>VALUE</ \t>"));
  EXPECT_FALSE(Accepts(compiled, "<a>b>VALUE</a>b>"));
  EXPECT_FALSE(Accepts(compiled, "</bad>VALUE<//bad>"));
}

TEST(DynamicTagGrammarTest, RejectsNameLanguageWithoutAnySafeName) {
  const auto grammar = Grammar::FromEBNF(R"(
    root ::= DynamicTag("<", name, ">", content, "</", ">")
    name ::= "" | [ \t]+
    content ::= "VALUE"
  )");

  EXPECT_ANY_THROW(Compile(grammar));
}

TEST(DynamicTagGrammarTest, MultiByteDelimiterHasAnUnambiguousCaptureBoundary) {
  const auto grammar = Grammar::FromEBNF(R"(
    root ::= DynamicTag("<", name, "::", content, "</", ">")
    name ::= [^]*
    content ::= [A-Z]*
  )");
  const auto compiled = Compile(grammar);

  EXPECT_TRUE(Accepts(compiled, "<a:b::VALUE</a:b>"));
  EXPECT_FALSE(Accepts(compiled, "<a:::VALUE</a:>"));
  EXPECT_FALSE(Accepts(compiled, "<a::b::VALUE</a::b>"));
}

TEST(DynamicTagGrammarTest, BothOpeningAndClosingSuffixesBoundTheName) {
  const auto grammar = Grammar::FromEBNF(R"(
    root ::= DynamicTag("<", name, ":", content, "</", ">")
    name ::= [^]*
    content ::= "VALUE"
  )");
  const auto compiled = Compile(grammar);

  EXPECT_TRUE(Accepts(compiled, "<key:VALUE</key>"));
  EXPECT_FALSE(Accepts(compiled, "<a>b:VALUE</a>b>"));
}

TEST(DynamicTagGrammarTest, MultiByteSuffixOnlyRejectsActualBoundaryOverlap) {
  const auto safe = Grammar::FromEBNF(R"(
    root ::= DynamicTag("<", name, "ab", content, "</", "ab")
    name ::= [^]*
    content ::= "VALUE"
  )");
  EXPECT_TRUE(Accepts(Compile(safe), "<aabVALUE</aab"));

  const auto overlapping = Grammar::FromEBNF(R"(
    root ::= DynamicTag("<", name, "aa", content, "</", "aa")
    name ::= [^]*
    content ::= "VALUE"
  )");
  EXPECT_FALSE(Accepts(Compile(overlapping), "<aaaVALUE</aaa"));
}

TEST(DynamicTagGrammarTest, OpeningAndClosingPrefixesAreDistinguishableSymmetrically) {
  const auto grammar = Grammar::FromEBNF(R"(
    root ::= DynamicTag("</", name, ">", content, "<", ">")
    name ::= [^>]+
    content ::= "VALUE"
  )");
  const auto compiled = Compile(grammar);

  EXPECT_TRUE(Accepts(compiled, "</key>VALUE<key>"));
  EXPECT_FALSE(Accepts(compiled, "<//key>VALUE</key>"));
}

TEST(DynamicTagGrammarTest, DelimiterSafetyMatchesTheDocumentedNameLanguage) {
  const auto grammar = Grammar::FromEBNF(R"(
    root ::= DynamicTag("<", name, "ab", content, "</", ">")
    name ::= [^]*
    content ::= "VALUE"
  )");
  const auto compiled = Compile(grammar);
  const std::string alphabet = "ab/x ";

  std::vector<std::string> names{""};
  for (int length = 1; length <= 4; ++length) {
    const size_t begin = names.size();
    std::vector<std::string> next;
    if (length == 1) {
      next.push_back("");
    } else {
      for (size_t index = begin; index-- > 0;) {
        if (names[index].size() == static_cast<size_t>(length - 1)) {
          next.push_back(names[index]);
        }
      }
    }
    for (const auto& prefix : next) {
      for (char byte : alphabet) {
        names.push_back(prefix + byte);
      }
    }
  }

  for (const std::string& name : names) {
    const bool has_non_whitespace =
        std::any_of(name.begin(), name.end(), [](char byte) { return byte != ' '; });
    const bool expected = !name.empty() && has_non_whitespace && name.front() != '/' &&
                          name.find("ab") == std::string::npos;
    EXPECT_EQ(Accepts(compiled, "<" + name + "abVALUE</" + name + ">"), expected) << name;
  }
}

TEST(DynamicTagGrammarTest, ExpandsLargeRegularNameRepetitions) {
  const auto grammar = Grammar::FromEBNF(R"(
    root ::= DynamicTag("<", name, ">", content, "</", ">")
    name ::= [a-z]{1, 1200}
    content ::= "VALUE"
  )");
  const auto compiled = Compile(grammar);
  const std::string name(1200, 'a');

  EXPECT_TRUE(Accepts(compiled, "<" + name + ">VALUE</" + name + ">"));
  EXPECT_FALSE(Accepts(compiled, "<" + name + ">VALUE</short>"));
}

TEST(DynamicTagGrammarTest, PreservesBoundedRepetitionLanguageForCompoundNames) {
  const auto grammar = Grammar::FromEBNF(R"(
    root ::= DynamicTag("<", name, ">", content, "</", ">")
    name ::= part{2, 4}
    part ::= "a" | "bc"
    content ::= "VALUE"
  )");
  const auto compiled = Compile(grammar);

  EXPECT_TRUE(Accepts(compiled, "<abc>VALUE</abc>"));
  EXPECT_TRUE(Accepts(compiled, "<bcbcbcbc>VALUE</bcbcbcbc>"));
  EXPECT_FALSE(Accepts(compiled, "<a>VALUE</a>"));
  EXPECT_FALSE(Accepts(compiled, "<aaaaa>VALUE</aaaaa>"));
}

TEST(DynamicTagGrammarTest, RejectsNonRegularNameRecursion) {
  const auto grammar = Grammar::FromEBNF(R"(
    root ::= DynamicTag("<", name, ">", content, "</", ">")
    name ::= "x" | "a" name "b"
    content ::= "VALUE"
  )");

  EXPECT_ANY_THROW(Compile(grammar));
}

TEST(DynamicTagGrammarTest, RejectsRuntimeMetadataOnNameRules) {
  const auto grammar = Grammar::FromEBNF(R"(
    root ::= DynamicTag("<", name, ">", content, "</", ">")
    name[max_tokens=1, max_chars=2, lazy, capture="name", temperature=0.5] ::= [a-z]+
    content ::= "VALUE"
  )");

  EXPECT_ANY_THROW(Compile(grammar));
}

TEST(DynamicTagGrammarTest, TokenMaskHandlesTokensCrossingEveryBoundary) {
  const std::vector<std::string> vocab = {
      "<key>VALUE</key>", "<key>VALUE</wrong>", "<other></other>"
  };
  const TokenizerInfo tokenizer_info(vocab, VocabType::RAW, std::nullopt, std::vector<int32_t>{});
  const auto compiled = Compile(BasicDynamicTagGrammar(), tokenizer_info);
  GrammarMatcher matcher(compiled, std::nullopt, true);

  const auto masked = GetMaskedTokens(&matcher, tokenizer_info);
  EXPECT_EQ(std::find(masked.begin(), masked.end(), 0), masked.end());
  EXPECT_NE(std::find(masked.begin(), masked.end(), 1), masked.end());
  EXPECT_EQ(std::find(masked.begin(), masked.end(), 2), masked.end());
  EXPECT_TRUE(matcher.AcceptToken(0));
  EXPECT_TRUE(matcher.IsCompleted());
}

TEST(DynamicTagGrammarTest, TokenMaskCrossesAnOrdinaryParentRule) {
  const auto grammar = Grammar::FromEBNF(R"(
    root ::= "prefix:" element ":suffix"
    element ::= DynamicTag("<", name, ">", content, "</", ">")
    name ::= [a-z_][a-z0-9_]*
    content ::= [A-Z]*
  )");
  const std::vector<std::string> vocab = {
      "prefix:<key>VALUE</key>:suffix", "prefix:<key>VALUE</wrong>:suffix"
  };
  const TokenizerInfo tokenizer_info(vocab, VocabType::RAW, std::nullopt, std::vector<int32_t>{});
  const auto compiled = Compile(grammar, tokenizer_info);
  GrammarMatcher matcher(compiled, std::nullopt, true);

  const auto masked = GetMaskedTokens(&matcher, tokenizer_info);
  EXPECT_EQ(std::find(masked.begin(), masked.end(), 0), masked.end());
  EXPECT_NE(std::find(masked.begin(), masked.end(), 1), masked.end());
  EXPECT_TRUE(matcher.AcceptToken(0));
  EXPECT_TRUE(matcher.IsCompleted());
}

TEST(DynamicTagGrammarTest, CaptureStartSurvivesAtomicAndBytePathMerge) {
  const auto grammar = Grammar::FromEBNF(R"(
    root ::= Token(0) element | "<trigger>NO"
    element ::= DynamicTag("", name, ":", content, "/", "")
    name ::= [a-z]+
    content ::= [A-Z]*
  )");
  const TokenizerInfo tokenizer_info(
      {"<trigger>"}, VocabType::RAW, std::nullopt, std::vector<int32_t>{}
  );
  const auto compiled = Compile(grammar, tokenizer_info);

  GrammarMatcher matching(compiled, std::nullopt, true);
  ASSERT_TRUE(matching.AcceptToken(0));
  EXPECT_TRUE(matching.AcceptString("key:VALUE/key"));
  EXPECT_TRUE(matching.IsCompleted());

  GrammarMatcher mismatching(compiled, std::nullopt, true);
  ASSERT_TRUE(mismatching.AcceptToken(0));
  EXPECT_FALSE(mismatching.AcceptString("key:VALUE/wrong"));
}

TEST(DynamicTagGrammarTest, TokenMaskKeepsDynamicEqualityBranchLocalAcrossUnion) {
  const auto grammar =
      Grammar::Union({Grammar::FromEBNF(R"(root ::= "plain")"), BasicDynamicTagGrammar()});
  const std::vector<std::string> vocab = {"plain", "<key>VALUE</key>", "<key>VALUE</wrong>"};
  const TokenizerInfo tokenizer_info(vocab, VocabType::RAW, std::nullopt, std::vector<int32_t>{});
  const auto compiled = Compile(grammar, tokenizer_info);
  GrammarMatcher matcher(compiled, std::nullopt, true);

  const auto masked = GetMaskedTokens(&matcher, tokenizer_info);
  EXPECT_EQ(std::find(masked.begin(), masked.end(), 0), masked.end());
  EXPECT_EQ(std::find(masked.begin(), masked.end(), 1), masked.end());
  EXPECT_NE(std::find(masked.begin(), masked.end(), 2), masked.end());
}

TEST(DynamicTagGrammarTest, TokenMaskTracksNestedCapturesInsideOneToken) {
  const auto grammar = Grammar::FromEBNF(R"(
    root ::= element
    element ::= DynamicTag("<", name, ">", content, "</", ">")
    name ::= [a-z]+
    content ::= [A-Z]* | element+
  )");
  const std::vector<std::string> vocab = {
      "<outer><left>A</left><right>B</right></outer>",
      "<outer><left>A</wrong><right>B</right></outer>",
      "<outer><left>A</left><right>B</right></wrong>"
  };
  const TokenizerInfo tokenizer_info(vocab, VocabType::RAW, std::nullopt, std::vector<int32_t>{});
  const auto compiled = Compile(grammar, tokenizer_info);
  GrammarMatcher matcher(compiled, std::nullopt, true);

  const auto masked = GetMaskedTokens(&matcher, tokenizer_info);
  EXPECT_EQ(std::find(masked.begin(), masked.end(), 0), masked.end());
  EXPECT_NE(std::find(masked.begin(), masked.end(), 1), masked.end());
  EXPECT_NE(std::find(masked.begin(), masked.end(), 2), masked.end());
}

TEST(DynamicTagGrammarTest, TokenMaskUsesTheOccurrenceLocalCapturedName) {
  const std::vector<std::string> vocab = {"alpha>", "beta>", "wrong>"};
  const TokenizerInfo tokenizer_info(vocab, VocabType::RAW, std::nullopt, std::vector<int32_t>{});
  const auto compiled = Compile(BasicDynamicTagGrammar(), tokenizer_info);
  GrammarMatcher matcher(compiled, std::nullopt, true);
  ASSERT_TRUE(matcher.AcceptString("<alpha>VALUE</"));

  const auto masked = GetMaskedTokens(&matcher, tokenizer_info);
  EXPECT_EQ(std::find(masked.begin(), masked.end(), 0), masked.end());
  EXPECT_NE(std::find(masked.begin(), masked.end(), 1), masked.end());
  EXPECT_NE(std::find(masked.begin(), masked.end(), 2), masked.end());
}

TEST(DynamicTagGrammarTest, BackReferenceMaskSlicesTheCompleteByteAlphabetSafely) {
  std::vector<std::string> vocab;
  vocab.reserve(258);
  for (int32_t byte = 0; byte < 256; ++byte) {
    vocab.push_back(std::string(1, static_cast<char>(byte)) + "suffix");
  }
  const int32_t matching_token = vocab.size();
  vocab.push_back("x>");
  const int32_t wrong_same_prefix_token = vocab.size();
  vocab.push_back("xwrong>");

  const TokenizerInfo tokenizer_info(vocab, VocabType::RAW, std::nullopt, std::vector<int32_t>{});
  const auto compiled = Compile(BasicDynamicTagGrammar(), tokenizer_info);
  GrammarMatcher matcher(compiled, std::nullopt, true);
  ASSERT_TRUE(matcher.AcceptString("<x>VALUE</"));

  const auto masked = GetMaskedTokens(&matcher, tokenizer_info);
  EXPECT_EQ(std::find(masked.begin(), masked.end(), matching_token), masked.end());
  EXPECT_NE(std::find(masked.begin(), masked.end(), wrong_same_prefix_token), masked.end());
  for (int32_t token_id = 0; token_id < 256; ++token_id) {
    EXPECT_NE(std::find(masked.begin(), masked.end(), token_id), masked.end()) << token_id;
  }
}

TEST(DynamicTagGrammarTest, BackReferenceMaskSlicesUtf8FirstBytesSafely) {
  const std::vector<std::string> vocab = {"城市>", "天气>", "城wrong>", "x>"};
  const TokenizerInfo tokenizer_info(vocab, VocabType::RAW, std::nullopt, std::vector<int32_t>{});
  const auto grammar = Grammar::FromEBNF(R"(
    root ::= DynamicTag("<", name, ">", content, "</", ">")
    name ::= [^>/]+
    content ::= "VALUE"
  )");
  const auto compiled = Compile(grammar, tokenizer_info);
  GrammarMatcher matcher(compiled, std::nullopt, true);
  ASSERT_TRUE(matcher.AcceptString("<城市>VALUE</"));

  const auto masked = GetMaskedTokens(&matcher, tokenizer_info);
  EXPECT_EQ(std::find(masked.begin(), masked.end(), 0), masked.end());
  EXPECT_NE(std::find(masked.begin(), masked.end(), 1), masked.end());
  EXPECT_NE(std::find(masked.begin(), masked.end(), 2), masked.end());
  EXPECT_NE(std::find(masked.begin(), masked.end(), 3), masked.end());
}

TEST(DynamicTagGrammarTest, BackReferenceFastPathNeverAdmitsStopOrSpecialTokensEarly) {
  const std::vector<std::string> vocab = {"key>", "<stop>", ""};
  const TokenizerInfo tokenizer_info(vocab, VocabType::RAW, std::nullopt, std::vector<int32_t>{1});
  const auto compiled = Compile(BasicDynamicTagGrammar(), tokenizer_info);
  GrammarMatcher matcher(compiled, std::nullopt, false);
  ASSERT_TRUE(matcher.AcceptString("<key>VALUE</"));

  const auto masked = GetMaskedTokens(&matcher, tokenizer_info);
  EXPECT_EQ(std::find(masked.begin(), masked.end(), 0), masked.end());
  EXPECT_NE(std::find(masked.begin(), masked.end(), 1), masked.end());
  EXPECT_NE(std::find(masked.begin(), masked.end(), 2), masked.end());
}

TEST(DynamicTagGrammarTest, RejectsTokenizerDependentNameGrammar) {
  const auto grammar = Grammar::FromEBNF(R"(
    root ::= DynamicTag("<", name, ">", content, "</", ">")
    name ::= Token(0) | [a-z]+
    content ::= "VALUE"
  )");
  const TokenizerInfo tokenizer_info(
      {"alpha"}, VocabType::RAW, std::nullopt, std::vector<int32_t>{}
  );
  EXPECT_ANY_THROW(Compile(grammar, tokenizer_info));
}

TEST(DynamicTagGrammarTest, ForkAndRollbackKeepIndependentCaptureState) {
  const std::vector<std::string> vocab = {"alpha>", "beta>"};
  const TokenizerInfo tokenizer_info(vocab, VocabType::RAW, std::nullopt, std::vector<int32_t>{});
  const auto compiled = Compile(BasicDynamicTagGrammar(), tokenizer_info);
  GrammarMatcher original(compiled, std::nullopt, true);
  ASSERT_TRUE(original.AcceptString("<alpha>VALUE</"));

  auto fork = original.Fork();
  EXPECT_TRUE(fork.AcceptToken(0));
  EXPECT_TRUE(fork.IsCompleted());
  fork.Rollback();
  EXPECT_TRUE(fork.AcceptToken(0));
  EXPECT_FALSE(original.AcceptToken(1));
  EXPECT_TRUE(original.AcceptToken(0));
}

TEST(DynamicTagGrammarTest, GrammarAndCompiledGrammarRoundTrip) {
  const auto grammar = BasicDynamicTagGrammar();
  const std::string printed = grammar.ToString();
  EXPECT_NE(printed.find("DynamicTag("), std::string::npos);
  const auto reparsed = Grammar::FromEBNF(printed);
  EXPECT_TRUE(Accepts(Compile(reparsed), "<key>VALUE</key>"));
  EXPECT_FALSE(Accepts(Compile(reparsed), "<key>VALUE</wrong>"));

  auto grammar_result = Grammar::DeserializeJSON(grammar.SerializeJSON());
  ASSERT_TRUE(std::holds_alternative<Grammar>(grammar_result));
  const auto recovered_grammar = std::get<Grammar>(grammar_result);
  EXPECT_EQ(recovered_grammar.SerializeJSON(), grammar.SerializeJSON());

  const TokenizerInfo tokenizer_info(
      {"key>", "wrong>"}, VocabType::RAW, std::nullopt, std::vector<int32_t>{}
  );
  const auto compiled = Compile(recovered_grammar, tokenizer_info);
  auto compiled_result = CompiledGrammar::DeserializeJSON(compiled.SerializeJSON(), tokenizer_info);
  ASSERT_TRUE(std::holds_alternative<CompiledGrammar>(compiled_result));
  const auto recovered_compiled = std::get<CompiledGrammar>(compiled_result);
  EXPECT_EQ(recovered_compiled.SerializeJSON(), compiled.SerializeJSON());

  GrammarMatcher matcher(recovered_compiled, std::nullopt, true);
  ASSERT_TRUE(matcher.AcceptString("<key>VALUE</"));
  const auto masked = GetMaskedTokens(&matcher, tokenizer_info);
  EXPECT_EQ(std::find(masked.begin(), masked.end(), 0), masked.end());
  EXPECT_NE(std::find(masked.begin(), masked.end(), 1), masked.end());
}

TEST(DynamicTagGrammarTest, SharedCompiledGrammarIsThreadSafe) {
  const std::vector<std::string> vocab = {"alpha>", "beta>", "wrong>"};
  const TokenizerInfo tokenizer_info(vocab, VocabType::RAW, std::nullopt, std::vector<int32_t>{});
  const auto compiled = Compile(BasicDynamicTagGrammar(), tokenizer_info);
  std::vector<std::future<bool>> futures;
  futures.reserve(kNumThreads);

  for (int thread_id = 0; thread_id < kNumThreads; ++thread_id) {
    futures.push_back(std::async(std::launch::async, [compiled, thread_id] {
      const std::string name = thread_id % 2 == 0 ? "alpha" : "beta";
      for (int iteration = 0; iteration < kThreadIterations; ++iteration) {
        GrammarMatcher matcher(compiled, std::nullopt, true);
        if (!matcher.AcceptString("<" + name + ">VALUE</" + name + ">") || !matcher.IsCompleted()) {
          return false;
        }
        GrammarMatcher invalid(compiled, std::nullopt, true);
        if (invalid.AcceptString("<" + name + ">VALUE</wrong>")) {
          return false;
        }
      }
      return true;
    }));
  }

  for (auto& future : futures) {
    EXPECT_TRUE(future.get());
  }
}

TEST(DynamicTagGrammarTest, UniqueKeyStateIsThreadLocal) {
  const auto compiled = Compile(UniqueKeyDynamicTagGrammar());
  std::vector<std::future<bool>> futures;
  futures.reserve(kNumThreads);

  for (int thread_id = 0; thread_id < kNumThreads; ++thread_id) {
    futures.push_back(std::async(std::launch::async, [compiled, thread_id] {
      const std::string first = thread_id % 2 == 0 ? "alpha" : "beta";
      for (int iteration = 0; iteration < kThreadIterations; ++iteration) {
        GrammarMatcher valid(compiled, std::nullopt, true);
        if (!valid.AcceptString("<" + first + ">A</" + first + "><other>B</other>") ||
            !valid.IsCompleted()) {
          return false;
        }
        GrammarMatcher duplicate(compiled, std::nullopt, true);
        if (duplicate.AcceptString(
                "<" + first + ">A</" + first + "><" + first + ">B</" + first + ">"
            )) {
          return false;
        }
      }
      return true;
    }));
  }

  for (auto& future : futures) {
    EXPECT_TRUE(future.get());
  }
}

TEST(DynamicTagGrammarTest, BatchMaskingUsesOnlyMatcherLocalCaptureState) {
  const std::vector<std::string> vocab = {"alpha>", "beta>", "wrong>"};
  const TokenizerInfo tokenizer_info(vocab, VocabType::RAW, std::nullopt, std::vector<int32_t>{});
  const auto compiled = Compile(BasicDynamicTagGrammar(), tokenizer_info);
  std::vector<GrammarMatcher> matchers;
  for (int index = 0; index < 16; ++index) {
    const std::string name = index % 2 == 0 ? "alpha" : "beta";
    GrammarMatcher matcher(compiled, std::nullopt, true);
    EXPECT_TRUE(matcher.AcceptString("<" + name + ">VALUE</"));
    matchers.push_back(std::move(matcher));
  }

  const int32_t bitmask_size = GetBitmaskSize(tokenizer_info.GetVocabSize());
  std::vector<int32_t> bitmask_data(matchers.size() * bitmask_size);
  int64_t shape[2] = {static_cast<int64_t>(matchers.size()), bitmask_size};
  DLTensor bitmask{};
  bitmask.data = bitmask_data.data();
  bitmask.device = DLDevice{kDLCPU, 0};
  bitmask.ndim = 2;
  bitmask.dtype = GetBitmaskDLType();
  bitmask.shape = shape;

  BatchGrammarMatcher batch_matcher(4);
  batch_matcher.BatchFillNextTokenBitmask(&matchers, &bitmask);
  for (int index = 0; index < static_cast<int>(matchers.size()); ++index) {
    std::vector<int> masked;
    _DebugGetMaskedTokensFromBitmask(&masked, bitmask, tokenizer_info.GetVocabSize(), index);
    const int expected = index % 2;
    EXPECT_EQ(std::find(masked.begin(), masked.end(), expected), masked.end());
    EXPECT_NE(std::find(masked.begin(), masked.end(), 1 - expected), masked.end());
    EXPECT_NE(std::find(masked.begin(), masked.end(), 2), masked.end());
  }
}

}  // namespace
}  // namespace xgrammar
