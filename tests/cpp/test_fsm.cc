/**
 * \file tests/cpp/test_fsm.cc
 * \brief Test FSM operations.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <utility>

#include "fsm.h"
#include "fsm_builder.h"
#include "support/logging.h"

using namespace xgrammar;

TEST(XGrammarFSMTest, BasicBuildTest) {
  std::cout << "--------- Basic Build Test Starts! -----------" << std::endl;
  std::cout << "--------- Basic Build Test1 -----------" << std::endl;
  auto fsm_wse = RegexFSMBuilder::Build("abcd\\n").Unwrap();
  std::string test_str = "abcd\n";
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  std::cout << "--------- Basic Build Test2 -----------" << std::endl;
  fsm_wse = RegexFSMBuilder::Build("[-a-z\\n]").Unwrap();
  test_str = "abcd-\n";
  for (const auto& character : test_str) {
    EXPECT_TRUE([&]() -> bool {
      for (const auto& edge : fsm_wse.GetFsm().GetEdges(0)) {
        if (edge.min <= int(character) && edge.max >= int(character)) {
          return true;
        }
      }
      return false;
    }());
  }
  std::cout << "--------- Basic Build Test3 -----------" << std::endl;
  fsm_wse = RegexFSMBuilder::Build("[\\d]").Unwrap();
  test_str = "1234567890";
  for (const auto& character : test_str) {
    EXPECT_TRUE([&]() -> bool {
      for (const auto& edge : fsm_wse.GetFsm().GetEdges(0)) {
        if (edge.min <= int(character) && edge.max >= int(character)) {
          return true;
        }
      }
      return false;
    }());
  }
  std::cout << "--------- Basic Build Test4 -----------" << std::endl;
  fsm_wse = RegexFSMBuilder::Build("[^\\d]").Unwrap();
  test_str = "1234567890";
  for (const auto& character : test_str) {
    EXPECT_TRUE([&]() -> bool {
      for (const auto& edge : fsm_wse.GetFsm().GetEdges(0)) {
        if (edge.min <= int(character) && edge.max >= int(character)) {
          return false;
        }
      }
      return true;
    }());
  }
  test_str = "abz";
  for (const auto& character : test_str) {
    EXPECT_TRUE([&]() -> bool {
      for (const auto& edge : fsm_wse.GetFsm().GetEdges(0)) {
        if (edge.min <= int(character) && edge.max >= int(character)) {
          return true;
        }
      }
      std::cout << character << std::endl;
      return false;
    }());
  }
  std::cout << "--------- Basic Build Test5 -----------" << std::endl;
  fsm_wse = RegexFSMBuilder::Build("你好a").Unwrap();
  test_str = "你好a";
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  std::cout << "--------- Basic Build Test6 -----------" << std::endl;
  fsm_wse = RegexFSMBuilder::Build("(())()()").Unwrap();
  test_str = "";
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  std::cout << "--------- Basic Build Test7 -----------" << std::endl;
  fsm_wse = RegexFSMBuilder::Build("[abcdabcdxyzxyz]").Unwrap();
  test_str = "a";
  std::cout << fsm_wse << std::endl;
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  EXPECT_FALSE(fsm_wse.AcceptString("e"));
  std::cout << fsm_wse << std::endl;
  EXPECT_EQ(fsm_wse.GetFsm().GetEdges(0).size(), 2);
  std::cout << "Basic Build Test Passed!" << std::endl;
}

TEST(XGrammarFSMTest, ConnectionTest) {
  std::cout << "--------- Connection Test Starts! -----------" << std::endl;
  std::cout << "--------- Connection Test1 -----------" << std::endl;
  auto fsm_wse = RegexFSMBuilder::Build(" [a-zA-Z0-9]--").Unwrap();
  std::string test_str = " a--";
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  std::cout << "--------- Connection Test2 -----------" << std::endl;
  fsm_wse = RegexFSMBuilder::Build("aaa|[\\d]").Unwrap();
  test_str = "aaa";
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  test_str = "1";
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  std::cout << "--------- Connection Test3 -----------" << std::endl;
  auto result = RegexFSMBuilder::Build("(([\\d]|[\\w])|aaa)");
  EXPECT_FALSE(result.IsErr()) << std::move(result).UnwrapErr().what();
  fsm_wse = std::move(result).Unwrap();
  test_str = "aaa";
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  test_str = "1";
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  test_str = "1a";
  EXPECT_FALSE(fsm_wse.AcceptString(test_str));
  std::cout << "Connection Test Passed!" << std::endl;
}

TEST(XGrammarFSMTest, SymbolTest) {
  std::cout << "--------- Symbol Test Starts! -----------" << std::endl;
  std::cout << "--------- Symbol Test1 -----------" << std::endl;
  auto fsm_wse = RegexFSMBuilder::Build("1[\\d]+").Unwrap();
  std::string test_str[2] = {"1111", "1"};
  EXPECT_TRUE(fsm_wse.AcceptString(test_str[0]));
  EXPECT_FALSE(fsm_wse.AcceptString(test_str[1]));
  std::cout << "--------- Symbol Test2 -----------" << std::endl;
  fsm_wse = RegexFSMBuilder::Build("1[1]*").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString(test_str[0]));
  EXPECT_TRUE(fsm_wse.AcceptString(test_str[1]));
  std::cout << "--------- Symbol Test3 -----------" << std::endl;
  fsm_wse = RegexFSMBuilder::Build("1[\\d]?").Unwrap();
  EXPECT_FALSE(fsm_wse.AcceptString(test_str[0]));
  EXPECT_TRUE(fsm_wse.AcceptString(test_str[1]));
  std::string test3 = "11";
  EXPECT_TRUE(fsm_wse.AcceptString(test3));
  std::cout << "--------- Symbol Test4 -----------" << std::endl;
  fsm_wse = RegexFSMBuilder::Build(" * * + ? *").Unwrap();
  test_str[0] = " ";
  test_str[1] = "      ";
  for (const auto& str : test_str) {
    EXPECT_TRUE(fsm_wse.AcceptString(str));
  }
  std::cout << "Symbol Test Passed!" << std::endl;
}

TEST(XGrammarFSMTest, IntegratedTest) {
  std::cout << "--------- Integrated Test Starts! -----------" << std::endl;
  auto fsm_wse = RegexFSMBuilder::Build("((naive|bbb|[\\d]+)*[\\w])|  +").Unwrap();
  std::string test_str[5] = {"naive1", "bbbnaive114514W", "    ", "123", "_"};
  for (const auto& str : test_str) {
    EXPECT_TRUE(fsm_wse.AcceptString(str));
  }
  std::string test_str2[5] = {"naive", "bbbbbb", "naive   ", "123 ", "aaa"};
  for (const auto& str : test_str2) {
    EXPECT_FALSE(fsm_wse.AcceptString(str));
  }
  std::cout << "--------- Integrated Test Passed! -----------" << std::endl;
}

TEST(XGrammarFSMTest, FunctionTest) {
  std::cout << "--------- Function Test Starts! -----------" << std::endl;
  std::cout << "--------- Function Test1 -----------" << std::endl;
  auto fsm_wse = RegexFSMBuilder::Build("[\\d\\d\\d]+123").Unwrap();
  std::string test_str = "123456123";
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  auto compact_fsm = fsm_wse.GetFsm().ToCompact();
  CompactFSMWithStartEnd compact_fsm_wse(compact_fsm, fsm_wse.GetStart(), fsm_wse.GetEnds());
  EXPECT_TRUE(compact_fsm_wse.AcceptString(test_str));
  fsm_wse = FSMWithStartEnd(compact_fsm.ToFSM(), fsm_wse.GetStart(), fsm_wse.GetEnds());
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  std::cout << "--------- Function Test2 -----------" << std::endl;
  fsm_wse = RegexFSMBuilder::Build("([abc]|[\\d])+").Unwrap();
  test_str = "abc3";
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  fsm_wse = std::move(fsm_wse.ToDFA()).Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  EXPECT_TRUE([&]() -> bool {
    for (const auto& edges : fsm_wse.GetFsm().GetEdges()) {
      for (const auto& edge : edges) {
        if (edge.IsEpsilon()) {
          return false;
        }
      }
    }
    return true;
  }());
  EXPECT_TRUE([&]() -> bool {
    for (const auto& edges : fsm_wse.GetFsm().GetEdges()) {
      std::unordered_set<int> rules;
      std::unordered_set<int> chars;
      for (const auto& edge : edges) {
        if (edge.IsRuleRef()) {
          if (rules.find(edge.GetRefRuleId()) != rules.end()) {
            return false;
          }
          rules.insert(edge.GetRefRuleId());
          continue;
        }
        for (int i = edge.min; i <= edge.max; i++) {
          if (chars.find(i) != chars.end()) {
            return false;
          }
          chars.insert(i);
        }
      }
    }
    return true;
  }());
  std::cout << "--------- Function Test3 -----------" << std::endl;
  fsm_wse = std::move(fsm_wse.MinimizeDFA()).Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  EXPECT_EQ(fsm_wse.GetFsm().GetEdges().size(), 2);
  std::cout << "--------- Function Test4 -----------" << std::endl;
  fsm_wse = std::move(fsm_wse.Not()).Unwrap();
  EXPECT_FALSE(fsm_wse.AcceptString(test_str));
  test_str = "abcd";
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  std::cout << "--------- Function Test5 -----------" << std::endl;
  fsm_wse = RegexFSMBuilder::Build("[\\d]{1,5}").Unwrap();
  std::string test_strs[2] = {"123", "12345"};
  for (const auto& str : test_strs) {
    EXPECT_TRUE(fsm_wse.AcceptString(str));
  }
  test_strs[0] = "123456";
  test_strs[1] = "1234567";
  for (const auto& str : test_strs) {
    EXPECT_FALSE(fsm_wse.AcceptString(str));
  }
  fsm_wse = RegexFSMBuilder::Build("[\\d]{6}").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("123456"));
  EXPECT_FALSE(fsm_wse.AcceptString("1234567"));
  fsm_wse = RegexFSMBuilder::Build("[\\d]{6, }").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("123456"));
  EXPECT_TRUE(fsm_wse.AcceptString("1234567"));
  std::cout << "--------- Function Test6 -----------" << std::endl;
  fsm_wse = RegexFSMBuilder::Build("[a][b][c][d]").Unwrap();
  test_str = "abcd";
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  fsm_wse = fsm_wse.SimplifyEpsilon();
  std::cout << fsm_wse << std::endl;
  EXPECT_EQ(fsm_wse.GetFsm().NumStates(), 5);
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  std::cout << "--------- Function Test7 -----------" << std::endl;
  fsm_wse = RegexFSMBuilder::Build("abc|abd").Unwrap();
  test_str = "abc";
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  fsm_wse = fsm_wse.SimplifyEpsilon();
  fsm_wse = fsm_wse.MergeEquivalentSuccessors();
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  test_str = "abcd";
  EXPECT_FALSE(fsm_wse.AcceptString(test_str));
  EXPECT_EQ(fsm_wse.GetFsm().NumStates(), 4);
  std::cout << "--------- Function Test8 -----------" << std::endl;
  fsm_wse = RegexFSMBuilder::Build("acd|bcd").Unwrap();
  test_str = "acd";
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  fsm_wse = fsm_wse.SimplifyEpsilon();
  fsm_wse = fsm_wse.MergeEquivalentSuccessors();
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  test_str = "abcd";
  EXPECT_FALSE(fsm_wse.AcceptString(test_str));
  EXPECT_EQ(fsm_wse.GetFsm().NumStates(), 4);
  XGRAMMAR_LOG(INFO) << fsm_wse;
  std::cout << "--------- Function Test9 -----------" << std::endl;
  fsm_wse = RegexFSMBuilder::Build("ab*").Unwrap();
  test_str = "abbb";
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  fsm_wse = fsm_wse.SimplifyEpsilon();
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  EXPECT_EQ(fsm_wse.GetFsm().NumStates(), 2);
  std::cout << "--------- Function Test10 -----------" << std::endl;
  const auto fsm_left = RegexFSMBuilder::Build("[c-f]+").Unwrap();
  const auto fsm_right = RegexFSMBuilder::Build("[d-h]*").Unwrap();
  std::cout << fsm_left << std::endl;
  std::cout << fsm_right << std::endl;
  fsm_wse = FSMWithStartEnd::Intersect(fsm_left, fsm_right).Unwrap();
  std::cout << fsm_wse << std::endl;
  EXPECT_TRUE(fsm_wse.AcceptString("de"));
  EXPECT_TRUE(fsm_wse.AcceptString("def"));
  EXPECT_FALSE(fsm_wse.AcceptString(""));
  EXPECT_FALSE(fsm_wse.AcceptString("cd"));
  std::cout << "--------- Function Test Passed! -----------" << std::endl;
}

TEST(XGrammarFSMTest, EfficiencyTest) {
  std::cout << "--------- Efficiency Test Starts! -----------" << std::endl;
  // i.e ([a-z]0123456789){10}. Use this way to test the performance.
  auto fsm_wse = RegexFSMBuilder::Build(
                     "(a0123456789|a0123456789|b0123456789|b0123456789|c0123456789|"
                     "c0123456789|d0123456789|d0123456789|e0123456789|e0123456789|"
                     "f0123456789|f0123456789|g0123456789|g0123456789|h0123456789|"
                     "h0123456789|i0123456789|i0123456789|j0123456789|j0123456789|"
                     "k0123456789|k0123456789|l0123456789|l0123456789|m0123456789|"
                     "m0123456789|n0123456789|n0123456789|o0123456789|o0123456789|"
                     "p0123456789|p0123456789|q0123456789|q0123456789|r0123456789|"
                     "r0123456789|s0123456789|s0123456789|t0123456789|t0123456789|"
                     "u0123456789|u0123456789|v0123456789|v0123456789|w0123456789|"
                     "w0123456789|x0123456789|x0123456789|y0123456789|y0123456789|"
                     "z0123456789|z0123456789)(a0123456789|a0123456789|b0123456789|"
                     "b0123456789|c0123456789|c0123456789|d0123456789|d0123456789|"
                     "e0123456789|e0123456789|f0123456789|f0123456789|g0123456789|"
                     "g0123456789|h0123456789|h0123456789|i0123456789|i0123456789|"
                     "j0123456789|j0123456789|k0123456789|k0123456789|l0123456789|"
                     "l0123456789|m0123456789|m0123456789|n0123456789|n0123456789|"
                     "o0123456789|o0123456789|p0123456789|p0123456789|q0123456789|"
                     "q0123456789|r0123456789|r0123456789|s0123456789|s0123456789|"
                     "t0123456789|t0123456789|u0123456789|u0123456789|v0123456789|"
                     "v0123456789|w0123456789|w0123456789|x0123456789|x0123456789|"
                     "y0123456789|y0123456789|z0123456789|z0123456789)(a0123456789|"
                     "a0123456789|b0123456789|b0123456789|c0123456789|c0123456789|"
                     "d0123456789|d0123456789|e0123456789|e0123456789|f0123456789|"
                     "f0123456789|g0123456789|g0123456789|h0123456789|h0123456789|"
                     "i0123456789|i0123456789|j0123456789|j0123456789|k0123456789|"
                     "k0123456789|l0123456789|l0123456789|m0123456789|m0123456789|"
                     "n0123456789|n0123456789|o0123456789|o0123456789|p0123456789|"
                     "p0123456789|q0123456789|q0123456789|r0123456789|r0123456789|"
                     "s0123456789|s0123456789|t0123456789|t0123456789|u0123456789|"
                     "u0123456789|v0123456789|v0123456789|w0123456789|w0123456789|"
                     "x0123456789|x0123456789|y0123456789|y0123456789|z0123456789|"
                     "z0123456789)(a0123456789|a0123456789|b0123456789|b0123456789|"
                     "c0123456789|c0123456789|d0123456789|d0123456789|e0123456789|"
                     "e0123456789|f0123456789|f0123456789|g0123456789|g0123456789|"
                     "h0123456789|h0123456789|i0123456789|i0123456789|j0123456789|"
                     "j0123456789|k0123456789|k0123456789|l0123456789|l0123456789|"
                     "m0123456789|m0123456789|n0123456789|n0123456789|o0123456789|"
                     "o0123456789|p0123456789|p0123456789|q0123456789|q0123456789|"
                     "r0123456789|r0123456789|s0123456789|s0123456789|t0123456789|"
                     "t0123456789|u0123456789|u0123456789|v0123456789|v0123456789|"
                     "w0123456789|w0123456789|x0123456789|x0123456789|y0123456789|"
                     "y0123456789|z0123456789|z0123456789)(a0123456789|a0123456789|"
                     "b0123456789|b0123456789|c0123456789|c0123456789|d0123456789|"
                     "d0123456789|e0123456789|e0123456789|f0123456789|f0123456789|"
                     "g0123456789|g0123456789|h0123456789|h0123456789|i0123456789|"
                     "i0123456789|j0123456789|j0123456789|k0123456789|k0123456789|"
                     "l0123456789|l0123456789|m0123456789|m0123456789|n0123456789|"
                     "n0123456789|o0123456789|o0123456789|p0123456789|p0123456789|"
                     "q0123456789|q0123456789|r0123456789|r0123456789|s0123456789|"
                     "s0123456789|t0123456789|t0123456789|u0123456789|u0123456789|"
                     "v0123456789|v0123456789|w0123456789|w0123456789|x0123456789|"
                     "x0123456789|y0123456789|y0123456789|z0123456789|z0123456789)("
                     "a0123456789|a0123456789|b0123456789|b0123456789|c0123456789|"
                     "c0123456789|d0123456789|d0123456789|e0123456789|e0123456789|"
                     "f0123456789|f0123456789|g0123456789|g0123456789|h0123456789|"
                     "h0123456789|i0123456789|i0123456789|j0123456789|j0123456789|"
                     "k0123456789|k0123456789|l0123456789|l0123456789|m0123456789|"
                     "m0123456789|n0123456789|n0123456789|o0123456789|o0123456789|"
                     "p0123456789|p0123456789|q0123456789|q0123456789|r0123456789|"
                     "r0123456789|s0123456789|s0123456789|t0123456789|t0123456789|"
                     "u0123456789|u0123456789|v0123456789|v0123456789|w0123456789|"
                     "w0123456789|x0123456789|x0123456789|y0123456789|y0123456789|"
                     "z0123456789|z0123456789)(a0123456789|a0123456789|b0123456789|"
                     "b0123456789|c0123456789|c0123456789|d0123456789|d0123456789|"
                     "e0123456789|e0123456789|f0123456789|f0123456789|g0123456789|"
                     "g0123456789|h0123456789|h0123456789|i0123456789|i0123456789|"
                     "j0123456789|j0123456789|k0123456789|k0123456789|l0123456789|"
                     "l0123456789|m0123456789|m0123456789|n0123456789|n0123456789|"
                     "o0123456789|o0123456789|p0123456789|p0123456789|q0123456789|"
                     "q0123456789|r0123456789|r0123456789|s0123456789|s0123456789|"
                     "t0123456789|t0123456789|u0123456789|u0123456789|v0123456789|"
                     "v0123456789|w0123456789|w0123456789|x0123456789|x0123456789|"
                     "y0123456789|y0123456789|z0123456789|z0123456789)(a0123456789|"
                     "a0123456789|b0123456789|b0123456789|c0123456789|c0123456789|"
                     "d0123456789|d0123456789|e0123456789|e0123456789|f0123456789|"
                     "f0123456789|g0123456789|g0123456789|h0123456789|h0123456789|"
                     "i0123456789|i0123456789|j0123456789|j0123456789|k0123456789|"
                     "k0123456789|l0123456789|l0123456789|m0123456789|m0123456789|"
                     "n0123456789|n0123456789|o0123456789|o0123456789|p0123456789|"
                     "p0123456789|q0123456789|q0123456789|r0123456789|r0123456789|"
                     "s0123456789|s0123456789|t0123456789|t0123456789|u0123456789|"
                     "u0123456789|v0123456789|v0123456789|w0123456789|w0123456789|"
                     "x0123456789|x0123456789|y0123456789|y0123456789|z0123456789|"
                     "z0123456789)(a0123456789|a0123456789|b0123456789|b0123456789|"
                     "c0123456789|c0123456789|d0123456789|d0123456789|e0123456789|"
                     "e0123456789|f0123456789|f0123456789|g0123456789|g0123456789|"
                     "h0123456789|h0123456789|i0123456789|i0123456789|j0123456789|"
                     "j0123456789|k0123456789|k0123456789|l0123456789|l0123456789|"
                     "m0123456789|m0123456789|n0123456789|n0123456789|o0123456789|"
                     "o0123456789|p0123456789|p0123456789|q0123456789|q0123456789|"
                     "r0123456789|r0123456789|s0123456789|s0123456789|t0123456789|"
                     "t0123456789|u0123456789|u0123456789|v0123456789|v0123456789|"
                     "w0123456789|w0123456789|x0123456789|x0123456789|y0123456789|"
                     "y0123456789|z0123456789|z0123456789)(a0123456789|a0123456789|"
                     "b0123456789|b0123456789|c0123456789|c0123456789|d0123456789|"
                     "d0123456789|e0123456789|e0123456789|f0123456789|f0123456789|"
                     "g0123456789|g0123456789|h0123456789|h0123456789|i0123456789|"
                     "i0123456789|j0123456789|j0123456789|k0123456789|k0123456789|"
                     "l0123456789|l0123456789|m0123456789|m0123456789|n0123456789|"
                     "n0123456789|o0123456789|o0123456789|p0123456789|p0123456789|"
                     "q0123456789|q0123456789|r0123456789|r0123456789|s0123456789|"
                     "s0123456789|t0123456789|t0123456789|u0123456789|u0123456789|"
                     "v0123456789|v0123456789|w0123456789|w0123456789|x0123456789|"
                     "x0123456789|y0123456789|y0123456789|z0123456789|z0123456789)"
  )
                     .Unwrap();
  std::cout << "Initial Node Numbers:" << fsm_wse.GetFsm().NumStates() << std::endl;
  auto time_start = std::chrono::high_resolution_clock::now();
  fsm_wse = fsm_wse.SimplifyEpsilon();
  auto time_end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
  std::cout << "Time taken to simplify epsilon: " << duration.count() << " ms" << std::endl;
  std::cout << "After SimplifyEpsilon Node Numbers:" << fsm_wse.GetFsm().NumStates() << std::endl;
  time_start = std::chrono::high_resolution_clock::now();
  fsm_wse = fsm_wse.MergeEquivalentSuccessors();
  time_end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
  std::cout << "Time taken to simplify transition: " << duration.count() << " ms" << std::endl;
  std::cout << "After SimplifyTransition Node Numbers:" << fsm_wse.GetFsm().NumStates()
            << std::endl;
  time_start = std::chrono::high_resolution_clock::now();
  fsm_wse = std::move(fsm_wse.ToDFA()).Unwrap();
  time_end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
  std::cout << "Time taken to convert to DFA: " << duration.count() << " ms" << std::endl;
  std::cout << "After ToDFA Node Numbers:" << fsm_wse.GetFsm().NumStates() << std::endl;
  time_start = std::chrono::high_resolution_clock::now();
  fsm_wse = std::move(fsm_wse.MinimizeDFA()).Unwrap();
  time_end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
  std::cout << "Time taken to minimize DFA: " << duration.count() << " ms" << std::endl;
  EXPECT_EQ(fsm_wse.GetFsm().NumStates(), 111);
  std::cout << "--------- Efficiency Test Passed! -----------" << std::endl;
}

TEST(XGrammarFSMTest, TestEmail) {
  std::string email_pattern = R"((\w+)(\.\w+)*@(\w+)(\.\w+)+)";
  auto fsm_wse = RegexFSMBuilder::Build(email_pattern).Unwrap();
  std::string valid_emails[5] = {
      "asnjdaj_19032910@google.com.test",
      "12393089340190@a.b.c.d.f.e.org.test",
      "as____________as@abc.me.test",
      "ooooohhhhh@123456.test",
      "ajidoa@a.test"
  };
  for (const auto& email : valid_emails) {
    EXPECT_TRUE(fsm_wse.AcceptString(email)) << "Failed for email: " << email;
  }

  std::string invalid_emails[5] = {
      "@google.test", "hello@", "hello@.test", "+++asd@b.test", "hello"
  };
  for (const auto& email : invalid_emails) {
    EXPECT_FALSE(fsm_wse.AcceptString(email)) << "Failed for email: " << email;
  }
}

TEST(XGrammarFSMTest, TestTime) {
  std::string time_pattern = R"((\d{1,2}):(\d{2})(:(\d{2}))?)";
  auto fsm_wse = RegexFSMBuilder::Build(time_pattern).Unwrap();
  std::string valid_times[5] = {"1:34", "23:59", "00:00", "01:02:03", "23:59:59"};
  for (const auto& time : valid_times) {
    EXPECT_TRUE(fsm_wse.AcceptString(time)) << "Failed for time: " << time;
  }
  std::string invalid_times[9] = {
      "19", "12:6", "12:34:", "12:34:5", "12:34:567", "12:123", "12:", ":34:23", "::"
  };
  for (const auto& time : invalid_times) {
    EXPECT_FALSE(fsm_wse.AcceptString(time)) << "Failed for time: " << time;
  }
}

TEST(XGrammarFSMTest, MergingNodesTest) {
  FSMWithStartEnd fsm_wse;
  for (int i = 0; i < 10; i++) {
    fsm_wse.AddState();
  }
  fsm_wse.SetStartState(0);
  fsm_wse.AddEndState(9);
  fsm_wse.GetFsm().AddEdge(0, 1, 'a', 'a');
  fsm_wse.GetFsm().AddEdge(0, 2, 'a', 'a');
  fsm_wse.GetFsm().AddEdge(1, 3, 'b', 'b');
  fsm_wse.GetFsm().AddEdge(1, 3, 'c', 'c');
  fsm_wse.GetFsm().AddEdge(1, 4, 'b', 'b');
  fsm_wse.GetFsm().AddEdge(1, 4, 'c', 'c');
  fsm_wse.GetFsm().AddEdge(2, 5, 'b', 'b');
  fsm_wse.GetFsm().AddEdge(2, 5, 'c', 'c');
  fsm_wse.GetFsm().AddEdge(2, 6, 'b', 'b');
  fsm_wse.GetFsm().AddEdge(2, 6, 'c', 'c');
  fsm_wse.GetFsm().AddEdge(3, 7, 'd', 'd');
  fsm_wse.GetFsm().AddEdge(4, 7, 'd', 'd');
  fsm_wse.GetFsm().AddEdge(5, 8, 'd', 'd');
  fsm_wse.GetFsm().AddEdge(6, 8, 'd', 'd');
  fsm_wse.GetFsm().AddEdge(7, 9, 'e', 'e');
  fsm_wse.GetFsm().AddEdge(8, 9, 'e', 'e');
  fsm_wse = fsm_wse.MergeEquivalentSuccessors();
  std::string expected_fsm = R"(FSM(num_states=5, start=3, end=[4], edges=[
0: ['d'->2]
1: ['b'->0, 'c'->0]
2: ['e'->4]
3: ['a'->1]
4: []
]))";
  EXPECT_EQ(fsm_wse.ToString(), expected_fsm);
  EXPECT_EQ(fsm_wse.GetFsm().NumStates(), 5);
}

TEST(XGrammarFSMTest, EpsilonSimplificationTest) {
  FSMWithStartEnd fsm_wse;
  for (int i = 0; i < 10; i++) {
    fsm_wse.AddState();
  }
  fsm_wse.SetStartState(0);
  fsm_wse.AddEndState(9);
  fsm_wse.GetFsm().AddEpsilonEdge(0, 1);
  fsm_wse.GetFsm().AddEpsilonEdge(0, 2);
  fsm_wse.GetFsm().AddEdge(1, 3, 'b', 'b');
  fsm_wse.GetFsm().AddEpsilonEdge(1, 3);
  fsm_wse.GetFsm().AddEdge(1, 4, 'b', 'b');
  fsm_wse.GetFsm().AddEdge(3, 3, 'c', 'c');
  fsm_wse.GetFsm().AddEpsilonEdge(2, 5);
  fsm_wse.GetFsm().AddEdge(2, 5, 'c', 'c');
  fsm_wse.GetFsm().AddEdge(2, 6, 'b', 'b');
  fsm_wse.GetFsm().AddEdge(2, 6, 'c', 'c');
  fsm_wse.GetFsm().AddEpsilonEdge(3, 7);
  fsm_wse.GetFsm().AddEpsilonEdge(4, 7);
  fsm_wse.GetFsm().AddEpsilonEdge(5, 8);
  fsm_wse.GetFsm().AddEpsilonEdge(6, 8);
  fsm_wse.GetFsm().AddEpsilonEdge(7, 9);
  fsm_wse.GetFsm().AddEpsilonEdge(8, 9);
  fsm_wse = fsm_wse.SimplifyEpsilon();
  std::string expected_fsm = R"(FSM(num_states=3, start=0, end=[1], edges=[
0: [Eps->1, Eps->2, 'b'->1, 'b'->2, 'c'->1]
1: []
2: [Eps->1, 'c'->2]
]))";
  EXPECT_EQ(fsm_wse.ToString(), expected_fsm);
  EXPECT_EQ(fsm_wse.GetFsm().NumStates(), 3);
}
