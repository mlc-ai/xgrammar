/**
 * \file tests/cpp/test_fsm.cc
 * \brief Test FSM operations.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

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
  fsm_wse = fsm_wse.MergeEquivalentStates();
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  test_str = "abcd";
  EXPECT_FALSE(fsm_wse.AcceptString(test_str));
  EXPECT_EQ(fsm_wse.GetFsm().NumStates(), 4);
  std::cout << "--------- Function Test8 -----------" << std::endl;
  fsm_wse = RegexFSMBuilder::Build("acd|bcd").Unwrap();
  test_str = "acd";
  EXPECT_TRUE(fsm_wse.AcceptString(test_str));
  fsm_wse = fsm_wse.SimplifyEpsilon();
  fsm_wse = fsm_wse.MergeEquivalentStates();
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
  fsm_wse = fsm_wse.MergeEquivalentStates();
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
  fsm_wse = fsm_wse.MergeEquivalentStates();
  std::string expected_fsm = R"(FSM(num_states=5, start=0, end=[4], edges=[
0: ['a'->1]
1: ['b'->2, 'c'->2]
2: ['d'->3]
3: ['e'->4]
4: []
]))";
  EXPECT_EQ(fsm_wse.ToString(), expected_fsm);
  EXPECT_EQ(fsm_wse.GetFsm().NumStates(), 5);
}

TEST(XGrammarFSMTest, MergeEquivalentStatesNoCrossRuleChaining) {
  FSMWithStartEnd fsm_wse;
  for (int i = 0; i < 7; ++i) {
    fsm_wse.AddState();
  }
  fsm_wse.SetStartState(0);
  fsm_wse.AddEndState(6);

  // 2 and 3 are equivalent successors of 0 under 'x' (Case 1).
  fsm_wse.GetFsm().AddEdge(0, 2, 'x', 'x');
  fsm_wse.GetFsm().AddEdge(0, 3, 'x', 'x');
  // 1 is another predecessor of 4 under 'a' (Case 2 candidate with 2).
  fsm_wse.GetFsm().AddEdge(0, 1, 'y', 'y');

  fsm_wse.GetFsm().AddEdge(1, 4, 'a', 'a');
  fsm_wse.GetFsm().AddEdge(2, 4, 'a', 'a');
  fsm_wse.GetFsm().AddEdge(3, 5, 'b', 'b');
  fsm_wse.GetFsm().AddEdge(4, 6, 'm', 'm');
  fsm_wse.GetFsm().AddEdge(5, 6, 'n', 'n');

  auto merged = fsm_wse.MergeEquivalentStates();

  // Still accepts original strings.
  EXPECT_TRUE(merged.AcceptString("xam"));
  EXPECT_TRUE(merged.AcceptString("xbn"));
  EXPECT_TRUE(merged.AcceptString("yam"));
  // Should not over-merge and introduce this path.
  EXPECT_FALSE(merged.AcceptString("ybn"));
}

TEST(XGrammarFSMTest, MergeEquivalentStatesHandlesEmptySmallAndStateLimit) {
  FSMWithStartEnd empty_fsm(FSM(0), 0, {});
  auto merged_empty_fsm = empty_fsm.MergeEquivalentStates();
  EXPECT_EQ(merged_empty_fsm.NumStates(), 0);
  EXPECT_EQ(merged_empty_fsm.GetStart(), 0);
  EXPECT_TRUE(merged_empty_fsm.GetEnds().empty());

  FSMWithStartEnd small_fsm(FSM(3), 0, {2});
  small_fsm.GetFsm().AddEdge(0, 1, 'a', 'a');
  small_fsm.GetFsm().AddEdge(1, 2, 'b', 'b');
  EXPECT_EQ(small_fsm.MergeEquivalentStates().ToString(), small_fsm.ToString());

  FSMWithStartEnd limited_fsm(FSM(4), 0, {1, 2});
  limited_fsm.GetFsm().AddEdge(0, 1, 'a', 'a');
  limited_fsm.GetFsm().AddEdge(0, 2, 'a', 'a');
  EXPECT_EQ(limited_fsm.MergeEquivalentStates(3).NumStates(), 4);
  EXPECT_EQ(limited_fsm.MergeEquivalentStates().NumStates(), 3);
}

TEST(XGrammarFSMTest, MergeEquivalentStatesMergesLeafStatesByEndStatus) {
  FSMWithStartEnd fsm;
  int start_state = fsm.AddState();
  fsm.SetStartState(start_state);
  for (int state = 0; state < 4; ++state) {
    fsm.AddState();
  }
  fsm.AddEndState(1);
  fsm.AddEndState(2);
  fsm.GetFsm().AddEdge(start_state, 1, 'a', 'a');
  fsm.GetFsm().AddEdge(start_state, 2, 'a', 'a');
  fsm.GetFsm().AddEdge(start_state, 3, 'b', 'b');
  fsm.GetFsm().AddEdge(start_state, 4, 'c', 'c');

  auto merged = fsm.MergeEquivalentStates();
  EXPECT_EQ(merged.GetFsm().NumStates(), 3);
  EXPECT_TRUE(merged.AcceptString("a"));
  EXPECT_FALSE(merged.AcceptString("b"));
  EXPECT_FALSE(merged.AcceptString("c"));
  EXPECT_EQ(merged.MergeEquivalentStates().ToString(), merged.ToString());
}

TEST(XGrammarFSMTest, MergeEquivalentStatesDeepCommonPrefix) {
  constexpr int kNumberOfCopies = 64;
  const std::string accepted_string = "abcdefghijklmnopqrstuvwx";

  FSMWithStartEnd fsm;
  int start_state = fsm.AddState();
  fsm.SetStartState(start_state);
  for (int copy = 0; copy < kNumberOfCopies; ++copy) {
    int current_state = start_state;
    for (char character : accepted_string) {
      int next_state = fsm.AddState();
      fsm.GetFsm().AddEdge(current_state, next_state, character, character);
      current_state = next_state;
    }
    fsm.AddEndState(current_state);
  }

  auto merged = fsm.MergeEquivalentStates();
  EXPECT_EQ(merged.GetFsm().NumStates(), accepted_string.size() + 1);
  EXPECT_TRUE(merged.AcceptString(accepted_string));
  EXPECT_FALSE(merged.AcceptString(accepted_string.substr(0, accepted_string.size() - 1)));
  EXPECT_EQ(merged.MergeEquivalentStates().ToString(), merged.ToString());
}

TEST(XGrammarFSMTest, MergeEquivalentStatesDeepCommonSuffix) {
  constexpr int kNumberOfBranches = 64;
  const std::string common_suffix = "shared_suffix";

  FSMWithStartEnd fsm;
  int start_state = fsm.AddState();
  fsm.SetStartState(start_state);
  for (int branch = 0; branch < kNumberOfBranches; ++branch) {
    int current_state = start_state;
    int first_character = '!' + branch;
    int next_state = fsm.AddState();
    fsm.GetFsm().AddEdge(current_state, next_state, first_character, first_character);
    current_state = next_state;
    for (char character : common_suffix) {
      next_state = fsm.AddState();
      fsm.GetFsm().AddEdge(current_state, next_state, character, character);
      current_state = next_state;
    }
    fsm.AddEndState(current_state);
  }

  auto merged = fsm.MergeEquivalentStates();
  EXPECT_EQ(merged.GetFsm().NumStates(), common_suffix.size() + 2);
  for (int branch = 0; branch < kNumberOfBranches; ++branch) {
    EXPECT_TRUE(merged.AcceptString(std::string(1, static_cast<char>('!' + branch)) + common_suffix)
    );
  }
  EXPECT_FALSE(merged.AcceptString(std::string(1, 'a') + common_suffix.substr(1)));
  EXPECT_EQ(merged.MergeEquivalentStates().ToString(), merged.ToString());
}

TEST(XGrammarFSMTest, MergeEquivalentStatesHandlesCyclesAndDifferentEndStatus) {
  FSMWithStartEnd cyclic_fsm;
  for (int state = 0; state < 5; ++state) {
    cyclic_fsm.AddState();
  }
  cyclic_fsm.SetStartState(0);
  cyclic_fsm.AddEndState(3);
  cyclic_fsm.AddEndState(4);
  cyclic_fsm.GetFsm().AddEdge(0, 1, 'a', 'a');
  cyclic_fsm.GetFsm().AddEdge(0, 2, 'a', 'a');
  cyclic_fsm.GetFsm().AddEdge(1, 1, 'b', 'b');
  cyclic_fsm.GetFsm().AddEdge(2, 2, 'b', 'b');
  cyclic_fsm.GetFsm().AddEdge(1, 3, 'c', 'c');
  cyclic_fsm.GetFsm().AddEdge(2, 4, 'd', 'd');

  auto merged_cycle = cyclic_fsm.MergeEquivalentStates();
  EXPECT_EQ(merged_cycle.GetFsm().NumStates(), 4);
  EXPECT_TRUE(merged_cycle.AcceptString("ac"));
  EXPECT_TRUE(merged_cycle.AcceptString("abbbc"));
  EXPECT_TRUE(merged_cycle.AcceptString("ad"));
  EXPECT_TRUE(merged_cycle.AcceptString("abbbd"));
  EXPECT_FALSE(merged_cycle.AcceptString("ab"));

  FSMWithStartEnd different_end_status_fsm;
  for (int state = 0; state < 4; ++state) {
    different_end_status_fsm.AddState();
  }
  different_end_status_fsm.SetStartState(0);
  different_end_status_fsm.AddEndState(1);
  different_end_status_fsm.AddEndState(3);
  different_end_status_fsm.GetFsm().AddEdge(0, 1, 'a', 'a');
  different_end_status_fsm.GetFsm().AddEdge(0, 2, 'a', 'a');
  different_end_status_fsm.GetFsm().AddEdge(2, 3, 'b', 'b');

  auto merged_end_status = different_end_status_fsm.MergeEquivalentStates();
  EXPECT_EQ(merged_end_status.GetFsm().NumStates(), 3);
  EXPECT_TRUE(merged_end_status.AcceptString("a"));
  EXPECT_TRUE(merged_end_status.AcceptString("ab"));
}

TEST(XGrammarFSMTest, MergeEquivalentStatesHandlesMultipleAndSpecialEdges) {
  FSMWithStartEnd multiple_edges_fsm;
  for (int state = 0; state < 4; ++state) {
    multiple_edges_fsm.AddState();
  }
  multiple_edges_fsm.SetStartState(0);
  multiple_edges_fsm.AddEndState(1);
  multiple_edges_fsm.AddEndState(2);
  multiple_edges_fsm.GetFsm().AddEdge(0, 1, 'a', 'c');
  multiple_edges_fsm.GetFsm().AddEdge(0, 1, 'x', 'z');
  multiple_edges_fsm.GetFsm().AddEdge(0, 2, 'a', 'c');
  multiple_edges_fsm.GetFsm().AddEdge(0, 2, 'x', 'z');
  multiple_edges_fsm.GetFsm().AddEdge(3, 3, 'q', 'q');

  auto merged_ranges = multiple_edges_fsm.MergeEquivalentStates();
  EXPECT_EQ(merged_ranges.GetFsm().NumStates(), 3);
  EXPECT_TRUE(merged_ranges.AcceptString("b"));
  EXPECT_TRUE(merged_ranges.AcceptString("y"));
  EXPECT_FALSE(merged_ranges.AcceptString("q"));

  FSMWithStartEnd special_edges_fsm;
  for (int state = 0; state < 5; ++state) {
    special_edges_fsm.AddState();
  }
  special_edges_fsm.SetStartState(0);
  special_edges_fsm.AddEndState(3);
  special_edges_fsm.AddEndState(4);
  special_edges_fsm.GetFsm().AddRuleEdge(0, 1, 7);
  special_edges_fsm.GetFsm().AddRuleEdge(0, 2, 7);
  special_edges_fsm.GetFsm().AddEOSEdge(1, 3);
  special_edges_fsm.GetFsm().AddEOSEdge(2, 4);

  auto merged_special_edges = special_edges_fsm.MergeEquivalentStates();
  EXPECT_EQ(merged_special_edges.GetFsm().NumStates(), 3);
  EXPECT_EQ(
      merged_special_edges.ToString(),
      R"(FSM(num_states=3, start=0, end=[2], edges=[
0: [Rule(7)->1]
1: [EOS->2]
2: []
]))"
  );
  EXPECT_EQ(
      merged_special_edges.MergeEquivalentStates().ToString(), merged_special_edges.ToString()
  );
}

TEST(XGrammarFSMTest, MergeEquivalentStatesCanonicalizesDuplicateAndSelfEdges) {
  FSMWithStartEnd fsm(FSM(5), 0, {3, 4});
  fsm.GetFsm().AddEdge(0, 1, 'a', 'a');
  fsm.GetFsm().AddEdge(0, 1, 'a', 'a');
  fsm.GetFsm().AddEdge(0, 2, 'a', 'a');
  fsm.GetFsm().AddEpsilonEdge(1, 1);
  fsm.GetFsm().AddEpsilonEdge(2, 2);
  fsm.GetFsm().AddEdge(1, 3, 'b', 'b');
  fsm.GetFsm().AddEdge(2, 4, 'b', 'b');

  auto merged = fsm.MergeEquivalentStates();
  EXPECT_EQ(
      merged.ToString(),
      R"(FSM(num_states=3, start=0, end=[2], edges=[
0: ['a'->1]
1: ['b'->2]
2: []
]))"
  );
  EXPECT_TRUE(merged.AcceptString("ab"));
  EXPECT_FALSE(merged.AcceptString("a"));
  EXPECT_EQ(merged.MergeEquivalentStates().ToString(), merged.ToString());
}

TEST(XGrammarFSMTest, MergeEquivalentStatesDoesNotChainLeafAndSuccessorMerges) {
  // 3 and 4 are equivalent successors of 2 (Case 1, merge by equal reaching paths). 1 and 3 are
  // both non-accepting leaves (merge by equal continuations). Chaining both groups through the
  // shared state 3 would give the class {1, 3, 4}, and the dead leaf 1 would wrongly gain the
  // continuation 'w' of state 4.
  FSMWithStartEnd fsm(FSM(6), 0, {5});
  fsm.GetFsm().AddEdge(0, 1, 'y', 'y');
  fsm.GetFsm().AddEdge(0, 2, 'z', 'z');
  fsm.GetFsm().AddEdge(2, 3, 'x', 'x');
  fsm.GetFsm().AddEdge(2, 4, 'x', 'x');
  fsm.GetFsm().AddEdge(4, 5, 'w', 'w');

  auto merged = fsm.MergeEquivalentStates();
  EXPECT_TRUE(merged.AcceptString("zxw"));
  EXPECT_FALSE(merged.AcceptString("yw"));
  EXPECT_FALSE(merged.AcceptString("y"));
}

TEST(XGrammarFSMTest, MergeEquivalentStatesDoesNotMergeStartByReachingPaths) {
  // 0 and 1 look like equivalent successors of 1 (single predecessor 1, equal incoming labels).
  // But 0 is the start state and is also reached "for free" by the empty string, so merging it
  // with 1 would wrongly accept "c".
  FSMWithStartEnd fsm(FSM(3), 0, {2});
  fsm.GetFsm().AddEdge(1, 0, 'a', 'a');
  fsm.GetFsm().AddEdge(1, 1, 'a', 'a');
  fsm.GetFsm().AddEdge(1, 2, 'c', 'c');

  auto merged = fsm.MergeEquivalentStates();
  EXPECT_FALSE(merged.AcceptString(""));
  EXPECT_FALSE(merged.AcceptString("c"));
  EXPECT_EQ(merged.GetFsm().NumStates(), 3);
}

TEST(XGrammarFSMTest, MergeEquivalentStatesRandomizedPreservesLanguage) {
  std::vector<std::string> test_strings = {""};
  std::vector<std::string> strings_at_current_length = {""};
  for (int length = 1; length <= 4; ++length) {
    std::vector<std::string> strings_at_next_length;
    for (const auto& prefix : strings_at_current_length) {
      for (char character : std::string("abc")) {
        strings_at_next_length.push_back(prefix + character);
      }
    }
    test_strings.insert(
        test_strings.end(), strings_at_next_length.begin(), strings_at_next_length.end()
    );
    strings_at_current_length = std::move(strings_at_next_length);
  }

  std::mt19937 random_generator(20260728);
  for (int test_case = 0; test_case < 300; ++test_case) {
    SCOPED_TRACE(test_case);
    int number_of_states = 4 + random_generator() % 9;
    FSMWithStartEnd fsm;
    for (int state = 0; state < number_of_states; ++state) {
      fsm.AddState();
    }
    fsm.SetStartState(random_generator() % number_of_states);
    for (int state = 0; state < number_of_states; ++state) {
      if (random_generator() % 4 == 0) {
        fsm.AddEndState(state);
      }
    }
    for (int source = 0; source < number_of_states; ++source) {
      int number_of_edges = random_generator() % 5;
      for (int edge_index = 0; edge_index < number_of_edges; ++edge_index) {
        int target = random_generator() % number_of_states;
        if (random_generator() % 8 == 0) {
          fsm.GetFsm().AddEpsilonEdge(source, target);
        } else {
          int character = 'a' + random_generator() % 3;
          fsm.GetFsm().AddEdge(source, target, character, character);
        }
      }
      if (random_generator() % 10 == 0 && !fsm.GetFsm().GetEdges(source).empty()) {
        const auto duplicate_edge = fsm.GetFsm().GetEdges(source).front();
        fsm.GetFsm().AddEdge(source, duplicate_edge.target, duplicate_edge.min, duplicate_edge.max);
      }
    }

    std::vector<bool> accepted_before;
    accepted_before.reserve(test_strings.size());
    for (const auto& test_string : test_strings) {
      accepted_before.push_back(fsm.AcceptString(test_string));
    }

    auto merged = fsm.MergeEquivalentStates();
    EXPECT_LE(merged.GetFsm().NumStates(), fsm.GetFsm().NumStates());
    for (size_t index = 0; index < test_strings.size(); ++index) {
      EXPECT_EQ(merged.AcceptString(test_strings[index]), accepted_before[index])
          << "String: " << test_strings[index];
    }
    EXPECT_EQ(merged.MergeEquivalentStates().ToString(), merged.ToString());
  }
}

TEST(XGrammarFSMTest, SimplifyEpsilonPreservesAcceptance) {
  // (ab)+ : the accepting state's only edge is an epsilon back to the non-accepting start.
  // Merging the two states would wrongly make the start state accepting.
  auto fsm_wse = RegexFSMBuilder::Build("(ab)+").Unwrap();
  fsm_wse = fsm_wse.SimplifyEpsilon();
  EXPECT_FALSE(fsm_wse.AcceptString(""));
  EXPECT_FALSE(fsm_wse.AcceptString("a"));
  EXPECT_TRUE(fsm_wse.AcceptString("ab"));
  EXPECT_TRUE(fsm_wse.AcceptString("abab"));

  fsm_wse = RegexFSMBuilder::Build("a+").Unwrap();
  fsm_wse = fsm_wse.SimplifyEpsilon();
  EXPECT_FALSE(fsm_wse.AcceptString(""));
  EXPECT_TRUE(fsm_wse.AcceptString("a"));
  EXPECT_TRUE(fsm_wse.AcceptString("aaa"));

  // Hand-built: accepting state 1 has a single epsilon edge to non-accepting state 2, and
  // state 2 is also reachable via another path (0 -b-> 2). Merging 1 and 2 would wrongly
  // accept "b". The language is {"a", "ac", "bc"}.
  FSMWithStartEnd manual_fsm;
  for (int i = 0; i < 4; i++) {
    manual_fsm.AddState();
  }
  manual_fsm.SetStartState(0);
  manual_fsm.AddEndState(1);
  manual_fsm.AddEndState(3);
  manual_fsm.GetFsm().AddEdge(0, 1, 'a', 'a');
  manual_fsm.GetFsm().AddEpsilonEdge(1, 2);
  manual_fsm.GetFsm().AddEdge(0, 2, 'b', 'b');
  manual_fsm.GetFsm().AddEdge(2, 3, 'c', 'c');
  manual_fsm = manual_fsm.SimplifyEpsilon();
  EXPECT_TRUE(manual_fsm.AcceptString("a"));
  EXPECT_TRUE(manual_fsm.AcceptString("ac"));
  EXPECT_TRUE(manual_fsm.AcceptString("bc"));
  EXPECT_FALSE(manual_fsm.AcceptString("b"));
  EXPECT_FALSE(manual_fsm.AcceptString(""));

  // The safe direction still merges: non-accepting state with a single epsilon edge into an
  // accepting state.
  FSMWithStartEnd mergeable_fsm;
  for (int i = 0; i < 3; i++) {
    mergeable_fsm.AddState();
  }
  mergeable_fsm.SetStartState(0);
  mergeable_fsm.AddEndState(2);
  mergeable_fsm.GetFsm().AddEdge(0, 1, 'a', 'a');
  mergeable_fsm.GetFsm().AddEpsilonEdge(1, 2);
  mergeable_fsm = mergeable_fsm.SimplifyEpsilon();
  EXPECT_EQ(mergeable_fsm.GetFsm().NumStates(), 2);
  EXPECT_TRUE(mergeable_fsm.AcceptString("a"));
  EXPECT_FALSE(mergeable_fsm.AcceptString(""));
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
