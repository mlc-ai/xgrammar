#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <optional>

#include "fsm.h"
#include "fsm_builder.h"
#include "fsm_parser.h"
using namespace xgrammar;

FSMBuilder builder;
TEST(XGrammarFSMTest, BasicBuildTest) {
  std::cout << "--------- Basic Build Test Starts! -----------" << std::endl;
  std::cout << "--------- Basic Build Test1 -----------" << std::endl;
  auto fsm_wse = builder.BuildFSMFromRegex("abcd\\n").Unwrap();
  std::string test_str = "abcd\n";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  std::cout << "--------- Basic Build Test2 -----------" << std::endl;
  fsm_wse = builder.BuildFSMFromRegex("[-a-z\\n]").Unwrap();
  test_str = "abcd-\n";
  for (const auto& character : test_str) {
    EXPECT_TRUE([&]() -> bool {
      for (const auto& edge : fsm_wse.fsm.edges[0]) {
        if (edge.min <= int(character) && edge.max >= int(character)) {
          return true;
        }
      }
      return false;
    }());
  }
  std::cout << "--------- Basic Build Test3 -----------" << std::endl;
  fsm_wse = builder.BuildFSMFromRegex("[\\d]").Unwrap();
  test_str = "1234567890";
  for (const auto& character : test_str) {
    EXPECT_TRUE([&]() -> bool {
      for (const auto& edge : fsm_wse.fsm.edges[0]) {
        if (edge.min <= int(character) && edge.max >= int(character)) {
          return true;
        }
      }
      return false;
    }());
  }
  std::cout << "--------- Basic Build Test4 -----------" << std::endl;
  fsm_wse = builder.BuildFSMFromRegex("[^\\d]").Unwrap();
  test_str = "1234567890";
  for (const auto& character : test_str) {
    EXPECT_TRUE([&]() -> bool {
      for (const auto& edge : fsm_wse.fsm.edges[0]) {
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
      for (const auto& edge : fsm_wse.fsm.edges[0]) {
        if (edge.min <= int(character) && edge.max >= int(character)) {
          return true;
        }
      }
      std::cout << character << std::endl;
      return false;
    }());
  }
  std::cout << "--------- Basic Build Test5 -----------" << std::endl;
  fsm_wse = builder.BuildFSMFromRegex("你好a").Unwrap();
  test_str = "你好a";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  std::cout << "--------- Basic Build Test6 -----------" << std::endl;
  fsm_wse = builder.BuildFSMFromRegex("(())()()").Unwrap();
  test_str = "";
  EXPECT_FALSE(fsm_wse.Check(test_str));
  std::cout << "--------- Basic Build Test7 -----------" << std::endl;
  fsm_wse = builder.BuildFSMFromRegex("[abcdabcdxyzxyz]").Unwrap();
  test_str = "a";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  EXPECT_FALSE(fsm_wse.Check("e"));
  std::cout << fsm_wse << std::endl;
  EXPECT_EQ(fsm_wse.fsm.edges[0].size(), 2);
  std::cout << "Basic Build Test Passed!" << std::endl;
}

TEST(XGrammarFSMTest, ConnectionTest) {
  std::cout << "--------- Connection Test Starts! -----------" << std::endl;
  std::cout << "--------- Connection Test1 -----------" << std::endl;
  auto fsm_wse = builder.BuildFSMFromRegex(" [a-zA-Z0-9]--").Unwrap();
  std::string test_str = " a--";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  std::cout << "--------- Connection Test2 -----------" << std::endl;
  fsm_wse = builder.BuildFSMFromRegex("aaa|[\\d]").Unwrap();
  test_str = "aaa";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  test_str = "1";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  std::cout << "--------- Connection Test3 -----------" << std::endl;
  if (builder.BuildFSMFromRegex("(([\\d]|[\\w])|aaa)").IsErr()) {
    std::cout << builder.BuildFSMFromRegex("(([\\d]|[\\w])|aaa)").UnwrapErr()->what() << std::endl;
  }
  fsm_wse = builder.BuildFSMFromRegex("(([\\d]|[\\w])|aaa)").Unwrap();
  test_str = "aaa";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  test_str = "1";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  test_str = "1a";
  EXPECT_FALSE(fsm_wse.Check(test_str));
  std::cout << "Connection Test Passed!" << std::endl;
}

TEST(XGrammarFSMTest, SymbolTest) {
  std::cout << "--------- Symbol Test Starts! -----------" << std::endl;
  std::cout << "--------- Symbol Test1 -----------" << std::endl;
  auto fsm_wse = builder.BuildFSMFromRegex("1[\\d]+").Unwrap();
  std::string test_str[2] = {"1111", "1"};
  EXPECT_TRUE(fsm_wse.Check(test_str[0]));
  EXPECT_FALSE(fsm_wse.Check(test_str[1]));
  std::cout << "--------- Symbol Test2 -----------" << std::endl;
  fsm_wse = builder.BuildFSMFromRegex("1[1]*").Unwrap();
  EXPECT_TRUE(fsm_wse.Check(test_str[0]));
  EXPECT_TRUE(fsm_wse.Check(test_str[1]));
  std::cout << "--------- Symbol Test3 -----------" << std::endl;
  fsm_wse = builder.BuildFSMFromRegex("1[\\d]?").Unwrap();
  EXPECT_FALSE(fsm_wse.Check(test_str[0]));
  EXPECT_TRUE(fsm_wse.Check(test_str[1]));
  std::string test3 = "11";
  EXPECT_TRUE(fsm_wse.Check(test3));
  std::cout << "--------- Symbol Test4 -----------" << std::endl;
  fsm_wse = builder.BuildFSMFromRegex(" * * + ? *").Unwrap();
  test_str[0] = " ";
  test_str[1] = "      ";
  for (const auto& str : test_str) {
    EXPECT_TRUE(fsm_wse.Check(str));
  }
  std::cout << "Symbol Test Passed!" << std::endl;
}

TEST(XGrammarFSMTest, IntegratedTest) {
  std::cout << "--------- Integrated Test Starts! -----------" << std::endl;
  auto fsm_wse = builder.BuildFSMFromRegex("((naive|bbb|[\\d]+)*[\\w])|  +").Unwrap();
  std::string test_str[5] = {"naive1", "bbbnaive114514W", "    ", "123", "_"};
  for (const auto& str : test_str) {
    EXPECT_TRUE(fsm_wse.Check(str));
  }
  std::string test_str2[5] = {"naive", "bbbbbb", "naive   ", "123 ", "aaa"};
  for (const auto& str : test_str2) {
    EXPECT_FALSE(fsm_wse.Check(str));
  }
  std::cout << "--------- Integrated Test Passed! -----------" << std::endl;
}

TEST(XGrammarFSMTest, FunctionTest) {
  std::cout << "--------- Function Test Starts! -----------" << std::endl;
  std::cout << "--------- Function Test1 -----------" << std::endl;
  auto fsm_wse = builder.BuildFSMFromRegex("[\\d\\d\\d]+123").Unwrap();
  std::string test_str = "123456123";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  auto compact_fsm = fsm_wse.fsm.ToCompact();
  CompactFSMWithStartEnd compact_fsm_wse;
  compact_fsm_wse.fsm = compact_fsm;
  compact_fsm_wse.start = fsm_wse.start;
  compact_fsm_wse.ends = fsm_wse.ends;
  EXPECT_TRUE(compact_fsm_wse.Check(test_str));
  fsm_wse.fsm = compact_fsm_wse.fsm.ToFSM();
  EXPECT_TRUE(fsm_wse.Check(test_str));
  std::cout << "--------- Function Test2 -----------" << std::endl;
  fsm_wse = builder.BuildFSMFromRegex("([abc]|[\\d])+").Unwrap();
  test_str = "abc3";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  fsm_wse = fsm_wse.ToDFA();
  EXPECT_TRUE(fsm_wse.Check(test_str));
  EXPECT_TRUE([&]() -> bool {
    for (const auto& edges : fsm_wse.fsm.edges) {
      for (const auto& edge : edges) {
        if (edge.IsEpsilon()) {
          return false;
        }
      }
    }
    return true;
  }());
  EXPECT_TRUE([&]() -> bool {
    for (const auto& edges : fsm_wse.fsm.edges) {
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
  fsm_wse = fsm_wse.MinimizeDFA();
  EXPECT_TRUE(fsm_wse.Check(test_str));
  EXPECT_EQ(fsm_wse.fsm.edges.size(), 2);
  std::cout << "--------- Function Test4 -----------" << std::endl;
  fsm_wse = fsm_wse.Not();
  EXPECT_FALSE(fsm_wse.Check(test_str));
  test_str = "abcd";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  std::cout << "--------- Function Test5 -----------" << std::endl;
  fsm_wse = builder.BuildFSMFromRegex("[\\d]{1,  5}").Unwrap();
  std::string test_strs[2] = {"123", "12345"};
  for (const auto& str : test_strs) {
    EXPECT_TRUE(fsm_wse.Check(str));
  }
  test_strs[0] = "123456";
  test_strs[1] = "1234567";
  for (const auto& str : test_strs) {
    EXPECT_FALSE(fsm_wse.Check(str));
  }
  fsm_wse = builder.BuildFSMFromRegex("[\\d]{6}").Unwrap();
  EXPECT_TRUE(fsm_wse.Check("123456"));
  EXPECT_FALSE(fsm_wse.Check("1234567"));
  fsm_wse = builder.BuildFSMFromRegex("[\\d]{6, }").Unwrap();
  EXPECT_TRUE(fsm_wse.Check("123456"));
  EXPECT_TRUE(fsm_wse.Check("1234567"));
  std::cout << "--------- Function Test6 -----------" << std::endl;
  fsm_wse = builder.BuildFSMFromRegex("[a][b][c][d]").Unwrap();
  test_str = "abcd";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  fsm_wse.SimplifyEpsilon();
  EXPECT_EQ(fsm_wse.NumNodes(), 5);
  EXPECT_TRUE(fsm_wse.Check(test_str));
  std::cout << "--------- Function Test7 -----------" << std::endl;
  fsm_wse = builder.BuildFSMFromRegex("abc|abd").Unwrap();
  test_str = "abc";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  fsm_wse.SimplifyTransition();
  fsm_wse.SimplifyEpsilon();
  EXPECT_TRUE(fsm_wse.Check(test_str));
  test_str = "abcd";
  EXPECT_FALSE(fsm_wse.Check(test_str));
  EXPECT_EQ(fsm_wse.NumNodes(), 4);
  std::cout << "--------- Function Test8 -----------" << std::endl;
  fsm_wse = builder.BuildFSMFromRegex("acd|bcd").Unwrap();
  test_str = "acd";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  fsm_wse.SimplifyTransition();
  fsm_wse.SimplifyEpsilon();
  EXPECT_TRUE(fsm_wse.Check(test_str));
  test_str = "abcd";
  EXPECT_FALSE(fsm_wse.Check(test_str));
  EXPECT_EQ(fsm_wse.NumNodes(), 4);
  std::cout << "--------- Function Test9 -----------" << std::endl;
  fsm_wse = builder.BuildFSMFromRegex("ab*").Unwrap();
  test_str = "abbb";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  fsm_wse.SimplifyEpsilon();
  EXPECT_TRUE(fsm_wse.Check(test_str));
  EXPECT_EQ(fsm_wse.NumNodes(), 2);
  std::cout << "--------- Function Test Passed! -----------" << std::endl;
}

TEST(XGrammarFSMTest, EfficiencyTest) {
  std::cout << "--------- Efficiency Test Starts! -----------" << std::endl;
  // i.e ([a-z]0123456789){10}. Use this way to test the performance.
  auto fsm_wse = builder
                     .BuildFSMFromRegex(
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
  std::cout << "Initial Node Numbers:" << fsm_wse.NumNodes() << std::endl;
  auto time_start = std::chrono::high_resolution_clock::now();
  fsm_wse.SimplifyEpsilon();
  auto time_end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
  std::cout << "Time taken to simplify epsilon: " << duration.count() << " ms" << std::endl;
  std::cout << "After SimplifyEpsilon Node Numbers:" << fsm_wse.NumNodes() << std::endl;
  time_start = std::chrono::high_resolution_clock::now();
  fsm_wse.SimplifyTransition();
  time_end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
  std::cout << "Time taken to simplify transition: " << duration.count() << " ms" << std::endl;
  std::cout << "After SimplifyTransition Node Numbers:" << fsm_wse.NumNodes() << std::endl;
  time_start = std::chrono::high_resolution_clock::now();
  fsm_wse = fsm_wse.ToDFA();
  time_end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
  std::cout << "Time taken to convert to DFA: " << duration.count() << " ms" << std::endl;
  std::cout << "After ToDFA Node Numbers:" << fsm_wse.NumNodes() << std::endl;
  time_start = std::chrono::high_resolution_clock::now();
  fsm_wse = fsm_wse.MinimizeDFA();
  time_end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
  std::cout << "Time taken to minimize DFA: " << duration.count() << " ms" << std::endl;
  EXPECT_EQ(fsm_wse.NumNodes(), 111);
  std::cout << "--------- Efficiency Test Passed! -----------" << std::endl;
}

TEST(XGrammarFSMTest, BuildTrieTest) {
  std::vector<std::string> patterns = {"hello", "hi", "哈哈", "哈", "hili", "good"};
  auto fsm = BuildTrie(patterns);

  // Test1: The printed result of FSM

  // Test2: The printed result of CompactFSM
  CompactFSMWithStartEnd compact_fsm;
  compact_fsm.start = fsm.StartNode();
  compact_fsm.ends = fsm.ends;
  compact_fsm.fsm = fsm.fsm.ToCompact();

  // Test3: Walk through the FSM
  int state = fsm.StartNode();
  EXPECT_EQ(state, 0);

  // Test "hello"
  state = fsm.StartNode();
  EXPECT_EQ(fsm.LegacyTransitionOnDFA(state, 'h'), 1);
  EXPECT_EQ(fsm.LegacyTransitionOnDFA(1, 'e'), 2);
  EXPECT_EQ(fsm.LegacyTransitionOnDFA(2, 'l'), 3);
  EXPECT_EQ(fsm.LegacyTransitionOnDFA(3, 'l'), 4);
  EXPECT_EQ(fsm.LegacyTransitionOnDFA(4, 'o'), 5);
  EXPECT_TRUE(fsm.IsEndNode(5));

  // Test "hil"
  state = fsm.StartNode();
  EXPECT_EQ(fsm.LegacyTransitionOnDFA(state, 'h'), 1);
  EXPECT_EQ(fsm.LegacyTransitionOnDFA(1, 'i'), 6);
  EXPECT_EQ(fsm.LegacyTransitionOnDFA(6, 'l'), 13);
  EXPECT_FALSE(fsm.IsEndNode(13));

  // Test walk failure
  state = fsm.StartNode();
  EXPECT_EQ(fsm.LegacyTransitionOnDFA(state, 'g'), 15);
  EXPECT_EQ(fsm.LegacyTransitionOnDFA(15, 'o'), 16);
  EXPECT_EQ(fsm.LegacyTransitionOnDFA(16, 'e'), -1);
}

TEST(XGrammarFSMTest, RuleToFSMTest) {
  std::string simple_grammar = R"(
  main::="hello"|((rule1)+rule2)
  rule1::= ("a"|"b")+rule2
  rule2::= "c"
  rule2::= "abc\"\"d")";
  FSMGroup fsm_group = GrammarToFSMs(simple_grammar, "main").Unwrap();
  EXPECT_EQ(fsm_group.Size(), 3);
  EXPECT_EQ(fsm_group.GetRootRuleName(), "main");

  int32_t main_id = fsm_group.GetRuleID("main");
  int32_t rule1_id = fsm_group.GetRuleID("rule1");
  int32_t rule2_id = fsm_group.GetRuleID("rule2");
  FSMWithStartEnd fsm_main = fsm_group.GetFSM(main_id);
  FSMWithStartEnd fsm_rule1 = fsm_group.GetFSM(rule1_id);
  FSMWithStartEnd fsm_rule2 = fsm_group.GetFSM(rule2_id);
  // Test "hello"
  int32_t main_start = fsm_main.StartNode();
  std::unordered_set<int> current_states = {main_start};
  std::unordered_set<int> next_states;
  fsm_main.fsm.GetEpsilonClosure(&current_states);
  fsm_main.Transition(current_states, 'h', &next_states);
  current_states = next_states;
  fsm_main.Transition(current_states, 'e', &next_states);
  current_states = next_states;
  fsm_main.Transition(current_states, 'l', &next_states);
  current_states = next_states;
  fsm_main.Transition(current_states, 'l', &next_states);
  current_states = next_states;
  fsm_main.Transition(current_states, 'o', &next_states);
  current_states = next_states;
  EXPECT_TRUE(std::find_if(current_states.begin(), current_states.end(), [&](int state) {
                return fsm_main.IsEndNode(state);
              }) != current_states.end());
  // Test "(rule1)+rule2"
  current_states = {main_start};
  fsm_main.fsm.GetEpsilonClosure(&current_states);
  fsm_main.Transition(current_states, rule1_id, &next_states, true);
  current_states = next_states;
  fsm_main.Transition(current_states, rule1_id, &next_states, true);
  current_states = next_states;
  fsm_main.Transition(current_states, rule2_id, &next_states, true);
  current_states = next_states;
  EXPECT_TRUE(std::find_if(current_states.begin(), current_states.end(), [&](int state) {
                return fsm_main.IsEndNode(state);
              }) != current_states.end());

  current_states = {main_start};
  fsm_main.fsm.GetEpsilonClosure(&current_states);
  fsm_main.Transition(current_states, rule1_id, &next_states, true);
  current_states = next_states;
  fsm_main.Transition(current_states, rule1_id, &next_states, true);
  current_states = next_states;
  EXPECT_FALSE(std::find_if(current_states.begin(), current_states.end(), [&](int state) {
                 return fsm_main.IsEndNode(state);
               }) != current_states.end());

  // Test multiple definitions of the same rule
  int16_t rule2_start = fsm_rule2.StartNode();
  current_states = {rule2_start};
  fsm_rule2.fsm.GetEpsilonClosure(&current_states);
  fsm_rule2.Transition(current_states, 'c', &next_states);
  current_states = next_states;
  EXPECT_TRUE(std::find_if(current_states.begin(), current_states.end(), [&](int state) {
                return fsm_rule2.IsEndNode(state);
              }) != current_states.end());
  current_states = {rule2_start};
  fsm_rule2.fsm.GetEpsilonClosure(&current_states);
  fsm_rule2.Transition(current_states, 'a', &next_states);
  current_states = next_states;
  fsm_rule2.Transition(current_states, 'b', &next_states);
  current_states = next_states;
  fsm_rule2.Transition(current_states, 'c', &next_states);
  current_states = next_states;
  fsm_rule2.Transition(current_states, '\"', &next_states);
  current_states = next_states;
  fsm_rule2.Transition(current_states, '\"', &next_states);
  current_states = next_states;
  fsm_rule2.Transition(current_states, 'd', &next_states);
  current_states = next_states;
  EXPECT_TRUE(std::find_if(current_states.begin(), current_states.end(), [&](int state) {
                return fsm_rule2.IsEndNode(state);
              }) != current_states.end());

  std::string regex_grammar = R"(
  main::= rule1+ /a/?
  rule1 ::= /[\d]/+
  rule1 ::= /abc/"abc")";
  fsm_group = GrammarToFSMs(regex_grammar, "main").Unwrap();
  main_id = fsm_group.GetRuleID("main");
  rule1_id = fsm_group.GetRuleID("rule1");
  fsm_main = fsm_group.GetFSM(main_id);
  fsm_rule1 = fsm_group.GetFSM(rule1_id);

  // Test "123"
  main_start = fsm_main.StartNode();
  int32_t rule1_start = fsm_rule1.StartNode();
  current_states = {main_start};
  fsm_main.fsm.GetEpsilonClosure(&current_states);
  fsm_main.Transition(current_states, rule1_id, &next_states, true);
  current_states = next_states;
  fsm_main.Transition(current_states, rule1_id, &next_states, true);
  current_states = next_states;
  EXPECT_TRUE(std::find_if(current_states.begin(), current_states.end(), [&](int state) {
                return fsm_main.IsEndNode(state);
              }) != current_states.end());
  fsm_main.Transition(current_states, 'a', &next_states);
  current_states = next_states;
  EXPECT_TRUE(std::find_if(current_states.begin(), current_states.end(), [&](int state) {
                return fsm_main.IsEndNode(state);
              }) != current_states.end());

  // Test rule1
  current_states = {rule1_start};
  fsm_rule1.fsm.GetEpsilonClosure(&current_states);
  fsm_rule1.Transition(current_states, '1', &next_states);
  current_states = next_states;
  fsm_rule1.Transition(current_states, '2', &next_states);
  current_states = next_states;
  fsm_rule1.Transition(current_states, '3', &next_states);
  current_states = next_states;
  EXPECT_TRUE(std::find_if(current_states.begin(), current_states.end(), [&](int state) {
                return fsm_rule1.IsEndNode(state);
              }) != current_states.end());

  current_states = {rule1_start};
  fsm_rule1.fsm.GetEpsilonClosure(&current_states);
  fsm_rule1.Transition(current_states, 'a', &next_states);
  current_states = next_states;
  fsm_rule1.Transition(current_states, 'b', &next_states);
  current_states = next_states;
  fsm_rule1.Transition(current_states, 'c', &next_states);
  current_states = next_states;
  EXPECT_FALSE(std::find_if(current_states.begin(), current_states.end(), [&](int state) {
                 return fsm_rule1.IsEndNode(state);
               }) != current_states.end());

  fsm_rule1.Transition(current_states, 'a', &next_states);
  current_states = next_states;
  fsm_rule1.Transition(current_states, 'b', &next_states);
  current_states = next_states;
  fsm_rule1.Transition(current_states, 'c', &next_states);
  current_states = next_states;
  EXPECT_TRUE(std::find_if(current_states.begin(), current_states.end(), [&](int state) {
                return fsm_rule1.IsEndNode(state);
              }) != current_states.end());

  std::string repeated_grammar = R"(
    main ::= "a" | "b"
    main ::= "c" | "d"
    main ::= /[h-z]/ | ("e" | ("f" | "g"))
    main ::= /[a-z]/
  )";
  fsm_group = GrammarToFSMs(repeated_grammar, "main").Unwrap();
  fsm_group.Simplify();
  fsm_group.ToMinimizedDFA(50);
  fsm_main = fsm_group.GetFSM(0);
  EXPECT_EQ(fsm_main.NumNodes(), 2);
}

TEST(XGrammarFSMTest, FSMAdvanceTest) {
  std::string simple_grammar = R"(
    root ::= rule1+
    rule1 ::= /[a-z]/ rule1*
  )";

  EarleyParserWithFSM parser(simple_grammar, "root");
  EXPECT_TRUE(parser.Advance('a'));
  EXPECT_TRUE(parser.IsAcceptStopToken());
  EXPECT_TRUE(parser.Advance('b'));
  EXPECT_TRUE(parser.IsAcceptStopToken());
  EXPECT_TRUE(parser.Advance('c'));
  EXPECT_TRUE(parser.IsAcceptStopToken());

  std::string basic_float_grammar = R"(
    root ::= "-"? int ("." int)?
    int ::= /[0-9]+/
  )";
  EarleyParserWithFSM parser2(basic_float_grammar, "root");
  std::cout << parser2;
  EXPECT_TRUE(parser2.Advance('-'));
  EXPECT_FALSE(parser2.IsAcceptStopToken());
  EXPECT_TRUE(parser2.Advance('1'));
  EXPECT_TRUE(parser2.IsAcceptStopToken());
  EXPECT_TRUE(parser2.Advance('2'));
  EXPECT_TRUE(parser2.IsAcceptStopToken());
  EXPECT_TRUE(parser2.Advance('.'));
  EXPECT_FALSE(parser2.IsAcceptStopToken());
  EXPECT_TRUE(parser2.Advance('3'));
  EXPECT_TRUE(parser2.IsAcceptStopToken());

  std::string basic_ascii_string_grammar = R"(
    string ::= "\"" chars "\""
    chars ::= char*
    char ::= /[!-\[\]-~]/ | escape
    escape ::= "\\" /[!-~]/
  )";

  EarleyParserWithFSM parser3(basic_ascii_string_grammar, "string");
  EXPECT_FALSE(parser3.Advance(' '));
  EXPECT_TRUE(parser3.Advance('"'));
  EXPECT_FALSE(parser3.IsAcceptStopToken());
  EXPECT_TRUE(parser3.Advance('a'));
  EXPECT_FALSE(parser3.IsAcceptStopToken());
  EXPECT_TRUE(parser3.Advance('"'));
  EXPECT_TRUE(parser3.IsAcceptStopToken());
  parser3.Reset();
  EXPECT_TRUE(parser3.Advance('"'));
  EXPECT_FALSE(parser3.IsAcceptStopToken());
  EXPECT_TRUE(parser3.Advance('\\'));
  EXPECT_FALSE(parser3.IsAcceptStopToken());
  EXPECT_TRUE(parser3.Advance('\"'));
  EXPECT_FALSE(parser3.IsAcceptStopToken());
  EXPECT_TRUE(parser3.Advance('\\'));
  EXPECT_FALSE(parser3.IsAcceptStopToken());
  EXPECT_TRUE(parser3.Advance('n'));
  EXPECT_FALSE(parser3.IsAcceptStopToken());
  EXPECT_TRUE(parser3.Advance('"'));
  EXPECT_TRUE(parser3.IsAcceptStopToken());
}

TEST(XGrammarFSMTest, FSMNullableTest) {
  std::string test_empty_grammar = R"(
    root ::= rule1+ rule2?
    rule1 ::= ("" | /[a-z]/)+
    rule2 ::= /[0-9]/
    rule3 ::= rule3*
    rule4 ::= /[a-z]/?
  )";
  EarleyParserWithFSM parser(test_empty_grammar, "root");
  std::cout << parser;
  EXPECT_TRUE(parser.IsFsmNullable(0));
  EXPECT_TRUE(parser.IsFsmNullable(1));
  EXPECT_FALSE(parser.IsFsmNullable(2));
  EXPECT_TRUE(parser.IsFsmNullable(3));
  EXPECT_TRUE(parser.IsFsmNullable(4));
}

TEST(XGrammarFSMTest, FSMToDFA) {
  std::string test_grammar = R"(
    root ::= "[" rule1? "]"
    rule1 ::= /[0-9]/
  )";
  EarleyParserWithFSM parser(test_grammar, "root");
  std::cout << parser;
}

TEST(XGrammarFSMTest, BasicJsonGrammarTest) {
  std::string test_empty_grammar = R"(
    root ::= Empty* Json Empty*
    Json ::= Array | Object
    Array ::= "[" Element? "]"
    Object ::= "{" ObjectElement? "}"
    ObjectElement ::=  Empty* String Empty* ":" Empty* Value Empty* ( "," ObjectElement)?
    Element ::= Empty* Value Empty* ("," Element)?
    Value ::= String | Int | Float | Object | Array | Bool | "Null"
    Float ::= sign? Int "." Int
    Int ::= sign? /[0-9]/+
    String ::= "\"" (char | escaped)* "\""
    char ::= /[ !#-\[\]-~]/
    sign ::= "+" | "-"
    escaped ::= "\\\""  | "\\\/"  | "\\n"  | "\\b"  | "\\f"  | "\\r" | "\\t" | "\\u" HEX
    HEX ::= /[0-9a-fA-F]{4}/
    Bool ::= "true" | "false"
    Null ::= "null"
    Empty ::= " " | "\n" | "\r" | "\t"
  )";

  EarleyParserWithFSM parser(test_empty_grammar, "root");
  std::cout << parser;
  std::string test_str = R"({"key1": "value1", "key2": 123, "key3": [1, 2, 3]})";
  for (const auto& ch : test_str) {
    std::cout << ch;
    EXPECT_FALSE(parser.IsAcceptStopToken());
    EXPECT_TRUE(parser.Advance(ch));
  }
  EXPECT_TRUE(parser.IsAcceptStopToken());
  parser.Reset();
  for (const auto& ch : test_str) {
    EXPECT_FALSE(parser.IsAcceptStopToken());
    EXPECT_TRUE(parser.Advance(ch));
  }
  EXPECT_TRUE(parser.IsAcceptStopToken());
  parser.Reset();
  test_str = R"(
{
    "web-app": {
    "servlet": [
        {
        "servlet-name": "cofaxCDS",
        "servlet-class": "org.cofax.cds.CDSServlet",
        "init-param": {
            "configGlossary:installationAt": "Philadelphia, PA",
            "configGlossary:adminEmail": "ksm@pobox.com",
            "configGlossary:poweredBy": "Cofax",
            "configGlossary:poweredByIcon": "/images/cofax.gif",
            "configGlossary:staticPath": "/content/static",
            "templateProcessorClass": "org.cofax.WysiwygTemplate",
            "templateLoaderClass": "org.cofax.FilesTemplateLoader",
            "templatePath": "templates",
            "templateOverridePath": "",
            "defaultListTemplate": "listTemplate.htm",
            "defaultFileTemplate": "articleTemplate.htm",
            "useJSP": false,
            "jspListTemplate": "listTemplate.jsp",
            "jspFileTemplate": "articleTemplate.jsp",
            "cachePackageTagsTrack": 200,
            "cachePackageTagsStore": 200,
            "cachePackageTagsRefresh": 60,
            "cacheTemplatesTrack": 100,
            "cacheTemplatesStore": 50,
            "cacheTemplatesRefresh": 15,
            "cachePagesTrack": 200,
            "cachePagesStore": 100,
            "cachePagesRefresh": 10,
            "cachePagesDirtyRead": 10,
            "searchEngineListTemplate": "forSearchEnginesList.htm",
            "searchEngineFileTemplate": "forSearchEngines.htm",
            "searchEngineRobotsDb": "WEB-INF/robots.db",
            "useDataStore": true,
            "dataStoreClass": "org.cofax.SqlDataStore",
            "redirectionClass": "org.cofax.SqlRedirection",
            "dataStoreName": "cofax",
            "dataStoreDriver": "com.microsoft.jdbc.sqlserver.SQLServerDriver",
            "dataStoreUrl": "jdbc:microsoft:sqlserver://LOCALHOST:1433;DatabaseName=goon",
            "dataStoreUser": "sa",
            "dataStorePassword": "dataStoreTestQuery",
            "dataStoreTestQuery": "SET NOCOUNT ON;select test='test';",
            "dataStoreLogFile": "/usr/local/tomcat/logs/datastore.log",
            "dataStoreInitConns": 10,
            "dataStoreMaxConns": 100,
            "dataStoreConnUsageLimit": 100,
            "dataStoreLogLevel": "debug",
            "maxUrlLength": 500
        }
        },
        {
        "servlet-name": "cofaxEmail",
        "servlet-class": "org.cofax.cds.EmailServlet",
        "init-param": {
            "mailHost": "mail1",
            "mailHostOverride": "mail2"
        }
        },
        {
        "servlet-name": "cofaxAdmin",
        "servlet-class": "org.cofax.cds.AdminServlet"
        },
        {
        "servlet-name": "fileServlet",
        "servlet-class": "org.cofax.cds.FileServlet"
        },
        {
        "servlet-name": "cofaxTools",
        "servlet-class": "org.cofax.cms.CofaxToolsServlet",
        "init-param": {
            "templatePath": "toolstemplates/",
            "log": 1,
            "logLocation": "/usr/local/tomcat/logs/CofaxTools.log",
            "logMaxSize": "",
            "dataLog": 1,
            "dataLogLocation": "/usr/local/tomcat/logs/dataLog.log",
            "dataLogMaxSize": "",
            "removePageCache": "/content/admin/remove?cache=pages&id=",
            "removeTemplateCache": "/content/admin/remove?cache=templates&id=",
            "fileTransferFolder": "/usr/local/tomcat/webapps/content/fileTransferFolder",
            "lookInContext": 1,
            "adminGroupID": 4,
            "betaServer": true
        }
        }
    ],
    "servlet-mapping": {
        "cofaxCDS": "/",
        "cofaxEmail": "/cofaxutil/aemail/*",
        "cofaxAdmin": "/admin/*",
        "fileServlet": "/static/*",
        "cofaxTools": "/tools/*"
    },
    "taglib": {
        "taglib-uri": "cofax.tld",
        "taglib-location": "/WEB-INF/tlds/cofax.tld"
    }
    }
})";
  for (const auto& ch : test_str) {
    std::cout << ch;
    EXPECT_FALSE(parser.IsAcceptStopToken());
    EXPECT_TRUE(parser.Advance(ch));
  }
  EXPECT_TRUE(parser.IsAcceptStopToken());
}

TEST(XGrammarFSMTest, LookAheadTest) {
  std::string test_grammar = R"(
    root ::= rule1? ((rule2))(=((((("abc"))))))
    rule1 ::= "a" | ("b")(=/[a-z]/)
    rule2 ::= "c" | "d"(=rule1)
    rule3 ::= "e" | "f"
  )";
  EarleyParserWithFSM parser(test_grammar, "root");
  std::cout << parser;
  EXPECT_TRUE(parser.GetLookaheadFSM(3) == std::nullopt);
  const auto& lookahead_fsm_optional = parser.GetLookaheadFSM(1);
  EXPECT_TRUE(lookahead_fsm_optional.has_value());
  const auto& lookahead_fsm = lookahead_fsm_optional.value();
  EXPECT_EQ(lookahead_fsm.Print(), R"(FSM(num_nodes=2, start=0, end=[1, ], edges=[
0: [(97, 122)->1]
1: []
]))");
}
