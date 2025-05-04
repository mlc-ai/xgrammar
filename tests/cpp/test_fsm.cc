#include <gtest/gtest.h>

#include <chrono>

#include "fsm.h"
using namespace xgrammar;
TEST(XGrammarFSMTest, BasicBuildTest) {
  std::cout << "--------- Basic Build Test Starts! -----------" << std::endl;
  std::cout << "--------- Basic Build Test1 -----------" << std::endl;
  auto fsm_wse = RegexToFSM("abcd\\n").Unwrap();
  std::string test_str = "abcd\n";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  std::cout << "--------- Basic Build Test2 -----------" << std::endl;
  fsm_wse = RegexToFSM("[-a-z\\n]").Unwrap();
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
  fsm_wse = RegexToFSM("[\\d]").Unwrap();
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
  fsm_wse = RegexToFSM("[^\\d]").Unwrap();
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
  fsm_wse = RegexToFSM("你好a").Unwrap();
  test_str = "你好a";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  std::cout << "--------- Basic Build Test6 -----------" << std::endl;
  fsm_wse = RegexToFSM("(())()()").Unwrap();
  test_str = "";
  EXPECT_FALSE(fsm_wse.Check(test_str));
  std::cout << "--------- Basic Build Test7 -----------" << std::endl;
  fsm_wse = RegexToFSM("[abcdabcdxyzxyz]").Unwrap();
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
  auto fsm_wse = RegexToFSM(" [a-zA-Z0-9]--").Unwrap();
  std::string test_str = " a--";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  std::cout << "--------- Connection Test2 -----------" << std::endl;
  fsm_wse = RegexToFSM("aaa|[\\d]").Unwrap();
  test_str = "aaa";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  test_str = "1";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  std::cout << "--------- Connection Test3 -----------" << std::endl;
  if (RegexToFSM("(([\\d]|[\\w])|aaa)").IsErr()) {
    std::cout << RegexToFSM("(([\\d]|[\\w])|aaa)").UnwrapErr()->what() << std::endl;
  }
  fsm_wse = RegexToFSM("(([\\d]|[\\w])|aaa)").Unwrap();
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
  auto fsm_wse = RegexToFSM("1[\\d]+").Unwrap();
  std::string test_str[2] = {"1111", "1"};
  EXPECT_TRUE(fsm_wse.Check(test_str[0]));
  EXPECT_FALSE(fsm_wse.Check(test_str[1]));
  std::cout << "--------- Symbol Test2 -----------" << std::endl;
  fsm_wse = RegexToFSM("1[1]*").Unwrap();
  EXPECT_TRUE(fsm_wse.Check(test_str[0]));
  EXPECT_TRUE(fsm_wse.Check(test_str[1]));
  std::cout << "--------- Symbol Test3 -----------" << std::endl;
  fsm_wse = RegexToFSM("1[\\d]?").Unwrap();
  EXPECT_FALSE(fsm_wse.Check(test_str[0]));
  EXPECT_TRUE(fsm_wse.Check(test_str[1]));
  std::string test3 = "11";
  EXPECT_TRUE(fsm_wse.Check(test3));
  std::cout << "--------- Symbol Test4 -----------" << std::endl;
  fsm_wse = RegexToFSM(" * * + ? *").Unwrap();
  test_str[0] = " ";
  test_str[1] = "      ";
  for (const auto& str : test_str) {
    EXPECT_TRUE(fsm_wse.Check(str));
  }
  std::cout << "Symbol Test Passed!" << std::endl;
}

TEST(XGrammarFSMTest, IntegratedTest) {
  std::cout << "--------- Integrated Test Starts! -----------" << std::endl;
  auto fsm_wse = RegexToFSM("((naive|bbb|[\\d]+)*[\\w])|  +").Unwrap();
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
  auto fsm_wse = RegexToFSM("[\\d\\d\\d]+123").Unwrap();
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
  fsm_wse = RegexToFSM("([abc]|[\\d])+").Unwrap();
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
  EXPECT_EQ(fsm_wse.fsm.edges.size(), 3);
  std::cout << "--------- Function Test4 -----------" << std::endl;
  fsm_wse = fsm_wse.Not();
  EXPECT_FALSE(fsm_wse.Check(test_str));
  test_str = "abcd";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  std::cout << "--------- Function Test5 -----------" << std::endl;
  fsm_wse = RegexToFSM("[\\d]{1,  5}").Unwrap();
  std::string test_strs[2] = {"123", "12345"};
  for (const auto& str : test_strs) {
    EXPECT_TRUE(fsm_wse.Check(str));
  }
  test_strs[0] = "123456";
  test_strs[1] = "1234567";
  for (const auto& str : test_strs) {
    EXPECT_FALSE(fsm_wse.Check(str));
  }
  fsm_wse = RegexToFSM("[\\d]{6}").Unwrap();
  EXPECT_TRUE(fsm_wse.Check("123456"));
  EXPECT_FALSE(fsm_wse.Check("1234567"));
  fsm_wse = RegexToFSM("[\\d]{6, }").Unwrap();
  EXPECT_TRUE(fsm_wse.Check("123456"));
  EXPECT_TRUE(fsm_wse.Check("1234567"));
  std::cout << "--------- Function Test6 -----------" << std::endl;
  fsm_wse = RegexToFSM("[a][b][c][d]").Unwrap();
  test_str = "abcd";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  fsm_wse.SimplifyEpsilon();
  EXPECT_EQ(fsm_wse.NumNodes(), 5);
  EXPECT_TRUE(fsm_wse.Check(test_str));
  std::cout << "--------- Function Test7 -----------" << std::endl;
  fsm_wse = RegexToFSM("abc|abd").Unwrap();
  test_str = "abc";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  fsm_wse.SimplifyTransition();
  fsm_wse.SimplifyEpsilon();
  EXPECT_TRUE(fsm_wse.Check(test_str));
  test_str = "abcd";
  EXPECT_FALSE(fsm_wse.Check(test_str));
  EXPECT_EQ(fsm_wse.NumNodes(), 4);
  std::cout << "--------- Function Test8 -----------" << std::endl;
  fsm_wse = RegexToFSM("acd|bcd").Unwrap();
  test_str = "acd";
  EXPECT_TRUE(fsm_wse.Check(test_str));
  fsm_wse.SimplifyTransition();
  fsm_wse.SimplifyEpsilon();
  EXPECT_TRUE(fsm_wse.Check(test_str));
  test_str = "abcd";
  EXPECT_FALSE(fsm_wse.Check(test_str));
  EXPECT_EQ(fsm_wse.NumNodes(), 4);
  std::cout << "--------- Function Test9 -----------" << std::endl;
  fsm_wse = RegexToFSM("ab*").Unwrap();
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
  auto fsm_wse = RegexToFSM(
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
  EXPECT_EQ(fsm.Transition(state, 'h'), 1);
  EXPECT_EQ(fsm.Transition(1, 'e'), 2);
  EXPECT_EQ(fsm.Transition(2, 'l'), 3);
  EXPECT_EQ(fsm.Transition(3, 'l'), 4);
  EXPECT_EQ(fsm.Transition(4, 'o'), 5);
  EXPECT_TRUE(fsm.IsEndNode(5));

  // Test "hil"
  state = fsm.StartNode();
  EXPECT_EQ(fsm.Transition(state, 'h'), 1);
  EXPECT_EQ(fsm.Transition(1, 'i'), 6);
  EXPECT_EQ(fsm.Transition(6, 'l'), 13);
  EXPECT_FALSE(fsm.IsEndNode(13));

  // Test walk failure
  state = fsm.StartNode();
  EXPECT_EQ(fsm.Transition(state, 'g'), 15);
  EXPECT_EQ(fsm.Transition(15, 'o'), 16);
  EXPECT_EQ(fsm.Transition(16, 'e'), -1);
}
