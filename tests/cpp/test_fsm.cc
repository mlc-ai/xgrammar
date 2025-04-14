#include <gtest/gtest.h>

#include "fsm.h"
using namespace xgrammar;
TEST(XGrammarFSMTest, BasicBuildTest) {
  std::cout << "--------- Basic Build Test Starts! -----------" << std::endl;
  std::cout << "--------- Basic Build Test1 -----------" << std::endl;
  auto fsm_wse = RegexToFSM("\"abcd\\n\"");
  std::string test_str = "abcd\n";
  assert(fsm_wse.Check(test_str));
  std::cout << "--------- Basic Build Test2 -----------" << std::endl;
  try {
    fsm_wse = RegexToFSM("\"\\W\"");
    assert(false);
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
  }
  std::cout << "--------- Basic Build Test3 -----------" << std::endl;
  fsm_wse = RegexToFSM("[-a-z\\n]");
  test_str = "abcd-\n";
  assert([&]() -> bool {
    for (const auto& character : test_str) {
      for (const auto& edge : fsm_wse.fsm.edges[0]) {
        if (edge.min <= int(character) && edge.max >= int(character)) {
          return true;
        }
      }
      return false;
    }
  }());
  std::cout << "--------- Basic Build Test4 -----------" << std::endl;
  fsm_wse = RegexToFSM("[\\d]");
  test_str = "1234567890";
  assert([&]() -> bool {
    for (const auto& character : test_str) {
      for (const auto& edge : fsm_wse.fsm.edges[0]) {
        if (edge.min <= int(character) && edge.max >= int(character)) {
          return true;
        }
      }
      return false;
    }
  }());
  std::cout << "--------- Basic Build Test5 -----------" << std::endl;
  fsm_wse = RegexToFSM("[^\\d]");
  test_str = "1234567890";
  assert([&]() -> bool {
    for (const auto& character : test_str) {
      for (const auto& edge : fsm_wse.fsm.edges[0]) {
        if (edge.min <= int(character) && edge.max >= int(character)) {
          return false;
        }
      }
      return true;
    }
  }());

  test_str = "abz";
  assert([&]() -> bool {
    for (const auto& character : test_str) {
      for (const auto& edge : fsm_wse.fsm.edges[0]) {
        if (edge.min <= int(character) && edge.max >= int(character)) {
          return true;
        }
      }
      std::cout << character << std::endl;
      return false;
    }
  }());

  std::cout << "--------- Basic Build Test6 -----------" << std::endl;
  fsm_wse = RegexToFSM("\"你好a\"");
  test_str = "你好a";
  assert(fsm_wse.Check(test_str) == true);
  std::cout << "Basic Build Test Passed!" << std::endl;
}

TEST(XGrammarFSMTest, ConnectionTest) {
  std::cout << "--------- Connection Test Starts! -----------" << std::endl;
  std::cout << "--------- Connection Test1 -----------" << std::endl;
  auto fsm_wse = RegexToFSM("\" \"[a-zA-Z0-9]\"--\"");
  std::string test_str = " a--";
  assert(fsm_wse.Check(test_str) == true);
  std::cout << "--------- Connection Test2 -----------" << std::endl;
  fsm_wse = RegexToFSM("\"aaa\" | [\\d]");
  test_str = "aaa";
  assert(fsm_wse.Check(test_str) == true);
  test_str = "1";
  assert(fsm_wse.Check(test_str) == true);
  std::cout << "--------- Connection Test3 -----------" << std::endl;
  fsm_wse = RegexToFSM("(([\\d]|[\\w]) | \"aaa\")");
  test_str = "aaa";
  assert(fsm_wse.Check(test_str) == true);
  test_str = "1";
  assert(fsm_wse.Check(test_str) == true);
  test_str = "1a";
  assert(fsm_wse.Check(test_str) == false);
  std::cout << "--------- Connection Test4 -----------" << std::endl;
  fsm_wse = RegexToFSM("[\\d] & [123]");
  test_str = "1";
  assert(fsm_wse.Check(test_str) == true);
  test_str = "5";
  assert(fsm_wse.Check(test_str) == false);
  std::cout << "Connection Test Passed!" << std::endl;
}

TEST(XGrammarFSMTest, SymbolTest) {
  std::cout << "--------- Symbol Test Starts! -----------" << std::endl;
  std::cout << "--------- Symbol Test1 -----------" << std::endl;
  auto fsm_wse = RegexToFSM("\"1\"[\\d]+");
  std::string test_str[2] = {"1111", "1"};
  assert(fsm_wse.Check(test_str[0]) == true);
  assert(fsm_wse.Check(test_str[1]) == false);
  std::cout << "--------- Symbol Test2 -----------" << std::endl;
  fsm_wse = RegexToFSM("\"1\"[1]*");
  assert(fsm_wse.Check(test_str[0]) == true);
  assert(fsm_wse.Check(test_str[1]) == true);
  std::cout << "--------- Symbol Test3 -----------" << std::endl;
  fsm_wse = RegexToFSM("\"1\"[\\d]?");
  assert(fsm_wse.Check(test_str[0]) == false);
  assert(fsm_wse.Check(test_str[1]) == true);
  std::string test3 = "11";
  assert(fsm_wse.Check(test3) == true);
  std::cout << "--------- Symbol Test4 -----------" << std::endl;
  fsm_wse = RegexToFSM("\" \"*\" \"*\" \"+\" \"?\" \"*");
  test_str[0] = " ";
  test_str[1] = "      ";
  for (const auto& str : test_str) {
    EXPECT_TRUE(fsm_wse.Check(str));
  }
  std::cout << "Symbol Test Passed!" << std::endl;
}

TEST(XGrammarFSMTest, IntegratedTest) {
  std::cout << "--------- Integrated Test Starts! -----------" << std::endl;
  auto fsm_wse = RegexToFSM("((\"naive\" | \"bbb\" | [\\d]+)* [\\w]) | \"  \"+");
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
  auto fsm_wse = RegexToFSM("[\\d\\d\\d]+\"123\"");
  std::string test_str = "123456123";
  assert(fsm_wse.Check(test_str) == true);
  auto compact_fsm = fsm_wse.fsm.ToCompact();
  CompactFSMWithStartEnd compact_fsm_wse;
  compact_fsm_wse.fsm = compact_fsm;
  compact_fsm_wse.start = fsm_wse.start;
  compact_fsm_wse.ends = fsm_wse.ends;
  assert(compact_fsm_wse.Check(test_str) == true);
  fsm_wse.fsm = compact_fsm_wse.fsm.ToFSM();
  assert(fsm_wse.Check(test_str) == true);
  std::cout << "--------- Function Test2 -----------" << std::endl;
  fsm_wse = RegexToFSM("([abc] | [\\d])+");
  test_str = "abc3";
  assert(fsm_wse.Check(test_str) == true);
  fsm_wse = fsm_wse.ToDFA();
  assert(fsm_wse.Check(test_str) == true);
  assert([&]() -> bool {
    for (const auto& edges : fsm_wse.fsm.edges) {
      for (const auto& edge : edges) {
        if (edge.IsEpsilon()) {
          return false;
        }
      }
    }
    return true;
  }());
  assert([&]() -> bool {
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
  assert(fsm_wse.Check(test_str) == true);
  assert(fsm_wse.fsm.edges.size() == 3);
  std::cout << "--------- Function Test4 -----------" << std::endl;
  fsm_wse = fsm_wse.Not();
  assert(fsm_wse.Check(test_str) == false);
  test_str = "abcd";
  assert(fsm_wse.Check(test_str) == true);
  std::cout << "--------- Function Test5 -----------" << std::endl;
  fsm_wse = RegexToFSM("[\\d]{1,  5}");
  std::string test_strs[2] = {"123", "12345"};
  for (const auto& str : test_strs) {
    EXPECT_TRUE(fsm_wse.Check(str));
  }
  test_strs[0] = "123456";
  test_strs[1] = "1234567";
  for (const auto& str : test_strs) {
    EXPECT_FALSE(fsm_wse.Check(str));
  }
  std::cout << "--------- Function Test6 -----------" << std::endl;
  fsm_wse = RegexToFSM("[a][b][c][d]");
  test_str = "abcd";
  assert(fsm_wse.Check(test_str) == true);
  fsm_wse.SimplifyEpsilon();
  assert(fsm_wse.NumNodes() == 5);
  assert(fsm_wse.Check(test_str) == true);
  std::cout << "--------- Function Test7 -----------" << std::endl;
  fsm_wse = RegexToFSM("\"abc\" | \"abd\"");
  test_str = "abc";
  assert(fsm_wse.Check(test_str) == true);
  fsm_wse.SimplifyTransition();
  fsm_wse.SimplifyEpsilon();
  assert(fsm_wse.Check(test_str) == true);
  test_str = "abcd";
  assert(fsm_wse.Check(test_str) == false);
  assert(fsm_wse.NumNodes() == 4);
  std::cout << "--------- Function Test8 -----------" << std::endl;
  fsm_wse = RegexToFSM("\"acd\" | \"bcd\"");
  test_str = "acd";
  assert(fsm_wse.Check(test_str) == true);
  fsm_wse.SimplifyTransition();
  fsm_wse.SimplifyEpsilon();
  assert(fsm_wse.Check(test_str) == true);
  test_str = "abcd";
  assert(fsm_wse.Check(test_str) == false);
  assert(fsm_wse.NumNodes() == 4);
  std::cout << "--------- Function Test Passed! -----------" << std::endl;
}
