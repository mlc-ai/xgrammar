#include <gtest/gtest.h>

#include "fsm.h"
using namespace xgrammar;
TEST(XGrammarFSMTest, BasicBuildTest) {
  std::cout << "--------- Basic Build Test Starts! -----------" << std::endl;
  std::cout << "--------- Basic Build Test1 -----------" << std::endl;
  auto fsm_wse = RegexToFSM("\"abcd\\n\"");
  std::string test_str = "abcd\n";
  std::vector<int> result;
  std::vector<int> from({fsm_wse.start});
  for (const auto& character : test_str) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) == fsm_wse.ends.end()) {
        std::cout << "Node: " << node << " is not in the end set." << std::endl;
        return false;
      }
    }
    return true;
  }());
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
  for (const auto& character : test_str) {
    assert([&]() -> bool {
      for (const auto& edge : fsm_wse.fsm.edges[0]) {
        if (edge.min <= int(character) && edge.max >= int(character)) {
          return true;
        }
      }
      return false;
    }());
  }
  std::cout << "--------- Basic Build Test4 -----------" << std::endl;
  fsm_wse = RegexToFSM("[\\d]");
  test_str = "1234567890";
  for (const auto& character : test_str) {
    assert([&]() -> bool {
      for (const auto& edge : fsm_wse.fsm.edges[0]) {
        if (edge.min <= int(character) && edge.max >= int(character)) {
          return true;
        }
      }
      return false;
    }());
  }
  std::cout << "--------- Basic Build Test5 -----------" << std::endl;
  fsm_wse = RegexToFSM("[^\\d]");
  test_str = "1234567890";
  for (const auto& character : test_str) {
    assert([&]() -> bool {
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
    assert([&]() -> bool {
      for (const auto& edge : fsm_wse.fsm.edges[0]) {
        if (edge.min <= int(character) && edge.max >= int(character)) {
          return true;
        }
      }
      std::cout << character << std::endl;
      return false;
    }());
  }
  std::cout << "--------- Basic Build Test6 -----------" << std::endl;
  fsm_wse = RegexToFSM("\"你好a\"");
  test_str = "你好a";
  result.clear();
  from.clear();
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) == fsm_wse.ends.end()) {
        std::cout << "Node: " << node << " is not in the end set." << std::endl;
        return false;
      }
    }
    return true;
  }());
  std::cout << "Basic Build Test Passed!" << std::endl;
}

TEST(XGrammarFSMTest, ConnectionTest) {
  std::cout << "--------- Connection Test Starts! -----------" << std::endl;
  std::cout << "--------- Connection Test1 -----------" << std::endl;
  auto fsm_wse = RegexToFSM("\" \"[a-zA-Z0-9]\"--\"");
  std::string test_str = " a--";
  std::vector<int> result;
  std::vector<int> from;
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        return true;
      }
    }
    return false;
  }());
  std::cout << "--------- Connection Test2 -----------" << std::endl;
  fsm_wse = RegexToFSM("\"aaa\" | [\\d]");
  test_str = "aaa";
  result.clear();
  from.clear();
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        return true;
      }
    }
    return false;
  }());
  test_str = "1";
  result.clear();
  from.clear();
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        return true;
      }
    }
    return false;
  }());
  std::cout << "--------- Connection Test3 -----------" << std::endl;
  fsm_wse = RegexToFSM("(([\\d]|[\\w]) | \"aaa\")");
  test_str = "aaa";
  result.clear();
  from.clear();
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        return true;
      }
    }
    return false;
  }());
  test_str = "1";
  result.clear();
  from.clear();
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) == fsm_wse.ends.end()) {
        std::cout << "Node: " << node << " is not in the end set." << std::endl;
        return false;
      }
    }
    return true;
  }());
  test_str = "1a";
  result.clear();
  from.clear();
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        std::cout << "Node: " << node << " is in the end set." << std::endl;
        return false;
      }
    }
    return true;
  }());
  std::cout << "--------- Connection Test4 -----------" << std::endl;
  fsm_wse = RegexToFSM("[\\d] & [123]");
  test_str = "1";
  result.clear();
  from.clear();
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        return true;
      }
    }
    return false;
  }());
  test_str = "5";
  result.clear();
  from.clear();
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        return false;
      }
    }
    return true;
  }());
  std::cout << "Connection Test Passed!" << std::endl;
}

TEST(XGrammarFSMTest, SymbolTest) {
  std::cout << "--------- Symbol Test Starts! -----------" << std::endl;
  std::cout << "--------- Symbol Test1 -----------" << std::endl;
  auto fsm_wse = RegexToFSM("\"1\"[\\d]+");
  std::string test_str[2] = {"1111", "1"};
  std::vector<int> result;
  std::vector<int> from;
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str[0]) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        return true;
      }
    }
    return false;
  }());
  result.clear();
  from.clear();
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str[1]) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        std::cout << "Node: " << node << " is in the end set." << std::endl;
        return false;
      }
    }
    return true;
  }());
  std::cout << "--------- Symbol Test2 -----------" << std::endl;
  fsm_wse = RegexToFSM("\"1\"[1]*");
  result.clear();
  from.clear();
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str[0]) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        return true;
      }
    }
    return false;
  }());
  result.clear();
  from.clear();
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str[1]) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        return true;
      }
    }
    return false;
  }());
  std::cout << "--------- Symbol Test3 -----------" << std::endl;
  fsm_wse = RegexToFSM("\"1\"[\\d]?");
  result.clear();
  from.clear();
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str[0]) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        return false;
      }
    }
    return true;
  }());
  result.clear();
  from.clear();
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str[1]) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        return true;
      }
    }
    return false;
  }());
  std::string test3 = "11";
  result.clear();
  from.clear();
  from.push_back(fsm_wse.start);
  for (const auto& character : test3) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        return true;
      }
    }
    return false;
  }());
  std::cout << "--------- Symbol Test4 -----------" << std::endl;
  fsm_wse = RegexToFSM("\" \"*\" \"*\" \"+\" \"?\" \"*");
  test_str[0] = " ";
  test_str[1] = "      ";
  for (const auto str : test_str) {
    result.clear();
    from.clear();
    from.push_back(fsm_wse.start);
    for (const auto& character : str) {
      fsm_wse.fsm.Advance(from, int(character), &result);
      from = result;
    }
    assert([&]() -> bool {
      for (const auto& node : from) {
        if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
          return true;
        }
      }
      return false;
    }());
  }
  std::cout << "Symbol Test Passed!" << std::endl;
}

TEST(XGrammarFSMTest, IntegratedTest) {
  std::cout << "--------- Integrated Test Starts! -----------" << std::endl;
  auto fsm_wse = RegexToFSM("((\"naive\" | \"bbb\" | [\\d]+)* [\\w]) | \"  \"+");
  std::string test_str[5] = {"naive1", "bbbnaive114514W", "    ", "123", "_"};
  std::vector<int> result;
  std::vector<int> from;
  for (const auto& str : test_str) {
    result.clear();
    from.clear();
    from.push_back(fsm_wse.start);
    for (const auto& character : str) {
      fsm_wse.fsm.Advance(from, int(character), &result);
      from = result;
    }
    assert([&]() -> bool {
      for (const auto& node : from) {
        if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
          return true;
        }
      }
      return false;
    }());
  }
  std::string test_str2[5] = {"naive", "bbbbbb", "naive   ", "123 ", "aaa"};
  for (const auto& str : test_str2) {
    result.clear();
    from.clear();
    from.push_back(fsm_wse.start);
    for (const auto& character : str) {
      fsm_wse.fsm.Advance(from, int(character), &result);
      from = result;
    }
    assert([&]() -> bool {
      for (const auto& node : from) {
        if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
          return false;
        }
      }
      return true;
    }());
  }
  std::cout << "--------- Integrated Test Passed! -----------" << std::endl;
}

TEST(XGrammarFSMTest, FunctionTest) {
  std::cout << "--------- Function Test Starts! -----------" << std::endl;
  std::cout << "--------- Function Test1 -----------" << std::endl;
  auto fsm_wse = RegexToFSM("[\\d\\d\\d]+\"123\"");
  std::string test_str = "123456123";
  std::vector<int> result;
  std::vector<int> from;
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        return true;
      }
    }
    return false;
  }());
  auto compact_fsm = fsm_wse.fsm.ToCompact();
  CompactFSMWithStartEnd compact_fsm_wse;
  compact_fsm_wse.fsm = compact_fsm;
  compact_fsm_wse.start = fsm_wse.start;
  compact_fsm_wse.ends = fsm_wse.ends;
  result.clear();
  from.clear();
  from.push_back(compact_fsm_wse.start);
  for (const auto& character : test_str) {
    compact_fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (compact_fsm_wse.ends.find(node) != compact_fsm_wse.ends.end()) {
        return true;
      }
    }
    return false;
  }());
  fsm_wse.fsm = compact_fsm_wse.fsm.ToFSM();
  result.clear();
  from.clear();
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        return true;
      }
    }
    return false;
  }());
  std::cout << "--------- Function Test2 -----------" << std::endl;
  fsm_wse = RegexToFSM("([abc] | [\\d])+");
  test_str = "abc3";
  result.clear();
  from.clear();
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        return true;
      }
    }
    return false;
  }());
  result.clear();
  from.clear();
  fsm_wse = fsm_wse.TODFA();
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        return true;
      }
    }
    return false;
  }());
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
  std::cout << "--------- Function Test3 -----------" << std::endl;
  result.clear();
  from.clear();
  fsm_wse = fsm_wse.MinimizeDFA();
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        return true;
      }
    }
    return false;
  }());
  assert(fsm_wse.fsm.edges.size() == 3);
  std::cout << "--------- Function Test4 -----------" << std::endl;
  fsm_wse = fsm_wse.Not();
  result.clear();
  from.clear();
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        return false;
      }
    }
    return true;
  }());
  result.clear();
  from.clear();
  test_str = "abcd";
  from.push_back(fsm_wse.start);
  for (const auto& character : test_str) {
    fsm_wse.fsm.Advance(from, int(character), &result);
    from = result;
  }
  assert([&]() -> bool {
    for (const auto& node : from) {
      if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
        return true;
      }
    }
    return false;
  }());
  std::cout << "--------- Function Test5 -----------" << std::endl;
  fsm_wse = RegexToFSM("[\\d]{1,  5}");
  std::string test_strs[2] = {"123", "12345"};
  for (const auto& str : test_strs) {
    result.clear();
    from.clear();
    from.push_back(fsm_wse.start);
    for (const auto& character : str) {
      fsm_wse.fsm.Advance(from, int(character), &result);
      from = result;
    }
    assert([&]() -> bool {
      for (const auto& node : from) {
        if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
          return true;
        }
      }
      return false;
    }());
  }
  test_strs[0] = "123456";
  test_strs[1] = "1234567";
  for (const auto& str : test_strs) {
    result.clear();
    from.clear();
    from.push_back(fsm_wse.start);
    for (const auto& character : str) {
      fsm_wse.fsm.Advance(from, int(character), &result);
      from = result;
    }
    assert([&]() -> bool {
      for (const auto& node : from) {
        if (fsm_wse.ends.find(node) != fsm_wse.ends.end()) {
          return false;
        }
      }
      return true;
    }());
  }

  std::cout << "--------- Function Test Passed! -----------" << std::endl;
}
