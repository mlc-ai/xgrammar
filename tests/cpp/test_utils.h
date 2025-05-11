#ifndef XGRAMMAR_TESTS_CPP_TEST_UTILS_H_
#define XGRAMMAR_TESTS_CPP_TEST_UTILS_H_

#include <gmock/gmock.h>  // for ::testing::ContainsRegex
#include <gtest/gtest.h>

#include <regex>
#include <string>

/**
 * @brief Macro to test that a statement throws an exception with a message matching a regex
 * pattern.
 * @param statement The statement that should throw an exception.
 * @param expected_exception The type of exception expected to be thrown.
 * @param msg_regex Regular expression pattern that the exception message should contain.
 */
#define XGRAMMAR_EXPECT_THROW(statement, expected_exception, msg_regex) \
  EXPECT_THROW(                                                         \
      {                                                                 \
        try {                                                           \
          statement;                                                    \
        } catch (const expected_exception& e) {                         \
          EXPECT_THAT(e.what(), ::testing::ContainsRegex(msg_regex));   \
          throw; /* rethrow for EXPECT_THROW to catch */                \
        }                                                               \
      },                                                                \
      expected_exception                                                \
  )

/**
 * @brief Macro to test that an std::any value equals an expected value of a specific type.
 * @param any_val The std::any value to test.
 * @param type The expected type of the value stored in std::any.
 * @param val2 The expected value to compare against.
 */
#define XGRAMMAR_EXPECT_ANY_EQ(any_val, type_name, val2) \
  do {                                                   \
    EXPECT_TRUE(any_val.has_value());                    \
    EXPECT_TRUE(any_val.type() == typeid(type_name));    \
    EXPECT_EQ(std::any_cast<type_name>(any_val), val2);  \
  } while (0)

#endif  // XGRAMMAR_TESTS_CPP_TEST_UTILS_H_
