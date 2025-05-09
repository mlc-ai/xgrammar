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
#endif  // XGRAMMAR_TESTS_CPP_TEST_UTILS_H_
