/*!
 * Copyright (c) 2026 by Contributors
 * \file tests/cpp/test_compact_2d_array.cc
 * \brief Indirect append correctness and amortized-growth regressions.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "support/compact_2d_array.h"

namespace xgrammar {
namespace {

struct CopyCounted {
  int value;
  int64_t* operations;

  CopyCounted(int value, int64_t* operations) : value(value), operations(operations) {}
  CopyCounted(const CopyCounted& other) : value(other.value), operations(other.operations) {
    ++*operations;
  }
  CopyCounted(CopyCounted&& other) noexcept : value(other.value), operations(other.operations) {
    ++*operations;
  }
  CopyCounted& operator=(const CopyCounted&) = default;
  CopyCounted& operator=(CopyCounted&&) = default;
};

TEST(Compact2DArrayTest, IndirectAppendHasAmortizedLinearRelocations) {
  // Count copies/moves instead of measuring wall-clock latency. Exact reserve
  // previously relocated the full retained history at every new high-water mark.
  for (bool rollback : {false, true}) {
    SCOPED_TRACE(rollback);
    int64_t operations = 0;
    const CopyCounted external(42, &operations);
    const std::vector<const CopyCounted*> incoming{&external};
    Compact2DArray<CopyCounted> array;
    constexpr int kRows = 1024;
    for (int i = 0; i < kRows; ++i) {
      EXPECT_EQ(array.PushBackIndirect(incoming), i);
      if (rollback) {
        array.PushBackIndirect(incoming);
        array.PushBackIndirect(incoming);
        array.PopBack(2);
      }
      ASSERT_EQ(array.size(), i + 1);
    }
    // Loose linear bound includes payload insertion plus geometric relocation;
    // it does not depend on a particular std::vector growth factor.
    EXPECT_LT(operations, 10 * kRows);
    for (int i = 0; i < kRows; ++i) EXPECT_EQ(array[i][0].value, 42);
  }
}

TEST(Compact2DArrayTest, IndirectEmptyAndMultipleElementRowsSurviveRollback) {
  Compact2DArray<int> array;
  const int first = 7;
  const int second = 19;
  EXPECT_EQ(array.PushBackIndirect({}), 0);
  EXPECT_EQ(array[0].size(), 0);
  for (int i = 0; i < 100; ++i) {
    const int index = array.PushBackIndirect({&second, &first, &second});
    ASSERT_EQ(array[index].size(), 3);
    EXPECT_EQ(array[index][0], 19);
    EXPECT_EQ(array[index][1], 7);
    EXPECT_EQ(array[index][2], 19);
    array.PushBackIndirect({});
    array.PushBackIndirect({&first});
    array.PopBack(2);
    ASSERT_EQ(array.size(), i + 2);
  }
  array.PopBack(100);
  ASSERT_EQ(array.size(), 1);
  EXPECT_EQ(array[0].size(), 0);
  EXPECT_EQ(array.PushBackIndirect({&first}), 1);
  EXPECT_EQ(array[1][0], 7);
}

}  // namespace
}  // namespace xgrammar
