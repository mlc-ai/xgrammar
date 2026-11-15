#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "support/container.h"

using namespace xgrammar;

TEST(XGrammarContainerTest, ListEraseReleasesValue) {
  List<std::shared_ptr<int>> list;
  auto value = std::make_shared<int>(42);

  auto it = list.PushBack(value);
  EXPECT_EQ(value.use_count(), 2);

  list.Erase(it);
  // The list must not keep the erased value alive in its recycled slot.
  EXPECT_EQ(value.use_count(), 1);
}

TEST(XGrammarContainerTest, ListEraseThenReuseSlot) {
  List<std::shared_ptr<int>> list;
  std::vector<std::shared_ptr<int>> values;
  for (int i = 0; i < 4; ++i) {
    values.push_back(std::make_shared<int>(i));
    list.PushBack(values.back());
  }

  // Erase from the front, as an LRU eviction does.
  for (auto it = list.begin(); it != list.end();) {
    it = list.Erase(it);
  }
  for (const auto& v : values) {
    EXPECT_EQ(v.use_count(), 1);
  }

  // Recycled slots still work and hold only the new values.
  auto fresh = std::make_shared<int>(7);
  auto it = list.PushBack(fresh);
  EXPECT_EQ(**it, 7);
  EXPECT_EQ(fresh.use_count(), 2);
  EXPECT_EQ(list.begin(), it);
}
