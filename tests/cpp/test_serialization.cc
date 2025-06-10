#include <gtest/gtest.h>

#include <cstdint>
#include <unordered_set>
#include <vector>

#include "fsm.h"
#include "picojson.h"
#include "reflection/json.h"
#include "support/csr_array.h"

using namespace xgrammar;

TEST(XGrammarReflectionTest, JSONSerialization) {
  // FSMedge is delegated to a int64_t object
  const auto edge = FSMEdge{1, 2, 3};
  auto deserialized_edge = FSMEdge{};

  auto json_obj = AutoJSONSerialize(edge);
  AutoJSONDeserialize(deserialized_edge, json_obj);
  ASSERT_EQ(edge, deserialized_edge);
  ASSERT_TRUE(json_obj.is<int64_t>());

  // CSRArray use a data_ and indptr_ structure
  auto array = CSRArray<int>{};
  array.Insert({0, 1, 2, 3});
  array.Insert({4, 5, 6, 7});
  auto deserialized_array = CSRArray<int>{};

  auto json_array = AutoJSONSerialize(array);
  AutoJSONDeserialize(deserialized_array, json_array);
  ASSERT_EQ(array, deserialized_array);
  ASSERT_TRUE(json_array.is<picojson::object>());
  for (const auto& [key, value] : json_array.get<picojson::object>()) {
    ASSERT_TRUE(value.is<picojson::array>());
    auto& array = value.get<picojson::array>();
    if (key == "data_") {
      ASSERT_EQ(array.size(), 8);
      int i = 0;
      for (const auto& item : array) {
        ASSERT_TRUE(item.is<int64_t>());
        ASSERT_EQ(item.get<int64_t>(), i);
        i++;
      }
    } else {
      ASSERT_TRUE(key == "indptr_");
      ASSERT_EQ(array.size(), 3);
    }
  }

  // C++ standard library types
  auto native_structure = std::vector<std::unordered_set<double>>{{1.0, 2.0, 3.0}, {4.0, 5.0}};
  auto json_native = AutoJSONSerialize(native_structure);
  auto deserialized_native_structure = std::vector<std::unordered_set<double>>{};
  AutoJSONDeserialize(deserialized_native_structure, json_native);
  ASSERT_EQ(native_structure, deserialized_native_structure);

  // optional serialization
  auto optional_value = std::optional<int>{42};
  auto deserialized_optional = std::optional<int>{};

  auto json_optional = AutoJSONSerialize(optional_value);
  AutoJSONDeserialize(deserialized_optional, json_optional);
  ASSERT_TRUE(deserialized_optional.has_value());
  ASSERT_EQ(*deserialized_optional, 42);

  optional_value.reset();
  json_optional = AutoJSONSerialize(optional_value);
  AutoJSONDeserialize(deserialized_optional, json_optional);
  ASSERT_FALSE(deserialized_optional.has_value());
}
