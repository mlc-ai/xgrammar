#include <gtest/gtest.h>
#include <picojson.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "fsm.h"
#include "support/compact_2d_array.h"
#include "support/reflection/json_serializer.h"

namespace xgrammar {

bool operator==(const Compact2DArray<int>& lhs, const Compact2DArray<int>& rhs) {
  if (lhs.size() != rhs.size()) return false;
  const std::size_t indptr_size = lhs.size() + 1;
  const auto* lhs_indptr = lhs.indptr();
  const auto* rhs_indptr = rhs.indptr();
  for (std::size_t i = 0; i < indptr_size; ++i) {
    if (lhs_indptr[i] != rhs_indptr[i]) return false;
  }
  const auto data_size = *std::max_element(lhs_indptr, lhs_indptr + indptr_size);
  const auto* lhs_data = lhs.data();
  const auto* rhs_data = rhs.data();
  for (int i = 0; i < int(data_size); ++i) {
    if (lhs_data[i] != rhs_data[i]) return false;
  }
  return true;
}

}  // namespace xgrammar

TEST(XGrammarReflectionTest, JSONSerialization) {
  using namespace xgrammar;

  const auto edge = FSMEdge{1, 2, 3};
  auto deserialized_edge = FSMEdge{};

  auto json_obj = AutoSerializeJSONValue(edge);
  AutoDeserializeJSONValue(deserialized_edge, json_obj);
  ASSERT_EQ(edge, deserialized_edge);

  // Compact2DArray use a data_ and indptr_ structure
  auto array = Compact2DArray<int>{};
  array.PushBack({0, 1, 2, 3});
  array.PushBack({4, 5, 6, 7});
  auto deserialized_array = Compact2DArray<int>{};

  auto json_array = AutoSerializeJSONValue(array);
  AutoDeserializeJSONValue(deserialized_array, json_array);
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
  auto json_native = AutoSerializeJSONValue(native_structure);
  auto deserialized_native_structure = std::vector<std::unordered_set<double>>{};
  AutoDeserializeJSONValue(deserialized_native_structure, json_native);
  ASSERT_EQ(native_structure, deserialized_native_structure);

  // optional serialization
  auto optional_value = std::optional<int>{42};
  auto deserialized_optional = std::optional<int>{};

  auto json_optional = AutoSerializeJSONValue(optional_value);
  AutoDeserializeJSONValue(deserialized_optional, json_optional);
  ASSERT_TRUE(deserialized_optional.has_value());
  ASSERT_EQ(*deserialized_optional, 42);

  optional_value.reset();
  json_optional = AutoSerializeJSONValue(optional_value);
  AutoDeserializeJSONValue(deserialized_optional, json_optional);
  ASSERT_FALSE(deserialized_optional.has_value());
}
