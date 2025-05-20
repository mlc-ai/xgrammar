#pragma once
#include <optional>

#include "picojson.h"
#include "serialize.h"

namespace xgrammar {

namespace details {
template <typename, typename = void>
struct has_json_serialize_member : std::false_type {};

template <typename T>
struct has_json_serialize_member<T, std::void_t<decltype(std::declval<const T&>().JSONSerialize())>>
    : std::true_type {
  static_assert(
      std::is_same_v<decltype(std::declval<const T&>().JSONSerialize()), picojson::value>,
      "JSONSerialize must be a const method returning picojson::value"
  );
};

template <typename, typename = void>
struct has_json_serialize_global : std::false_type {};

template <typename T>
struct has_json_serialize_global<T, std::void_t<decltype(JSONSerialize(std::declval<const T&>()))>>
    : std::true_type {
  static_assert(
      std::is_same_v<decltype(JSONSerialize(std::declval<const T&>())), picojson::value>,
      "JSONSerialize must be a global function returning picojson::value"
  );
};

template <typename, typename = void>
struct has_begin_end : std::false_type {};

template <typename T>
struct has_begin_end<
    T,
    std::void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>()))>>
    : std::true_type {};

template <typename, typename = void>
struct is_optional : std::false_type {};

template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};

}  // namespace details

template <typename T>
inline picojson::value AutoJSONSerialize(const T& value);

template <typename T>
inline picojson::value TraitJSONSerialize(const T& value) {
  if constexpr (std::is_base_of_v<member_offset, member_trait<T>>) {
    constexpr auto& kOffset = member_trait<T>::kOffset;
    picojson::array arr;
    arr.reserve(std::tuple_size_v<std::decay_t<decltype(kOffset)>>);
    std::apply(
        [&arr, &value](auto... members) {
          ((arr.push_back(AutoJSONSerialize(value.*members))), ...);
        },
        kOffset
    );
    return picojson::value(std::move(arr));
  } else if constexpr (std::is_base_of_v<member_table, member_trait<T>>) {
    constexpr auto& kTable = member_trait<T>::kTable;
    picojson::object obj;
    std::apply(
        [&obj, &value](auto... pair) {
          ((obj.try_emplace(pair.first, AutoJSONSerialize(value.*pair.second))), ...);
        },
        kTable
    );
    return picojson::value(std::move(obj));
  } else if constexpr (std::is_base_of_v<member_delegate, member_trait<T>>) {
    // cast to the delegate type
    return JSONSerialize(member_trait<T>::into(value));
  } else {
    // should give an error in this case
    static_assert(
        details::false_v<T>,
        "Invalid trait type: should be derived from "
        "member_offset, member_table or member_delegate"
    );
    return picojson::value{};
  }
}

template <typename T>
inline picojson::value AutoJSONSerialize(const T& value) {
  // always prefer user-defined JSONSerialize
  if constexpr (details::has_json_serialize_member<T>::value) {
    return value.JSONSerialize();
  } else if constexpr (details::has_json_serialize_global<T>::value) {
    return JSONSerialize(value);
  } else if constexpr (std::is_same_v<T, bool>) {
    return picojson::value(value);
  } else if constexpr (std::is_integral_v<T>) {
    return picojson::value(static_cast<int64_t>(value));
  } else if constexpr (std::is_floating_point_v<T>) {
    return picojson::value(static_cast<double>(value));
  } else if constexpr (std::is_same_v<T, std::string>) {
    return picojson::value(value);
  } else if constexpr (details::is_optional<T>::value) {
    if (value.has_value()) {
      return AutoJSONSerialize(*value);
    } else {
      return picojson::value{};
    }
  } else if constexpr (details::has_begin_end<T>::value) {
    picojson::array arr;
    arr.reserve(std::size(value));
    for (const auto& item : value) {
      arr.push_back(AutoJSONSerialize(item));
    }
    return picojson::value(std::move(arr));
  } else if constexpr (member_trait<T>::value) {
    return TraitJSONSerialize(value);
  } else {
    // should give an error in this case
    static_assert(details::false_v<T>, "Cannot serialize this type");
    return picojson::value{};
  }
}

}  // namespace xgrammar
