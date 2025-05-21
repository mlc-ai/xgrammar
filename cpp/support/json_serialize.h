#pragma once
#include <cstddef>
#include <optional>
#include <type_traits>
#include <unordered_set>

#include "dynamic_bitset.h"
#include "logging.h"
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
struct has_json_deserialize_member : std::false_type {};

template <typename T>
struct has_json_deserialize_member<T, std::void_t<decltype(T::JSONDeserialize(picojson::value{}))>>
    : std::true_type {
  static_assert(
      std::is_same_v<decltype(T::JSONDeserialize(picojson::value{})), T>,
      "JSONDeserialize must be a static method returning T"
  );
};

template <typename T, typename = void>
struct has_json_deserialize_global : std::false_type {};

template <typename T>
struct has_json_deserialize_global<
    T,
    std::void_t<decltype(JSONDeserialize(std::declval<T&>(), picojson::value{}))>>
    : std::true_type {
  static_assert(
      std::is_same_v<decltype(JSONDeserialize(std::declval<T>(), picojson::value{})), void>,
      "JSONDeserialize must be a global function returning void"
  );
  static_assert(
      std::is_default_constructible_v<T>,
      "global deserializer can only apply to a default constructible type"
  );
};

template <typename>
struct is_optional : std::false_type {};

template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};

template <typename>
struct is_vector : std::false_type {};

template <typename... R>
struct is_vector<std::vector<R...>> : std::true_type {};

template <typename>
struct is_unordered_map : std::false_type {};

template <typename... R>
struct is_unordered_map<std::unordered_map<R...>> : std::true_type {};

template <typename T>
struct is_unordered_set : std::false_type {};

template <typename... R>
struct is_unordered_set<std::unordered_set<R...>> : std::true_type {};

template <typename T>
inline const T& json_as(const picojson::value& value) {
  XGRAMMAR_CHECK(value.is<T>()) << "Wrong type in JSONDeserialize";
  return value.get<T>();
}

inline const picojson::value& json_member(const picojson::object& value, const std::string& name) {
  auto it = value.find(name);
  XGRAMMAR_CHECK(it != value.end()) << "Missing member in JSONDeserialize";
  return it->second;
}

template <typename Fn, typename Tuple, std::size_t... Idx>
inline constexpr void visit_tuple(Fn&& f, const Tuple& t, std::index_sequence<Idx...>) {
  (f(std::get<Idx>(t), Idx), ...);
}

template <typename Fn, typename Tuple>
inline constexpr void visit_tuple(Fn&& f, const Tuple& t) {
  return visit_tuple(
      std::forward<Fn>(f), t, std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>()
  );
}

}  // namespace details

template <typename T>
inline picojson::value AutoJSONSerialize(const T& value);

template <typename T>
inline T AutoJSONDeserialize(const picojson::value& value);

template <typename T>
inline void AutoJSONDeserialize(T& result, const picojson::value& value);

template <typename T>
inline void TraitJSONDeserialize(T& result, const picojson::value& value);

template <typename T>
inline picojson::value TraitJSONSerialize(const T& value) {
  using Trait = member_trait<T>;
  if constexpr (std::is_base_of_v<member_array, Trait>) {
    picojson::array arr;
    arr.reserve(std::tuple_size_v<std::decay_t<decltype(Trait::kArray)>>);
    details::visit_tuple(
        [&arr, &value](auto member_ptr, size_t) {
          arr.push_back(AutoJSONSerialize(value.*member_ptr));
        },
        Trait::kArray
    );
    return picojson::value(std::move(arr));
  } else if constexpr (std::is_base_of_v<member_table, Trait>) {
    picojson::object obj;
    obj.reserve(std::tuple_size_v<std::decay_t<decltype(Trait::kTable)>>);
    details::visit_tuple(
        [&obj, &value](auto pair, size_t) {
          auto&& [name, member_ptr] = pair;
          obj[name] = AutoJSONSerialize(value.*member_ptr);
        },
        Trait::kTable
    );
    return picojson::value(std::move(obj));
  } else if constexpr (std::is_base_of_v<member_delegate, Trait>) {
    using Delegate = typename Trait::delegate_type;
    return AutoJSONSerialize(static_cast<Delegate>(value));
  } else if constexpr (std::is_base_of_v<member_subclass, Trait>) {
    constexpr auto kSubclass = Trait::kSubclass;
    return AutoJSONSerialize(value.*kSubclass);
  } else {
    // should give an error in this case
    static_assert(details::false_v<T>, "Invalid trait type");
    return picojson::value{};
  }
}

template <typename T>
inline T TraitJSONDeserialize(const picojson::value& value) {
  using Trait = member_trait<T>;
  if constexpr (std::is_base_of_v<member_array, Trait>) {
    T result;
    // fallback to result + value
    TraitJSONDeserialize(result, value);
    return result;
  } else if constexpr (std::is_base_of_v<member_table, Trait>) {
    T result;
    // fallback to result + value
    TraitJSONDeserialize(result, value);
    return result;
  } else if constexpr (std::is_base_of_v<member_delegate, Trait>) {
    // cast to the delegate type
    using Delegate = typename Trait::delegate_type;
    return static_cast<T>(AutoJSONDeserialize<Delegate>(value));
  } else if constexpr (std::is_base_of_v<member_subclass, Trait>) {
    // fallback to result + value
    T result;
    TraitJSONDeserialize(result, value);
    return result;
  } else {
    // should give an error in this case
    static_assert(details::false_v<T>, "Invalid trait type");
    return T{};
  }
}

template <typename T>
inline void TraitJSONDeserialize(T& result, const picojson::value& value) {
  using Trait = member_trait<T>;
  if constexpr (std::is_base_of_v<member_array, Trait>) {
    const auto& arr = details::json_as<picojson::array>(value);
    XGRAMMAR_CHECK(arr.size() == std::tuple_size_v<std::decay_t<decltype(Trait::kArray)>>)
        << "Wrong number of elements in array in JSONDeserialize";
    details::visit_tuple(
        [&arr, &result](auto member_ptr, size_t idx) {
          using U = std::decay_t<decltype(result.*member_ptr)>;
          result.*member_ptr = AutoJSONDeserialize<U>(arr[idx]);
        },
        Trait::kArray
    );
  } else if constexpr (std::is_base_of_v<member_table, Trait>) {
    const auto& obj = details::json_as<picojson::object>(value);
    XGRAMMAR_CHECK(obj.size() == std::tuple_size_v<std::decay_t<decltype(Trait::kTable)>>)
        << "Wrong number of members in object in JSONDeserialize";
    details::visit_tuple(
        [&obj, &result](auto pair, size_t) {
          auto&& [name, member_ptr] = pair;
          using U = std::decay_t<decltype(result.*member_ptr)>;
          result.*member_ptr = AutoJSONDeserialize<U>(details::json_member(obj, name));
        },
        Trait::kTable
    );
  } else if constexpr (std::is_base_of_v<member_delegate, Trait>) {
    // cast to the delegate type
    using Delegate = typename Trait::delegate_type;
    result = static_cast<T>(AutoJSONDeserialize<Delegate>(value));
  } else if constexpr (std::is_base_of_v<member_subclass, Trait>) {
    constexpr auto kSubclass = Trait::kSubclass;
    return AutoJSONDeserialize(result.*kSubclass, value);
  } else {
    // should give an error in this case
    static_assert(details::false_v<T>, "Invalid trait type");
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
  } else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
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
  } else if constexpr (details::is_vector<T>::value) {
    picojson::array arr;
    arr.reserve(value.size());
    for (const auto& item : value) {
      arr.push_back(AutoJSONSerialize(item));
    }
    return picojson::value(std::move(arr));
  } else if constexpr (details::is_unordered_set<T>::value) {
    picojson::array arr;
    arr.reserve(value.size());
    for (const auto& item : value) {
      arr.push_back(AutoJSONSerialize(item));
    }
    return picojson::value(std::move(arr));
  } else if constexpr (details::is_unordered_map<T>::value) {
    picojson::array arr;
    arr.reserve(value.size() * 2);
    for (const auto& [key, item] : value) {
      arr.push_back(AutoJSONSerialize(key));
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

template <typename T>
inline T AutoJSONDeserialize(const picojson::value& value) {
  if constexpr (details::has_json_deserialize_member<T>::value) {
    return T::JSONDeserialize(value);
  } else if constexpr (details::has_json_deserialize_global<T>::value) {
    T obj;
    JSONDeserialize(obj, value);
    return obj;
  } else if constexpr (std::is_same_v<T, bool>) {
    return details::json_as<bool>(value);
  } else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
    return static_cast<T>(details::json_as<int64_t>(value));
  } else if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(details::json_as<double>(value));
  } else if constexpr (std::is_same_v<T, std::string>) {
    return details::json_as<std::string>(value);
  } else if constexpr (details::is_optional<T>::value || details::is_vector<T>::value ||
                       details::is_unordered_set<T>::value || details::is_unordered_map<T>::value) {
    T result;
    AutoJSONDeserialize(result, value);
    return result;
  } else if constexpr (member_trait<T>::value) {
    return TraitJSONDeserialize<T>(value);
  } else {
    // should give an error in this case
    static_assert(details::false_v<T>, "Cannot deserialize this type");
    return T{};
  }
}

template <typename T>
inline void AutoJSONDeserialize(T& result, const picojson::value& value) {
  if constexpr (details::has_json_deserialize_member<T>::value) {
    result = T::JSONDeserialize(value);
  } else if constexpr (details::has_json_deserialize_global<T>::value) {
    JSONDeserialize(result, value);
  } else if constexpr (std::is_same_v<T, bool>) {
    result = details::json_as<bool>(value);
  } else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
    result = static_cast<T>(details::json_as<int64_t>(value));
  } else if constexpr (std::is_floating_point_v<T>) {
    result = static_cast<T>(details::json_as<double>(value));
  } else if constexpr (std::is_same_v<T, std::string>) {
    result = details::json_as<std::string>(value);
  } else if constexpr (details::is_optional<T>::value) {
    if (value.is<picojson::null>()) {
      result.reset();
    } else {
      AutoJSONDeserialize(result.emplace(), value);
    }
  } else if constexpr (details::is_vector<T>::value) {
    result.clear();
    const auto& arr = details::json_as<picojson::array>(value);
    result.reserve(arr.size());
    for (const auto& item : details::json_as<picojson::array>(value)) {
      result.emplace_back(AutoJSONDeserialize<typename T::value_type>(item));
    }
  } else if constexpr (details::is_unordered_set<T>::value) {
    result.clear();
    const auto& arr = details::json_as<picojson::array>(value);
    result.reserve(arr.size());
    for (const auto& item : arr) {
      result.emplace(AutoJSONDeserialize<typename T::value_type>(item));
    }
  } else if constexpr (details::is_unordered_map<T>::value) {
    const auto& arr = details::json_as<picojson::array>(value);
    XGRAMMAR_CHECK(arr.size() % 2 == 0) << "Wrong number of elements in array in JSONDeserialize";
    result.clear();
    result.reserve(arr.size() / 2);
    for (size_t i = 0; i < arr.size(); i += 2) {
      result.try_emplace(
          AutoJSONDeserialize<typename T::key_type>(arr[i]),
          AutoJSONDeserialize<typename T::mapped_type>(arr[i + 1])
      );
    }
  } else if constexpr (member_trait<T>::value) {
    return TraitJSONDeserialize<T>(result, value);
  } else {
    // should give an error in this case
    static_assert(details::false_v<T>, "Cannot deserialize this type");
    return T{};
  }
}

static_assert(details::has_json_serialize_member<DynamicBitset>::value);

}  // namespace xgrammar
