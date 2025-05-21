#pragma once
#include <cstddef>
#include <type_traits>

#include "../support/logging.h"
#include "picojson.h"
#include "serialize.h"
#include "utils.h"

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
      std::is_same_v<decltype(JSONDeserialize(std::declval<T&>(), picojson::value{})), void>,
      "JSONDeserialize must be a global function returning void"
  );
  static_assert(
      std::is_default_constructible_v<T>,
      "global deserializer can only apply to a default constructible type"
  );
};

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
  using Functor = details::member_functor<T>;
  static_assert(Functor::value != member_registry::kNone, "No member registry found");
  if constexpr (Functor::value == member_registry::kConfig) {
    if constexpr (Functor::has_names) {
      // normal named struct
      picojson::object obj;
      obj.reserve(Functor::member_count);
      details::visit_tuple(
          [&obj, &value](auto member_ptr, size_t idx) {
            const char* name = Functor::names[idx];
            obj[name] = AutoJSONSerialize(value.*member_ptr);
          },
          Functor::members
      );
      return picojson::value(std::move(obj));
    } else if constexpr (Functor::member_count == 1) {
      // optimize for single member unnamed structs
      constexpr auto member_ptr = std::get<0>(Functor::members);
      return AutoJSONSerialize(value.*member_ptr);
    } else {
      // normal unnamed struct
      picojson::array arr;
      arr.reserve(Functor::member_count);
      details::visit_tuple(
          [&arr, &value](auto member_ptr, size_t) {
            arr.push_back(AutoJSONSerialize(value.*member_ptr));
          },
          Functor::members
      );
      return picojson::value(std::move(arr));
    }
  } else if constexpr (Functor::value == member_registry::kDelegate) {
    // just cast to the delegate type
    return AutoJSONSerialize(Functor::into(value));
  } else {
    // should give an error in this case
    static_assert(details::false_v<T>, "Invalid trait type");
    return picojson::value{};
  }
}

template <typename T>
inline T TraitJSONDeserialize(const picojson::value& value) {
  using Functor = details::member_functor<T>;
  static_assert(Functor::value != member_registry::kNone, "No member registry found");
  if constexpr (Functor::value == member_registry::kConfig) {
    // fallback to result + value
    T result;
    TraitJSONDeserialize(result, value);
    return result;
  } else if constexpr (Functor::value == member_registry::kDelegate) {
    // cast to the delegate type
    using delegate_type = typename Functor::delegate_type;
    return Functor::from(AutoJSONDeserialize<delegate_type>(value));
  } else {
    // should give an error in this case
    static_assert(details::false_v<T>, "Invalid trait type");
    return T{};
  }
}

template <typename T>
inline void TraitJSONDeserialize(T& result, const picojson::value& value) {
  using Functor = details::member_functor<T>;
  static_assert(Functor::value != member_registry::kNone, "No member registry found");
  if constexpr (Functor::value == member_registry::kConfig) {
    if constexpr (Functor::has_names) {
      // normal named struct
      const auto& obj = details::json_as<picojson::object>(value);
      XGRAMMAR_CHECK(obj.size() == Functor::member_count)
          << "Wrong number of members in object in JSONDeserialize"
          << " expected " << Functor::member_count << " but got " << obj.size();
      details::visit_tuple(
          [&obj, &result](auto member_ptr, size_t idx) {
            const char* name = Functor::names[idx];
            AutoJSONDeserialize(result.*member_ptr, details::json_member(obj, name));
          },
          Functor::members
      );
    } else if constexpr (Functor::member_count == 1) {
      // optimize for single member unnamed structs
      constexpr auto member_ptr = std::get<0>(Functor::members);
      using U = std::decay_t<decltype(result.*member_ptr)>;
      result.*member_ptr = AutoJSONDeserialize<U>(value);
    } else {
      // normal unnamed struct
      const auto& arr = details::json_as<picojson::array>(value);
      XGRAMMAR_CHECK(arr.size() == Functor::member_count)
          << "Wrong number of elements in array in JSONDeserialize"
          << " expected " << Functor::member_count << " but got " << arr.size();
      details::visit_tuple(
          [&arr, &result](auto member_ptr, size_t idx) {
            AutoJSONDeserialize(result.*member_ptr, arr[idx]);
          },
          Functor::members
      );
    }
  } else if constexpr (Functor::value == member_registry::kDelegate) {
    // cast to the delegate type
    using Delegate = typename Functor::delegate_type;
    result = Functor::from(AutoJSONDeserialize<Delegate>(value));
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
  } else if constexpr (member_trait<T>::value != member_registry::kNone) {
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
  } else if constexpr (member_trait<T>::value != member_registry::kNone) {
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
  } else if constexpr (member_trait<T>::value != member_registry::kNone) {
    return TraitJSONDeserialize<T>(result, value);
  } else {
    // should give an error in this case
    static_assert(details::false_v<T>, "Cannot deserialize this type");
    return T{};
  }
}

}  // namespace xgrammar
