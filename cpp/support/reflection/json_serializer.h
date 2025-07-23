#ifndef XGRAMMAR_REFLECTION_JSON_H_
#define XGRAMMAR_REFLECTION_JSON_H_

#include <picojson.h>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "../logging.h"
#include "reflection.h"

namespace xgrammar {

namespace details {

template <typename, typename = void>
struct has_serialize_json_member : std::false_type {};

template <typename T>
struct has_serialize_json_member<
    T,
    std::void_t<decltype(std::declval<const T&>().SerializeJSONValue())>> : std::true_type {
  static_assert(
      std::is_same_v<decltype(std::declval<const T&>().SerializeJSONValue()), picojson::value>,
      "SerializeJSONValue must be a const method returning picojson::value"
  );
};

template <typename, typename = void>
struct has_serialize_json_global : std::false_type {};

template <typename T>
struct has_serialize_json_global<
    T,
    std::void_t<decltype(SerializeJSONValue(std::declval<const T&>()))>> : std::true_type {
  static_assert(
      std::is_same_v<decltype(SerializeJSONValue(std::declval<const T&>())), picojson::value>,
      "SerializeJSONValue must be a global function returning picojson::value"
  );
};

template <typename, typename = void>
struct has_deserialize_json_member : std::false_type {};

template <typename T>
struct has_deserialize_json_member<
    T,
    std::void_t<decltype(T::DeserializeJSONValue(picojson::value{}))>> : std::true_type {
  static_assert(
      std::is_same_v<decltype(T::DeserializeJSONValue(picojson::value{})), T>,
      "DeserializeJSONValue must be a static method returning T"
  );
};

template <typename T, typename = void>
struct has_deserialize_json_global : std::false_type {};

template <typename T>
struct has_deserialize_json_global<
    T,
    std::void_t<decltype(DeserializeJSONValue(std::declval<T&>(), picojson::value{}))>>
    : std::true_type {
  static_assert(
      std::is_same_v<decltype(DeserializeJSONValue(std::declval<T&>(), picojson::value{})), void>,
      "DeserializeJSONValue must be a global function returning void"
  );
  static_assert(
      std::is_default_constructible_v<T>,
      "global deserializer can only apply to a default constructible type"
  );
};

template <typename T>
inline const T& json_as(const picojson::value& value) {
  XGRAMMAR_CHECK(value.is<T>()) << "Wrong type in DeserializeJSONValue";
  return value.get<T>();
}

inline const picojson::value& json_member(const picojson::object& value, const std::string& name) {
  auto it = value.find(name);
  XGRAMMAR_CHECK(it != value.end()) << "Missing member in DeserializeJSONValue";
  return it->second;
}

}  // namespace details

template <typename T>
inline picojson::value AutoSerializeJSONValue(const T& value);

template <typename T>
inline void AutoDeserializeJSONValue(T& result, const picojson::value& value);

template <typename T>
inline picojson::value TraitSerializeJSONValue(const T& value);

template <typename T>
inline void TraitDeserializeJSONValue(T& result, const picojson::value& value);

template <typename T>
inline picojson::value TraitSerializeJSONValue(const T& value) {
  using Functor = details::member_functor<T>;
  if constexpr (Functor::value == member_type::kConfig) {
    if constexpr (Functor::has_names) {
      // normal named struct
      picojson::object obj;
      obj.reserve(Functor::member_count);
      details::visit_config<T>([&](auto ptr, const char* name, std::size_t) {
        obj[name] = AutoSerializeJSONValue(value.*ptr);
      });
      return picojson::value(std::move(obj));
    } else if constexpr (Functor::member_count == 1) {
      // optimize for single member unnamed structs
      constexpr auto member_ptr = std::get<0>(Functor::members);
      return AutoSerializeJSONValue(value.*member_ptr);
    } else {
      // normal unnamed struct
      picojson::array arr;
      arr.resize(Functor::member_count);
      details::visit_config<T>([&](auto ptr, const char*, std::size_t idx) {
        arr[idx] = AutoSerializeJSONValue(value.*ptr);
      });
      return picojson::value(std::move(arr));
    }
  } else {
    // should give an error in this case
    static_assert(details::false_v<T>, "Invalid trait type");
    return picojson::value{};
  }
}

template <typename T>
inline void TraitDeserializeJSONValue(T& result, const picojson::value& value) {
  using Functor = details::member_functor<T>;
  if constexpr (Functor::value == member_type::kConfig) {
    if constexpr (Functor::has_names) {
      // normal named struct
      const auto& obj = details::json_as<picojson::object>(value);
      XGRAMMAR_CHECK(obj.size() == Functor::member_count)
          << "Wrong number of members in object in DeserializeJSONValue" << " expected "
          << Functor::member_count << " but got " << obj.size();
      details::visit_config<T>([&](auto ptr, const char* name, std::size_t) {
        AutoDeserializeJSONValue(result.*ptr, details::json_member(obj, name));
      });
    } else if constexpr (Functor::member_count == 1) {
      // optimize for single member unnamed structs
      constexpr auto member_ptr = std::get<0>(Functor::members);
      AutoDeserializeJSONValue(result.*member_ptr, value);
    } else {
      // normal unnamed struct
      const auto& arr = details::json_as<picojson::array>(value);
      XGRAMMAR_CHECK(arr.size() == Functor::member_count)
          << "Wrong number of elements in array in DeserializeJSONValue" << " expected "
          << Functor::member_count << " but got " << arr.size();
      details::visit_config<T>([&arr, &result](auto ptr, const char*, size_t idx) {
        AutoDeserializeJSONValue(result.*ptr, arr[idx]);
      });
    }
  } else {
    // should give an error in this case
    static_assert(details::false_v<T>, "Invalid trait type");
  }
}

template <typename T>
inline picojson::value AutoSerializeJSONValue(const T& value) {
  // always prefer user-defined SerializeJSONValue
  if constexpr (details::has_serialize_json_member<T>::value) {
    return value.SerializeJSONValue();
  } else if constexpr (details::has_serialize_json_global<T>::value) {
    return SerializeJSONValue(value);
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
      return AutoSerializeJSONValue(*value);
    } else {
      return picojson::value{};
    }
  } else if constexpr (details::is_vector<T>::value) {
    picojson::array arr;
    arr.reserve(value.size());
    for (const auto& item : value) {
      arr.push_back(AutoSerializeJSONValue(item));
    }
    return picojson::value(std::move(arr));
  } else if constexpr (details::is_unordered_set<T>::value) {
    std::vector<const typename T::value_type*> ptr_vec;
    ptr_vec.reserve(value.size());
    for (const auto& item : value) ptr_vec.push_back(&item);
    std::sort(ptr_vec.begin(), ptr_vec.end(), [](const auto* a, const auto* b) { return *a < *b; });
    picojson::array arr;
    arr.reserve(value.size());
    for (const auto* ptr : ptr_vec) {
      arr.push_back(AutoSerializeJSONValue(*ptr));
    }
    return picojson::value(std::move(arr));
  } else if constexpr (details::is_unordered_map<T>::value) {
    std::vector<const typename T::value_type*> ptr_vec;
    ptr_vec.reserve(value.size());
    for (const auto& item : value) ptr_vec.push_back(&item);
    std::sort(ptr_vec.begin(), ptr_vec.end(), [](const auto* a, const auto* b) {
      return a->first < b->first;
    });
    picojson::array arr;
    arr.reserve(value.size() * 2);
    for (const auto* ptr : ptr_vec) {
      const auto& [key, item] = *ptr;
      arr.push_back(AutoSerializeJSONValue(key));
      arr.push_back(AutoSerializeJSONValue(item));
    }
    return picojson::value(std::move(arr));
  } else if constexpr (member_trait<T>::value != member_type::kNone) {
    return TraitSerializeJSONValue(value);
  } else {
    // should give an error in this case
    static_assert(details::false_v<T>, "Cannot serialize this type");
    return picojson::value{};
  }
}

template <typename T>
inline void AutoDeserializeJSONValue(T& result, const picojson::value& value) {
  static_assert(!std::is_const_v<T>, "Cannot deserialize into a const type");
  if constexpr (details::has_deserialize_json_member<T>::value) {
    result = T::DeserializeJSONValue(value);
  } else if constexpr (details::has_deserialize_json_global<T>::value) {
    DeserializeJSONValue(result, value);
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
      AutoDeserializeJSONValue(result.emplace(), value);
    }
  } else if constexpr (details::is_vector<T>::value) {
    result.clear();
    const auto& arr = details::json_as<picojson::array>(value);
    result.reserve(arr.size());
    for (const auto& item : details::json_as<picojson::array>(value)) {
      auto& item_value = result.emplace_back();
      AutoDeserializeJSONValue(item_value, item);
    }
  } else if constexpr (details::is_unordered_set<T>::value) {
    result.clear();
    const auto& arr = details::json_as<picojson::array>(value);
    result.reserve(arr.size());
    for (const auto& item : arr) {
      typename T::value_type item_value;
      AutoDeserializeJSONValue(item_value, item);
      result.emplace(std::move(item_value));
    }
  } else if constexpr (details::is_unordered_map<T>::value) {
    const auto& arr = details::json_as<picojson::array>(value);
    XGRAMMAR_CHECK(arr.size() % 2 == 0)
        << "Wrong number of elements in array in DeserializeJSONValue";
    result.clear();
    result.reserve(arr.size() / 2);
    for (size_t i = 0; i < arr.size(); i += 2) {
      // typename T::value_type item_value;
      typename T::key_type key_value;
      AutoDeserializeJSONValue(key_value, arr[i + 0]);
      typename T::mapped_type item_value;
      AutoDeserializeJSONValue(item_value, arr[i + 1]);
      result.emplace(std::move(key_value), std::move(item_value));
    }
  } else if constexpr (member_trait<T>::value != member_type::kNone) {
    return TraitDeserializeJSONValue(result, value);
  } else {
    // should give an error in this case
    static_assert(details::false_v<T>, "Cannot deserialize this type");
  }
}

}  // namespace xgrammar

#endif  // XGRAMMAR_REFLECTION_JSON_H_
