#pragma once
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace xgrammar::details {

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

}  // namespace xgrammar::details
