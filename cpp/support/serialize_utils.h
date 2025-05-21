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

}  // namespace xgrammar::details
