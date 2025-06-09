// IWYU pragma: private
#pragma once
#include <type_traits>

namespace xgrammar::details {

template <typename, typename = void>
struct has_equality : std::false_type {};

template <typename T>
struct has_equality<T, std::void_t<decltype(std::declval<T>() == std::declval<T>())>>
    : std::true_type {};

}  // namespace xgrammar::details
