#pragma once
#include <tuple>
#include <type_traits>

#include "serialize.h"

namespace xgrammar {

namespace details {

template <typename, typename = void>
struct has_equality : std::false_type {};

template <typename T>
struct has_equality<T, std::void_t<decltype(std::declval<T>() == std::declval<T>())>>
    : std::true_type {};

}  // namespace details

template <bool SkipDefault = false, typename T>
inline bool TraitCompareEq(const T& lhs, const T& rhs) {
  using Trait = member_trait<T>;
  // a shortcut for non-default compare
  if constexpr (SkipDefault) {
    if (&lhs == &rhs) return true;
  }

  if constexpr (!SkipDefault && details::has_equality<T>::value) {
    return lhs == rhs;
  } else if constexpr (std::is_base_of_v<member_array, Trait>) {
    return std::apply(
        [&lhs, &rhs](auto&&... args) { return ((lhs.*args == rhs.*args) && ...); }, Trait::kArray
    );
  } else if constexpr (std::is_base_of_v<member_table, Trait>) {
    return std::apply(
        [&lhs, &rhs](auto&&... pairs) {
          return ((lhs.*std::get<1>(pairs) == rhs.*std::get<1>(pairs)) && ...);
        },
        Trait::kTable
    );
  } else if constexpr (std::is_base_of_v<member_delegate, Trait>) {
    using Delegate = typename Trait::delegate_type;
    return TraitCompareEq(static_cast<Delegate>(lhs), static_cast<Delegate>(rhs));
  } else if constexpr (std::is_base_of_v<member_subclass, Trait>) {
    constexpr auto kSubclass = Trait::kSubclass;
    return TraitCompareEq(lhs.*kSubclass, rhs.*kSubclass);
  } else {
    static_assert(details::false_v<T>, "Cannot compare this type");
    return false;
  }
}

#define XGRAMMAR_GENERATE_EQUALITY(T)                                \
  bool operator==(const T& lhs, const T& rhs) {                      \
    /* skip default compare to prevent infinite recursion  */        \
    return xgrammar::TraitCompareEq</*SkipDefault=*/true>(lhs, rhs); \
  }                                                                  \
  static_assert(true, "Don't forget the semicolon after the macro")

}  // namespace xgrammar
