#pragma once
#include <tuple>

#include "serialize.h"

namespace xgrammar {

namespace details {

template <typename, typename = void>
struct has_equality : std::false_type {};

template <typename T>
struct has_equality<T, std::void_t<decltype(std::declval<T>() == std::declval<T>())>>
    : std::true_type {};

}  // namespace details

template <bool AllowDefault = true, typename T>
inline bool TraitCompareEq(const T& lhs, const T& rhs) {
  using Trait = member_trait<T>;
  if constexpr (AllowDefault && details::has_equality<T>::value) {
    return lhs == rhs;
  } else if constexpr (std::is_base_of_v<member_array, Trait>) {
    return std::apply(
        [&lhs, &rhs](auto&&... args) { return ((lhs.*args == rhs.*args) && ...); }, Trait::kTable
    );
  } else if constexpr (std::is_base_of_v<member_table, Trait>) {
    return std::apply(
        [&lhs, &rhs](auto&&... pairs) {
          // we just care about pair.first, so we can use std::ignore
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

}  // namespace xgrammar
