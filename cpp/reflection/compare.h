#pragma once
#include <tuple>

#include "details/compare.h"  // IWYU pragma: export
#include "reflection.h"

namespace xgrammar {

template <bool SkipDefault = false, typename T>
inline bool TraitCompareEq(const T& lhs, const T& rhs) {
  using Functor = details::member_functor<T>;
  // skip default is useful for automatically generated equality operators
  // this will prevent infinite recursion
  if constexpr (!SkipDefault && details::has_equality<T>::value) {
    return lhs == rhs;
  } else if constexpr (Functor::value == member_type::kConfig) {
    return std::apply(
        [&lhs, &rhs](auto&&... member_ptr) {
          return (TraitCompareEq(lhs.*member_ptr, rhs.*member_ptr) && ...);
        },
        Functor::members
    );
  } else if constexpr (Functor::value == member_type::kDelegate) {
    return TraitCompareEq(Functor::into(lhs), Functor::into(rhs));
  } else {
    static_assert(details::false_v<T>, "Cannot compare this type");
    return false;
  }
}

#define XGRAMMAR_GENERATE_EQUALITY(T)                                                \
  bool operator==(const T& lhs, const T& rhs) {                                      \
    /* skip default compare to prevent infinite recursion  */                        \
    return &lhs == &rhs || xgrammar::TraitCompareEq</*SkipDefault=*/true>(lhs, rhs); \
  }                                                                                  \
  static_assert(true, "Don't forget the semicolon after the macro")

}  // namespace xgrammar
