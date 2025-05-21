#pragma once
#include <tuple>
#include <type_traits>
#include <utility>

namespace xgrammar {

namespace details {

// We cannot use `static_assert(false)` even in unreachable code in `if constexpr`.
// See https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2593r1.html
// for more details.
// TL;DR: We use the following `false_v` as a workaround.
template <typename T>
inline constexpr bool false_v = false;

// Make the table for member pointers, packing each 2 elements into a pair.
// Note that we don't allow empty tables now (that's uncommon).
template <typename X, typename Y, typename... Args>
inline constexpr auto make_table(X first, Y second, Args... args) {
  // pack each 2 elements into a pair
  static_assert(sizeof...(args) % 2 == 0, "member table must be even");
  static_assert(std::is_same_v<X, const char*>, "first member must be a c-string");
  static_assert(std::is_member_pointer_v<Y>, "second member must be a member pointer");
  const auto pair = std::make_pair(first, second);
  if constexpr (sizeof...(args) == 0) {
    return std::make_tuple(pair);
  } else {
    return std::tuple_cat(std::make_tuple(pair), make_table(args...));
  }
}

}  // namespace details

// base trait for member traits
// for true_types, we define a few patterns to describe the member traits.
template <typename T>
struct member_trait : std::false_type {};

// Implementation should contains `kArray` as the tuple of member pointers.
struct member_array : std::true_type {};

// Implementation should contains `kTable` as a tuple of pairs of (name, member pointer).
struct member_table : std::true_type {};

// Implementation should contains type `delegate_type` as the type of the delegate.
struct member_delegate : std::true_type {};

// Implementation should contains `kSubclass` as the member pointer to the subclass.
// This will delegate the serialization to the member (usually the only one) of this class.
struct member_subclass : std::true_type {};

#define XGRAMMAR_MEMBER_TABLE_TEMPLATE(Type, ...)                                \
  struct member_trait<Type> : member_table {                                     \
    static constexpr auto kTable = ::xgrammar::details::make_table(__VA_ARGS__); \
  }

#define XGRAMMAR_MEMBER_ARRAY_TEMPLATE(Type, ...)                \
  struct member_trait<Type> : member_array {                     \
    static constexpr auto kArray = std::make_tuple(__VA_ARGS__); \
  }

#define XGRAMMAR_MEMBER_DELEGATE_TEMPLATE(Type, Delegate) \
  struct member_trait<Type> : member_delegate {           \
    using delegate_type = Delegate;                       \
  }

#define XGRAMMAR_MEMBER_SUBCLASS_TEMPLATE(Type, MemberPtr) \
  struct member_trait<Type> : member_subclass {            \
    static constexpr auto kSubclass = MemberPtr;           \
  }

#define XGRAMMAR_MEMBER_TABLE(Type, ...) \
  template <>                            \
  XGRAMMAR_MEMBER_TABLE_TEMPLATE(Type, __VA_ARGS__)

#define XGRAMMAR_MEMBER_ARRAY(Type, ...) \
  template <>                            \
  XGRAMMAR_MEMBER_ARRAY_TEMPLATE(Type, __VA_ARGS__)

#define XGRAMMAR_MEMBER_DELEGATE(Type, Delegate) \
  template <>                                    \
  XGRAMMAR_MEMBER_DELEGATE_TEMPLATE(Type, Delegate)

#define XGRAMMAR_MEMBER_SUBCLASS(Type, MemberPtr) \
  template <>                                     \
  XGRAMMAR_MEMBER_SUBCLASS_TEMPLATE(Type, MemberPtr)

}  // namespace xgrammar
