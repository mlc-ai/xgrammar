#pragma once
#include <tuple>
#include <type_traits>
#include <utility>

namespace xgrammar {

struct member_array : std::true_type {};

struct member_table : std::true_type {};

struct member_delegate : std::true_type {};

struct member_subclass : std::true_type {};

template <typename T>
struct member_trait : std::false_type {};

namespace details {

template <typename T>
inline constexpr bool false_v = false;

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

template <auto... MemberPtrs>
struct member_array_impl : member_array {
  static constexpr auto kOffset = std::make_tuple(MemberPtrs...);
};

template <typename T, typename D>
struct member_delegate_impl : member_delegate {
  using delegate_type = D;
};

template <typename T, auto MemberPtr>
struct member_subclass_impl : member_subclass {
  static constexpr auto kSubclass = MemberPtr;
};

#define XGRAMMAR_MEMBER_TABLE_TEMPLATE(Type, ...)                                \
  struct member_trait<Type> : member_table {                                     \
    static constexpr auto kTable = ::xgrammar::details::make_table(__VA_ARGS__); \
  }

#define XGRAMMAR_MEMBER_ARRAY_TEMPLATE(Type, ...)              \
  struct member_trait<Type> : member_array_impl<__VA_ARGS__> { \
    using member_array_impl<__VA_ARGS__>::kOffset;             \
  }

#define XGRAMMAR_MEMBER_DELEGATE_TEMPLATE(Type, Delegate)            \
  struct member_trait<Type> : member_delegate_impl<Type, Delegate> { \
    using member_delegate_impl<Type, Delegate>::delegate_type;       \
  }

#define XGRAMMAR_MEMBER_SUBCLASS_TEMPLATE(Type, MEMBER)                   \
  struct member_trait<Type> : member_subclass_impl<Type, &Type::MEMBER> { \
    using member_subclass_impl<Type, &Type::MEMBER>::kSubclass;           \
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

#define XGRAMMAR_MEMBER_SUBCLASS(Type, MEMBER) \
  template <>                                  \
  XGRAMMAR_MEMBER_SUBCLASS_TEMPLATE(Type, MEMBER)

}  // namespace xgrammar
