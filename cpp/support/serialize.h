#pragma once
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace xgrammar {

struct member_offset : std::true_type {};

struct member_table : std::true_type {};

struct member_delegate : std::true_type {};

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
struct member_offset_impl : member_offset {
  static constexpr auto kOffset = std::make_tuple(MemberPtrs...);
};

template <typename T, typename D>
struct member_delegate_type_impl : member_delegate {
  static constexpr D into(const T& value) { return static_cast<D>(value); }
  static constexpr T from(const D& value) { return static_cast<T>(value); }
  // in case we need rvalue version due to performance concerns
  static constexpr D into(T&& value) { return static_cast<D>(value); }
  static constexpr T from(D&& value) { return static_cast<T>(value); }
};

template <typename T>
struct member_delegate_type_pimpl : member_delegate {
  using D = typename T::Impl;  // pimpl type
  // special case: dereference the shared_ptr (should never be null)
  static constexpr const D& into(const T& value) { return *value; }
  // special case: need to construct from a shared_ptr
  static constexpr T from(const std::shared_ptr<D>& value) { return T{value}; }
};

template <typename T, auto MemberPtr>
struct member_delegate_object_impl : member_delegate {
  static_assert(std::is_member_object_pointer_v<decltype(MemberPtr)>);
  // use perfect forwarding to allow move-only types, like std::unique_ptr
  template <typename V>
  static constexpr T try_cast_from(V&& value) {
    if constexpr (std::is_convertible_v<V, T>) {
      return static_cast<T>(std::forward<V>(value));
    } else if constexpr (std::is_default_constructible_v<T>) {
      T obj;
      obj.*MemberPtr = std::forward<V>(value);
      return obj;  // RVO here, no copy
    } else {
      static_assert(details::false_v<T>, "Cannot construct object from member pointer!");
      return T{};  // unreachable
    }
  }

  using D = std::decay_t<decltype(std::declval<const T&>().*MemberPtr)>;
  static constexpr const D& into(const T& value) { return value.*MemberPtr; }
  static constexpr T from(const D& value) { return try_cast_from(value); }
  static constexpr T from(D&& value) { return try_cast_from(std::move(value)); }
};

#define XGRAMMAR_MEMBER_TABLE_TEMPLATE(Type, ...)                                \
  struct member_trait<Type> : member_table {                                     \
    static constexpr auto kTable = ::xgrammar::details::make_table(__VA_ARGS__); \
  }

#define XGRAMMAR_MEMBER_OFFSET_TEMPLATE(Type, ...)              \
  struct member_trait<Type> : member_offset_impl<__VA_ARGS__> { \
    using member_offset_impl<__VA_ARGS__>::kOffset;             \
  }

#define XGRAMMAR_MEMBER_DELEGATE_TYPE_TEMPLATE(Type, Delegate)            \
  struct member_trait<Type> : member_delegate_type_impl<Type, Delegate> { \
    using member_delegate_impl<Type, Delegate>::into;                     \
    using member_delegate_impl<Type, Delegate>::from;                     \
  }

#define XGRAMMAR_MEMBER_DELEGATE_OBJECT_TEMPLATE(Type, MEMBER)                   \
  struct member_trait<Type> : member_delegate_object_impl<Type, &Type::MEMBER> { \
    using member_delegate_object_impl<Type, &Type::MEMBER>::into;                \
    using member_delegate_object_impl<Type, &Type::MEMBER>::from;                \
  }

#define XGRAMMAR_MEMBER_TABLE(Type, ...) \
  template <>                            \
  XGRAMMAR_MEMBER_TABLE_TEMPLATE(Type, __VA_ARGS__)

#define XGRAMMAR_MEMBER_OFFSET(Type, ...) \
  template <>                             \
  XGRAMMAR_MEMBER_OFFSET_TEMPLATE(Type, __VA_ARGS__)

#define XGRAMMAR_MEMBER_DELEGATE_TYPE(Type, Delegate) \
  template <>                                         \
  XGRAMMAR_MEMBER_DELEGATE_TYPE_TEMPLATE(Type, Delegate)

#define XGRAMMAR_MEMBER_DELEGATE_OBJECT(Type, MEMBER) \
  template <>                                         \
  XGRAMMAR_MEMBER_DELEGATE_OBJECT_TEMPLATE(Type, MEMBER)

}  // namespace xgrammar
