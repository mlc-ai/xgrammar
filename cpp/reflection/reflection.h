#pragma once
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace xgrammar {

template <typename T>
struct member_trait;

enum class member_registry {
  kNone = 0,      // this is default, which has no member trait
  kConfig = 1,    // this is a config with member pointers
  kDelegate = 2,  // this is a delegate member, which is forwarded to delegate
};

namespace details {

// We cannot use `static_assert(false)` even in unreachable code in `if constexpr`.
// See https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2593r1.html
// for more details.
// TL;DR: We use the following `false_v` as a workaround.
template <typename T>
inline constexpr bool false_v = false;

template <typename>
struct is_std_array : std::false_type {};

template <typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

template <typename T>
struct is_std_tuple : std::false_type {};

template <typename... R>
struct is_std_tuple<std::tuple<R...>> : std::true_type {};

// Note that we don't allow empty tables now (that's uncommon).
template <typename X, typename Y, typename... Args>
inline constexpr auto make_member_table(X first, Y second, Args... args) {
  static_assert(sizeof...(args) % 2 == 0, "member table must be even");
  static_assert(std::is_same_v<X, const char*>, "first member must be a c-string");
  static_assert(std::is_member_pointer_v<Y>, "second member must be a member pointer");
  if constexpr (sizeof...(args) == 0) {
    return std::make_tuple(second);
  } else {
    return std::tuple_cat(std::make_tuple(second), make_member_table(args...));
  }
}

template <size_t... Idx, typename Tuple>
inline constexpr auto make_name_table_aux(std::index_sequence<Idx...>, Tuple tuple) {
  return std::array{std::get<Idx * 2>(tuple)...};
}

template <typename... Args>
inline constexpr auto make_name_table(Args... args) {
  constexpr auto N = sizeof...(args);
  static_assert(N % 2 == 0, "name table must be even");
  return make_name_table_aux(std::make_index_sequence<N / 2>{}, std::make_tuple(args...));
}

template <typename T, member_registry R = member_trait<T>::value>
struct member_functor {
  static_assert(false_v<T>, "This specialization should never be used");
  static constexpr auto members = std::tuple{};
  static constexpr auto names = std::array<const char*, 0>{};
  static constexpr auto member_count = 0;
  static constexpr auto has_names = false;
};

template <typename T>
struct member_functor<T, member_registry::kNone> {
  static constexpr auto value = member_registry::kNone;
  static constexpr auto members = std::tuple{};
  static constexpr auto names = std::array<const char*, 0>{};
  static constexpr auto member_count = 0;
  static constexpr auto has_names = false;
};

template <typename T>
struct member_functor<T, member_registry::kConfig> {
  using _trait_type = member_trait<T>;
  using _members_t = std::decay_t<decltype(_trait_type::members)>;
  using _names_t = std::decay_t<decltype(_trait_type::names)>;
  static constexpr auto value = member_registry::kConfig;
  static constexpr auto members = _trait_type::members;
  static constexpr auto names = _trait_type::names;
  static constexpr auto member_count = std::tuple_size_v<_members_t>;
  static constexpr auto has_names = names.size() == member_count;
  // some static_asserts to check the member list and name list
  static_assert(is_std_tuple<_members_t>::value, "Member list must be a tuple");
  static_assert(is_std_array<_names_t>::value, "Name list must be an array");
  static_assert(member_count > 0, "Member list must not be empty");
  static_assert(
      names.size() == member_count || names.size() == 0,
      "Name list must be empty or have the same size as member list"
  );
};

template <typename T>
struct member_functor<T, member_registry::kDelegate> {
  using _trait_type = member_trait<T>;
  using U = typename _trait_type::delegate_type;
  using delegate_type = U;
  static constexpr auto value = member_registry::kDelegate;
  static constexpr auto members = std::tuple{};
  static constexpr auto names = std::array<const char*, 0>{};
  static constexpr auto member_count = 0;
  static constexpr auto has_names = false;
  static U into(const T& obj) { return static_cast<U>(obj); }
  static T from(const U& obj) { return static_cast<T>(obj); }
};

}  // namespace details

// base trait for member traits
template <typename T>
struct member_trait {
  static constexpr auto value = member_registry::kNone;
};

#define XGRAMMAR_MEMBER_TABLE_TEMPLATE(Type, ...)                            \
  struct member_trait<Type> {                                                \
    static constexpr auto value = member_registry::kConfig;                  \
    static constexpr auto members = details::make_member_table(__VA_ARGS__); \
    static constexpr auto names = details::make_name_table(__VA_ARGS__);     \
  }

#define XGRAMMAR_MEMBER_ARRAY_TEMPLATE(Type, ...)                 \
  struct member_trait<Type> {                                     \
    static constexpr auto value = member_registry::kConfig;       \
    static constexpr auto members = std::make_tuple(__VA_ARGS__); \
    static constexpr auto names = std::array<const char*, 0>{};   \
  }

#define XGRAMMAR_MEMBER_DELEGATE_TEMPLATE(Type, Delegate)     \
  struct member_trait<Type> {                                 \
    static constexpr auto value = member_registry::kDelegate; \
    using delegate_type = Delegate;                           \
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

}  // namespace xgrammar
