#pragma once
#include "reflection_details.h"  // IWYU pragma: export

namespace xgrammar {

// base trait for member traits
template <typename T>
struct member_trait {
  static constexpr auto value = member_type::kNone;
};

/*!
 * @brief Macros to define member traits for types.
 *
 * These macros are used to define the structural information of types
 * for serialization and reflection purposes.
 *
 * - `XGRAMMAR_MEMBER_TABLE`: Defines a type with a table of (name, member pointer) pairs.
 * - `XGRAMMAR_MEMBER_ARRAY`: Defines a type with an array of member pointer.
 * - `XGRAMMAR_MEMBER_DELEGATE`: Defines a type that can be converted from/into another type.
 *
 * For template types, use the version with `_TEMPLATE` suffix instead.
 *
 * For real-world examples, see `tests/cpp/test_reflection.h` and `tests/cpp/test_reflection.cc`.
 */

#define XGRAMMAR_MEMBER_TABLE_TEMPLATE(Type, ...)                            \
  struct member_trait<Type> {                                                \
    static constexpr auto value = member_type::kConfig;                      \
    static constexpr auto members = details::make_member_table(__VA_ARGS__); \
    static constexpr auto names = details::make_name_table(__VA_ARGS__);     \
  }

#define XGRAMMAR_MEMBER_ARRAY_TEMPLATE(Type, ...)                 \
  struct member_trait<Type> {                                     \
    static constexpr auto value = member_type::kConfig;           \
    static constexpr auto members = std::make_tuple(__VA_ARGS__); \
    static constexpr auto names = std::array<const char*, 0>{};   \
  }

#define XGRAMMAR_MEMBER_DELEGATE_TEMPLATE(Type, Delegate) \
  struct member_trait<Type> {                             \
    static constexpr auto value = member_type::kDelegate; \
    using delegate_type = Delegate;                       \
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
