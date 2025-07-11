#ifndef XGRAMMAR_REFLECTION_H_
#define XGRAMMAR_REFLECTION_H_

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
 *
 * For template types, use the version with `_TEMPLATE` suffix instead.
 *
 * @example
 *
 * ```cpp
 * // @note Example of using XGRAMMAR_MEMBER_TABLE to register (name, member pointer) pairs
 * // You can use any string as the name, and it will be used in serialization.
 * struct SimpleClass {
 *   int a;
 *   double b;
 * };
 * XGRAMMAR_MEMBER_TABLE(SimpleClass, "name_a", &SimpleClass::a, "name_b", &SimpleClass::b);
 *
 *
 * // @note Example of using XGRAMMAR_MEMBER_ARRAY to register member pointers
 * // In a derived class, you can use the same macro to register members from the base class
 * struct Derived: SimpleClass {
 *   std::string c;
 * };
 * XGRAMMAR_MEMBER_ARRAY(Derived, &Derived::a, &Derived::b, &Derived::c);
 *
 *
 * // @note Example of using XGRAMMAR_MEMBER_ARRAY_TEMPLATE to register members in a template type
 * // If the default constructor/member is private, you need to declare a friend for member_trait.
 * template <typename T>
 * struct TemplateClass {
 * private:
 *   T value;
 *   TemplateClass() = default;
 *   friend struct member_trait<TemplateClass>;
 * };
 * template <typename T>
 * XGRAMMAR_MEMBER_ARRAY_TEMPLATE(TemplateClass<T>, &TemplateClass<T>::value);
 * ```
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

#define XGRAMMAR_MEMBER_TABLE(Type, ...) \
  template <>                            \
  XGRAMMAR_MEMBER_TABLE_TEMPLATE(Type, __VA_ARGS__)

#define XGRAMMAR_MEMBER_ARRAY(Type, ...) \
  template <>                            \
  XGRAMMAR_MEMBER_ARRAY_TEMPLATE(Type, __VA_ARGS__)

}  // namespace xgrammar

#endif  // XGRAMMAR_REFLECTION_H_
