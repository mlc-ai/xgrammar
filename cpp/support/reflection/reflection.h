#ifndef XGRAMMAR_SUPPORT_REFLECTION_REFLECTION_H_
#define XGRAMMAR_SUPPORT_REFLECTION_REFLECTION_H_

#include "reflection_details.h"  // IWYU pragma: export

namespace xgrammar {

/**
 * \brief Base trait for member traits.
 *
 * \tparam T the type whose members are being reflected
 * \details Provides a default trait indicating no members.
 */
template <typename T>
struct member_trait {
  static constexpr auto value = member_type::kNone;
};

/**
 * \brief Macros to define member traits for types.
 * \details These macros are used to define the structural information of types
 * for serialization and reflection purposes.
 *
 * Macros:
 *   - \c XGRAMMAR_MEMBER_TABLE: Defines a type with a table of (name, member pointer) pairs.
 *   - \c XGRAMMAR_MEMBER_ARRAY: Defines a type with an array of member pointers.
 *
 * Use the `_TEMPLATE` variants for template types.
 *
 * \example
 * \code{.cpp}
 * // Example of using XGRAMMAR_MEMBER_TABLE to register (name, member pointer) pairs
 * struct SimpleClass {
 *   int a;
 *   double b;
 * };
 * XGRAMMAR_MEMBER_TABLE(SimpleClass, "name_a", &SimpleClass::a, "name_b", &SimpleClass::b);
 *
 * // Or register members as an array with XGRAMMAR_MEMBER_ARRAY
 * XGRAMMAR_MEMBER_ARRAY(SimpleClass, &SimpleClass::a, &SimpleClass::b);
 *
 * // Example of using XGRAMMAR_MEMBER_ARRAY to register members from a derived class
 * struct Derived : SimpleClass {
 *   std::string c;
 * };
 * XGRAMMAR_MEMBER_TABLE(Derived, "name_a", &Derived::a, "name_b", &Derived::b, "name_c",
 * &Derived::c);
 *
 * // Example of using XGRAMMAR_MEMBER_ARRAY_TEMPLATE for a template type
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
 * \endcode
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

#endif  // XGRAMMAR_SUPPORT_REFLECTION_REFLECTION_H_
