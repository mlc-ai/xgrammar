/*!
 * Copyright (c) 2024 by Contributors
 * \file xgrammar/support/utils.h
 * \brief Utility functions.
 */
#ifndef XGRAMMAR_SUPPORT_UTILS_H_
#define XGRAMMAR_SUPPORT_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>

#include "logging.h"

namespace xgrammar {

/*!
 * \brief Hash and combine value into seed.
 * \ref https://www.boost.org/doc/libs/1_84_0/boost/intrusive/detail/hash_combine.hpp
 */
inline void HashCombineBinary(uint32_t& seed, uint32_t value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/*!
 * \brief Find the hash sum of several uint32_t args.
 */
template <typename... Args>
inline uint32_t HashCombine(Args... args) {
  uint32_t seed = 0;
  (..., HashCombineBinary(seed, args));
  return seed;
}

// Sometimes GCC fails to detect some branches will not return, such as when we use LOG(FATAL)
// to raise an error. This macro manually mark them as unreachable to avoid warnings.
#ifdef __GNUC__
#define XGRAMMAR_UNREACHABLE() __builtin_unreachable()
#else
#define XGRAMMAR_UNREACHABLE()
#endif

/******************* MemorySize Procotol *******************/

template <typename T, typename = std::enable_if_t<std::is_trivially_copyable_v<T>>>
inline constexpr std::size_t MemorySize(const T& value) {
  return 0;
}

/*!
 * \brief Compute the memory consumption in heap memory. This function is specialized for
 * containers.
 * \tparam Container The container type.
 * \param container The container.
 * \return The memory consumption in heap memory of the container.
 */
template <typename T>
inline constexpr std::size_t MemorySize(const std::vector<T>& container) {
  std::size_t result = sizeof(T) * container.size();
  for (const auto& item : container) {
    result += MemorySize(item);
  }
  return result;
}

template <typename T>
inline constexpr std::size_t MemorySize(const std::unordered_set<T>& container) {
  return sizeof(T) * container.size();
}

/*!
 * \brief Compute the memory consumption in heap memory. This function is specialized for
 * std::optional.
 * \tparam Tp The type of the optional.
 * \param range The optional.
 * \return The memory consumption in heap memory of the optional.
 */
template <typename T>
inline constexpr std::size_t MemorySize(const std::optional<T>& optional_value) {
  return optional_value.has_value() ? MemorySize(*optional_value) : 0;
}

/*!
 * \brief An error class that contains a type. The type can be an enum.
 */
template <typename T>
class TypedError : public std::runtime_error {
 public:
  explicit TypedError(T type, const std::string& msg) : std::runtime_error(msg), type_(type) {}
  const T& Type() const noexcept { return type_; }

 private:
  T type_;
};

namespace detail {

/*!
 * \brief Check if the parameter pack has exactly one type X. The generic version is false.
 */
template <typename X, typename... Args>
constexpr bool is_exactly_one_type = false;

/*!
 * \brief Partial specialization of is_exactly_one_type that returns true if the parameter pack
 * has exactly one type Arg when it is exactly one X.
 */
template <typename X, typename Arg>
constexpr bool is_exactly_one_type<X, Arg> = std::is_same_v<std::decay_t<Arg>, X>;

}  // namespace detail

/*!
 * \brief An always-move Result type similar to Rust's Result, representing either success (Ok) or
 * failure (Err). It always uses move semantics for the success and error values.
 * \tparam T The type of the success value
 * \tparam E The type of the error value
 *
 * \note The Ok and Err constructor, and all methods of this class (except for ValueRef and ErrRef)
 * accept only rvalue references as parameters for performance reasons. You should use std::move to
 * convert a Result to an rvalue reference before invoking these methods. Examples for move
 * semantics are shown below.
 *
 * \example Construct a success result with a rvalue reference
 * \code
 * T value;
 * return Result<T, std::string>::Ok(std::move(value));
 * \endcode
 * \example Construct a error result with a rvalue reference of std::runtime_error
 * \code
 * std::runtime_error error_msg = std::runtime_error("Error");
 * return Result<T>::Err(std::move(error_msg));
 * \endcode
 * \example Construct a error result with a std::runtime_error object constructed with a string
 * \code
 * std::string error_msg = "Error";
 * return Result<T>::Err(std::move(error_msg));
 * \endcode
 * \example Unwrap the rvalue reference of the result
 * \code
 * Result<T> result = func();
 * if (result.IsOk()) {
 *   T result_val = std::move(result).Unwrap();
 * } else {
 *   std::runtime_error error_msg = std::move(result).UnwrapErr();
 * }
 * \endcode
 */
template <typename T, typename E = std::runtime_error>
class Result {
 private:
  static_assert(!std::is_same_v<T, E>, "T and E cannot be the same type");

 public:
  /*! \brief Construct a success Result by moving T */
  static Result Ok(T&& value) { return Result(std::in_place_type<T>, std::move(value)); }

  /*!
   * \brief Construct a success Result by invoking T's constructor.
   * \note This method cannot accept T as the only argument, therefore avoiding user passing const
   * T& and invoke the copy constructor.
   */
  template <typename... Args, typename = std::enable_if_t<!detail::is_exactly_one_type<T, Args...>>>
  static Result Ok(Args&&... args) {
    return Result(std::in_place_type<T>, std::forward<Args>(args)...);
  }

  /*! \brief Construct an error Result by moving E */
  static Result Err(E&& value) { return Result(std::in_place_type<E>, std::move(value)); }

  /*!
   * \brief Construct an error Result by invoking E's constructor.
   * \note This method cannot accept E as the only argument, therefore avoiding user passing const
   * E& and invoke the copy constructor.
   */
  template <typename... Args, typename = std::enable_if_t<!detail::is_exactly_one_type<E, Args...>>>
  static Result Err(Args&&... args) {
    return Result(std::in_place_type<E>, std::forward<Args>(args)...);
  }

  /*! \brief Check if Result contains success value */
  bool IsOk() const { return std::holds_alternative<T>(data_); }

  /*! \brief Check if Result contains error */
  bool IsErr() const { return std::holds_alternative<E>(data_); }

  /*! \brief Get the success value. It assumes (or checks if in debug mode) the result is ok. */
  T Unwrap() && {
    XGRAMMAR_DCHECK(IsOk()) << "Called Unwrap() on an Err value";
    return std::get<T>(std::move(data_));
  }

  /*! \brief Get the error value. It assumes (or checks if in debug mode) the result is an error. */
  E UnwrapErr() && {
    XGRAMMAR_DCHECK(IsErr()) << "Called UnwrapErr() on an Ok value";
    return std::get<E>(std::move(data_));
  }

  /*! \brief Get the success value if present, otherwise return the provided default */
  T UnwrapOr(T default_value) && {
    return IsOk() ? std::get<T>(std::move(data_)) : std::move(default_value);
  }

  /*!
   * \brief Get the success value, or throw E if it is an error.
   * \note It's useful when exposing Result values to Python.
   */
  T UnwrapOrThrow() && {
    if (!IsOk()) {
      throw std::get<E>(std::move(data_));
    }
    return std::get<T>(std::move(data_));
  }

  /*! \brief Map success value to new type using provided function */
  template <typename F, typename U = std::decay_t<std::invoke_result_t<F, T>>>
  Result<U, E> Map(F&& f) && {
    if (IsOk()) {
      return Result<U, E>::Ok(f(std::get<T>(std::move(data_))));
    }
    return Result<U, E>::Err(std::get<E>(std::move(data_)));
  }

  /*! \brief Map error value to new type using provided function */
  template <typename F, typename V = std::decay_t<std::invoke_result_t<F, E>>>
  Result<T, V> MapErr(F&& f) && {
    if (IsErr()) {
      return Result<T, V>::Err(f(std::get<E>(std::move(data_))));
    }
    return Result<T, V>::Ok(std::get<T>(std::move(data_)));
  }

  /*!
   * \brief Convert a Result<U, V> to a Result<T, E>. U should be convertible to T, and V should be
   * convertible to E.
   */
  template <typename U, typename V>
  static Result<T, E> Convert(Result<U, V>&& result) {
    if (result.IsOk()) {
      return Result<T, E>::Ok(std::move(result).Unwrap());
    }
    return Result<T, E>::Err(std::move(result).UnwrapErr());
  }

  /*! \brief Get a std::variant<T, E> from the result. */
  std::variant<T, E> ToVariant() && { return std::move(data_); }

  /*!
   * \brief Get a reference to the success value. It assumes (or checks if in debug mode) the
   * result is ok.
   */
  T& ValueRef() & {
    XGRAMMAR_DCHECK(IsOk()) << "Called ValueRef() on an Err value";
    return std::get<T>(data_);
  }

  /*!
   * \brief Get a reference to the error value. It assumes (or checks if in debug mode) the
   * result is an error.
   */
  E& ErrRef() & {
    XGRAMMAR_DCHECK(IsErr()) << "Called ErrRef() on an Ok value";
    return std::get<E>(data_);
  }

 private:
  // in-place construct T in variant
  template <typename... Args>
  explicit Result(std::in_place_type_t<T>, Args&&... args)
      : data_(std::in_place_type<T>, std::forward<Args>(args)...) {}

  // in-place construct E in variant
  template <typename... Args>
  explicit Result(std::in_place_type_t<E>, Args&&... args)
      : data_(std::in_place_type<E>, std::forward<Args>(args)...) {}

  std::variant<T, E> data_;
};

}  // namespace xgrammar

namespace std {

template <typename T, typename U>
struct hash<std::pair<T, U>> {
  size_t operator()(const std::pair<T, U>& pair) const {
    return xgrammar::HashCombine(std::hash<T>{}(pair.first), std::hash<U>{}(pair.second));
  }
};

template <typename... Args>
struct hash<std::tuple<Args...>> {
  size_t operator()(const std::tuple<Args...>& tuple) const {
    return std::apply(
        [](const Args&... args) { return xgrammar::HashCombine(std::hash<Args>{}(args)...); }, tuple
    );
  }
};

template <typename T>
struct hash<std::vector<T>> {
  size_t operator()(const std::vector<T>& vec) const {
    uint32_t seed = 0;
    for (const auto& item : vec) {
      xgrammar::HashCombineBinary(seed, std::hash<T>{}(item));
    }
    return seed;
  }
};

}  // namespace std

#endif  // XGRAMMAR_SUPPORT_UTILS_H_
