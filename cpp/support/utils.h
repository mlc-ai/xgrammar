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
#include <stdexcept>
#include <tuple>
#include <type_traits>
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

/*!
 * \brief A partial result type that can be used to construct a Result. Holds a result value or an
 * error value.
 * \tparam T The type of the value
 * \tparam IsOk Whether the result is ok
 */
template <typename T, bool IsOk>
struct PartialResult {
  template <typename... Args>
  PartialResult(Args&&... args) : value(std::forward<Args>(args)...) {}
  T value;
};

template <typename E = std::runtime_error, typename... Args>
inline PartialResult<E, false> ResultErr(Args&&... args) {
  return PartialResult<E, false>{std::forward<Args>(args)...};
}

template <typename T, typename... Args>
inline PartialResult<T, true> ResultOk(Args&&... args) {
  return PartialResult<T, true>{std::forward<Args>(args)...};
}

template <typename T>
inline PartialResult<T&&, true> ResultOk(T&& value) {
  return PartialResult<T&&, true>{std::forward<T>(value)};
}

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
  /*! \brief Default constructor is deleted to avoid accidental use */
  Result() = delete;

  /*! \brief Construct from Result::Err */
  template <typename V, typename = std::enable_if_t<std::is_same_v<std::decay_t<V>, E>>>
  Result(PartialResult<V, false>&& partial_result)
      : data_(std::in_place_type<E>, std::forward<V>(partial_result.value)) {}

  /*! \brief Construct from Result::Ok */
  template <typename U, typename = std::enable_if_t<std::is_same_v<std::decay_t<U>, T>>>
  Result(PartialResult<U, true>&& partial_result)
      : data_(std::in_place_type<T>, std::forward<U>(partial_result.value)) {}

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

  /*! \brief Map success value to new type using provided function */
  template <typename F, typename U = std::decay_t<std::invoke_result_t<F, T>>>
  Result<U, E> Map(F&& f) && {
    if (IsOk()) {
      return ResultOk(f(std::get<T>(std::move(data_))));
    }
    return ResultErr(std::get<E>(std::move(data_)));
  }

  /*! \brief Map error value to new type using provided function */
  template <typename F, typename V = std::decay_t<std::invoke_result_t<F, E>>>
  Result<T, V> MapErr(F&& f) && {
    if (IsErr()) {
      return ResultErr(f(std::get<E>(std::move(data_))));
    }
    return ResultOk(std::get<T>(std::move(data_)));
  }

  /*!
   * \brief Convert a Result<U, V> to a Result<T, E>. U should be convertible to T, and V should be
   * convertible to E.
   */
  template <typename U, typename V>
  static Result<T, E> Convert(Result<U, V>&& result) {
    if (result.IsOk()) {
      return ResultOk(std::move(result).Unwrap());
    }
    return ResultErr(std::move(result).UnwrapErr());
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
