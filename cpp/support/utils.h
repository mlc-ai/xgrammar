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
#include <tuple>
#include <type_traits>

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

// Return the memory consumption in heap memory of a container.
template <typename Container>
inline constexpr std::size_t MemorySize(const Container& container) {
  using Element_t = std::decay_t<decltype(*std::begin(container))>;
  static_assert(std::is_trivially_copyable_v<Element_t>, "Element type must be trivial");
  static_assert(!std::is_trivially_copyable_v<Container>, "Container type must not be trivial");
  return sizeof(Element_t) * std::size(container);
}

template <typename Tp>
inline constexpr std::size_t MemorySize(const std::optional<Tp>& range) {
  return range.has_value() ? MemorySize(*range) : 0;
}

/*!
 * \brief A Result type similar to Rust's Result, representing either success (Ok) or failure (Err).
 * \tparam T The type of the success value
 */
template <typename T>
class Result {
 public:
  /*! \brief Construct a success Result */
  static Result Ok(T value) { return Result(std::move(value), nullptr); }

  /*! \brief Construct an error Result */
  static Result Err(std::shared_ptr<Error> error) { return Result(std::nullopt, std::move(error)); }

  /*! \brief Check if Result contains success value */
  bool IsOk() const { return value_.has_value(); }

  /*! \brief Check if Result contains error */
  bool IsErr() const { return error_ != nullptr; }

  /*! \brief Get the success value, or terminate if this is an error */
  const T& Unwrap() const& {
    if (!IsOk()) {
      XGRAMMAR_LOG(FATAL) << "Called Unwrap() on an Err value";
      XGRAMMAR_UNREACHABLE();
    }
    return *value_;
  }

  /*! \brief Get the success value, or terminate if this is an error */
  T&& Unwrap() && {
    if (!IsOk()) {
      XGRAMMAR_LOG(FATAL) << "Called Unwrap() on an Err value";
      XGRAMMAR_UNREACHABLE();
    }
    return std::move(*value_);
  }

  /*! \brief Get the error value as a pointer, or terminate if this is not an error */
  std::shared_ptr<Error> UnwrapErr() const& {
    if (!IsErr()) {
      XGRAMMAR_LOG(FATAL) << "Called UnwrapErr() on an Ok value";
      XGRAMMAR_UNREACHABLE();
    }
    return error_;
  }

  /*! \brief Get the error value as a pointer, or terminate if this is not an error */
  std::shared_ptr<Error> UnwrapErr() && {
    if (!IsErr()) {
      XGRAMMAR_LOG(FATAL) << "Called UnwrapErr() on an Ok value";
      XGRAMMAR_UNREACHABLE();
    }
    return std::move(error_);
  }

  /*! \brief Get the success value if present, otherwise return the provided default */
  T UnwrapOr(T default_value) const { return IsOk() ? *value_ : default_value; }

  /*! \brief Map success value to new type using provided function */
  template <typename U, typename F>
  Result<U> Map(F&& f) const {
    if (IsOk()) {
      return Result<U>::Ok(f(*value_));
    }
    return Result<U>::Err(error_);
  }

  /*! \brief Map error value to new type using provided function */
  template <typename F>
  Result<T> MapErr(F&& f) const {
    if (IsErr()) {
      return Result<T>::Err(f(error_));
    }
    return Result<T>::Ok(*value_);
  }

 private:
  Result(std::optional<T> value, std::shared_ptr<Error> error)
      : value_(std::move(value)), error_(std::move(error)) {}

  std::optional<T> value_;
  std::shared_ptr<Error> error_;
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
