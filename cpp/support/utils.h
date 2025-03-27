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
#include <optional>
#include <tuple>
#include <type_traits>

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
