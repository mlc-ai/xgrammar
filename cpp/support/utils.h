/*!
 * Copyright (c) 2024 by Contributors
 * \file xgrammar/support/utils.h
 * \brief Utility functions.
 */
#ifndef XGRAMMAR_SUPPORT_UTILS_H_
#define XGRAMMAR_SUPPORT_UTILS_H_

#include <cstdint>
#include <functional>
#include <tuple>

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
uint32_t HashCombine(Args... args) {
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

}  // namespace std

#endif  // XGRAMMAR_SUPPORT_UTILS_H_
