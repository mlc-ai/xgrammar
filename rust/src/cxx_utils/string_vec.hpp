#ifndef XGRAMMAR_RS_CXX_UTILS_STRING_VEC_H_
#define XGRAMMAR_RS_CXX_UTILS_STRING_VEC_H_

#include <memory>
#include <string>
#include <vector>

namespace cxx_utils {

inline std::unique_ptr<std::vector<std::string>> new_string_vector() {
  return std::make_unique<std::vector<std::string>>();
}

inline void string_vec_reserve(std::vector<std::string>& vec, size_t n) {
  vec.reserve(n);
}

inline void string_vec_push_bytes(
    std::vector<std::string>& vec,
    const char* ptr,
    size_t len
) {
  vec.emplace_back(ptr, ptr + len);
}

} // namespace cxx_utils

#endif // XGRAMMAR_RS_CXX_UTILS_STRING_VEC_H_
