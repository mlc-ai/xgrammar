#ifndef XGRAMMAR_RS_CXX_UTILS_STRUCTURAL_TAG_H_
#define XGRAMMAR_RS_CXX_UTILS_STRUCTURAL_TAG_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xgrammar/xgrammar.h"

namespace cxx_utils {

inline std::unique_ptr<std::vector<xgrammar::StructuralTagItem>>
new_structural_tag_vector() {
  return std::make_unique<std::vector<xgrammar::StructuralTagItem>>();
}

inline void structural_tag_vec_reserve(
    std::vector<xgrammar::StructuralTagItem>& vector,
    size_t reserve_size
) {
  vector.reserve(reserve_size);
}

inline void structural_tag_vec_push(
    std::vector<xgrammar::StructuralTagItem>& vector,
    const std::string& begin,
    const std::string& schema,
    const std::string& end
) {
  vector.emplace_back(xgrammar::StructuralTagItem{begin, schema, end});
}

} // namespace cxx_utils

#endif // XGRAMMAR_RS_CXX_UTILS_STRUCTURAL_TAG_H_
