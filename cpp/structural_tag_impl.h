/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/structural_tag_impl.h
 * \brief The implementation header for the structural tag.
 */

#ifndef XGRAMMAR_STRUCTURAL_TAG_IMPL_H_
#define XGRAMMAR_STRUCTURAL_TAG_IMPL_H_

#include <xgrammar/structural_tag.h>

namespace xgrammar {

// Grammar StructuralTagToGrammar(
//     const std::vector<StructuralTagItem>& tags, const std::vector<std::string>& triggers
// );

Result<Grammar> StructuralTagToGrammar(const StructuralTag& structural_tag);

Result<Grammar> StructuralTagToGrammar(const std::string& structural_tag_json);

}  // namespace xgrammar

#endif  // XGRAMMAR_STRUCTURAL_TAG_IMPL_H_
