/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/builtin_structural_tag.h
 */

#ifndef XGRAMMAR_BUILTIN_STRUCTURAL_TAG_H_
#define XGRAMMAR_BUILTIN_STRUCTURAL_TAG_H_

#include <string>

namespace xgrammar {

/*!
 * \brief Generate built-in structural tag json by model.
 * \param model Built-in style name.
 * \param input_dict_json JSON string that contains tools, builtin_tools, reasoning and
 * force_empty_reasoning.
 * \return StructuralTag JSON string.
 */
std::string GetBuiltinStructuralTagJSON(
    const std::string& model, const std::string& input_dict_json
);

}  // namespace xgrammar

#endif  // XGRAMMAR_BUILTIN_STRUCTURAL_TAG_H_
