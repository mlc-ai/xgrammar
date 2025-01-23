/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/structural_tag.h
 * \brief The header for the definition of the structural tag.
 */
#ifndef XGRAMMAR_STRUCTURAL_TAG_H_
#define XGRAMMAR_STRUCTURAL_TAG_H_

#include <xgrammar/xgrammar.h>

#include <string>
#include <vector>

namespace xgrammar {

Grammar StructuralTagToGrammar(
    const std::vector<StructuralTagItem>& tags, const std::vector<std::string>& triggers
);

}  // namespace xgrammar

#endif  // XGRAMMAR_STRUCTURAL_TAG_H_
