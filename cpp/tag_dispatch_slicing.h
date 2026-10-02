/*!
 * Copyright (c) 2026 by Contributors
 * \file xgrammar/tag_dispatch_slicing.h
 * \brief Vocabulary preprocessing for tag-dispatch token masks.
 */
#ifndef XGRAMMAR_TAG_DISPATCH_SLICING_H_
#define XGRAMMAR_TAG_DISPATCH_SLICING_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "support/dynamic_bitset.h"

namespace xgrammar {

/*!
 * \brief Mark tokens containing no complete pattern starting at byte offset one or later.
 * \note Bits are indexed by vocabulary position, not token ID. An empty token is always marked,
 * matching the tag-dispatch mask fast path. Patterns include both triggers and excluded strings.
 */
DynamicBitset BuildTagDispatchSecondSlicingBitset(
    const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
    const std::vector<std::string>& patterns
);

}  // namespace xgrammar

#endif  // XGRAMMAR_TAG_DISPATCH_SLICING_H_
