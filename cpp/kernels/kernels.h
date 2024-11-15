/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/kernels/kernels.h
 * \brief The header for the support of grammar-guided generation.
 */

#ifndef XGRAMMAR_KERNELS_KERNELS_H_
#define XGRAMMAR_KERNELS_KERNELS_H_

namespace xgrammar {

enum class DTypeFlag : int { DTYPE_FLOAT16 = 0, DTYPE_FLOAT32 = 1, DTYPE_FLOAT64 = 2 };

void apply_token_bitmask_inplace(
    int* bitmask, void* logits, DTypeFlag dtype_flag, int batch_size, int vocab_size
);

}  // namespace xgrammar

#endif  // XGRAMMAR_KERNELS_KERNELS_H_
