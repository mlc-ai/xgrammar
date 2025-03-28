// Copyright © 2024 Apple Inc.

#include "apply_xgrammar_token_bitmask.h"

#include <iostream>
#include <sstream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace my_ext {

///////////////////////////////////////////////////////////////////////////////
// Operation Implementation
///////////////////////////////////////////////////////////////////////////////

mx::array apply_xgrammar_token_bitmask(
    const mx::array& bitmask, const mx::array& logits, mx::StreamOrDevice s
) {
  // Ensure bitmask is uint32
  if (bitmask.dtype() != mx::uint32) {
    throw std::runtime_error("Bitmask must be uint32");
  }

  // Ensure logits is float16 or float32
  if (logits.dtype() != mx::float16 && logits.dtype() != mx::float32) {
    throw std::runtime_error("Logits must be float16 or float32");
  }

  // Ensure the bitmask has enough bits to cover the logits
  size_t bits_needed = logits.size();
  size_t int32s_needed = (bits_needed + 31) / 32;  // Ceiling division by 32
  if (bitmask.size() < int32s_needed) {
    std::ostringstream error_msg;
    error_msg << "Bitmask has insufficient bits: got " << (bitmask.size() * 32) << " bits ("
              << bitmask.size() << " int32s), but need " << bits_needed << " bits ("
              << int32s_needed << " int32s)";
    throw std::runtime_error(error_msg.str());
  }

  // Construct the array as the output of the primitive
  return mx::array(
      logits.shape(),
      logits.dtype(),
      std::make_shared<ApplyXGrammarTokenBitmask>(to_stream(s)),
      {bitmask, logits}
  );
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Common Backend Implementation
///////////////////////////////////////////////////////////////////////////////

template <typename T>
void apply_xgrammar_token_bitmask_cpu_impl(
    const mx::array& bitmask, const mx::array& logits, mx::array& out, mx::Stream stream
) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  // Get the CPU command encoder and register input and output arrays
  auto& encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(bitmask);
  encoder.set_input_array(logits);
  encoder.set_output_array(out);

  // Launch the CPU kernel
  encoder.dispatch([bitmask_ptr = bitmask.data<uint32_t>(),
                    logits_ptr = logits.data<T>(),
                    out_ptr = out.data<T>(),
                    size = out.size()]() {
    // For each element
    for (size_t i = 0; i < size; i++) {
      // Get the uint32 containing the bit we want to check
      uint32_t mask_word = bitmask_ptr[i / 32];
      // Get the bit position within that uint32
      int32_t bit_pos = i % 32;
      // Check if the bit is set
      bool bit_set = (mask_word >> bit_pos) & 1;
      T logit = logits_ptr[i];

      // If bit is 1, keep the logit; otherwise set to -inf
      out_ptr[i] = bit_set ? logit : -std::numeric_limits<T>::infinity();
    }
  });
}

void ApplyXGrammarTokenBitmask::eval_cpu(
    const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs
) {
  auto& bitmask = inputs[0];
  auto& logits = inputs[1];
  auto& out = outputs[0];

  // Dispatch to the correct dtype
  if (out.dtype() == mx::float32) {
    return apply_xgrammar_token_bitmask_cpu_impl<float>(bitmask, logits, out, stream());
  } else if (out.dtype() == mx::float16) {
    return apply_xgrammar_token_bitmask_cpu_impl<mx::float16_t>(bitmask, logits, out, stream());
  } else {
    throw std::runtime_error("ApplyXGrammarTokenBitmask only supports float16 and float32 logits");
  }
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Metal Backend Implementation
///////////////////////////////////////////////////////////////////////////////

#ifdef _METAL_

void ApplyXGrammarTokenBitmask::eval_gpu(
    const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs
) {
  auto& bitmask = inputs[0];
  auto& logits = inputs[1];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  // Allocate output memory
  out.set_data(mx::allocator::malloc(out.nbytes()));

  // Resolve name of kernel
  std::ostringstream kname;
  kname << "apply_xgrammar_token_bitmask_";
  kname << type_to_name(out);

  // Make sure the metal library is available
  d.register_library("mlx_ext");

  // Make a kernel from this metal library
  auto kernel = d.get_kernel(kname.str(), "mlx_ext");

  // Prepare to encode kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Encode input arrays to kernel
  compute_encoder.set_input_array(bitmask, 0);
  compute_encoder.set_input_array(logits, 1);

  // Encode output arrays to kernel
  compute_encoder.set_output_array(out, 2);

  // Launch the grid
  size_t nelem = out.size();
  size_t tgp_size = std::min(nelem, kernel->maxTotalThreadsPerThreadgroup());
  MTL::Size group_dims = MTL::Size(tgp_size, 1, 1);
  MTL::Size grid_dims = MTL::Size(nelem, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

#else

void ApplyXGrammarTokenBitmask::eval_gpu(
    const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs
) {
  throw std::runtime_error("ApplyXGrammarTokenBitmask has no GPU implementation.");
}

#endif

///////////////////////////////////////////////////////////////////////////////
// Primitive Transforms
///////////////////////////////////////////////////////////////////////////////

std::vector<mx::array> ApplyXGrammarTokenBitmask::jvp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& tangents,
    const std::vector<int>& argnums
) {
  // Only propagate gradients through logits, not bitmask
  if (argnums.size() == 1 && argnums[0] == 1) {
    return {apply_xgrammar_token_bitmask(primals[0], tangents[0], stream())};
  }
  return {mx::zeros_like(primals[1])};
}

std::vector<mx::array> ApplyXGrammarTokenBitmask::
    vjp(const std::vector<mx::array>& primals,
        const std::vector<mx::array>& cotangents,
        const std::vector<int>& argnums,
        const std::vector<mx::array>&) {
  // Only propagate gradients through logits, not bitmask
  if (argnums.size() == 1 && argnums[0] == 1) {
    return {apply_xgrammar_token_bitmask(primals[0], cotangents[0], stream())};
  }
  return {mx::zeros_like(primals[0])};
}

std::pair<std::vector<mx::array>, std::vector<int>> ApplyXGrammarTokenBitmask::vmap(
    const std::vector<mx::array>& inputs, const std::vector<int>& axes
) {
  throw std::runtime_error("ApplyXGrammarTokenBitmask has no vmap implementation.");
}

bool ApplyXGrammarTokenBitmask::is_equivalent(const Primitive& other) const {
  return true;  // No parameters to compare
}

}  // namespace my_ext
