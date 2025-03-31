#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename T>
[[kernel]] void apply_token_bitmask_float32(
    device const uint32_t* bitmask [[buffer(0)]],
    device const float* logits [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  // Get the uint32 containing the bit we want to check
  uint32_t mask_word = bitmask[index / 32];
  // Get the bit position within that uint32
  int32_t bit_pos = index % 32;
  // Check if the bit is set
  bool bit_set = (mask_word >> bit_pos) & 1;
  float logit = logits[index];

  // If bit is 1, keep the logit; otherwise set to -inf
  out[index] = bit_set ? logit : -INFINITY;
}

template <typename T>
[[kernel]] void apply_token_bitmask_float16(
    device const uint32_t* bitmask [[buffer(0)]],
    device const half* logits [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  // Get the uint32 containing the bit we want to check
  uint32_t mask_word = bitmask[index / 32];
  // Get the bit position within that uint32
  int32_t bit_pos = index % 32;
  // Check if the bit is set
  bool bit_set = (mask_word >> bit_pos) & 1;
  half logit = logits[index];

  // If bit is 1, keep the logit; otherwise set to -inf
  out[index] = bit_set ? logit : -INFINITY;
}

#define instantiate_apply_token_bitmask(type_name, type) \
  instantiate_kernel(                                             \
      "apply_token_bitmask_" #type_name,                 \
      apply_token_bitmask_##type_name,                   \
      type)

instantiate_apply_token_bitmask(float32, float)
    instantiate_apply_token_bitmask(float16, half)
