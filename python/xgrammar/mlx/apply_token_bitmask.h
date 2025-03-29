#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace my_ext {

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

/**
 * Apply a bitmask to vocabulary logits
 * For each position, if the corresponding bit in bitmask is 1,
 * keep the logit value; otherwise set to -inf
 *
 * Args:
 *   bitmask: Array of int32 where each bit corresponds to a token
 *   logits: Array of float16 or float32 containing vocabulary logits
 *   stream: Stream on which to schedule the operation
 */
mx::array apply_token_bitmask(
    const mx::array& bitmask, const mx::array& logits, mx::StreamOrDevice s = {}
);

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

class ApplyXGrammarTokenBitmask : public mx::Primitive {
 public:
  explicit ApplyXGrammarTokenBitmask(mx::Stream stream) : mx::Primitive(stream) {};

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;

  /** The Jacobian-vector product. */
  std::vector<mx::array> jvp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& tangents,
      const std::vector<int>& argnums
  ) override;

  /** The vector-Jacobian product. */
  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs
  ) override;

  /** Vectorize primitive along given axis */
  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs, const std::vector<int>& axes
  ) override;

  /** Print the primitive. */
  void print(std::ostream& os) override { os << "ApplyXGrammarTokenBitmask"; }

  /** Equivalence check **/
  bool is_equivalent(const mx::Primitive& other) const override;
};

}  // namespace my_ext
