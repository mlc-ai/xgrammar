#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "apply_token_bitmask.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(extension, m) {
  m.doc() = "XGrammar extension for MLX";

  m.def(
      "apply_token_bitmask",
      &my_ext::apply_token_bitmask,
      "bitmask"_a,
      "logits"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
          Apply a bitmask to vocabulary logits
          For each position, if the corresponding bit in bitmask is 1,
          keep the logit value; otherwise set to -inf

          Args:
              bitmask (array): Array of uint32 where each bit corresponds to a token
              logits (array): Array of float16 or float32 containing vocabulary logits
              stream (Stream, optional): Stream on which to schedule the operation

          Returns:
              array: Array with same shape and dtype as logits, where values are either
                    the original logit value (if corresponding bit is 1) or -inf
      )"
  );
}
