/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/muse.cc
 * \brief Muse ATEM XML parameter format.
 */
#include "../json_schema_converter_ext.h"

namespace xgrammar {
namespace converter_ext {

XMLWrapper GetMuseXMLWrapper() {
  return {"<atem:parameter name=\"", "\">", "", "</atem:parameter>"};
}

}  // namespace converter_ext
}  // namespace xgrammar
