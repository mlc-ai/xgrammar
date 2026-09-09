/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/glm.cc
 * \brief GLM XML parameter format.
 */
#include "xml_tool_calling.h"

namespace xgrammar {
namespace converter_ext {

XMLWrapperParts GetGLMXMLWrapper() {
  return {"<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"};
}

}  // namespace converter_ext
}  // namespace xgrammar
