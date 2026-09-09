/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/deepseek.cc
 * \brief DeepSeek XML parameter format.
 */
#include "xml_tool_calling.h"

namespace xgrammar {
namespace converter_ext {

XMLWrapperParts GetDeepSeekXMLWrapper() {
  return {"<｜DSML｜parameter name=\"", "", "", "</｜DSML｜parameter>"};
}

const XMLKeySuffix& GetDeepSeekXMLKeySuffix() {
  // TODO(Linzhang): We do not validate the string's value, and we accept both.
  static const XMLKeySuffix suffix = {"\" string=\"", {"true", "false"}, "\">"};
  return suffix;
}

}  // namespace converter_ext
}  // namespace xgrammar
