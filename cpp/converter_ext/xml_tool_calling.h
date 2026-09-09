/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/xml_tool_calling.h
 * \brief Internal helpers for model-specific XML parameter formats.
 */
#ifndef XGRAMMAR_CONVERTER_EXT_XML_TOOL_CALLING_H_
#define XGRAMMAR_CONVERTER_EXT_XML_TOOL_CALLING_H_

#include <array>
#include <vector>

namespace xgrammar {
namespace converter_ext {

// Key prefix, key suffix, value prefix, and closing suffix, in XMLWrapper field order.
using XMLWrapperParts = std::array<const char*, 4>;

XMLWrapperParts GetQwenXMLWrapper();
XMLWrapperParts GetMiniMaxXMLWrapper();
XMLWrapperParts GetDeepSeekXMLWrapper();
XMLWrapperParts GetGLMXMLWrapper();
XMLWrapperParts GetCohereXMLWrapper();
XMLWrapperParts GetKimiK3XMLWrapper();

struct XMLKeySuffix {
  const char* prefix;
  std::vector<const char*> values;
  const char* suffix;
};

const XMLKeySuffix& GetDeepSeekXMLKeySuffix();
const XMLKeySuffix& GetKimiK3XMLKeySuffix();

}  // namespace converter_ext
}  // namespace xgrammar

#endif  // XGRAMMAR_CONVERTER_EXT_XML_TOOL_CALLING_H_
