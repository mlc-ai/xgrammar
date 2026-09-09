/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/qwen.cc
 * \brief Qwen XML parameter format.
 */
#include "xml_tool_calling.h"

namespace xgrammar {
namespace converter_ext {

XMLWrapperParts GetQwenXMLWrapper() { return {"<parameter=", ">", "", "</parameter>"}; }

}  // namespace converter_ext
}  // namespace xgrammar
