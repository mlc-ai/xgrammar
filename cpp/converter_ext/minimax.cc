/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/minimax.cc
 * \brief MiniMax XML parameter format.
 */
#include "xml_tool_calling.h"

namespace xgrammar {
namespace converter_ext {

XMLWrapperParts GetMiniMaxXMLWrapper() { return {"<parameter name=\"", "\">", "", "</parameter>"}; }

}  // namespace converter_ext
}  // namespace xgrammar
