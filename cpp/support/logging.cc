/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/support/logging.cc
 */
#include "logging.h"
#if (XGRAMMAR_LOG_CUSTOMIZE == 0)
namespace xgrammar {

LogFatal::Entry& LogFatal::GetEntry() {
  static thread_local LogFatal::Entry result;
  return result;
}

const char* LogMessage::level_strings_[] = {
    ": Debug: ",    // XGRAMMAR_LOG_LEVEL_DEBUG
    ": ",           // XGRAMMAR_LOG_LEVEL_INFO
    ": Warning: ",  // XGRAMMAR_LOG_LEVEL_WARNING
    ": Error: ",    // XGRAMMAR_LOG_LEVEL_ERROR
};

}  // namespace xgrammar
#endif  // XGRAMMAR_LOG_CUSTOMIZE
