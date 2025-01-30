/*!
 * Copyright (c) 2024 by Contributors
 * \file xgrammar/support/cpptrace.h
 * \details This file is an encapsulation of the cpptrace library. It helps debugging. This file
 * can only be included when XGRAMMAR_ENABLE_CPPTRACE is set to ON, and only support Linux and
 * RelWithDebugInfo or Debug build.
 */
#ifndef XGRAMMAR_SUPPORT_CPPTRACE_H_
#define XGRAMMAR_SUPPORT_CPPTRACE_H_

#include <cpptrace/cpptrace.hpp>

namespace xgrammar {

inline void PrintTrace() { cpptrace::generate_trace().print(); }

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_CPPTRACE_H_
