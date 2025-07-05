#ifndef XGRAMMAR_CONFIG_H_
#define XGRAMMAR_CONFIG_H_

#include <string>

namespace xgrammar {

void SetMaxRecursionDepth(int max_recursion_depth);

int GetMaxRecursionDepth();

std::string GetSerializationVersion();

}  // namespace xgrammar

#endif  // XGRAMMAR_CONFIG_H_
