/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/muse.cc
 * \brief Muse ATEM XML parameter format and attribute escaping.
 */
#include <array>
#include <string>
#include <utility>

#include "../json_schema_converter_ext.h"

namespace xgrammar {
namespace converter_ext {

XMLWrapper GetMuseXMLWrapper() {
  return {"<atem:parameter name=\"", "\">", "", "</atem:parameter>"};
}

namespace {

// Replace every non-overlapping occurrence of `from`, scanning left to right like Jinja's
// `replace` filter.
std::string ReplaceAll(const std::string& text, const std::string& from, const std::string& to) {
  std::string result;
  result.reserve(text.size());
  size_t start = 0;
  for (size_t pos = text.find(from); pos != std::string::npos; pos = text.find(from, start)) {
    result.append(text, start, pos - start);
    result += to;
    start = pos + from.size();
  }
  result.append(text, start, std::string::npos);
  return result;
}

}  // namespace

std::string EscapeMuseAttrValue(const std::string& value) {
  // The chat template's esc_attr: esc, then esc_header, then the XML attribute entities, applied
  // in this order.
  static const std::array<std::pair<const char*, const char*>, 11> kReplacements = {{
      // esc: break special tokens and media placeholders.
      {"<|", "< |"},
      {"<image>", "< image>"},
      {"<video>", "< video>"},
      {"<audio>", "< audio>"},
      // esc_header: keep the attribute on one line.
      {"\r", " "},
      {"\n", " "},
      // esc_attr: XML attribute entities.
      {"&", "&amp;"},
      {"\"", "&quot;"},
      {"'", "&#39;"},
      {"<", "&lt;"},
      {">", "&gt;"},
  }};
  std::string escaped = value;
  for (const auto& [from, to] : kReplacements) {
    escaped = ReplaceAll(escaped, from, to);
  }
  return escaped;
}

}  // namespace converter_ext
}  // namespace xgrammar
