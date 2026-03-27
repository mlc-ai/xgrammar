/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/builtin_structural_tag.cc
 */

#include <picojson.h>
#include <xgrammar/builtin_structural_tag.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "structural_tag.h"
#include "support/logging.h"

namespace xgrammar {

using FunctionDef = std::pair<std::string, std::string>;

const std::vector<std::string> kThinkExcludeTokens = {"<think>", "</think>"};

std::string StructuralTagToJSON(const StructuralTag& structural_tag) {
  picojson::object obj;
  obj["type"] = picojson::value(StructuralTag::type);
  obj["format"] = FormatToJSONValue(structural_tag.format);
  return picojson::value(std::move(obj)).serialize(false);
}

bool GetBoolWithDefault(const picojson::object& obj, const std::string& key, bool default_value) {
  auto it = obj.find(key);
  if (it == obj.end()) {
    return default_value;
  }
  XGRAMMAR_CHECK(it->second.is<bool>())
      << "The '" << key << "' key in the input_dict must be a boolean.";
  return it->second.get<bool>();
}

std::vector<FunctionDef> ParseFunctionList(
    const picojson::value& input_value, const std::string& key
) {
  std::vector<FunctionDef> functions;
  if (!input_value.is<picojson::object>()) {
    return functions;
  }
  const auto& obj = input_value.get<picojson::object>();
  auto list_it = obj.find(key);
  if (list_it == obj.end()) {
    return functions;
  }
  XGRAMMAR_CHECK(list_it->second.is<picojson::array>())
      << "The '" << key << "' key in the input_dict must be a list.";
  for (const auto& tool : list_it->second.get<picojson::array>()) {
    if (!tool.is<picojson::object>()) {
      continue;
    }
    const auto& tool_obj = tool.get<picojson::object>();
    auto function_it = tool_obj.find("function");
    if (function_it == tool_obj.end() || !function_it->second.is<picojson::object>()) {
      continue;
    }
    const auto& function_obj = function_it->second.get<picojson::object>();
    auto name_it = function_obj.find("name");
    XGRAMMAR_CHECK(name_it != function_obj.end() && name_it->second.is<std::string>())
        << "Each function in the '" << key << "' list must have 'name' key.";

    bool use_true_schema = false;
    auto strict_it = function_obj.find("strict");
    if (strict_it != function_obj.end() && strict_it->second.is<bool>() &&
        strict_it->second.get<bool>() == false) {
      use_true_schema = true;
    }
    auto parameters_it = function_obj.find("parameters");
    if (parameters_it == function_obj.end()) {
      use_true_schema = true;
    }

    std::string schema_json;
    if (use_true_schema) {
      schema_json = "true";
    } else {
      XGRAMMAR_CHECK(
          parameters_it->second.is<picojson::object>() || parameters_it->second.is<bool>()
      ) << "The 'parameters' key in each tool must be a dict or a boolean.";
      schema_json = parameters_it->second.serialize(false);
    }
    functions.push_back({name_it->second.get<std::string>(), std::move(schema_json)});
  }
  return functions;
}

Format BuildSuffixTriggeredOrAnyText(
    const std::vector<TagFormat>& tags, const std::vector<std::string>& triggers
) {
  if (!tags.empty()) {
    return TriggeredTagsFormat(triggers, tags, kThinkExcludeTokens, false, false);
  }
  return AnyTextFormat(kThinkExcludeTokens);
}

StructuralTag BuildLlamaStructuralTag(
    const std::vector<FunctionDef>& tools, bool reasoning, bool force_empty_reasoning
) {
  std::vector<TagFormat> tags;
  tags.reserve(tools.size());
  for (const auto& [name, schema] : tools) {
    tags.emplace_back(
        "{\"name\": \"" + name + "\", \"parameters\": ",
        std::make_shared<Format>(JSONSchemaFormat(schema)),
        std::vector<std::string>{"}"}
    );
  }
  Format suffix_tag = BuildSuffixTriggeredOrAnyText(tags, {"{\"name\": "});
  if (!reasoning) {
    return StructuralTag(std::move(suffix_tag));
  }
  Format prefix_tag = force_empty_reasoning ? Format(ConstStringFormat("<think>\n\n</think>"))
                                            : Format(TagFormat(
                                                  "<think>",
                                                  std::make_shared<Format>(AnyTextFormat({})),
                                                  std::vector<std::string>{"</think>"}
                                              ));
  return StructuralTag(SequenceFormat({std::move(prefix_tag), std::move(suffix_tag)}));
}

StructuralTag BuildKimiStructuralTag(
    const std::vector<FunctionDef>& tools, bool reasoning, bool force_empty_reasoning
) {
  std::vector<TagFormat> tags;
  tags.reserve(tools.size());
  for (const auto& [name, schema] : tools) {
    tags.emplace_back(
        "<|tool_call_begin|>functions." + name + ":",
        std::make_shared<Format>(SequenceFormat(
            {Format(RegexFormat("\\d+")),
             Format(ConstStringFormat("<|tool_call_argument_begin|>")),
             Format(JSONSchemaFormat(schema))}
        )),
        std::vector<std::string>{"<|tool_call_end|>"}
    );
  }
  Format suffix_tag = BuildSuffixTriggeredOrAnyText(tags, {"<|tool_call_begin|>"});
  if (!reasoning) {
    return StructuralTag(std::move(suffix_tag));
  }
  Format prefix_tag = force_empty_reasoning ? Format(ConstStringFormat("<think></think>"))
                                            : Format(TagFormat(
                                                  "<think>",
                                                  std::make_shared<Format>(AnyTextFormat({})),
                                                  std::vector<std::string>{"</think>"}
                                              ));
  return StructuralTag(SequenceFormat({std::move(prefix_tag), std::move(suffix_tag)}));
}

StructuralTag BuildDeepSeekR1StructuralTag(
    const std::vector<FunctionDef>& tools, bool reasoning, bool force_empty_reasoning
) {
  std::vector<TagFormat> tags;
  tags.reserve(tools.size());
  for (const auto& [name, schema] : tools) {
    tags.emplace_back(
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>" + name + "<｜tool▁sep｜>",
        std::make_shared<Format>(JSONSchemaFormat(schema)),
        std::vector<std::string>{"<｜tool▁call▁end｜>"}
    );
  }
  Format suffix_tag =
      BuildSuffixTriggeredOrAnyText(tags, {"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"});
  if (!reasoning) {
    return StructuralTag(std::move(suffix_tag));
  }
  Format prefix_tag = force_empty_reasoning ? Format(ConstStringFormat("</think>"))
                                            : Format(TagFormat(
                                                  "",
                                                  std::make_shared<Format>(AnyTextFormat({})),
                                                  std::vector<std::string>{"</think>"}
                                              ));
  return StructuralTag(SequenceFormat({std::move(prefix_tag), std::move(suffix_tag)}));
}

StructuralTag BuildQwenCoderStructuralTag(
    const std::vector<FunctionDef>& tools, bool reasoning, bool force_empty_reasoning
) {
  std::vector<TagFormat> tags;
  tags.reserve(tools.size());
  for (const auto& [name, schema] : tools) {
    tags.emplace_back(
        "<tool_call>\n<function=" + name + ">\n",
        std::make_shared<Format>(JSONSchemaFormat(schema, "qwen_xml")),
        std::vector<std::string>{"\n</function>\n</tool_call>"}
    );
  }
  Format suffix_tag = BuildSuffixTriggeredOrAnyText(tags, {"<tool_call>\n<function="});
  if (!reasoning) {
    return StructuralTag(std::move(suffix_tag));
  }
  Format prefix_tag = force_empty_reasoning ? Format(ConstStringFormat("<think>\n\n</think>"))
                                            : Format(TagFormat(
                                                  "<think>",
                                                  std::make_shared<Format>(AnyTextFormat({})),
                                                  std::vector<std::string>{"</think>"}
                                              ));
  return StructuralTag(SequenceFormat({std::move(prefix_tag), std::move(suffix_tag)}));
}

StructuralTag BuildQwenStructuralTag(
    const std::vector<FunctionDef>& tools, bool reasoning, bool force_empty_reasoning
) {
  std::vector<TagFormat> tags;
  tags.reserve(tools.size());
  for (const auto& [name, schema] : tools) {
    tags.emplace_back(
        "<tool_call>\n{\"name\": \"" + name + "\", \"arguments\": ",
        std::make_shared<Format>(JSONSchemaFormat(schema)),
        std::vector<std::string>{"}\n</tool_call>"}
    );
  }
  Format suffix_tag = BuildSuffixTriggeredOrAnyText(tags, {"<tool_call>"});
  if (!reasoning) {
    return StructuralTag(std::move(suffix_tag));
  }
  Format prefix_tag = force_empty_reasoning ? Format(ConstStringFormat("<think>\n\n</think>"))
                                            : Format(TagFormat(
                                                  "<think>",
                                                  std::make_shared<Format>(AnyTextFormat({})),
                                                  std::vector<std::string>{"</think>"}
                                              ));
  return StructuralTag(SequenceFormat({std::move(prefix_tag), std::move(suffix_tag)}));
}

StructuralTag BuildHarmonyStructuralTag(
    const std::vector<FunctionDef>& tools,
    const std::vector<FunctionDef>& builtin_tools,
    bool reasoning,
    bool force_empty_reasoning
) {
  std::vector<TagFormat> tags;
  if (reasoning) {
    if (force_empty_reasoning) {
      tags.emplace_back(
          "<|channel|>analysis<|message|>",
          std::make_shared<Format>(ConstStringFormat("<|end|>")),
          std::vector<std::string>{""}
      );
    } else {
      tags.emplace_back(
          "<|channel|>analysis<|message|>",
          std::make_shared<Format>(AnyTextFormat({})),
          std::vector<std::string>{"<|end|>"}
      );
    }
  }
  for (const auto& [name, schema] : tools) {
    tags.emplace_back(
        "<|channel|>commentary to=" + name + "<|constrain|>json<|message|>",
        std::make_shared<Format>(JSONSchemaFormat(schema)),
        std::vector<std::string>{"<|call|>"}
    );
  }
  for (const auto& [name, schema] : builtin_tools) {
    tags.emplace_back(
        "<|channel|>analysis to=" + name + "<|message|>",
        std::make_shared<Format>(JSONSchemaFormat(schema)),
        std::vector<std::string>{"<|call|>"}
    );
  }
  tags.emplace_back(
      "<|channel|>final<|message|>",
      std::make_shared<Format>(AnyTextFormat({})),
      std::vector<std::string>{"<|end|>"}
  );
  return StructuralTag(TagsWithSeparatorFormat(std::move(tags), "<|start|>assistant", false, false)
  );
}

StructuralTag BuildDeepSeekV32StructuralTag(
    const std::vector<FunctionDef>& tools, bool reasoning, bool force_empty_reasoning
) {
  std::vector<TagFormat> tags;
  tags.reserve(tools.size());
  for (const auto& [name, schema] : tools) {
    tags.emplace_back(
        "<｜DSML｜invoke name=\"" + name + "\">\n",
        std::make_shared<Format>(JSONSchemaFormat(schema, "deepseek_xml")),
        std::vector<std::string>{"</｜DSML｜invoke>\n"}
    );
  }
  if (!tags.empty()) {
    Format function_calling_tags = TagsWithSeparatorFormat(tags, "\n", true, false);
    Format suffix_tag = TriggeredTagsFormat(
        {"<｜DSML｜function_calls>"},
        {TagFormat(
            "<｜DSML｜function_calls>\n",
            std::make_shared<Format>(std::move(function_calling_tags)),
            std::vector<std::string>{"</｜DSML｜function_calls>\n"}
        )},
        kThinkExcludeTokens,
        false,
        false
    );
    if (!reasoning) {
      return StructuralTag(std::move(suffix_tag));
    }
    Format prefix_tag = force_empty_reasoning ? Format(ConstStringFormat("<think>\n\n</think>"))
                                              : Format(TagFormat(
                                                    "<think>",
                                                    std::make_shared<Format>(AnyTextFormat({})),
                                                    std::vector<std::string>{"</think>"}
                                                ));
    return StructuralTag(SequenceFormat({std::move(prefix_tag), std::move(suffix_tag)}));
  } else {
    Format suffix_tag = AnyTextFormat(kThinkExcludeTokens);
    if (!reasoning) {
      return StructuralTag(std::move(suffix_tag));
    }
    Format prefix_tag = force_empty_reasoning ? Format(ConstStringFormat("<think>\n\n</think>"))
                                              : Format(TagFormat(
                                                    "<think>",
                                                    std::make_shared<Format>(AnyTextFormat({})),
                                                    std::vector<std::string>{"</think>"}
                                                ));
    return StructuralTag(SequenceFormat({std::move(prefix_tag), std::move(suffix_tag)}));
  }
}

StructuralTag BuildMiniMaxStructuralTag(
    const std::vector<FunctionDef>& tools, bool reasoning, bool force_empty_reasoning
) {
  std::vector<TagFormat> tags;
  tags.reserve(tools.size());
  for (const auto& [name, schema] : tools) {
    tags.emplace_back(
        "<invoke name=\"" + name + "\">\n",
        std::make_shared<Format>(JSONSchemaFormat(schema, "minimax_xml")),
        std::vector<std::string>{"</invoke>\n"}
    );
  }
  if (!tags.empty()) {
    Format function_calling_tags = TagsWithSeparatorFormat(tags, "\n", true, false);
    Format suffix_tag = TriggeredTagsFormat(
        {"<minimax:tool_call>"},
        {TagFormat(
            "<minimax:tool_call>\n",
            std::make_shared<Format>(std::move(function_calling_tags)),
            std::vector<std::string>{"</minimax:tool_call>\n"}
        )},
        kThinkExcludeTokens,
        false,
        false
    );
    if (!reasoning) {
      return StructuralTag(std::move(suffix_tag));
    }
    Format prefix_tag = force_empty_reasoning ? Format(ConstStringFormat("<think>\n\n</think>"))
                                              : Format(TagFormat(
                                                    "<think>",
                                                    std::make_shared<Format>(AnyTextFormat({})),
                                                    std::vector<std::string>{"</think>"}
                                                ));
    return StructuralTag(SequenceFormat({std::move(prefix_tag), std::move(suffix_tag)}));
  } else {
    Format suffix_tag = AnyTextFormat(kThinkExcludeTokens);
    if (!reasoning) {
      return StructuralTag(std::move(suffix_tag));
    }
    Format prefix_tag = force_empty_reasoning ? Format(ConstStringFormat("<think>\n\n</think>"))
                                              : Format(TagFormat(
                                                    "<think>",
                                                    std::make_shared<Format>(AnyTextFormat({})),
                                                    std::vector<std::string>{"</think>"}
                                                ));
    return StructuralTag(SequenceFormat({std::move(prefix_tag), std::move(suffix_tag)}));
  }
}

StructuralTag BuildGLM47StructuralTag(
    const std::vector<FunctionDef>& tools, bool reasoning, bool force_empty_reasoning
) {
  std::vector<TagFormat> tags;
  tags.reserve(tools.size());
  for (const auto& [name, schema] : tools) {
    tags.emplace_back(
        "<tool_call>" + name,
        std::make_shared<Format>(JSONSchemaFormat(schema, "glm_xml")),
        std::vector<std::string>{"</tool_call>"}
    );
  }
  Format suffix_tag = BuildSuffixTriggeredOrAnyText(tags, {"<tool_call>"});
  if (!reasoning) {
    return StructuralTag(std::move(suffix_tag));
  }
  Format prefix_tag = force_empty_reasoning ? Format(ConstStringFormat("<think>\n\n</think>"))
                                            : Format(TagFormat(
                                                  "<think>",
                                                  std::make_shared<Format>(AnyTextFormat({})),
                                                  std::vector<std::string>{"</think>"}
                                              ));
  return StructuralTag(SequenceFormat({std::move(prefix_tag), std::move(suffix_tag)}));
}

std::string GetBuiltinStructuralTagJSON(
    const std::string& model, const std::string& input_dict_json
) {
  picojson::value input_value;
  std::string err = picojson::parse(input_value, input_dict_json);
  XGRAMMAR_CHECK(err.empty()) << "Failed to parse input_dict_json: " << err;
  XGRAMMAR_CHECK(input_value.is<picojson::object>()) << "input_dict_json must be a JSON object.";

  const auto& input_obj = input_value.get<picojson::object>();
  const bool reasoning = GetBoolWithDefault(input_obj, "reasoning", true);
  const bool force_empty_reasoning = GetBoolWithDefault(input_obj, "force_empty_reasoning", false);
  const auto tools = ParseFunctionList(input_value, "tools");
  const auto builtin_tools = ParseFunctionList(input_value, "builtin_tools");

  StructuralTag structural_tag = [&]() -> StructuralTag {
    if (model == "llama") {
      return BuildLlamaStructuralTag(tools, reasoning, force_empty_reasoning);
    }
    if (model == "kimi") {
      return BuildKimiStructuralTag(tools, reasoning, force_empty_reasoning);
    }
    if (model == "deepseek_r1") {
      return BuildDeepSeekR1StructuralTag(tools, reasoning, force_empty_reasoning);
    }
    if (model == "qwen_coder") {
      return BuildQwenCoderStructuralTag(tools, reasoning, force_empty_reasoning);
    }
    if (model == "qwen") {
      return BuildQwenStructuralTag(tools, reasoning, force_empty_reasoning);
    }
    if (model == "harmony") {
      return BuildHarmonyStructuralTag(tools, builtin_tools, reasoning, force_empty_reasoning);
    }
    if (model == "deepseek_v3_2") {
      return BuildDeepSeekV32StructuralTag(tools, reasoning, force_empty_reasoning);
    }
    if (model == "minimax") {
      return BuildMiniMaxStructuralTag(tools, reasoning, force_empty_reasoning);
    }
    if (model == "glm47") {
      return BuildGLM47StructuralTag(tools, reasoning, force_empty_reasoning);
    }
    XGRAMMAR_LOG(FATAL) << "Unknown format type: " << model
                        << ", support types: [llama, qwen, qwen_coder, kimi, deepseek_r1, "
                           "harmony, deepseek_v3_2, minimax, glm47]";
    XGRAMMAR_UNREACHABLE();
  }();

  return StructuralTagToJSON(structural_tag);
}

}  // namespace xgrammar
