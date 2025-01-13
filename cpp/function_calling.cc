/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/function_calling.cc
 */

#include <xgrammar/function_calling.h>

#include <string_view>
#include <unordered_map>

#include "picojson.h"
#include "support/logging.h"

namespace xgrammar {

/******************* Parser *******************/

std::vector<std::pair<std::string, std::unordered_map<std::string, std::string>>> parse_message(
    const std::string& input, bool ignore_error
) {
  std::vector<std::pair<std::string, std::unordered_map<std::string, std::string>>> tool_calls;
  size_t input_pos = 0;
  const size_t input_length = input.length();
  std::string_view input_view(input);

  while (input_pos < input_length) {
    if (input_pos + 10 >= input_length ||
        (input_view.substr(input_pos, 10) != "<function=" && input[input_pos] != '{')) {
      input_pos++;
      continue;
    }

    // function tag case, xml style tags.
    // e.g. <function=get_weather>{"location": "San Francisco, CA", "unit": "Fahrenheit"}</function>
    if (input_view.substr(input_pos, 10) == "<function=") {
      size_t name_start = input_pos + 10;
      size_t name_end = input.find('>', name_start);
      if (name_end == std::string::npos) {
        if (!ignore_error) {
          XGRAMMAR_LOG(FATAL) << "Malformed function tag: missing closing '>'";
        }
        input_pos++;
        continue;
      }

      size_t params_start = name_end + 1;
      size_t params_end = input.find("</function>", params_start);
      if (params_end == std::string::npos) {
        if (!ignore_error) {
          XGRAMMAR_LOG(FATAL) << "Malformed function tag: missing closing '</function>' tag";
        }
        input_pos++;
        continue;
      }

      std::string function_name = input.substr(name_start, name_end - name_start);
      std::string params_str = input.substr(params_start, params_end - params_start);

      // Replace single quotes with double quotes
      size_t quote_pos = 0;
      while ((quote_pos = params_str.find('\'', quote_pos)) != std::string::npos) {
        params_str.replace(quote_pos, 1, "\"");
        quote_pos++;
      }

      // parse JSON parameters
      picojson::value v;
      std::string err = picojson::parse(v, params_str);
      if (!err.empty()) {
        if (!ignore_error) {
          XGRAMMAR_LOG(FATAL) << "Failed to parse JSON parameters: " << err;
        }
        input_pos++;
        continue;
      }

      if (!v.is<picojson::object>()) {
        if (!ignore_error) {
          XGRAMMAR_LOG(FATAL) << "Parameters must be a JSON object";
        }
        input_pos++;
        continue;
      }

      // main processing, can handle string, double, bool, null types and convert to string
      const picojson::object& obj = v.get<picojson::object>();
      std::unordered_map<std::string, std::string> params;
      for (const auto& pair : obj) {
        if (pair.second.is<std::string>()) {
          params[pair.first] = pair.second.get<std::string>();
        } else if (pair.second.is<double>()) {
          params[pair.first] = std::to_string(pair.second.get<double>());
        } else if (pair.second.is<bool>()) {
          params[pair.first] = pair.second.get<bool>() ? "true" : "false";
        } else if (pair.second.is<picojson::null>()) {
          params[pair.first] = "null";
        } else {
          if (!ignore_error) {
            XGRAMMAR_LOG(FATAL) << "Invalid parameter type for field: " << pair.first;
          }
          continue;
        }
      }

      tool_calls.emplace_back(function_name, std::move(params));
      input_pos = params_end + 11;
      continue;
    }

    // JSON format case,
    // e.g. {"name": "get_current_conditions", "parameters": {"location": "San Francisco, CA",
    // "unit": "Fahrenheit"}}
    int bracket_count = 1;
    size_t json_end_pos;
    for (json_end_pos = input_pos + 1; json_end_pos < input_length && bracket_count > 0;
         json_end_pos++) {
      if (input[json_end_pos] == '{') {
        bracket_count++;
      } else if (input[json_end_pos] == '}') {
        bracket_count--;
      }
    }

    if (bracket_count != 0) {
      if (!ignore_error) {
        XGRAMMAR_LOG(FATAL) << "Unmatched JSON brackets";
      }
      input_pos++;
      continue;
    }

    std::string json_str = input.substr(input_pos, json_end_pos - input_pos);
    picojson::value v;
    std::string err = picojson::parse(v, json_str);

    if (!err.empty()) {
      if (!ignore_error) {
        XGRAMMAR_LOG(FATAL) << "Failed to parse JSON: " << err;
      }
      input_pos = json_end_pos;
      continue;
    }

    if (!v.is<picojson::object>()) {
      if (!ignore_error) {
        XGRAMMAR_LOG(FATAL) << "Top-level JSON must be an object";
      }
      input_pos = json_end_pos;
      continue;
    }

    const picojson::object& obj = v.get<picojson::object>();
    auto name_it = obj.find("name");
    auto params_it = obj.find("parameters");

    if (name_it == obj.end() || !name_it->second.is<std::string>()) {
      if (!ignore_error) {
        XGRAMMAR_LOG(FATAL) << "Invalid JSON format: missing or invalid name field";
      }
      input_pos = json_end_pos;
      continue;
    }

    if (params_it == obj.end() || !params_it->second.is<picojson::object>()) {
      if (!ignore_error) {
        XGRAMMAR_LOG(FATAL) << "Invalid JSON format: missing or invalid parameters field";
      }
      input_pos = json_end_pos;
      continue;
    }

    std::string name = name_it->second.get<std::string>();
    const picojson::object& params_obj = params_it->second.get<picojson::object>();
    std::unordered_map<std::string, std::string> params;

    for (const auto& pair : params_obj) {
      if (pair.second.is<std::string>()) {
        params[pair.first] = pair.second.get<std::string>();
      } else if (pair.second.is<double>()) {
        params[pair.first] = std::to_string(pair.second.get<double>());
      } else if (pair.second.is<bool>()) {
        params[pair.first] = pair.second.get<bool>() ? "true" : "false";
      } else if (pair.second.is<picojson::null>()) {
        params[pair.first] = "null";
      } else {
        if (!ignore_error) {
          XGRAMMAR_LOG(FATAL) << "Invalid parameter type for field: " << pair.first;
        }
        continue;
      }
    }

    tool_calls.emplace_back(name, std::move(params));
    input_pos = json_end_pos;
  }

  return tool_calls;
}

}  // namespace xgrammar
