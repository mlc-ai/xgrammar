#ifndef XGRAMMAR_EXCEPTION_H
#define XGRAMMAR_EXCEPTION_H

#include <stdexcept>
#include <variant>

namespace xgrammar {

struct DeserializeVersionError : std::runtime_error {
  DeserializeVersionError(const std::string& message)
      : std::runtime_error("Deserialize version error: " + message) {}
};

struct InvalidJSONError : std::runtime_error {
  InvalidJSONError(const std::string& message)
      : std::runtime_error("Invalid JSON error: " + message) {}
};

struct DeserializeFormatError : std::runtime_error {
  DeserializeFormatError(const std::string& message)
      : std::runtime_error("Deserialize format error: " + message) {}
};

using SerializationError =
    std::variant<DeserializeVersionError, InvalidJSONError, DeserializeFormatError>;

}  // namespace xgrammar

#endif  // XGRAMMAR_EXCEPTION_H
