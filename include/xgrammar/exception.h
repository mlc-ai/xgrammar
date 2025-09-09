#ifndef XGRAMMAR_EXCEPTION_H
#define XGRAMMAR_EXCEPTION_H

#include <stdexcept>
#include <string>
#include <variant>

namespace xgrammar {

/************** Exception Definitions **************/

/*!
 * \brief Exception thrown when the version in the serialized data does not follow the current
 * serialization version.
 */
struct DeserializeVersionError : std::runtime_error {
  DeserializeVersionError(const std::string& message)
      : std::runtime_error(std::string("Deserialize version error: ") + message) {}
};

/*!
 * \brief Exception thrown when the JSON is invalid.
 */
struct InvalidJSONError : std::runtime_error {
  InvalidJSONError(const std::string& message)
      : std::runtime_error(std::string("Invalid JSON error: ") + message) {}
};

/*!
 * \brief Exception thrown when the serialized data does not follow the expected format.
 */
struct DeserializeFormatError : std::runtime_error {
  DeserializeFormatError(const std::string& message)
      : std::runtime_error(std::string("Deserialize format error: ") + message) {}
};

/*!
 * \brief Exception thrown when the JSON schema is invalid or not satisfiable.
 */
struct InvalidJSONSchemaError : std::runtime_error {
  InvalidJSONSchemaError(const std::string& message)
      : std::runtime_error(std::string("Invalid JSON schema error: ") + message) {}
};

/*!
 * \brief Exception thrown when the structural tag is invalid.
 */
struct InvalidStructuralTagError : std::runtime_error {
  InvalidStructuralTagError(const std::string& message)
      : std::runtime_error(std::string("Invalid structural tag error: ") + message) {}
};

/************** Union Exceptions **************/

/*!
 * \brief Represents a serialization error.
 */
using SerializationError =
    std::variant<DeserializeVersionError, InvalidJSONError, DeserializeFormatError>;

/*!
 * \brief Represents an error from the structural tag conversion.
 */
using StructuralTagError =
    std::variant<InvalidJSONError, InvalidJSONSchemaError, InvalidStructuralTagError>;

}  // namespace xgrammar

#endif  // XGRAMMAR_EXCEPTION_H
