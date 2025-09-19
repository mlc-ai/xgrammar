# Supported JSON Schema

XGrammar provides common and robust support for converting JSON Schema into EBNF grammars, enabling precise, constraint-based generation and validation of JSON data. This document outlines the supported features. For more detailed information of JSON schema, please refer to [JSON Schema](https://json-schema.org).

## Generic Keywords

* `type`: All json schema types are supported, including `string`, `number`, `integer`, `object`, `array`, `boolean`, and `null`.

* `enum`

* `const`

## Applicators
* `anyOf`: Fully supported.

* `oneOf`: Be regarded as `anyOf`.

* `allOf`: Only one schema is supported now.

## Referencing
* `$ref`: Supports internal schema references. E.g. `"#/$defs/user"` refers to the `"user"` field of the `"$defs"` field of the global json schema object.

## String

* `minLength` / `maxLength`

* `pattern`: Regular expressions are supported. Lookahead and lookbehind zero-length assertions are ignored.

* `format`: All formats are supported, including:
  - `date`, `time`, `date-time`, `duration`, `email`, `ipv4`, `ipv6`, `hostname`, `uuid`, `uri`, `uri-reference`, `uri-template`, `json-pointer`, `relative-json-pointer`.

## Integer

* `minimum` / `maximum`

* `exclusiveMinimum` / `exclusiveMaximum`

## Number

* `minimum` / `maximum`: The precision is $10^{−6}$.

* `exclusiveMinimum` / `exclusiveMaximum`: The precision is $10^{−6}$.

## Array

* `items`

* `prefixItems`

* `minItems` / `maxItems`

* `unevaluatedItems`

## Object

* `properties`

* `required`

* `additionalProperties`

* `unevaluatedProperties`

* `minProperties` / `maxProperties`

* `patternProperties`: Only supports `additionalProperties=False` and will be ignored when `properties` or `required` is set.

* `propertyNames`: Only supports `additionalProperties=False` and will be ignored when `properties`, `required` or `patternProperties` is set.
