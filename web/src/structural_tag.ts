/**
 * Type definitions and helpers for structural tag support in web bindings.
 */

export type JSONSchemaValue = boolean | Record<string, unknown>;

export interface ConstStringFormat {
  type: "const_string";
  value: string;
}

export interface JSONSchemaFormat {
  type: "json_schema";
  json_schema: JSONSchemaValue;
}

export interface QwenXMLParameterFormat {
  type: "qwen_xml_parameter";
  json_schema: JSONSchemaValue;
}

export interface AnyTextFormat {
  type: "any_text";
}

export interface GrammarFormat {
  type: "grammar";
  grammar: string;
}

export interface RegexFormat {
  type: "regex";
  pattern: string;
}

export interface SequenceFormat {
  type: "sequence";
  elements: StructuralTagFormat[];
}

export interface OrFormat {
  type: "or";
  elements: StructuralTagFormat[];
}

export interface TagFormat {
  type: "tag";
  begin: string;
  content: StructuralTagFormat;
  end: string;
}

export interface TriggeredTagsFormat {
  type: "triggered_tags";
  triggers: string[];
  tags: TagFormat[];
  at_least_one?: boolean;
  stop_after_first?: boolean;
}

export interface TagsWithSeparatorFormat {
  type: "tags_with_separator";
  tags: TagFormat[];
  separator: string;
  at_least_one?: boolean;
  stop_after_first?: boolean;
}

export type StructuralTagFormat =
  | AnyTextFormat
  | ConstStringFormat
  | JSONSchemaFormat
  | GrammarFormat
  | RegexFormat
  | QwenXMLParameterFormat
  | OrFormat
  | SequenceFormat
  | TagFormat
  | TriggeredTagsFormat
  | TagsWithSeparatorFormat;

export interface StructuralTag {
  /**
   * The discriminant for structural tags. Optional to keep compatibility with the Python API.
   */
  type?: "structural_tag";
  /**
   * The structural tag format definition.
   */
  format: StructuralTagFormat;
}

export type StructuralTagLike = StructuralTag | StructuralTagFormat | Record<string, unknown> | string;

/**
 * A structural tag item used in the deprecated legacy API.
 */
export class StructuralTagItem {
  constructor(
    public begin: string,
    public schema: string | JSONSchemaValue,
    public end: string,
  ) { }

  /**
   * Convert the schema to a string representation.
   *
   * @returns The schema as a JSON string.
   */
  getSchemaAsString(): string {
    if (typeof this.schema === "string") {
      return this.schema;
    }
    return JSON.stringify(this.schema);
  }

  /**
   * Convert the schema to a JSON object or boolean.
   */
  getSchemaAsJSON(): JSONSchemaValue {
    if (typeof this.schema === "boolean") {
      return this.schema;
    }
    if (typeof this.schema === "string") {
      try {
        const parsed = JSON.parse(this.schema);
        if (typeof parsed === "boolean" || isPlainObject(parsed)) {
          return parsed as JSONSchemaValue;
        }
        throw new Error("Schema string must be a JSON object or boolean.");
      } catch (err) {
        throw new Error(
          `Failed to parse schema string "${this.schema}": ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }
    if (isPlainObject(this.schema)) {
      return this.schema as Record<string, unknown>;
    }
    throw new Error("Schema must be either a JSON string, object, or boolean.");
  }
}

/**
 * Convert legacy structural tag items to a StructuralTag definition.
 */
export function structuralTagFromLegacy(tags: StructuralTagItem[], triggers: string[]): StructuralTag {
  return {
    type: "structural_tag",
    format: {
      type: "triggered_tags",
      triggers: [...triggers],
      tags: tags.map((tag) => ({
        type: "tag",
        begin: tag.begin,
        end: tag.end,
        content: {
          type: "json_schema",
          json_schema: tag.getSchemaAsJSON(),
        },
      })),
    },
  };
}

/**
 * Normalize user input to a structural tag JSON string.
 */
export function structuralTagArgsToJSONString(
  structuralTagOrLegacy: StructuralTagLike | StructuralTagItem[],
  triggers?: string[],
): string {
  if (Array.isArray(structuralTagOrLegacy)) {
    if (!Array.isArray(triggers)) {
      throw new Error("Legacy structural tag usage requires both tags and triggers.");
    }
    const structuralTag = structuralTagFromLegacy(structuralTagOrLegacy, triggers);
    return JSON.stringify(structuralTag);
  }
  if (Array.isArray(triggers)) {
    throw new Error("The triggers argument is only supported with legacy StructuralTagItem usage.");
  }
  return structuralTagToJSONString(structuralTagOrLegacy);
}

const STRUCTURAL_TAG_FORMAT_TYPES = new Set([
  "any_text",
  "const_string",
  "grammar",
  "json_schema",
  "qwen_xml_parameter",
  "regex",
  "sequence",
  "or",
  "tag",
  "triggered_tags",
  "tags_with_separator",
]);

function structuralTagToJSONString(structuralTag: StructuralTagLike): string {
  if (typeof structuralTag === "string") {
    return structuralTag;
  }
  const normalized = structuralTagToObject(structuralTag);
  return JSON.stringify({
    type: "structural_tag",
    format: normalized.format,
  });
}

function structuralTagToObject(structuralTag: Exclude<StructuralTagLike, string>): StructuralTag {
  if (isPlainObject(structuralTag)) {
    if ("format" in structuralTag) {
      return normalizeStructuralTagObject(structuralTag);
    }
    if (isStructuralTagFormat(structuralTag)) {
      return { type: "structural_tag", format: structuralTag };
    }
  }
  if (isStructuralTagFormat(structuralTag)) {
    return { type: "structural_tag", format: structuralTag };
  }
  throw new Error("Structural tag input must be a JSON string, StructuralTag object, or format.");
}

function normalizeStructuralTagObject(value: Record<string, unknown>): StructuralTag {
  if (!("format" in value)) {
    throw new Error('Structural tag object must contain a "format" field.');
  }
  const typeField = value["type"];
  if (typeField !== undefined && typeField !== "structural_tag") {
    throw new Error('Structural tag "type" field must be "structural_tag" when present.');
  }
  const formatCandidate = (value as { format: unknown }).format;
  if (!isStructuralTagFormat(formatCandidate)) {
    throw new Error('Structural tag "format" field must be a valid structural tag format.');
  }
  return {
    type: "structural_tag",
    format: formatCandidate,
  };
}

function isStructuralTagFormat(value: unknown): value is StructuralTagFormat {
  if (!isPlainObject(value)) {
    return false;
  }
  const typeValue = (value as { type?: unknown }).type;
  if (typeof typeValue !== "string") {
    return false;
  }
  return STRUCTURAL_TAG_FORMAT_TYPES.has(typeValue);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
