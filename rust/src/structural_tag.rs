//! Structural tag definitions, the Rust counterpart of
//! `xgrammar.structural_tag` (pydantic models in Python, serde here).
//!
//! The only contract with the C++ engine is the serialized JSON produced by
//! [`StructuralTag::to_json`]; the wire format is identical to what the
//! pydantic models emit, so tags serialized by either language are accepted
//! by the other.
//!
//! Python's per-format classes map onto [`Format`] variants:
//! `ConstStringFormat(value="x")` becomes `Format::ConstString { value }`,
//! and so on. [`TagFormat`] stays a standalone struct because several formats
//! embed lists of tags.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::Result;

/// A token reference: either a token id or the token's string form
/// (Python's `Union[int, str]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TokenRef {
    /// A token id.
    Id(i64),
    /// The token string.
    Text(String),
}

impl From<i64> for TokenRef {
    fn from(id: i64) -> Self {
        TokenRef::Id(id)
    }
}

impl From<&str> for TokenRef {
    fn from(text: &str) -> Self {
        TokenRef::Text(text.to_string())
    }
}

/// How [`Format::JsonSchema`] content is rendered (JSON or a model-specific
/// XML dialect).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JsonSchemaStyle {
    /// Standard JSON.
    #[default]
    Json,
    /// Qwen XML: `<parameter=key>value</parameter>`.
    QwenXml,
    /// MiniMax XML: `<parameter name="key">value</parameter>`.
    MinimaxXml,
    /// DeepSeek XML (DeepSeek-v3.2).
    DeepseekXml,
    /// GLM XML: `<arg_key>key</arg_key><arg_value>value</arg_value>`.
    GlmXml,
}

/// The begin marker of a [`TagFormat`]: a string or a single token.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TagBegin {
    /// A literal begin string.
    Text(String),
    /// A single-token begin marker (`{"type": "token", ...}` on the wire).
    Token(TokenMarker),
}

impl From<&str> for TagBegin {
    fn from(text: &str) -> Self {
        TagBegin::Text(text.to_string())
    }
}

/// The end marker(s) of a [`TagFormat`]: a string, several alternative
/// strings, or a single token.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TagEnd {
    /// A literal end string.
    Text(String),
    /// Alternative end strings; any of them closes the tag.
    AnyOf(Vec<String>),
    /// A single-token end marker.
    Token(TokenMarker),
}

impl From<&str> for TagEnd {
    fn from(text: &str) -> Self {
        TagEnd::Text(text.to_string())
    }
}

/// A single-token format used inside [`TagBegin`]/[`TagEnd`]
/// (`{"type": "token", "token": ...}` on the wire, Python's `TokenFormat`).
#[derive(Debug, Clone, PartialEq)]
pub struct TokenMarker {
    /// The token id or string.
    pub token: TokenRef,
}

impl Serialize for TokenMarker {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Wire<'a> {
            #[serde(rename = "type")]
            type_: &'static str,
            token: &'a TokenRef,
        }
        Wire {
            type_: "token",
            token: &self.token,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TokenMarker {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Wire {
            #[serde(rename = "type")]
            type_: String,
            token: TokenRef,
        }
        let wire = Wire::deserialize(deserializer)?;
        if wire.type_ != "token" {
            return Err(serde::de::Error::custom(format!(
                "expected type \"token\", got {:?}",
                wire.type_
            )));
        }
        Ok(TokenMarker { token: wire.token })
    }
}

/// A tag: `begin content end` (Python's `TagFormat`).
#[derive(Debug, Clone, PartialEq)]
pub struct TagFormat {
    /// The begin marker.
    pub begin: TagBegin,
    /// The content between the markers.
    pub content: Box<Format>,
    /// The end marker(s).
    pub end: TagEnd,
}

impl TagFormat {
    /// Convenience constructor.
    pub fn new(begin: impl Into<TagBegin>, content: Format, end: impl Into<TagEnd>) -> Self {
        Self {
            begin: begin.into(),
            content: Box::new(content),
            end: end.into(),
        }
    }
}

impl Serialize for TagFormat {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Wire<'a> {
            #[serde(rename = "type")]
            type_: &'static str,
            begin: &'a TagBegin,
            content: &'a Format,
            end: &'a TagEnd,
        }
        Wire {
            type_: "tag",
            begin: &self.begin,
            content: &self.content,
            end: &self.end,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TagFormat {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Wire {
            #[serde(rename = "type", default)]
            type_: Option<String>,
            begin: TagBegin,
            content: Box<Format>,
            end: TagEnd,
        }
        let wire = Wire::deserialize(deserializer)?;
        if let Some(t) = &wire.type_ {
            if t != "tag" {
                return Err(serde::de::Error::custom(format!(
                    "expected type \"tag\", got {t:?}"
                )));
            }
        }
        Ok(TagFormat {
            begin: wire.begin,
            content: wire.content,
            end: wire.end,
        })
    }
}

impl From<TagFormat> for Format {
    fn from(tag: TagFormat) -> Self {
        Format::Tag {
            begin: tag.begin,
            content: tag.content,
            end: tag.end,
        }
    }
}

fn default_true() -> bool {
    true
}

/// A structural tag format, the discriminated union of every format kind
/// (Python's `Format`). The serialized form carries the kind in the `type`
/// field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Format {
    /// Matches any text, optionally excluding some substrings.
    AnyText {
        /// Strings that must not appear in the matched text.
        #[serde(default)]
        excludes: Vec<String>,
    },
    /// Matches a constant string.
    ConstString {
        /// The constant string.
        value: String,
    },
    /// Matches JSON (or a model-specific XML dialect) following a schema.
    JsonSchema {
        /// The JSON schema: `true`/`false` or a schema object.
        json_schema: serde_json::Value,
        /// The rendering style.
        #[serde(default)]
        style: JsonSchemaStyle,
        /// Whether object properties may appear in any order.
        #[serde(default)]
        any_order: bool,
        /// Max consecutive whitespace characters; `None` means no limit.
        #[serde(default)]
        max_whitespace_cnt: Option<i64>,
    },
    /// Matches an EBNF grammar.
    Grammar {
        /// The EBNF grammar text.
        grammar: String,
    },
    /// Matches a regular expression.
    Regex {
        /// The regex pattern.
        pattern: String,
    },
    /// Deprecated: use [`Format::JsonSchema`] with
    /// [`JsonSchemaStyle::QwenXml`]. Kept so old serialized tags load.
    QwenXmlParameter {
        /// The JSON schema for the function-call parameters.
        json_schema: serde_json::Value,
    },
    /// Matches one of the alternatives.
    Or {
        /// The alternatives.
        elements: Vec<Format>,
    },
    /// Matches a sequence of formats.
    Sequence {
        /// The sequence elements.
        elements: Vec<Format>,
    },
    /// Matches `begin content end`; see [`TagFormat`].
    Tag {
        /// The begin marker.
        begin: TagBegin,
        /// The content between the markers.
        content: Box<Format>,
        /// The end marker(s).
        end: TagEnd,
    },
    /// Free text until a trigger string dispatches into one of the tags.
    TriggeredTags {
        /// The trigger strings; each tag's begin must extend exactly one
        /// trigger.
        triggers: Vec<String>,
        /// The dispatchable tags (string `begin`s only).
        tags: Vec<TagFormat>,
        /// Whether at least one tag must be generated.
        #[serde(default)]
        at_least_one: bool,
        /// Whether matching stops after the first tag.
        #[serde(default)]
        stop_after_first: bool,
        /// Strings that must not appear in the free text.
        #[serde(default)]
        excludes: Vec<String>,
    },
    /// Like `TriggeredTags`, but dispatching on trigger tokens
    /// (token `begin`s only).
    TokenTriggeredTags {
        /// The trigger tokens.
        trigger_tokens: Vec<TokenRef>,
        /// The dispatchable tags.
        tags: Vec<TagFormat>,
        /// Tokens that must not appear in the free part.
        #[serde(default)]
        exclude_tokens: Vec<TokenRef>,
        /// Whether at least one tag must be generated.
        #[serde(default)]
        at_least_one: bool,
        /// Whether matching stops after the first tag.
        #[serde(default)]
        stop_after_first: bool,
    },
    /// Zero or more tags separated by `separator`, with no other text.
    TagsWithSeparator {
        /// The allowed tags.
        tags: Vec<TagFormat>,
        /// The separator between consecutive tags.
        separator: String,
        /// Whether at least one tag must be matched.
        #[serde(default)]
        at_least_one: bool,
        /// Whether matching stops after the first tag.
        #[serde(default)]
        stop_after_first: bool,
    },
    /// The content appears 0 or 1 time.
    Optional {
        /// The optional content.
        content: Box<Format>,
    },
    /// The content appears 1 or more times.
    Plus {
        /// The repeated content.
        content: Box<Format>,
    },
    /// The content appears 0 or more times.
    Star {
        /// The repeated content.
        content: Box<Format>,
    },
    /// The content appears between `min` and `max` times (inclusive).
    Repeat {
        /// Minimum number of occurrences.
        min: i64,
        /// Maximum number of occurrences; `-1` means unbounded.
        max: i64,
        /// The repeated content.
        content: Box<Format>,
    },
    /// Matches a single token.
    Token {
        /// The token id or string.
        token: TokenRef,
    },
    /// Matches a single token outside the excluded set.
    ExcludeToken {
        /// Tokens that must not be matched.
        #[serde(default)]
        exclude_tokens: Vec<TokenRef>,
    },
    /// Matches zero or more tokens outside the excluded set.
    AnyTokens {
        /// Tokens that must not appear.
        #[serde(default)]
        exclude_tokens: Vec<TokenRef>,
    },
    /// Free text where generating a pattern string dispatches into the
    /// corresponding format.
    Dispatch {
        /// `(pattern, content format)` pairs.
        rules: Vec<(String, Format)>,
        /// Whether to keep matching after the first dispatch.
        #[serde(default = "default_true", rename = "loop")]
        loop_: bool,
        /// Strings that must not appear in the free text.
        #[serde(default)]
        excludes: Vec<String>,
    },
    /// Free text where generating a pattern token dispatches into the
    /// corresponding format.
    TokenDispatch {
        /// `(pattern token, content format)` pairs.
        rules: Vec<(TokenRef, Format)>,
        /// Whether to keep matching after the first dispatch.
        #[serde(default = "default_true", rename = "loop")]
        loop_: bool,
        /// Tokens that must not appear in the free text.
        #[serde(default)]
        exclude_tokens: Vec<TokenRef>,
    },
}

/// Deprecated legacy structural tag item, kept for
/// [`StructuralTag::from_legacy`] (Python's `StructuralTagItem`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralTagItem {
    /// The begin tag.
    pub begin: String,
    /// The JSON schema (an object, or `true`/`false`).
    #[serde(rename = "schema")]
    pub schema: serde_json::Value,
    /// The end tag.
    pub end: String,
}

/// A complete structural tag, corresponding to
/// `"response_format": {"type": "structural_tag", "format": {...}}` in the
/// OpenAI-style API (Python's `StructuralTag`).
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralTag {
    /// The root format.
    pub format: Format,
}

impl StructuralTag {
    /// Wrap a [`Format`] into a structural tag.
    pub fn new(format: Format) -> Self {
        Self { format }
    }

    /// Serialize to the JSON wire format understood by
    /// [`crate::Grammar::from_structural_tag_json`] and the Python API.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Parse from the JSON wire format (Python's `StructuralTag.from_json`).
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    /// Convert legacy `(tags, triggers)` structural tags into the modern
    /// representation (Python's `StructuralTag.from_legacy_structural_tag`).
    pub fn from_legacy(tags: &[StructuralTagItem], triggers: &[String]) -> Self {
        StructuralTag::new(Format::TriggeredTags {
            triggers: triggers.to_vec(),
            tags: tags
                .iter()
                .map(|tag| {
                    TagFormat::new(
                        tag.begin.as_str(),
                        Format::JsonSchema {
                            json_schema: tag.schema.clone(),
                            style: JsonSchemaStyle::Json,
                            any_order: false,
                            max_whitespace_cnt: None,
                        },
                        tag.end.as_str(),
                    )
                })
                .collect(),
            at_least_one: false,
            stop_after_first: false,
            excludes: Vec::new(),
        })
    }
}

impl Serialize for StructuralTag {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Wire<'a> {
            #[serde(rename = "type")]
            type_: &'static str,
            format: &'a Format,
        }
        Wire {
            type_: "structural_tag",
            format: &self.format,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for StructuralTag {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Wire {
            #[serde(rename = "type")]
            type_: String,
            format: Format,
        }
        let wire = Wire::deserialize(deserializer)?;
        if wire.type_ != "structural_tag" {
            return Err(serde::de::Error::custom(format!(
                "expected type \"structural_tag\", got {:?}",
                wire.type_
            )));
        }
        Ok(StructuralTag {
            format: wire.format,
        })
    }
}
