/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/dynamic_tag_matcher.h
 * \brief Stateful validation for markup whose closing tag repeats a runtime-generated name.
 */

#ifndef XGRAMMAR_DYNAMIC_TAG_MATCHER_H_
#define XGRAMMAR_DYNAMIC_TAG_MATCHER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "support/reflection.h"

namespace xgrammar {

/*!
 * \brief Describes a namespaced element syntax that needs dynamic open/close-name matching.
 *
 * This constraint is intentionally kept outside the context-free grammar. A CFG can describe
 * both tag names independently, but cannot require an unbounded runtime-generated closing name to
 * equal the opening name. GrammarMatcher intersects its normal CFG result with this matcher.
 */
struct DynamicTagMatcherConfig {
  /*! \brief Prefix through the opening '<', e.g. `]<]minimax[>[<`. */
  std::string element_prefix;
  /*! \brief Marker immediately after element_prefix for a closing tag. */
  std::string close_marker = "/";
  /*! \brief Terminator for both opening and closing tags. */
  std::string tag_suffix = ">";
  /*! \brief Tag name whose opening form may contain attributes. Empty disables attributes. */
  std::string attribute_tag_name;
  /*! \brief Required parent of attribute_tag_name. Empty matches any parent. */
  std::string attribute_tag_parent;
  /*! \brief Required open-tag stack depth for the attribute tag. -1 matches any depth. */
  int32_t attribute_tag_parent_depth = -1;

  bool operator==(const DynamicTagMatcherConfig& other) const {
    return element_prefix == other.element_prefix && close_marker == other.close_marker &&
           tag_suffix == other.tag_suffix && attribute_tag_name == other.attribute_tag_name &&
           attribute_tag_parent == other.attribute_tag_parent &&
           attribute_tag_parent_depth == other.attribute_tag_parent_depth;
  }
};

XGRAMMAR_MEMBER_TABLE(
    DynamicTagMatcherConfig,
    "element_prefix",
    &DynamicTagMatcherConfig::element_prefix,
    "close_marker",
    &DynamicTagMatcherConfig::close_marker,
    "tag_suffix",
    &DynamicTagMatcherConfig::tag_suffix,
    "attribute_tag_name",
    &DynamicTagMatcherConfig::attribute_tag_name,
    "attribute_tag_parent",
    &DynamicTagMatcherConfig::attribute_tag_parent,
    "attribute_tag_parent_depth",
    &DynamicTagMatcherConfig::attribute_tag_parent_depth
);

/*! \brief Return the dynamic-tag syntax used by MiniMax M3. */
const DynamicTagMatcherConfig& GetMiniMaxM3DynamicTagMatcherConfig();

/*!
 * \brief Validate invariants required by DynamicTagMatcher and its token indexes.
 * \return An error message when the configuration is invalid, or std::nullopt otherwise.
 */
std::optional<std::string> ValidateDynamicTagMatcherConfig(const DynamicTagMatcherConfig& config);

/*! \brief Return the owned memory used by a dynamic-tag matcher configuration. */
std::size_t MemorySize(const DynamicTagMatcherConfig& config);

/*!
 * \brief Incrementally validates matching runtime-generated element names and element nesting.
 *
 * The matcher records opening names on a stack and accepts a closing name only when it equals the
 * stack top byte-for-byte. An element may contain scalar text or child elements, but not both;
 * this keeps CFG and runtime parsing aligned even when a user key regex includes the tag delimiter.
 * It is copyable so token-mask generation can cheaply test candidates without mutating the
 * committed state.
 */
class DynamicTagMatcher {
 public:
  enum class TokenMaskMode : uint8_t {
    kFullPrefix,
    kPrefixCompletion,
    kAfterPrefix,
    kTagSuffix,
    kRequiredByte,
    kValidateAccepted,
  };

  explicit DynamicTagMatcher(DynamicTagMatcherConfig config);

  /*! \brief Accept a byte sequence. Returns false without guaranteeing rollback on this object. */
  bool Accept(std::string_view text);

  /*! \brief Whether generation may terminate at the current position. */
  bool CanTerminate() const;

  /*! \brief Whether text can be accepted from the current state, without mutating it. */
  bool CanAccept(std::string_view text) const;

  /*! \brief Whether two matchers represent the same committed semantic state. */
  bool HasSameState(const DynamicTagMatcher& other) const;

  /*! \brief Whether accepting text can change this matcher's committed state. */
  bool NeedsStateUpdate(std::string_view text) const;

  /*! \brief How token-mask generation can narrow the tokens that need validation. */
  TokenMaskMode GetTokenMaskMode() const;

  /*! \brief Required first byte when GetTokenMaskMode() is kRequiredByte. */
  uint8_t GetRequiredNextByte() const;

  /*! \brief Heap memory owned by the immutable definition shared by matcher forks. */
  std::size_t SharedDefinitionMemorySize() const;

 private:
  enum class Mode : uint8_t {
    kText,
    kAfterPrefix,
    kOpeningName,
    kOpeningAttributes,
    kClosingName,
    kClosingSuffix,
  };

  enum class ContentKind : uint8_t {
    kEmpty,
    kText,
    kChildren,
  };

  struct Definition {
    DynamicTagMatcherConfig config;
    std::vector<size_t> prefix_failure;
  };

  struct OpenTagNode {
    std::string name;
    std::shared_ptr<const OpenTagNode> parent;
    size_t depth;
    ContentKind parent_content_kind;
  };

  // Common property names stay inline; long names spill in fixed-size immutable chunks.
  static constexpr size_t kPendingNameChunkCapacity = 32;

  struct PendingNameChunk {
    std::array<char, kPendingNameChunkCapacity> bytes;
    uint8_t size;
    std::shared_ptr<const PendingNameChunk> parent;
  };

  bool AcceptByte(uint8_t byte);
  bool AdvancePrefix(uint8_t byte);
  bool AcceptTextByte(uint8_t byte);
  void AppendPendingOpenNameByte(uint8_t byte);
  void SpillPendingOpenNameChunk();
  bool FinishOpeningTag();
  void FinishClosingTag();
  bool OpeningTagHasAttributes() const;
  bool CompletesElementPrefix(std::string_view text, size_t initial_match_length = 0) const;

  std::shared_ptr<const Definition> definition_;
  size_t prefix_match_length_ = 0;
  Mode mode_ = Mode::kText;
  std::shared_ptr<const PendingNameChunk> pending_open_name_chunks_;
  std::array<char, kPendingNameChunkCapacity> pending_open_name_chunk_{};
  uint8_t pending_open_name_chunk_size_ = 0;
  size_t pending_open_name_length_ = 0;
  bool pending_open_name_all_whitespace_ = true;
  bool pending_open_name_matches_attribute_ = false;
  std::shared_ptr<const OpenTagNode> open_tag_stack_;
  ContentKind open_tag_content_kind_ = ContentKind::kEmpty;
  size_t close_name_position_ = 0;
  size_t close_suffix_position_ = 0;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_DYNAMIC_TAG_MATCHER_H_
