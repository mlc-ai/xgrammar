/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/dynamic_tag_matcher.cc
 */

#include "dynamic_tag_matcher.h"

#include <algorithm>
#include <utility>

#include "support/logging.h"
#include "support/memory_size.h"

namespace xgrammar {

namespace {

constexpr bool IsASCIIWhitespace(uint8_t byte) {
  return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r' || byte == '\f' ||
         byte == '\v';
}

}  // namespace

const DynamicTagMatcherConfig& GetMiniMaxM3DynamicTagMatcherConfig() {
  static const DynamicTagMatcherConfig config{"]<]minimax[>[<", "/", ">", "invoke", "tool_call", 1};
  return config;
}

std::optional<std::string> ValidateDynamicTagMatcherConfig(const DynamicTagMatcherConfig& config) {
  if (config.element_prefix.empty()) {
    return "Dynamic tag element_prefix cannot be empty";
  }
  if (config.close_marker.size() != 1) {
    return "Dynamic tag close_marker must contain exactly one byte";
  }
  if (config.tag_suffix.size() != 1) {
    return "Dynamic tag tag_suffix must contain exactly one byte";
  }
  if (config.attribute_tag_parent_depth < -1) {
    return "Dynamic tag attribute_tag_parent_depth must be -1 or non-negative";
  }
  return std::nullopt;
}

std::size_t MemorySize(const DynamicTagMatcherConfig& config) {
  return MemorySize(config.element_prefix) + MemorySize(config.close_marker) +
         MemorySize(config.tag_suffix) + MemorySize(config.attribute_tag_name) +
         MemorySize(config.attribute_tag_parent);
}

DynamicTagMatcher::DynamicTagMatcher(DynamicTagMatcherConfig config) {
  if (auto error = ValidateDynamicTagMatcherConfig(config)) {
    XGRAMMAR_LOG(FATAL) << *error;
  }

  auto definition = std::make_shared<Definition>();
  definition->config = std::move(config);
  definition->prefix_failure.resize(definition->config.element_prefix.size(), 0);
  for (size_t i = 1, matched = 0; i < definition->config.element_prefix.size(); ++i) {
    while (matched > 0 &&
           definition->config.element_prefix[i] != definition->config.element_prefix[matched]) {
      matched = definition->prefix_failure[matched - 1];
    }
    if (definition->config.element_prefix[i] == definition->config.element_prefix[matched]) {
      ++matched;
    }
    definition->prefix_failure[i] = matched;
  }
  definition_ = std::move(definition);
}

bool DynamicTagMatcher::Accept(std::string_view text) {
  for (uint8_t byte : text) {
    if (!AcceptByte(byte)) {
      return false;
    }
  }
  return true;
}

bool DynamicTagMatcher::CanAccept(std::string_view text) const {
  if (text.empty()) {
    return true;
  }

  // Once an element has children, only whitespace, another child, or its closing tag may follow.
  // Let the full state machine resolve partial namespace prefixes in this uncommon boundary state.
  if (mode_ == Mode::kText && open_tag_content_kind_ == ContentKind::kChildren) {
    DynamicTagMatcher copy = *this;
    return copy.Accept(text);
  }

  size_t token_position = 0;
  bool validating_close = mode_ == Mode::kClosingName || mode_ == Mode::kClosingSuffix;
  size_t close_name_position = close_name_position_;
  size_t close_suffix_position = close_suffix_position_;
  if (mode_ == Mode::kAfterPrefix && open_tag_stack_ &&
      open_tag_content_kind_ == ContentKind::kText &&
      text.front() != definition_->config.close_marker.front()) {
    return false;
  }
  if (mode_ == Mode::kAfterPrefix && text.front() == definition_->config.close_marker.front()) {
    if (!open_tag_stack_) {
      return false;
    }
    validating_close = true;
    close_name_position = 0;
    close_suffix_position = 0;
    token_position = 1;
  }

  if (validating_close) {
    if (mode_ != Mode::kClosingSuffix) {
      const auto& expected_name = open_tag_stack_->name;
      while (token_position < text.size() && close_name_position < expected_name.size()) {
        if (static_cast<uint8_t>(text[token_position]) !=
            static_cast<uint8_t>(expected_name[close_name_position])) {
          return false;
        }
        ++token_position;
        ++close_name_position;
      }
      if (token_position == text.size()) {
        return true;
      }
    }

    const auto& suffix = definition_->config.tag_suffix;
    while (token_position < text.size() && close_suffix_position < suffix.size()) {
      if (static_cast<uint8_t>(text[token_position]) !=
          static_cast<uint8_t>(suffix[close_suffix_position])) {
        return false;
      }
      ++token_position;
      ++close_suffix_position;
    }
    if (token_position == text.size()) {
      return true;
    }

    // The close may expose a parent whose content is element-only. The remaining bytes therefore
    // need the full mixed-content and prefix validation.
    DynamicTagMatcher copy = *this;
    return copy.Accept(text);
  } else if (mode_ == Mode::kAfterPrefix || mode_ == Mode::kOpeningName ||
             mode_ == Mode::kOpeningAttributes) {
    const char suffix = definition_->config.tag_suffix.front();
    const size_t suffix_position = text.find(suffix);
    if (suffix_position == std::string_view::npos) {
      return true;
    }
    if (mode_ != Mode::kOpeningAttributes && pending_open_name_all_whitespace_ &&
        std::all_of(text.begin(), text.begin() + suffix_position, [](char byte) {
          return IsASCIIWhitespace(static_cast<uint8_t>(byte));
        })) {
      return false;
    }
    if (suffix_position + 1 == text.size()) {
      return true;
    }
    if (!CompletesElementPrefix(text.substr(suffix_position + 1))) {
      // The newly opened element can always start with scalar text or a partial prefix. Accept()
      // will record that state only for the token that is actually selected.
      return true;
    }
    // A compound token can finish an opening tag and immediately start its value or a child.
    DynamicTagMatcher copy = *this;
    return copy.Accept(text);
  } else if (!CompletesElementPrefix(text, prefix_match_length_)) {
    return true;
  }

  // A single token can contain multiple complete dynamic tags. This uncommon path needs the full
  // persistent state transition; ordinary mask candidates above remain allocation-free.
  DynamicTagMatcher copy = *this;
  return copy.Accept(text);
}

bool DynamicTagMatcher::HasSameState(const DynamicTagMatcher& other) const {
  return definition_ == other.definition_ && prefix_match_length_ == other.prefix_match_length_ &&
         mode_ == other.mode_ && pending_open_name_chunks_ == other.pending_open_name_chunks_ &&
         pending_open_name_chunk_size_ == other.pending_open_name_chunk_size_ &&
         std::equal(
             pending_open_name_chunk_.begin(),
             pending_open_name_chunk_.begin() + pending_open_name_chunk_size_,
             other.pending_open_name_chunk_.begin()
         ) &&
         pending_open_name_length_ == other.pending_open_name_length_ &&
         pending_open_name_all_whitespace_ == other.pending_open_name_all_whitespace_ &&
         pending_open_name_matches_attribute_ == other.pending_open_name_matches_attribute_ &&
         open_tag_stack_ == other.open_tag_stack_ &&
         open_tag_content_kind_ == other.open_tag_content_kind_ &&
         close_name_position_ == other.close_name_position_ &&
         close_suffix_position_ == other.close_suffix_position_;
}

bool DynamicTagMatcher::NeedsStateUpdate(std::string_view text) const {
  if (text.empty()) {
    return false;
  }
  if (mode_ != Mode::kText || prefix_match_length_ != 0) {
    return true;
  }
  if (open_tag_stack_ && open_tag_content_kind_ != ContentKind::kText &&
      std::any_of(text.begin(), text.end(), [](char byte) {
        return !IsASCIIWhitespace(static_cast<uint8_t>(byte));
      })) {
    return true;
  }
  return text.find(definition_->config.element_prefix.front()) != std::string_view::npos;
}

DynamicTagMatcher::TokenMaskMode DynamicTagMatcher::GetTokenMaskMode() const {
  switch (mode_) {
    case Mode::kText:
      if (open_tag_content_kind_ == ContentKind::kChildren) {
        if (prefix_match_length_ != 0) {
          return TokenMaskMode::kRequiredByte;
        }
        return TokenMaskMode::kValidateAccepted;
      }
      return prefix_match_length_ == 0 ? TokenMaskMode::kFullPrefix
                                       : TokenMaskMode::kPrefixCompletion;
    case Mode::kAfterPrefix:
      if (open_tag_stack_ && open_tag_content_kind_ == ContentKind::kText) {
        return TokenMaskMode::kRequiredByte;
      }
      return TokenMaskMode::kAfterPrefix;
    case Mode::kOpeningName:
    case Mode::kOpeningAttributes:
      return TokenMaskMode::kTagSuffix;
    case Mode::kClosingName:
    case Mode::kClosingSuffix:
      return TokenMaskMode::kRequiredByte;
  }
  XGRAMMAR_LOG(FATAL) << "Invalid dynamic-tag matcher mode";
  return TokenMaskMode::kRequiredByte;
}

std::size_t DynamicTagMatcher::SharedDefinitionMemorySize() const {
  return sizeof(Definition) + MemorySize(definition_->config) +
         MemorySize(definition_->prefix_failure);
}

uint8_t DynamicTagMatcher::GetRequiredNextByte() const {
  if (mode_ == Mode::kText) {
    XGRAMMAR_DCHECK(open_tag_content_kind_ == ContentKind::kChildren && prefix_match_length_ != 0);
    return static_cast<uint8_t>(definition_->config.element_prefix[prefix_match_length_]);
  }
  if (mode_ == Mode::kAfterPrefix) {
    XGRAMMAR_DCHECK(open_tag_stack_ && open_tag_content_kind_ == ContentKind::kText);
    return static_cast<uint8_t>(definition_->config.close_marker.front());
  }
  if (mode_ == Mode::kClosingName) {
    XGRAMMAR_DCHECK(open_tag_stack_ && close_name_position_ < open_tag_stack_->name.size());
    return static_cast<uint8_t>(open_tag_stack_->name[close_name_position_]);
  }
  XGRAMMAR_DCHECK(mode_ == Mode::kClosingSuffix);
  const auto& suffix = definition_->config.tag_suffix;
  XGRAMMAR_DCHECK(close_suffix_position_ < suffix.size());
  return static_cast<uint8_t>(suffix[close_suffix_position_]);
}

bool DynamicTagMatcher::CanTerminate() const {
  // A partial element prefix at EOF is ordinary text; only a fully entered element is incomplete.
  return mode_ == Mode::kText && !open_tag_stack_;
}

bool DynamicTagMatcher::AcceptByte(uint8_t byte) {
  const auto& config = definition_->config;
  switch (mode_) {
    case Mode::kText: {
      return AdvancePrefix(byte);
    }
    case Mode::kAfterPrefix: {
      if (byte == static_cast<uint8_t>(config.close_marker.front())) {
        if (!open_tag_stack_) {
          return false;
        }
        mode_ = Mode::kClosingName;
        close_name_position_ = 0;
        return true;
      }
      if (byte == static_cast<uint8_t>(config.tag_suffix.front())) {
        return false;
      }
      if (open_tag_stack_) {
        if (open_tag_content_kind_ == ContentKind::kText) {
          return false;
        }
        open_tag_content_kind_ = ContentKind::kChildren;
      }
      pending_open_name_chunks_.reset();
      pending_open_name_chunk_size_ = 0;
      pending_open_name_length_ = 0;
      pending_open_name_all_whitespace_ = true;
      pending_open_name_matches_attribute_ = !config.attribute_tag_name.empty();
      AppendPendingOpenNameByte(byte);
      mode_ = Mode::kOpeningName;
      return true;
    }
    case Mode::kOpeningName: {
      if (byte == static_cast<uint8_t>(config.tag_suffix.front())) {
        return FinishOpeningTag();
      }
      if (IsASCIIWhitespace(byte) && OpeningTagHasAttributes()) {
        mode_ = Mode::kOpeningAttributes;
        return true;
      }
      AppendPendingOpenNameByte(byte);
      return true;
    }
    case Mode::kOpeningAttributes: {
      if (byte == static_cast<uint8_t>(config.tag_suffix.front())) {
        return FinishOpeningTag();
      }
      return true;
    }
    case Mode::kClosingName: {
      const auto& expected = open_tag_stack_->name;
      if (close_name_position_ >= expected.size() ||
          byte != static_cast<uint8_t>(expected[close_name_position_])) {
        return false;
      }
      ++close_name_position_;
      if (close_name_position_ == expected.size()) {
        mode_ = Mode::kClosingSuffix;
        close_suffix_position_ = 0;
      }
      return true;
    }
    case Mode::kClosingSuffix: {
      if (byte != static_cast<uint8_t>(config.tag_suffix[close_suffix_position_])) {
        return false;
      }
      ++close_suffix_position_;
      if (close_suffix_position_ == config.tag_suffix.size()) {
        FinishClosingTag();
      }
      return true;
    }
  }
  return false;
}

void DynamicTagMatcher::AppendPendingOpenNameByte(uint8_t byte) {
  const auto& attribute_tag_name = definition_->config.attribute_tag_name;
  if (pending_open_name_matches_attribute_ &&
      (pending_open_name_length_ >= attribute_tag_name.size() ||
       byte != static_cast<uint8_t>(attribute_tag_name[pending_open_name_length_]))) {
    pending_open_name_matches_attribute_ = false;
  }
  ++pending_open_name_length_;
  pending_open_name_all_whitespace_ = pending_open_name_all_whitespace_ && IsASCIIWhitespace(byte);
  if (pending_open_name_chunk_size_ == kPendingNameChunkCapacity) {
    SpillPendingOpenNameChunk();
  }
  pending_open_name_chunk_[pending_open_name_chunk_size_++] = static_cast<char>(byte);
}

void DynamicTagMatcher::SpillPendingOpenNameChunk() {
  if (pending_open_name_chunk_size_ == 0) {
    return;
  }
  pending_open_name_chunks_ = std::make_shared<PendingNameChunk>(PendingNameChunk{
      pending_open_name_chunk_, pending_open_name_chunk_size_, pending_open_name_chunks_
  });
  pending_open_name_chunk_size_ = 0;
}

bool DynamicTagMatcher::AdvancePrefix(uint8_t byte) {
  const auto& config = definition_->config;
  const auto& prefix_failure = definition_->prefix_failure;
  const size_t old_match_length = prefix_match_length_;
  while (prefix_match_length_ > 0 &&
         byte != static_cast<uint8_t>(config.element_prefix[prefix_match_length_])) {
    prefix_match_length_ = prefix_failure[prefix_match_length_ - 1];
  }
  if (byte == static_cast<uint8_t>(config.element_prefix[prefix_match_length_])) {
    ++prefix_match_length_;
  }
  if (prefix_match_length_ == config.element_prefix.size()) {
    prefix_match_length_ = 0;
    mode_ = Mode::kAfterPrefix;
    return true;
  }

  // Bytes retained in prefix_match_length_ may still become a structural prefix. Everything
  // before that suffix is ordinary text and can now determine the parent's content kind.
  const size_t released_bytes = old_match_length + 1 - prefix_match_length_;
  for (size_t index = 0; index < released_bytes; ++index) {
    const uint8_t released =
        index < old_match_length ? static_cast<uint8_t>(config.element_prefix[index]) : byte;
    if (!AcceptTextByte(released)) {
      return false;
    }
  }
  return true;
}

bool DynamicTagMatcher::AcceptTextByte(uint8_t byte) {
  if (!open_tag_stack_ || IsASCIIWhitespace(byte)) {
    return true;
  }
  if (open_tag_content_kind_ == ContentKind::kChildren) {
    return false;
  }
  open_tag_content_kind_ = ContentKind::kText;
  return true;
}

bool DynamicTagMatcher::FinishOpeningTag() {
  XGRAMMAR_DCHECK(pending_open_name_length_ > 0);
  if (pending_open_name_all_whitespace_) {
    return false;
  }
  std::string open_name(pending_open_name_length_, '\0');
  size_t write_position = open_name.size();
  write_position -= pending_open_name_chunk_size_;
  std::copy(
      pending_open_name_chunk_.begin(),
      pending_open_name_chunk_.begin() + pending_open_name_chunk_size_,
      open_name.begin() + write_position
  );
  for (auto chunk = pending_open_name_chunks_; chunk; chunk = chunk->parent) {
    write_position -= chunk->size;
    std::copy(
        chunk->bytes.begin(), chunk->bytes.begin() + chunk->size, open_name.begin() + write_position
    );
  }
  XGRAMMAR_DCHECK(write_position == 0);
  open_tag_stack_ = std::make_shared<OpenTagNode>(OpenTagNode{
      std::move(open_name),
      open_tag_stack_,
      open_tag_stack_ ? open_tag_stack_->depth + 1 : 1,
      open_tag_content_kind_,
  });
  open_tag_content_kind_ = ContentKind::kEmpty;
  pending_open_name_chunks_.reset();
  pending_open_name_chunk_size_ = 0;
  pending_open_name_length_ = 0;
  pending_open_name_all_whitespace_ = true;
  pending_open_name_matches_attribute_ = false;
  mode_ = Mode::kText;
  prefix_match_length_ = 0;
  return true;
}

void DynamicTagMatcher::FinishClosingTag() {
  XGRAMMAR_DCHECK(open_tag_stack_);
  const ContentKind parent_content_kind = open_tag_stack_->parent_content_kind;
  open_tag_stack_ = open_tag_stack_->parent;
  open_tag_content_kind_ = open_tag_stack_ ? parent_content_kind : ContentKind::kEmpty;
  close_name_position_ = 0;
  close_suffix_position_ = 0;
  mode_ = Mode::kText;
  prefix_match_length_ = 0;
}

bool DynamicTagMatcher::OpeningTagHasAttributes() const {
  const auto& config = definition_->config;
  if (config.attribute_tag_name.empty() || !pending_open_name_matches_attribute_ ||
      pending_open_name_length_ != config.attribute_tag_name.size()) {
    return false;
  }
  const size_t depth = open_tag_stack_ ? open_tag_stack_->depth : 0;
  if (config.attribute_tag_parent_depth >= 0 &&
      depth != static_cast<size_t>(config.attribute_tag_parent_depth)) {
    return false;
  }
  return config.attribute_tag_parent.empty() ||
         (open_tag_stack_ && open_tag_stack_->name == config.attribute_tag_parent);
}

bool DynamicTagMatcher::CompletesElementPrefix(std::string_view text, size_t initial_match_length)
    const {
  const auto& prefix = definition_->config.element_prefix;
  const auto& prefix_failure = definition_->prefix_failure;
  size_t matched = initial_match_length;
  for (size_t index = 0; index < text.size(); ++index) {
    const uint8_t byte = static_cast<uint8_t>(text[index]);
    while (matched > 0 && byte != static_cast<uint8_t>(prefix[matched])) {
      matched = prefix_failure[matched - 1];
    }
    if (byte == static_cast<uint8_t>(prefix[matched])) {
      ++matched;
    }
    if (matched == prefix.size()) {
      return true;
    }
  }
  return false;
}

}  // namespace xgrammar
