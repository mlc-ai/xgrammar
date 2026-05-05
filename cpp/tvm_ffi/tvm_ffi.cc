/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/tvm_ffi/tvm_ffi.cc
 * \brief TVM-FFI bindings for xgrammar.
 */

#include <dlpack/dlpack.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/tvm_ffi.h>
#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "../grammar_functor.h"
#include "../json_schema_converter.h"
#include "../regex_converter.h"
#include "../support/utils.h"
#include "../testing.h"
#include "python_methods.h"
#include "xgrammar/exception.h"
#include "xgrammar/matcher.h"

namespace ffi = tvm::ffi;
namespace refl = tvm::ffi::reflection;

namespace xgrammar {

// ----- Error handling -----

// ----- Helpers: convert FFI types to/from xgrammar types -----

static std::string BytesToString(const tvm::ffi::Bytes& bytes) {
  std::string result(bytes.data(), bytes.size());
  return result;
}

// Convert ffi::Array<ffi::Any> to vector<string>; each element can be str or bytes (like
// accept_string).
static std::vector<std::string> ArrayAnyToVectorString(ffi::Array<ffi::Any> array) {
  std::vector<std::string> result;
  result.reserve(static_cast<size_t>(array.size()));
  for (int64_t i = 0; i < static_cast<int64_t>(array.size()); ++i) {
    ffi::AnyView view = array[i];
    if (view.as<ffi::Bytes>()) {
      result.push_back(BytesToString(view.cast<ffi::Bytes>()));
    } else if (view.as<ffi::String>()) {
      result.push_back(view.cast<ffi::String>());
    } else {
      TVM_FFI_THROW(RuntimeError) << "Unsupported type in encoded_vocab: expected str or bytes";
      XGRAMMAR_UNREACHABLE();
    }
  }
  return result;
}

static ffi::Array<ffi::Bytes> VectorStringToBytesArray(const std::vector<std::string>& string_vector
) {
  ffi::Array<ffi::Bytes> bytes_array;
  for (const auto& value : string_vector) {
    bytes_array.push_back(ffi::Bytes(value));
  }
  return bytes_array;
}

static std::optional<int64_t> OptionalIntFromView(ffi::AnyView v) {
  if (v == nullptr) return std::nullopt;
  return v.cast<int64_t>();
}

static std::optional<bool> OptionalBoolFromView(ffi::AnyView v) {
  if (v == nullptr) return std::nullopt;
  return static_cast<bool>(v.cast<int64_t>());
}

static std::optional<std::vector<int32_t>> OptionalInt32VectorFromView(ffi::AnyView v) {
  if (v == nullptr) return std::nullopt;
  ffi::Array<int64_t> array = v.cast<ffi::Array<int64_t>>();
  std::vector<int32_t> result;
  result.reserve(static_cast<size_t>(array.size()));
  for (int64_t i = 0; i < static_cast<int64_t>(array.size()); ++i)
    result.push_back(static_cast<int32_t>(array[i]));
  return result;
}

static std::optional<std::vector<int>> OptionalIntVectorFromView(ffi::AnyView v) {
  if (v == nullptr) return std::nullopt;
  ffi::Array<int64_t> array = v.cast<ffi::Array<int64_t>>();
  std::vector<int> result;
  result.reserve(static_cast<size_t>(array.size()));
  for (int64_t i = 0; i < static_cast<int64_t>(array.size()); ++i)
    result.push_back(static_cast<int>(array[i]));
  return result;
}

static std::optional<std::pair<std::string, std::string>> OptionalSeparatorsFromView(ffi::AnyView v
) {
  if (v == nullptr) return std::nullopt;
  ffi::Array<ffi::String> separators_array = v.cast<ffi::Array<ffi::String>>();
  if (separators_array.size() < 2) return std::nullopt;
  return std::make_pair(separators_array[0], separators_array[1]);
}

static std::variant<std::string, int32_t> ParseMaxThreads(ffi::AnyView max_threads_view) {
  if (max_threads_view == nullptr) return "auto";
  auto int_value = max_threads_view.as<int64_t>();
  if (int_value.has_value()) {
    return static_cast<int32_t>(int_value.value());
  }
  auto string_value = max_threads_view.as<ffi::String>();
  if (string_value.has_value()) {
    return string_value.value();
  }
  TVM_FFI_THROW(RuntimeError) << "Invalid max_threads value";
  XGRAMMAR_UNREACHABLE();
}

// Wrap std::exception into TVM-FFI error
#define XGRAMMAR_FFI_TRY_BEGIN() try {
#define XGRAMMAR_FFI_TRY_END()                   \
  }                                              \
  catch (const XGrammarError& e) {               \
    throw ffi::Error(e.GetType(), e.what(), ""); \
    XGRAMMAR_UNREACHABLE();                      \
  }

// ----- Object wrappers (hold xgrammar types, inherit ffi::Object) -----

class TokenizerInfoObj : public ffi::Object {
 public:
  TokenizerInfo value;

  explicit TokenizerInfoObj(TokenizerInfo v) : value(std::move(v)) {}

  TokenizerInfoObj(
      ffi::Array<ffi::Any> encoded_vocab,
      int64_t vocab_type,
      ffi::AnyView vocab_size_opt,
      ffi::AnyView stop_token_ids_opt,
      bool add_prefix_space
  )
      : value(NullObj{}) {
    XGRAMMAR_FFI_TRY_BEGIN();
    value = TokenizerInfo_Init(
        ArrayAnyToVectorString(encoded_vocab),
        static_cast<int>(vocab_type),
        OptionalIntFromView(vocab_size_opt),
        OptionalInt32VectorFromView(stop_token_ids_opt),
        add_prefix_space
    );
    XGRAMMAR_FFI_TRY_END();
  }

  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "xgrammar.tvm_ffi_binding.TokenizerInfo", TokenizerInfoObj, ffi::Object
  );
};

class GrammarObj : public ffi::Object {
 public:
  Grammar value;

  explicit GrammarObj(Grammar v) : value(std::move(v)) {}

  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("xgrammar.tvm_ffi_binding.Grammar", GrammarObj, ffi::Object);
};

class CompiledGrammarObj : public ffi::Object {
 public:
  CompiledGrammar value;

  explicit CompiledGrammarObj(CompiledGrammar v) : value(std::move(v)) {}

  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "xgrammar.tvm_ffi_binding.CompiledGrammar", CompiledGrammarObj, ffi::Object
  );
};

class GrammarCompilerObj : public ffi::Object {
 public:
  GrammarCompiler value;

  explicit GrammarCompilerObj(GrammarCompiler v) : value(std::move(v)) {}

  GrammarCompilerObj(
      ffi::ObjectRef tokenizer_ref,
      int64_t max_threads,
      bool cache_enabled,
      int64_t max_memory_bytes
  )
      : value(
            tokenizer_ref.as<TokenizerInfoObj>()->value,
            static_cast<int>(max_threads),
            cache_enabled,
            max_memory_bytes
        ) {}

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "xgrammar.tvm_ffi_binding.GrammarCompiler", GrammarCompilerObj, ffi::Object
  );
};

class GrammarMatcherObj : public ffi::Object {
 public:
  GrammarMatcher value;

  explicit GrammarMatcherObj(GrammarMatcher v) : value(std::move(v)) {}

  GrammarMatcherObj(
      ffi::ObjectRef compiled_grammar_ref,
      ffi::AnyView override_stop_tokens_opt,
      bool terminate_without_stop_token,
      int64_t max_rollback_tokens
  )
      : value(
            compiled_grammar_ref.as<CompiledGrammarObj>()->value,
            OptionalIntVectorFromView(override_stop_tokens_opt),
            terminate_without_stop_token,
            static_cast<int>(max_rollback_tokens)
        ) {}

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "xgrammar.tvm_ffi_binding.GrammarMatcher", GrammarMatcherObj, ffi::Object
  );
};

class BatchGrammarMatcherObj : public ffi::Object {
 public:
  BatchGrammarMatcher value;

  BatchGrammarMatcherObj() = default;
  explicit BatchGrammarMatcherObj(BatchGrammarMatcher v) : value(std::move(v)) {}

  explicit BatchGrammarMatcherObj(ffi::AnyView max_threads_view)
      : value(ParseMaxThreads(max_threads_view)) {}

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "xgrammar.tvm_ffi_binding.BatchGrammarMatcher", BatchGrammarMatcherObj, ffi::Object
  );
};

// ----- Registration: ObjectDef -----
// Custom constructors are handled via lambda wrappers below; TVM-FFI uses refl::init<Args...>()
// and we need to bridge FFI types to xgrammar types in those lambdas.

TVM_FFI_STATIC_INIT_BLOCK() {
  using O = ffi::ObjectRef;

  // TokenizerInfo: init(encoded_vocab, vocab_type, vocab_size_opt, stop_token_ids_opt,
  // add_prefix_space)
  refl::ObjectDef<TokenizerInfoObj>()
      .def(refl::init<ffi::Array<ffi::Any>, int64_t, ffi::AnyView, ffi::AnyView, bool>())
      .def(
          "vocab_type",
          [](const TokenizerInfoObj* o) {
            return static_cast<int64_t>(TokenizerInfo_GetVocabType(o->value));
          }
      )
      .def(
          "vocab_size",
          [](const TokenizerInfoObj* o) { return static_cast<int64_t>(o->value.GetVocabSize()); }
      )
      .def(
          "add_prefix_space", [](const TokenizerInfoObj* o) { return o->value.GetAddPrefixSpace(); }
      )
      .def(
          "decoded_vocab",
          [](const TokenizerInfoObj* o) {
            const auto& decoded_vocab = o->value.GetDecodedVocab();
            return VectorStringToBytesArray(decoded_vocab);
          }
      )
      .def(
          "stop_token_ids",
          [](const TokenizerInfoObj* o) {
            const auto& stop_token_ids = o->value.GetStopTokenIds();
            ffi::Array<int64_t> stop_token_ids_array;
            for (int32_t token_id : stop_token_ids)
              stop_token_ids_array.push_back(static_cast<int64_t>(token_id));
            return stop_token_ids_array;
          }
      )
      .def(
          "special_token_ids",
          [](const TokenizerInfoObj* o) {
            const auto& special_token_ids = o->value.GetSpecialTokenIds();
            ffi::Array<int64_t> special_token_ids_array;
            for (int32_t token_id : special_token_ids)
              special_token_ids_array.push_back(static_cast<int64_t>(token_id));
            return special_token_ids_array;
          }
      )
      .def(
          "dump_metadata",
          [](const TokenizerInfoObj* o) { return ffi::String(o->value.DumpMetadata()); }
      )
      .def_static(
          "from_vocab_and_metadata",
          [](ffi::Array<ffi::Any> encoded_vocab, ffi::String metadata) {
            XGRAMMAR_FFI_TRY_BEGIN();
            auto v = TokenizerInfo::FromVocabAndMetadata(
                ArrayAnyToVectorString(encoded_vocab), metadata
            );
            return ffi::ObjectRef(ffi::make_object<TokenizerInfoObj>(std::move(v)));
            XGRAMMAR_FFI_TRY_END();
          }
      )
      .def_static(
          "_detect_metadata_from_hf",
          [](ffi::String backend_str) {
            return ffi::String(TokenizerInfo::DetectMetadataFromHF(backend_str));
          }
      )
      .def(
          "serialize_json",
          [](const TokenizerInfoObj* o) { return ffi::String(o->value.SerializeJSON()); }
      )
      .def_static("deserialize_json", [](ffi::String json_string) {
        XGRAMMAR_FFI_TRY_BEGIN();
        auto r = TokenizerInfo::DeserializeJSON(json_string);
        if (std::holds_alternative<SerializationError>(r)) {
          const auto& err = std::get<SerializationError>(r);
          throw ffi::Error(GetTypeFromVariantError(err), GetMessageFromVariantError(err), "");
        }
        return ffi::ObjectRef(ffi::make_object<TokenizerInfoObj>(std::get<TokenizerInfo>(r)));
        XGRAMMAR_FFI_TRY_END();
      });

  // Grammar
  refl::ObjectDef<GrammarObj>()
      .def("to_string", [](const GrammarObj* o) { return ffi::String(o->value.ToString()); })
      .def_static(
          "from_ebnf",
          [](ffi::String ebnf_str, ffi::String root_rule_name) {
            return ffi::ObjectRef(
                ffi::make_object<GrammarObj>(Grammar::FromEBNF(ebnf_str, root_rule_name))
            );
          }
      )
      .def_static(
          "from_json_schema",
          [](ffi::String schema,
             bool any_whitespace,
             ffi::AnyView indent,
             ffi::AnyView separators,
             bool strict_mode,
             ffi::AnyView max_whitespace_cnt,
             bool print_converted_ebnf) {
            XGRAMMAR_FFI_TRY_BEGIN();
            auto g = Grammar::FromJSONSchema(
                schema,
                any_whitespace,
                OptionalIntFromView(indent),
                OptionalSeparatorsFromView(separators),
                strict_mode,
                OptionalIntFromView(max_whitespace_cnt),
                print_converted_ebnf
            );
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(std::move(g)));
            XGRAMMAR_FFI_TRY_END();
          }
      )
      .def_static(
          "from_regex",
          [](ffi::String regex, bool print_converted_ebnf) {
            return ffi::ObjectRef(
                ffi::make_object<GrammarObj>(Grammar::FromRegex(regex, print_converted_ebnf))
            );
          }
      )
      .def_static(
          "from_structural_tag",
          [](ffi::String structural_tag_json) {
            XGRAMMAR_FFI_TRY_BEGIN();
            Grammar grammar = Grammar_FromStructuralTag(structural_tag_json);
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(std::move(grammar)));
            XGRAMMAR_FFI_TRY_END();
          }
      )
      .def_static(
          "builtin_json_grammar",
          []() {
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(Grammar::BuiltinJSONGrammar()));
          }
      )
      .def_static(
          "union",
          [](ffi::Array<O> grammars) {
            std::vector<Grammar> grammar_list;
            grammar_list.reserve(static_cast<size_t>(grammars.size()));
            for (int64_t i = 0; i < static_cast<int64_t>(grammars.size()); ++i) {
              grammar_list.push_back(grammars[i].as<GrammarObj>()->value);
            }
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(Grammar::Union(grammar_list)));
          }
      )
      .def_static(
          "concat",
          [](ffi::Array<O> grammars) {
            std::vector<Grammar> grammar_list;
            grammar_list.reserve(static_cast<size_t>(grammars.size()));
            for (int64_t i = 0; i < static_cast<int64_t>(grammars.size()); ++i) {
              grammar_list.push_back(grammars[i].as<GrammarObj>()->value);
            }
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(Grammar::Concat(grammar_list)));
          }
      )
      .def(
          "serialize_json",
          [](const GrammarObj* o) { return ffi::String(o->value.SerializeJSON()); }
      )
      .def_static("deserialize_json", [](ffi::String json_string) {
        XGRAMMAR_FFI_TRY_BEGIN();
        Grammar grammar = Grammar_DeserializeJSON(json_string);
        return ffi::ObjectRef(ffi::make_object<GrammarObj>(std::move(grammar)));
        XGRAMMAR_FFI_TRY_END();
      });

  // CompiledGrammar
  refl::ObjectDef<CompiledGrammarObj>()
      .def(
          "grammar",
          [](const CompiledGrammarObj* o) {
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(o->value.GetGrammar()));
          }
      )
      .def(
          "tokenizer_info",
          [](const CompiledGrammarObj* o) {
            return ffi::ObjectRef(ffi::make_object<TokenizerInfoObj>(o->value.GetTokenizerInfo()));
          }
      )
      .def(
          "memory_size_bytes",
          [](const CompiledGrammarObj* o) {
            return static_cast<int64_t>(o->value.MemorySizeBytes());
          }
      )
      .def(
          "serialize_json",
          [](const CompiledGrammarObj* o) { return ffi::String(o->value.SerializeJSON()); }
      )
      .def_static("deserialize_json", [](ffi::String json_string, O tokenizer_ref) {
        XGRAMMAR_FFI_TRY_BEGIN();
        const TokenizerInfo& tokenizer_info = tokenizer_ref.as<TokenizerInfoObj>()->value;
        CompiledGrammar compiled_grammar =
            CompiledGrammar_DeserializeJSON(json_string, tokenizer_info);
        return ffi::ObjectRef(ffi::make_object<CompiledGrammarObj>(std::move(compiled_grammar)));
        XGRAMMAR_FFI_TRY_END();
      });

  // GrammarCompiler: init(tokenizer_info, max_threads, cache_enabled, max_memory_bytes)
  refl::ObjectDef<GrammarCompilerObj>()
      .def(refl::init<O, int64_t, bool, int64_t>())
      .def(
          "compile_json_schema",
          [](GrammarCompilerObj* o,
             ffi::String schema,
             bool any_whitespace,
             ffi::AnyView indent,
             ffi::AnyView separators,
             bool strict_mode,
             ffi::AnyView max_whitespace_cnt) {
            XGRAMMAR_FFI_TRY_BEGIN();
            CompiledGrammar cg = o->value.CompileJSONSchema(
                schema,
                any_whitespace,
                OptionalIntFromView(indent),
                OptionalSeparatorsFromView(separators),
                strict_mode,
                OptionalIntFromView(max_whitespace_cnt)
            );
            return ffi::ObjectRef(ffi::make_object<CompiledGrammarObj>(std::move(cg)));
            XGRAMMAR_FFI_TRY_END();
          }
      )
      .def(
          "compile_builtin_json_grammar",
          [](GrammarCompilerObj* o) {
            return ffi::ObjectRef(
                ffi::make_object<CompiledGrammarObj>(o->value.CompileBuiltinJSONGrammar())
            );
          }
      )
      .def(
          "compile_structural_tag",
          [](GrammarCompilerObj* o, ffi::String structural_tag_json) {
            return ffi::ObjectRef(ffi::make_object<CompiledGrammarObj>(
                o->value.CompileStructuralTag(structural_tag_json)
            ));
          }
      )
      .def(
          "compile_regex",
          [](GrammarCompilerObj* o, ffi::String regex) {
            return ffi::ObjectRef(ffi::make_object<CompiledGrammarObj>(o->value.CompileRegex(regex))
            );
          }
      )
      .def(
          "compile_grammar_ebnf",
          [](GrammarCompilerObj* o, O grammar_ref) {
            return ffi::ObjectRef(ffi::make_object<CompiledGrammarObj>(
                o->value.CompileGrammar(grammar_ref.as<GrammarObj>()->value)
            ));
          }
      )
      .def(
          "compile_grammar_from_strings",
          [](GrammarCompilerObj* o, ffi::String ebnf_str, ffi::String root_rule_name) {
            return ffi::ObjectRef(ffi::make_object<CompiledGrammarObj>(
                o->value.CompileGrammar(ebnf_str, root_rule_name)
            ));
          }
      )
      .def("clear_cache", [](GrammarCompilerObj* o) { o->value.ClearCache(); })
      .def(
          "get_cache_size_bytes",
          [](const GrammarCompilerObj* o) { return o->value.GetCacheSizeBytes(); }
      )
      .def("cache_limit_bytes", [](const GrammarCompilerObj* o) {
        return o->value.CacheLimitBytes();
      });

  // BatchGrammarMatcher: init(max_threads)
  refl::ObjectDef<BatchGrammarMatcherObj>()
      .def(refl::init<ffi::AnyView>())
      .def(
          "batch_fill_next_token_bitmask",
          [](BatchGrammarMatcherObj* o,
             ffi::Array<O> matchers_ref,
             ffi::AnyView batch_token_bitmask,
             ffi::AnyView indices,
             bool debug_print) {
            std::vector<GrammarMatcher> matchers;
            matchers.reserve(matchers_ref.size());
            for (int64_t i = 0; i < static_cast<int64_t>(matchers_ref.size()); ++i) {
              matchers.push_back(matchers_ref[i].as<GrammarMatcherObj>()->value);
            }
            DLTensor* bitmask = batch_token_bitmask.cast<DLTensor*>();
            o->value.BatchFillNextTokenBitmask(
                &matchers, bitmask, OptionalInt32VectorFromView(indices), debug_print
            );
          }
      )
      .def_static(
          "batch_accept_string",
          [](ffi::Array<O> matchers_ref, ffi::Array<ffi::Any> input_str_byte_union, bool debug_print
          ) {
            std::vector<GrammarMatcher> matchers;
            matchers.reserve(matchers_ref.size());
            for (int64_t i = 0; i < static_cast<int64_t>(matchers_ref.size()); ++i) {
              matchers.push_back(matchers_ref[i].as<GrammarMatcherObj>()->value);
            }
            std::vector<std::string> input_strings = ArrayAnyToVectorString(input_str_byte_union);
            std::vector<uint8_t> acceptance_results =
                BatchGrammarMatcher::BatchAcceptString(&matchers, input_strings, debug_print);
            ffi::Array<int64_t> acceptance_results_array;
            for (uint8_t acceptance_flag : acceptance_results)
              acceptance_results_array.push_back(static_cast<int64_t>(acceptance_flag));
            return acceptance_results_array;
          }
      )
      .def_static(
          "batch_accept_token",
          [](ffi::Array<O> matchers_ref, ffi::Array<int64_t> token_ids, bool debug_print) {
            std::vector<GrammarMatcher> matchers;
            matchers.reserve(matchers_ref.size());
            for (int64_t i = 0; i < static_cast<int64_t>(matchers_ref.size()); ++i) {
              matchers.push_back(matchers_ref[i].as<GrammarMatcherObj>()->value);
            }
            std::vector<int32_t> token_id_vector;
            token_id_vector.reserve(token_ids.size());
            for (int64_t i = 0; i < static_cast<int64_t>(token_ids.size()); ++i) {
              token_id_vector.push_back(static_cast<int32_t>(token_ids[i]));
            }
            std::vector<uint8_t> acceptance_results =
                BatchGrammarMatcher::BatchAcceptToken(&matchers, token_id_vector, debug_print);
            ffi::Array<int64_t> acceptance_results_array;
            for (uint8_t acceptance_flag : acceptance_results)
              acceptance_results_array.push_back(static_cast<int64_t>(acceptance_flag));
            return acceptance_results_array;
          }
      )
      .def_static("batch_rollback", [](ffi::Array<O> matchers_ref, ffi::Array<int64_t> num_tokens) {
        std::vector<GrammarMatcher> matchers;
        matchers.reserve(matchers_ref.size());
        for (int64_t i = 0; i < static_cast<int64_t>(matchers_ref.size()); ++i) {
          matchers.push_back(matchers_ref[i].as<GrammarMatcherObj>()->value);
        }
        std::vector<int> num_tokens_vector;
        num_tokens_vector.reserve(num_tokens.size());
        for (int64_t i = 0; i < static_cast<int64_t>(num_tokens.size()); ++i) {
          num_tokens_vector.push_back(static_cast<int>(num_tokens[i]));
        }
        BatchGrammarMatcher::BatchRollback(&matchers, num_tokens_vector);
      });

  // GrammarMatcher: init(compiled_grammar, override_stop_tokens_opt, terminate_without_stop,
  // max_rollback_tokens)
  refl::ObjectDef<GrammarMatcherObj>()
      .def(refl::init<O, ffi::AnyView, bool, int64_t>())
      .def(
          "accept_token",
          [](GrammarMatcherObj* o, int64_t token_id, bool debug_print) {
            return o->value.AcceptToken(static_cast<int32_t>(token_id), debug_print);
          }
      )
      .def(
          "accept_string",
          [](GrammarMatcherObj* o, ffi::Any input_str_bytes_union, bool debug_print) {
            ffi::AnyView view = input_str_bytes_union;
            if (view.as<ffi::Bytes>()) {
              return o->value.AcceptString(BytesToString(view.cast<ffi::Bytes>()), debug_print);
            } else if (view.as<ffi::String>()) {
              return o->value.AcceptString(view.cast<ffi::String>(), debug_print);
            } else {
              TVM_FFI_THROW(RuntimeError) << "Unsupported type in accept_string";
              XGRAMMAR_UNREACHABLE();
            }
          }
      )
      .def(
          "fill_next_token_bitmask",
          [](GrammarMatcherObj* o, ffi::AnyView token_bitmask, int64_t index, bool debug_print) {
            DLTensor* dlt = token_bitmask.cast<DLTensor*>();
            return o->value.FillNextTokenBitmask(dlt, static_cast<int>(index), debug_print);
          }
      )
      .def(
          "traverse_draft_tree",
          [](GrammarMatcherObj* o,
             ffi::AnyView retrieve_next_token,
             ffi::AnyView retrieve_next_sibling,
             ffi::AnyView draft_tokens,
             ffi::AnyView token_bitmask,
             double time_threshold) {
            DLTensor* retrieve_next_token_ptr = retrieve_next_token.cast<DLTensor*>();
            DLTensor* retrieve_next_sibling_ptr = retrieve_next_sibling.cast<DLTensor*>();
            DLTensor* draft_tokens_ptr = draft_tokens.cast<DLTensor*>();
            DLTensor* token_bitmask_ptr = token_bitmask.cast<DLTensor*>();
            return o->value.TraverseDraftTree(
                retrieve_next_token_ptr,
                retrieve_next_sibling_ptr,
                draft_tokens_ptr,
                token_bitmask_ptr,
                time_threshold
            );
          }
      )
      .def(
          "find_jump_forward_string",
          [](GrammarMatcherObj* o) { return ffi::String(o->value.FindJumpForwardString()); }
      )
      .def(
          "rollback",
          [](GrammarMatcherObj* o, int64_t num_tokens) {
            o->value.Rollback(static_cast<int>(num_tokens));
          }
      )
      .def(
          "fork",
          [](GrammarMatcherObj* o) {
            return ffi::ObjectRef(ffi::make_object<GrammarMatcherObj>(o->value.Fork()));
          }
      )
      .def("is_terminated", [](const GrammarMatcherObj* o) { return o->value.IsTerminated(); })
      .def("is_completed", [](const GrammarMatcherObj* o) { return o->value.IsCompleted(); })
      .def("reset", [](GrammarMatcherObj* o) { o->value.Reset(); })
      .def(
          "max_rollback_tokens",
          [](const GrammarMatcherObj* o) {
            return static_cast<int64_t>(o->value.GetMaxRollbackTokens());
          }
      )
      .def(
          "stop_token_ids",
          [](const GrammarMatcherObj* o) {
            const auto& stop_token_ids = o->value.GetStopTokenIds();
            ffi::Array<int64_t> stop_token_ids_array;
            for (int token_id : stop_token_ids)
              stop_token_ids_array.push_back(static_cast<int64_t>(token_id));
            return stop_token_ids_array;
          }
      )
      .def("_debug_print_internal_state", [](const GrammarMatcherObj* o) {
        return ffi::String(o->value._DebugPrintInternalState());
      });

  // ----- Global functions: testing, kernels, config, exceptions -----
  refl::GlobalDef()
      .def(
          "xgrammar.tvm_ffi_binding.testing._json_schema_to_ebnf",
          [](ffi::String schema,
             bool any_whitespace,
             ffi::AnyView indent,
             ffi::AnyView separators,
             bool strict_mode,
             ffi::AnyView max_whitespace_cnt) {
            return ffi::String(JSONSchemaToEBNF(
                schema,
                any_whitespace,
                OptionalIntFromView(indent),
                OptionalSeparatorsFromView(separators),
                strict_mode,
                OptionalIntFromView(max_whitespace_cnt),
                JSONFormat::kJSON
            ));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._regex_to_ebnf",
          [](ffi::String regex, ffi::AnyView with_rule_name_opt) {
            bool with_rule_name = OptionalBoolFromView(with_rule_name_opt).value_or(true);
            return ffi::String(RegexToEBNF(regex, with_rule_name));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._ebnf_to_grammar_no_normalization",
          [](ffi::String ebnf_str, ffi::String root_rule_name) {
            Grammar grammar = _EBNFToGrammarNoNormalization(ebnf_str, root_rule_name);
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(std::move(grammar)));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._get_masked_tokens_from_bitmask",
          [](int64_t token_bitmask_ptr, ffi::Array<int64_t> shape, int64_t vocab_size, int64_t index
          ) {
            std::vector<int64_t> shape_vector;
            for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i)
              shape_vector.push_back(shape[i]);
            std::vector<int> masked_tokens = Testing_DebugGetMaskedTokensFromBitmask(
                static_cast<intptr_t>(token_bitmask_ptr),
                shape_vector,
                static_cast<int32_t>(vocab_size),
                static_cast<int32_t>(index)
            );
            ffi::Array<int64_t> masked_tokens_array;
            for (int token_id : masked_tokens)
              masked_tokens_array.push_back(static_cast<int64_t>(token_id));
            return masked_tokens_array;
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._is_single_token_bitmask",
          [](int64_t token_bitmask_ptr, ffi::Array<int64_t> shape, int64_t vocab_size, int64_t index
          ) {
            std::vector<int64_t> shape_vector;
            for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i)
              shape_vector.push_back(shape[i]);
            auto single_token_result = Testing_IsSingleTokenBitmask(
                static_cast<intptr_t>(token_bitmask_ptr),
                shape_vector,
                static_cast<int32_t>(vocab_size),
                static_cast<int32_t>(index)
            );
            return ffi::Array<int64_t>{
                static_cast<int64_t>(single_token_result.first),
                static_cast<int64_t>(single_token_result.second)
            };
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._get_allow_empty_rule_ids",
          [](O compiled_grammar_ref) {
            const auto& compiled_grammar = compiled_grammar_ref.as<CompiledGrammarObj>()->value;
            std::vector<int32_t> allow_empty_rule_ids = GetAllowEmptyRuleIds(compiled_grammar);
            ffi::Array<int64_t> allow_empty_rule_ids_array;
            for (int32_t rule_id : allow_empty_rule_ids)
              allow_empty_rule_ids_array.push_back(static_cast<int64_t>(rule_id));
            return allow_empty_rule_ids_array;
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._generate_range_regex",
          [](ffi::AnyView start, ffi::AnyView end) {
            std::optional<int> start_value = OptionalIntFromView(start);
            std::optional<int> end_value = OptionalIntFromView(end);
            std::string regex_string = GenerateRangeRegex(start_value, end_value);
            regex_string.erase(
                std::remove(regex_string.begin(), regex_string.end(), '\0'), regex_string.end()
            );
            return ffi::String(regex_string);
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._generate_float_regex",
          [](ffi::AnyView start, ffi::AnyView end) {
            std::optional<double> start_value =
                start == nullptr ? std::nullopt : std::make_optional(start.cast<double>());
            std::optional<double> end_value =
                end == nullptr ? std::nullopt : std::make_optional(end.cast<double>());
            std::string regex_string = GenerateFloatRangeRegex(start_value, end_value);
            regex_string.erase(
                std::remove(regex_string.begin(), regex_string.end(), '\0'), regex_string.end()
            );
            return ffi::String(regex_string);
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._qwen_xml_tool_calling_to_ebnf",
          [](ffi::String schema) { return ffi::String(QwenXMLToolCallingToEBNF(schema)); }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._minimax_xml_tool_calling_to_ebnf",
          [](ffi::String schema) { return ffi::String(MiniMaxXMLToolCallingToEBNF(schema)); }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._deepseek_xml_tool_calling_to_ebnf",
          [](ffi::String schema) { return ffi::String(DeepSeekXMLToolCallingToEBNF(schema)); }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._glm_xml_tool_calling_to_ebnf",
          [](ffi::String schema) { return ffi::String(GlmXMLToolCallingToEBNF(schema)); }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._print_grammar_fsms",
          [](O grammar_ref) {
            return ffi::String(_PrintGrammarFSMs(grammar_ref.as<GrammarObj>()->value));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing.grammar_functor.structure_normalizer",
          [](O grammar_ref) {
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(
                StructureNormalizer::Apply(grammar_ref.as<GrammarObj>()->value)
            ));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing.grammar_functor.byte_string_fuser",
          [](O grammar_ref) {
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(
                ByteStringFuser::Apply(grammar_ref.as<GrammarObj>()->value)
            ));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing.grammar_functor.rule_inliner",
          [](O grammar_ref) {
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(
                RuleInliner::Apply(grammar_ref.as<GrammarObj>()->value)
            ));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing.grammar_functor.dead_code_eliminator",
          [](O grammar_ref) {
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(
                DeadCodeEliminator::Apply(grammar_ref.as<GrammarObj>()->value)
            ));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing.grammar_functor.lookahead_assertion_analyzer",
          [](O grammar_ref) {
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(
                LookaheadAssertionAnalyzer::Apply(grammar_ref.as<GrammarObj>()->value)
            ));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing.grammar_functor.grammar_optimizer",
          [](O grammar_ref) {
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(
                GrammarOptimizer::Apply(grammar_ref.as<GrammarObj>()->value)
            ));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing.grammar_functor.repetition_normalizer",
          [](O grammar_ref) {
            Grammar grammar = grammar_ref.as<GrammarObj>()->value;
            RepetitionNormalizer::Apply(&grammar);
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(std::move(grammar)));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.kernels.apply_token_bitmask_inplace_cpu",
          [](int64_t logits_ptr,
             ffi::Array<int64_t> logits_shape,
             ffi::Array<int64_t> logits_strides,
             int64_t bitmask_ptr,
             ffi::Array<int64_t> bitmask_shape,
             ffi::Array<int64_t> bitmask_strides,
             int64_t vocab_size,
             ffi::AnyView indices,
             ffi::String logit_type) {
            Kernels_ApplyTokenBitmaskInplaceCPU(
                static_cast<intptr_t>(logits_ptr),
                {logits_shape[0], logits_shape[1]},
                {logits_strides[0], logits_strides[1]},
                static_cast<intptr_t>(bitmask_ptr),
                {bitmask_shape[0], bitmask_shape[1]},
                {bitmask_strides[0], bitmask_strides[1]},
                static_cast<int>(vocab_size),
                OptionalIntVectorFromView(indices),
                logit_type
            );
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.config.set_max_recursion_depth",
          [](int64_t depth) { SetMaxRecursionDepth(static_cast<int>(depth)); }
      )
      .def(
          "xgrammar.tvm_ffi_binding.config.get_max_recursion_depth",
          []() { return static_cast<int64_t>(GetMaxRecursionDepth()); }
      )
      .def(
          "xgrammar.tvm_ffi_binding.config.get_serialization_version",
          []() { return ffi::String(GetSerializationVersion()); }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._traverse_draft_tree",
          [](ffi::AnyView retrieve_next_token,
             ffi::AnyView retrieve_next_sibling,
             ffi::AnyView draft_tokens,
             O matcher_ref,
             ffi::AnyView bitmask,
             ffi::AnyView time_threshold_opt) {
            XGRAMMAR_FFI_TRY_BEGIN();
            DLTensor* retrieve_next_token_ptr = retrieve_next_token.cast<DLTensor*>();
            DLTensor* retrieve_next_sibling_ptr = retrieve_next_sibling.cast<DLTensor*>();
            DLTensor* draft_tokens_ptr = draft_tokens.cast<DLTensor*>();
            DLTensor* bitmask_ptr = bitmask.cast<DLTensor*>();
            double time_threshold =
                time_threshold_opt == nullptr ? -1.0 : time_threshold_opt.cast<double>();
            GrammarMatcher& matcher =
                const_cast<GrammarMatcher&>(matcher_ref.as<GrammarMatcherObj>()->value);
            return matcher.TraverseDraftTree(
                retrieve_next_token_ptr,
                retrieve_next_sibling_ptr,
                draft_tokens_ptr,
                bitmask_ptr,
                time_threshold
            );
            XGRAMMAR_FFI_TRY_END();
          }
      );
}

}  // namespace xgrammar
