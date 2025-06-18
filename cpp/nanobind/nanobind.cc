/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/nanobind/nanobind.cc
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <xgrammar/xgrammar.h>

#include "../grammar_functor.h"
#include "../json_schema_converter.h"
#include "../regex_converter.h"
#include "../support/recursion_guard.h"
#include "../testing.h"
#include "python_methods.h"

namespace nb = nanobind;
using namespace xgrammar;
using namespace nb::literals;

std::vector<std::string> CommonEncodedVocabType(
    const nb::typed<nb::list, std::variant<std::string, nb::bytes>> encoded_vocab
) {
  std::vector<std::string> encoded_vocab_strs;
  encoded_vocab_strs.reserve(encoded_vocab.size());
  for (const auto& token : encoded_vocab) {
    if (nb::bytes result; nb::try_cast(token, result)) {
      encoded_vocab_strs.emplace_back(result.c_str());
    } else if (nb::str result; nb::try_cast(token, result)) {
      encoded_vocab_strs.emplace_back(result.c_str());
    } else {
      throw nb::type_error("Expected str or bytes for encoded_vocab");
    }
  }
  return encoded_vocab_strs;
}

std::vector<nanobind::bytes> TokenizerInfo_GetDecodedVocab(const TokenizerInfo& tokenizer) {
  const auto& decoded_vocab = tokenizer.GetDecodedVocab();
  std::vector<nanobind::bytes> py_result;
  py_result.reserve(decoded_vocab.size());
  for (const auto& item : decoded_vocab) {
    py_result.emplace_back(nanobind::bytes(item.c_str()));
  }
  return py_result;
}

NB_MODULE(xgrammar_bindings, m) {
  auto pyTokenizerInfo = nb::class_<TokenizerInfo>(m, "TokenizerInfo", R"doc(
    The tokenizer info contains the vocabulary, the type of the vocabulary, and necessary
    information for the grammar-guided generation.

    Note that although some tokenizers will encode the tokens in a special format, e.g.
    "<0x1B>" for "\u001b" in the ByteFallback tokenizer, and "Ġ" for " " in the Byte-Level BPE
    tokenizer, TokenizerInfo always decodes the vocabulary to the original format (e.g. "\u001b"
    and " ").

    Also note that some models (e.g. Phi-3 and Deepseek-V2) may pad the vocabulary to a multiple
    of 32. In this case, the model's vocab_size is larger than the tokenizer's vocabulary size.
    Please pass the model's vocab_size to the vocab_size parameter in the constructor, because
    this information is used to determine the size of the token mask.

    Parameters
    ----------
    encoded_vocab : Union[List[bytes], List[str]]
        The encoded vocabulary of the tokenizer.

    vocab_type : VocabType, default: VocabType.RAW
        The type of the vocabulary. See also VocabType.

    vocab_size : Optional[int], default: None
        The size of the vocabulary. If not provided, the vocabulary size will be len(encoded_vocab).

    stop_token_ids : Optional[List[int]], default: None
        The stop token ids. If not provided, the stop token ids will be auto detected (but may not
        be correct).

    add_prefix_space : bool, default: False
        Whether the tokenizer will prepend a space before the text in the tokenization process.
  )doc");
  pyTokenizerInfo
      .def(
          "__init__",
          [](TokenizerInfo* out,
             const nb::typed<nb::list, std::variant<std::string, nb::bytes>> encoded_vocab,
             int vocab_type,
             std::optional<int> vocab_size,
             std::optional<std::vector<int32_t>> stop_token_ids,
             bool add_prefix_space) {
            new (out) TokenizerInfo{TokenizerInfo_Init(
                CommonEncodedVocabType(encoded_vocab),
                vocab_type,
                vocab_size,
                std::move(stop_token_ids),
                add_prefix_space
            )};
          },
          "encoded_vocab"_a,
          "vocab_type"_a,
          "vocab_size"_a.none(),
          "stop_token_ids"_a.none(),
          "add_prefix_space"_a
      )
      .def_prop_ro("vocab_type", &TokenizerInfo_GetVocabType, "The type of the vocabulary.")
      .def_prop_ro("vocab_size", &TokenizerInfo::GetVocabSize, "The size of the vocabulary.")
      .def_prop_ro(
          "add_prefix_space",
          &TokenizerInfo::GetAddPrefixSpace,
          "Whether the tokenizer will prepend a space before the text in the tokenization process."
      )
      .def_prop_ro("decoded_vocab", &TokenizerInfo_GetDecodedVocab, R"doc(
        The decoded vocabulary of the tokenizer. This converts the tokens in the LLM's
        vocabulary back to the original format of the input text. E.g. for type ByteFallback,
        the token <0x1B> is converted back to "\u001b".
      )doc")
      .def_prop_ro("stop_token_ids", &TokenizerInfo::GetStopTokenIds, "The stop token ids.")
      .def_prop_ro("special_token_ids", &TokenizerInfo::GetSpecialTokenIds, R"doc(
        The special token ids. Special tokens include control tokens, reserved tokens,
        padded tokens, etc. Now it is automatically detected from the vocabulary.
      )doc")
      .def("dump_metadata", &TokenizerInfo::DumpMetadata, R"doc(
        Dump the metadata of the tokenizer to a json string. It can be used to construct the
        tokenizer info from the vocabulary and the metadata string.
      )doc")
      .def_static(
          "from_vocab_and_metadata",
          [](const nb::typed<nb::list, std::variant<std::string, nb::bytes>> encoded_vocab,
             const std::string& metadata) {
            return TokenizerInfo::FromVocabAndMetadata(
                CommonEncodedVocabType(encoded_vocab), metadata
            );
          },
          "encoded_vocab"_a,
          "metadata"_a,
          R"doc(
        Construct the tokenizer info from the vocabulary and the metadata string in json
        format.

        Parameters
        ----------
        encoded_vocab : List[Union[bytes, str]]
            The encoded vocabulary of the tokenizer.

        metadata : str
            The metadata string in json format.
      )doc"
      )
      .def_static("_detect_metadata_from_hf", &TokenizerInfo::DetectMetadataFromHF);

  auto pyGrammar = nb::class_<Grammar>(m, "Grammar", R"doc(
    This class represents a grammar object in XGrammar, and can be used later in the
    grammar-guided generation.

    The Grammar object supports context-free grammar (CFG). EBNF (extended Backus-Naur Form) is
    used as the format of the grammar. There are many specifications for EBNF in the literature,
    and we follow the specification of GBNF (GGML BNF) in
    https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md.

    When printed, the grammar will be converted to GBNF format.
  )doc");
  pyGrammar.def("to_string", &Grammar::ToString)
      .def_static(
          "from_ebnf",
          &Grammar::FromEBNF,
          "ebnf_string"_a,
          nb::kw_only(),
          "root_rule_name"_a = "root",
          R"doc(
        Construct a grammar from EBNF string. The EBNF string should follow the format
        in https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md.

        Parameters
        ----------
        ebnf_string : str
            The grammar string in EBNF format.

        root_rule_name : str, default: "root"
            The name of the root rule in the grammar.

        Raises
        ------
        RuntimeError
            When converting the regex pattern fails, with details about the parsing error.
      )doc"
      )
      .def_static(
          "from_json_schema",
          &Grammar::FromJSONSchema,
          "schema"_a,
          nb::kw_only(),
          "any_whitespace"_a = true,
          "indent"_a.none() = nb::none(),
          "separators"_a.none() = nb::none(),
          "strict_mode"_a = true,
          "print_converted_ebnf"_a = false,
          nb::call_guard<nb::gil_scoped_release>(),
          R"doc(
        Construct a grammar from JSON schema. Pydantic model or JSON schema string can be
        used to specify the schema.

        It allows any whitespace by default. If user want to specify the format of the JSON,
        set `any_whitespace` to False and use the `indent` and `separators` parameters. The
        meaning and the default values of the parameters follows the convention in json.dumps().

        It internally converts the JSON schema to a EBNF grammar.

        Parameters
        ----------
        schema : Union[str, Type[BaseModel], Dict[str, Any]]
            The schema string or Pydantic model or JSON schema dict.

        any_whitespace : bool, default: True
            Whether to use any whitespace. If True, the generated grammar will ignore the
            indent and separators parameters, and allow any whitespace.

        indent : Optional[int], default: None
            The number of spaces for indentation. If None, the output will be in one line.

            Note that specifying the indentation means forcing the LLM to generate JSON strings
            strictly formatted. However, some models may tend to generate JSON strings that
            are not strictly formatted. In this case, forcing the LLM to generate strictly
            formatted JSON strings may degrade the generation quality. See
            <https://github.com/sgl-project/sglang/issues/2216#issuecomment-2516192009> for more
            details.

        separators : Optional[Tuple[str, str]], default: None
            Two separators used in the schema: comma and colon. Examples: (",", ":"), (", ", ": ").
            If None, the default separators will be used: (",", ": ") when the indent is not None,
            and (", ", ": ") otherwise.

        strict_mode : bool, default: True
            Whether to use strict mode. In strict mode, the generated grammar will not allow
            properties and items that is not specified in the schema. This is equivalent to
            setting unevaluatedProperties and unevaluatedItems to false.

            This helps LLM to generate accurate output in the grammar-guided generation with JSON
            schema.

        print_converted_ebnf : bool, default: False
            If True, the converted EBNF string will be printed. For debugging purposes.

        Returns
        -------
        grammar : Grammar
            The constructed grammar.

        Raises
        ------
        RuntimeError
            When converting the json schema fails, with details about the parsing error.
      )doc"
      )
      .def_static(
          "from_regex",
          &Grammar::FromRegex,
          "regex_string"_a,
          nb::kw_only(),
          "print_converted_ebnf"_a = false,
          nb::call_guard<nb::gil_scoped_release>(),
          R"doc(
        Create a grammar from a regular expression string.

        Parameters
        ----------
        regex_string : str
            The regular expression pattern to create the grammar from.

        print_converted_ebnf : bool, default: False
            This method will convert the regex pattern to EBNF first. If this is true, the converted
            EBNF string will be printed. For debugging purposes. Default: False.

        Returns
        -------
        grammar : Grammar
            The constructed grammar from the regex pattern.

        Raises
        ------
        RuntimeError
            When parsing the regex pattern fails, with details about the parsing error.
      )doc"
      )
      .def_static(
          "from_structural_tag",
          &Grammar_FromStructuralTag,
          "tags"_a,
          "triggers"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          R"doc(
        Create a grammar from structural tags. The structural tag handles the dispatching
        of different grammars based on the tags and triggers: it initially allows any output,
        until a trigger is encountered, then dispatch to the corresponding tag; when the end tag
        is encountered, the grammar will allow any following output, until the next trigger is
        encountered.

        The tags parameter is used to specify the output pattern. It is especially useful for LLM
        function calling, where the pattern is:
        <function=func_name>{"arg1": ..., "arg2": ...}</function>.
        This pattern consists of three parts: a begin tag (<function=func_name>), a parameter list
        according to some schema ({"arg1": ..., "arg2": ...}), and an end tag (</function>). This
        pattern can be described in a StructuralTagItem with a begin tag, a schema, and an end tag.
        The structural tag is able to handle multiple such patterns by passing them into multiple
        tags.

        The triggers parameter is used to trigger the dispatching of different grammars. The trigger
        should be a prefix of a provided begin tag. When the trigger is encountered, the
        corresponding tag should be used to constrain the following output. There can be multiple
        tags matching the same trigger. Then if the trigger is encountered, the following output
        should match one of the tags. For example, in function calling, the triggers can be
        ["<function="]. Then if "<function=" is encountered, the following output must match one
        of the tags (e.g. <function=get_weather>{"city": "Beijing"}</function>).

        The corrrespondence of tags and triggers is automatically determined: all tags with the
        same trigger will be grouped together. User should make sure any trigger is not a prefix
        of another trigger: then the corrrespondence of tags and triggers will be ambiguous.

        To use this grammar in grammar-guided generation, the GrammarMatcher constructed from
        structural tag will generate a mask for each token. When the trigger is not encountered,
        the mask will likely be all-1 and not have to be used (fill_next_token_bitmask returns
        False, meaning no token is masked). When a trigger is encountered, the mask should be
        enforced (fill_next_token_bitmask will return True, meaning some token is masked) to the
        output logits.

        The benefit of this method is the token boundary between tags and triggers is automatically
        handled. The user does not need to worry about the token boundary.

        Parameters
        ----------
        tags : List[StructuralTagItem]
            The structural tags.

        triggers : List[str]
            The triggers.

        Examples
        --------
        >>> class Schema1(BaseModel):
        >>>     arg1: str
        >>>     arg2: int
        >>> class Schema2(BaseModel):
        >>>     arg3: float
        >>>     arg4: List[str]
        >>> tags = [
        >>>     StructuralTagItem(begin="<function=f>", schema=Schema1, end="</function>"),
        >>>     StructuralTagItem(begin="<function=g>", schema=Schema2, end="</function>"),
        >>> ]
        >>> triggers = ["<function="]
        >>> grammar = Grammar.from_structural_tag(tags, triggers)
      )doc"
      )
      .def_static("builtin_json_grammar", &Grammar::BuiltinJSONGrammar, R"doc(
        Get the grammar of standard JSON. This is compatible with the official JSON grammar
        specification in https://www.json.org/json-en.html.

        Returns
        -------
        grammar : Grammar
            The JSON grammar.
      )doc")
      .def_static(
          "union", &Grammar::Union, "grammars"_a, nb::call_guard<nb::gil_scoped_release>(), R"doc(
        Create a grammar that matches any of the grammars in the list. That is equivalent to
        using the `|` operator to concatenate the grammars in the list.

        Parameters
        ----------
        grammars : List[Grammar]
            The grammars to create the union of.

        Returns
        -------
        grammar : Grammar
            The union of the grammars.
      )doc"
      )
      .def_static(
          "concat", &Grammar::Concat, "grammars"_a, nb::call_guard<nb::gil_scoped_release>(), R"doc(
        Create a grammar that matches the concatenation of the grammars in the list. That is
        equivalent to using the `+` operator to concatenate the grammars in the list.

        Parameters
        ----------
        grammars : List[Grammar]
            The grammars to create the concatenation of.

        Returns
        -------
        grammar : Grammar
            The concatenation of the grammars.
      )doc"
      );

  auto pyCompiledGrammar = nb::class_<CompiledGrammar>(m, "CompiledGrammar", R"doc(
    This is the primary object to store compiled grammar.

    A CompiledGrammar can be used to construct GrammarMatcher
    to generate token masks efficiently.

    Note
    ----
    Do not construct this class directly, instead
    use :class:`GrammarCompiler` to construct the object.
      )doc");
  pyCompiledGrammar.def_prop_ro("grammar", &CompiledGrammar::GetGrammar, "The original grammar.")
      .def_prop_ro(
          "tokenizer_info",
          &CompiledGrammar::GetTokenizerInfo,
          "The tokenizer info associated with the compiled grammar."
      )
      .def_prop_ro(
          "memory_size_bytes",
          &CompiledGrammar::MemorySizeBytes,
          "The approximate memory usage of the compiled grammar in bytes."
      );

  auto pyGrammarCompiler = nb::class_<GrammarCompiler>(m, "GrammarCompiler", R"doc(
    The compiler for grammars. It is associated with a certain tokenizer info, and compiles
    grammars into CompiledGrammar with the tokenizer info. It allows parallel compilation with
    multiple threads, and has a cache to store the compilation result, avoiding compiling the
    same grammar multiple times.

    Parameters
    ----------
    tokenizer_info : TokenizerInfo
        The tokenizer info.

    max_threads : int, default: 8
        The maximum number of threads used to compile the grammar.

    cache_enabled : bool, default: True
        Whether to enable the cache.

    cache_limit_bytes : int, default: -1
        The maximum memory usage for the cache in the specified unit.
        Note that the actual memory usage may slightly exceed this value.
      )doc");
  pyGrammarCompiler
      .def(
          nb::init<const TokenizerInfo&, int, bool, long long>(),
          "tokenizer_info"_a,
          nb::kw_only(),
          "max_threads"_a = 8,
          "cache_enabled"_a = true,
          "cache_limit_bytes"_a = -1
      )
      .def(
          "compile_json_schema",
          &GrammarCompiler::CompileJSONSchema,
          nb::call_guard<nb::gil_scoped_release>(),
          "schema"_a,
          nb::kw_only(),
          "any_whitespace"_a = true,
          "indent"_a.none() = nb::none(),
          "separators"_a.none() = nb::none(),
          "strict_mode"_a = true,
          R"doc(
        Get CompiledGrammar from the specified JSON schema and format. The indent
        and separators parameters follow the same convention as in json.dumps().

        Parameters
        ----------
        schema : Union[str, Type[BaseModel], Dict[str, Any]]
            The schema string or Pydantic model or JSON schema dict.

        indent : Optional[int], default: None
            The number of spaces for indentation. If None, the output will be in one line.

        separators : Optional[Tuple[str, str]], default: None
            Two separators used in the schema: comma and colon. Examples: (",", ":"), (", ", ": ").
            If None, the default separators will be used: (",", ": ") when the indent is not None,
            and (", ", ": ") otherwise.

        strict_mode : bool, default: True
            Whether to use strict mode. In strict mode, the generated grammar will not allow
            properties and items that is not specified in the schema. This is equivalent to
            setting unevaluatedProperties and unevaluatedItems to false.

            This helps LLM to generate accurate output in the grammar-guided generation with JSON
            schema.

        Returns
        -------
        compiled_grammar : CompiledGrammar
            The compiled grammar.
      )doc"
      )
      .def(
          "compile_builtin_json_grammar",
          &GrammarCompiler::CompileBuiltinJSONGrammar,
          nb::call_guard<nb::gil_scoped_release>(),
          R"doc(
        Get CompiledGrammar from the standard JSON.

        Returns
        -------
        compiled_grammar : CompiledGrammar
            The compiled grammar.
      )doc"
      )
      .def(
          "compile_structural_tag",
          &GrammarCompiler_CompileStructuralTag,
          nb::call_guard<nb::gil_scoped_release>(),
          "tags"_a,
          "triggers"_a,
          R"doc(
        Compile a grammar from structural tags. See Grammar.from_structural_tag() for more
        details.

        Parameters
        ----------
        tags : List[StructuralTagItem]
            The structural tags.

        triggers : List[str]
            The triggers.

        Returns
        -------
        compiled_grammar : CompiledGrammar
            The compiled grammar.
      )doc"
      )
      .def(
          "compile_regex",
          &GrammarCompiler::CompileRegex,
          nb::call_guard<nb::gil_scoped_release>(),
          "regex"_a,
          R"doc(
        Get CompiledGrammar from the specified regex.

        Parameters
        ----------
        regex : str
            The regex string.

        Returns
        -------
        compiled_grammar : CompiledGrammar
            The compiled grammar.
      )doc"
      )
      .def("compile_grammar", &GrammarCompiler::CompileGrammar, "grammar"_a)
      .def("clear_cache", &GrammarCompiler::ClearCache, "Clear all cached compiled grammars.")
      .def(
          "get_cache_size_bytes",
          &GrammarCompiler::GetCacheSizeBytes,
          "The approximate memory usage of the cache in bytes."
      )
      .def_prop_ro("cache_limit_bytes", &GrammarCompiler::CacheLimitBytes, R"doc(
        The maximum memory usage for the cache in bytes.
        Returns -1 if the cache has no memory limit.
      )doc");

  auto pyGrammarMatcher = nb::class_<GrammarMatcher>(m, "GrammarMatcher");
  pyGrammarMatcher
      .def(
          nb::init<const CompiledGrammar&, std::optional<std::vector<int>>, bool, int>(),
          nb::arg("compiled_grammar"),
          nb::arg("override_stop_tokens").none(),
          nb::arg("terminate_without_stop_token"),
          nb::arg("max_rollback_tokens")
      )
      .def("accept_token", &GrammarMatcher::AcceptToken, nb::call_guard<nb::gil_scoped_release>())
      .def("accept_string", &GrammarMatcher::AcceptString, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "accept_string",
          [](GrammarMatcher& self, const nb::bytes& input_str, bool debug_print) {
            return self.AcceptString(input_str.c_str(), debug_print);
          },
          nb::call_guard<nb::gil_scoped_release>()
      )
      .def(
          "fill_next_token_bitmask",
          &GrammarMatcher_FillNextTokenBitmask,
          nb::call_guard<nb::gil_scoped_release>()
      )
      .def(
          "find_jump_forward_string",
          &GrammarMatcher::FindJumpForwardString,
          nb::call_guard<nb::gil_scoped_release>()
      )
      .def("rollback", &GrammarMatcher::Rollback, nb::call_guard<nb::gil_scoped_release>())
      .def("is_terminated", &GrammarMatcher::IsTerminated)
      .def("reset", &GrammarMatcher::Reset, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("max_rollback_tokens", &GrammarMatcher::GetMaxRollbackTokens)
      .def_prop_ro("stop_token_ids", &GrammarMatcher::GetStopTokenIds)
      .def("_debug_print_internal_state", &GrammarMatcher::_DebugPrintInternalState);

  auto pyTestingModule = m.def_submodule("testing");
  pyTestingModule
      .def(
          "_json_schema_to_ebnf",
          nb::overload_cast<
              const std::string&,
              bool,
              std::optional<int>,
              std::optional<std::pair<std::string, std::string>>,
              bool>(&JSONSchemaToEBNF),
          nb::arg("schema"),
          nb::arg("any_whitespace"),
          nb::arg("indent").none(),
          nb::arg("separators").none(),
          nb::arg("strict_mode")
      )
      .def("_regex_to_ebnf", &RegexToEBNF)
      .def("_ebnf_to_grammar_no_normalization", &_EBNFToGrammarNoNormalization)
      .def("_get_masked_tokens_from_bitmask", &Testing_DebugGetMaskedTokensFromBitmask)
      .def("_is_single_token_bitmask", &Testing_IsSingleTokenBitmask)
      .def("_get_allow_empty_rule_ids", &GetAllowEmptyRuleIds)
      .def(
          "_generate_range_regex",
          [](std::optional<int> start, std::optional<int> end) {
            std::string result = GenerateRangeRegex(start, end);
            result.erase(std::remove(result.begin(), result.end(), '\0'), result.end());
            return result;
          },
          nb::arg("start").none(),
          nb::arg("end").none()
      )
      .def(
          "_generate_float_regex",
          [](std::optional<double> start, std::optional<double> end) {
            std::string result = GenerateFloatRangeRegex(start, end);
            result.erase(std::remove(result.begin(), result.end(), '\0'), result.end());
            return result;
          },
          nb::arg("start").none(),
          nb::arg("end").none()
      );

  auto pyGrammarFunctorModule = pyTestingModule.def_submodule("grammar_functor");
  pyGrammarFunctorModule.def("structure_normalizer", &StructureNormalizer::Apply)
      .def("byte_string_fuser", &ByteStringFuser::Apply)
      .def("rule_inliner", &RuleInliner::Apply)
      .def("dead_code_eliminator", &DeadCodeEliminator::Apply)
      .def("lookahead_assertion_analyzer", &LookaheadAssertionAnalyzer::Apply);

  auto pyKernelsModule = m.def_submodule("kernels");
  pyKernelsModule.def(
      "apply_token_bitmask_inplace_cpu",
      &Kernels_ApplyTokenBitmaskInplaceCPU,
      nb::arg("logits_ptr"),
      nb::arg("logits_shape"),
      nb::arg("bitmask_ptr"),
      nb::arg("bitmask_shape"),
      nb::arg("vocab_size"),
      nb::arg("indices").none(),
      nb::call_guard<nb::gil_scoped_release>()
  );

  auto pyConfigModule = m.def_submodule("config");
  pyConfigModule
      .def(
          "set_max_recursion_depth",
          &RecursionGuard::SetMaxRecursionDepth,
          nb::call_guard<nb::gil_scoped_release>()
      )
      .def(
          "get_max_recursion_depth",
          &RecursionGuard::GetMaxRecursionDepth,
          nb::call_guard<nb::gil_scoped_release>()
      );
}
