#![allow(unsafe_op_in_unsafe_fn)]
// Suppress warnings from auto-generated FFI code
#![allow(unused_imports)]

use autocxx::prelude::*;

include_cpp! {
    #include "xgrammar/xgrammar.h"
    #include "dlpack/dlpack.h"
    #include "cxx_utils.hpp"
    #include "cxx_utils/testing.hpp"
    safety!(unsafe_ffi)
    // xgrammar/compiler.h
    generate!("xgrammar::CompiledGrammar")
    generate!("xgrammar::GrammarCompiler")

    // xgrammar/config.h
    generate!("xgrammar::SetMaxRecursionDepth")
    generate!("xgrammar::GetMaxRecursionDepth")
    generate!("xgrammar::GetSerializationVersion")

    // xgrammar/grammar.h
    generate!("xgrammar::StructuralTagItem")
    generate!("xgrammar::Grammar")

    // xgrammar/matcher.h
    generate!("xgrammar::GetBitmaskSize")
    generate!("xgrammar::GetBitmaskDLType")
    generate!("xgrammar::ApplyTokenBitmaskInplaceCPU")
    generate!("xgrammar::GrammarMatcher")
    generate!("xgrammar::BatchGrammarMatcher")

    // xgrammar/tokenizer_info.h
    generate!("xgrammar::VocabType")
    generate!("xgrammar::TokenizerInfo")

    // cxx_utils/string_vec.hpp
    generate!("cxx_utils::new_string_vector")
    generate!("cxx_utils::string_vec_reserve")
    generate!("cxx_utils::string_vec_push_bytes")

    // cxx_utils/tokenizer_info.hpp
    generate!("cxx_utils::make_tokenizer_info")
    generate!("cxx_utils::tokenizer_info_deserialize_json_or_error")

    // cxx_utils/structural_tag.hpp
    generate!("cxx_utils::new_structural_tag_vector")
    generate!("cxx_utils::structural_tag_vec_reserve")
    generate!("cxx_utils::structural_tag_vec_push")

    // cxx_utils/grammar.hpp
    generate!("cxx_utils::grammar_from_json_schema")
    generate!("cxx_utils::grammar_from_ebnf")
    generate!("cxx_utils::grammar_from_regex")
    generate!("cxx_utils::grammar_from_structural_tag")
    generate!("cxx_utils::new_grammar_vector")
    generate!("cxx_utils::grammar_vec_reserve")
    generate!("cxx_utils::grammar_vec_push")
    generate!("cxx_utils::grammar_deserialize_json_or_error")

    // cxx_utils/compiled_grammar.hpp
    generate!("cxx_utils::compiled_grammar_deserialize_json_or_error")

    // cxx_utils/grammar_compiler.hpp
    generate!("cxx_utils::make_grammar_compiler")
    generate!("cxx_utils::compiler_compile_json_schema")
    generate!("cxx_utils::compiler_compile_builtin_json")
    generate!("cxx_utils::compiler_compile_regex")
    generate!("cxx_utils::compiler_compile_structural_tag")
    generate!("cxx_utils::compiler_compile_grammar_or_error")

    // cxx_utils/matcher.hpp
    generate!("cxx_utils::make_grammar_matcher")
    generate!("cxx_utils::make_batch_grammar_matcher")
    generate!("cxx_utils::new_grammar_matcher_vector")
    generate!("cxx_utils::grammar_matcher_vec_reserve")
    generate!("cxx_utils::grammar_matcher_vec_push")
    generate!("cxx_utils::batch_matcher_batch_fill_next_token_bitmask")
    generate!("cxx_utils::batch_accept_token")
    generate!("cxx_utils::batch_accept_string")

    // cxx_utils/testing.hpp
    generate!("cxx_utils::ebnf_to_grammar_no_normalization")
    generate!("cxx_utils::json_schema_to_ebnf")
    generate!("cxx_utils::qwen_xml_tool_calling_to_ebnf")
    generate!("cxx_utils::get_masked_tokens_from_bitmask")
    generate!("cxx_utils::SingleTokenResult")
    generate!("cxx_utils::is_single_token_bitmask")

    // DLPack core types
    generate_pod!("DLTensor")
    generate!("DLManagedTensor")  // Has function pointer, not POD
    generate_pod!("DLDevice")
    generate_pod!("DLDataType")
    generate_pod!("DLDeviceType")
    generate_pod!("DLDataTypeCode")

}

// Re-export DLPack types for public use.
//
// Note: these are generated from `dlpack/dlpack.h` via autocxx/bindgen. We
// provide Rust-side docs here to avoid leaking Doxygen markup (e.g. `\brief`)
// into docs.rs.
/// DLPack data type descriptor (`DLDataType`).
pub use ffi::DLDataType;
/// DLPack data type code enum (`DLDataTypeCode`).
pub use ffi::DLDataTypeCode;
/// DLPack device descriptor (`DLDevice`) (ABI-compatible with `dlpack/dlpack.h`).
pub use ffi::DLDevice;
/// DLPack device type enum (`DLDeviceType`).
pub use ffi::DLDeviceType;
/// DLPack managed tensor (`DLManagedTensor`) (owns tensor + deleter).
pub use ffi::DLManagedTensor;
/// DLPack tensor view (`DLTensor`) (does not own memory).
pub use ffi::DLTensor;
#[allow(unused_imports)]
use ffi::{
    cxx_utils,
    xgrammar::{
        BatchGrammarMatcher as FFIBatchGrammarMatcher,
        CompiledGrammar as FFICompiledGrammar,
        GetBitmaskDLType as FFIGetBitmaskDLType,
        GetBitmaskSize as FFIGetBitmaskSize,
        GetMaxRecursionDepth as FFIGetMaxRecursionDepth,
        GetSerializationVersion as FFIGetSerializationVersion,
        Grammar as FFIGrammar, GrammarCompiler as FFIGrammarCompiler,
        GrammarMatcher as FFIGrammarMatcher,
        SetMaxRecursionDepth as FFISetMaxRecursionDepth,
        StructuralTagItem as FFIStructuralTagItem,
        TokenizerInfo as FFITokenizerInfo,
    },
};

mod compiler;
mod config;
mod grammar;
mod matcher;
mod tokenizer_info;

pub mod testing;

pub use autocxx::{
    c_int as cxx_int, c_longlong as cxx_longlong, c_ulong as cxx_ulong,
    c_ulonglong as cxx_ulonglong,
};
pub use compiler::{CompiledGrammar, GrammarCompiler};
pub use config::{
    get_max_recursion_depth, get_serialization_version, set_max_recursion_depth,
};
pub use cxx::UniquePtr as CxxUniquePtr;
pub use ffi::xgrammar::VocabType;
pub use grammar::{Grammar, StructuralTagItem};
pub use matcher::{
    BatchGrammarMatcher, GrammarMatcher, allocate_token_bitmask,
    get_bitmask_shape, reset_token_bitmask,
};
pub use tokenizer_info::TokenizerInfo;
