//! Opaque wrappers for the C++ object types registered by
//! `libxgrammar_bindings`.
//!
//! Every wrapper is a single strong reference (`ObjectArc`) to a C++-allocated
//! object; the Rust side never mirrors C++ field layouts and never allocates
//! these objects itself — instances only come from FFI factory calls. All six
//! types are declared `final` on the C++ side.
//!
//! The tvm-ffi derive macros generate the whole FFI contract: the lazy
//! type-key → type-index binding (`ObjectCore`), the `Any` conversions
//! (`AnyCompatible`, `TryFrom<Any>`) and the argument-passing impls.

use tvm_ffi::{Object, ObjectArc};

/// Declares `$Obj`/`$Ref` wrapping the C++ type registered under `$type_key`.
macro_rules! def_xgrammar_object {
    ($(#[$doc:meta])* $Obj:ident, $Ref:ident, $type_key:literal) => {
        // Never constructed from Rust by design: instances are only ever
        // allocated by the C++ side and referenced through `$Ref`.
        #[repr(C)]
        #[derive(tvm_ffi::derive::Object)]
        #[type_key = $type_key]
        #[type_final]
        #[allow(dead_code)]
        pub(crate) struct $Obj {
            base: Object,
        }

        $(#[$doc])*
        #[repr(C)]
        #[derive(tvm_ffi::derive::ObjectRef, Clone)]
        pub(crate) struct $Ref {
            data: ObjectArc<$Obj>,
        }

        impl $Ref {
            /// Pointer identity, the analogue of Python's `handle.same_as`.
            #[allow(dead_code)]
            pub(crate) fn same_as(&self, other: &Self) -> bool {
                unsafe {
                    ObjectArc::as_raw(&self.data) as usize
                        == ObjectArc::as_raw(&other.data) as usize
                }
            }
        }

        impl std::fmt::Debug for $Ref {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}({:p})", $type_key, unsafe {
                    ObjectArc::as_raw(&self.data)
                })
            }
        }
    };
}

def_xgrammar_object!(
    /// Raw handle to a C++ `xgrammar::TokenizerInfo`.
    RawTokenizerInfoObj,
    RawTokenizerInfo,
    "xgrammar.tvm_ffi_binding.TokenizerInfo"
);
def_xgrammar_object!(
    /// Raw handle to a C++ `xgrammar::Grammar`.
    RawGrammarObj,
    RawGrammar,
    "xgrammar.tvm_ffi_binding.Grammar"
);
def_xgrammar_object!(
    /// Raw handle to a C++ `xgrammar::CompiledGrammar`.
    RawCompiledGrammarObj,
    RawCompiledGrammar,
    "xgrammar.tvm_ffi_binding.CompiledGrammar"
);
def_xgrammar_object!(
    /// Raw handle to a C++ `xgrammar::GrammarCompiler`.
    RawGrammarCompilerObj,
    RawGrammarCompiler,
    "xgrammar.tvm_ffi_binding.GrammarCompiler"
);
def_xgrammar_object!(
    /// Raw handle to a C++ `xgrammar::GrammarMatcher`.
    RawGrammarMatcherObj,
    RawGrammarMatcher,
    "xgrammar.tvm_ffi_binding.GrammarMatcher"
);
def_xgrammar_object!(
    /// Raw handle to a C++ `xgrammar::BatchGrammarMatcher`.
    RawBatchGrammarMatcherObj,
    RawBatchGrammarMatcher,
    "xgrammar.tvm_ffi_binding.BatchGrammarMatcher"
);

// The wrapped C++ objects are either immutable after construction
// (TokenizerInfo, Grammar, CompiledGrammar) or internally synchronized
// (GrammarCompiler); their handles can be shared across threads. The mutable
// matcher types stay `Send`-only: the public wrappers around them add a
// `!Sync` marker.
unsafe impl Send for RawTokenizerInfo {}
unsafe impl Sync for RawTokenizerInfo {}
unsafe impl Send for RawGrammar {}
unsafe impl Sync for RawGrammar {}
unsafe impl Send for RawCompiledGrammar {}
unsafe impl Sync for RawCompiledGrammar {}
unsafe impl Send for RawGrammarCompiler {}
unsafe impl Sync for RawGrammarCompiler {}
unsafe impl Send for RawGrammarMatcher {}
unsafe impl Send for RawBatchGrammarMatcher {}
