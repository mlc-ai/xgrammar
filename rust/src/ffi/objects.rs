//! Opaque wrappers for the C++ object types registered by
//! `libxgrammar_bindings`.
//!
//! Every wrapper is a single strong reference (`ObjectArc`) to a C++-allocated
//! object; the Rust side never mirrors C++ field layouts and never allocates
//! these objects itself — instances only come back from FFI factory calls.
//! All six types are declared `final` on the C++ side, so the strict type
//! check is an exact `type_index` comparison.
//!
//! Hand-written rather than derived: at tvm-ffi v0.1.12 the
//! `#[derive(Object)]`/`#[derive(ObjectRef)]` macros expand to crate-private
//! paths and cannot be used from an external crate.

use std::sync::OnceLock;

use tvm_ffi::tvm_ffi_sys::{TVMFFIAny, TVMFFIByteArray, TVMFFIObject, TVMFFITypeKeyToIndex};
use tvm_ffi::{AnyCompatible, Object, ObjectArc, ObjectCore, ObjectRefCore, TypeIndex};

/// Declares `$Obj`/`$Ref` wrapping the C++ type registered under `$type_key`.
macro_rules! def_xgrammar_object {
    ($(#[$doc:meta])* $Obj:ident, $Ref:ident, $type_key:literal) => {
        // Never constructed from Rust by design: instances are only ever
        // allocated by the C++ side and referenced through `$Ref`.
        #[repr(C)]
        #[allow(dead_code)]
        pub(crate) struct $Obj {
            base: Object,
        }

        unsafe impl ObjectCore for $Obj {
            const TYPE_KEY: &'static str = $type_key;

            fn type_index() -> i32 {
                static INDEX: OnceLock<i32> = OnceLock::new();
                *INDEX.get_or_init(|| unsafe {
                    let key = TVMFFIByteArray::from_str($type_key);
                    let mut index: i32 = -1;
                    if TVMFFITypeKeyToIndex(&key, &mut index) != 0 {
                        panic!(
                            "xgrammar: type key `{}` is not registered; \
                             the bindings library must be loaded first",
                            $type_key
                        );
                    }
                    index
                })
            }

            unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject {
                Object::object_header_mut(&mut this.base)
            }
        }

        $(#[$doc])*
        #[derive(Clone)]
        pub(crate) struct $Ref {
            data: ObjectArc<$Obj>,
        }

        unsafe impl ObjectRefCore for $Ref {
            type ContainerType = $Obj;

            fn data(this: &Self) -> &ObjectArc<$Obj> {
                &this.data
            }

            fn into_data(this: Self) -> ObjectArc<$Obj> {
                this.data
            }

            fn from_data(data: ObjectArc<$Obj>) -> Self {
                Self { data }
            }
        }

        unsafe impl AnyCompatible for $Ref {
            fn type_str() -> String {
                $type_key.to_string()
            }

            unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
                // The C++ type is declared final: exact match is the correct
                // subtype check here.
                data.type_index == <$Obj as ObjectCore>::type_index()
            }

            unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
                data.type_index = <$Obj as ObjectCore>::type_index();
                data.small_str_len = 0;
                data.data_union.v_obj =
                    ObjectArc::as_raw(Self::data(src)) as *mut TVMFFIObject;
            }

            unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
                data.type_index = <$Obj as ObjectCore>::type_index();
                data.small_str_len = 0;
                data.data_union.v_obj =
                    ObjectArc::into_raw(Self::into_data(src)) as *mut TVMFFIObject;
            }

            unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
                let ptr = data.data_union.v_obj as *const $Obj;
                // `from_raw` adopts the reference without bumping the count;
                // clone to take our own reference and forget the adopted one
                // (`object::unsafe_::inc_ref` is crate-private in tvm-ffi).
                let adopted = ObjectArc::from_raw(ptr);
                let owned = adopted.clone();
                std::mem::forget(adopted);
                Self::from_data(owned)
            }

            unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
                let ptr = data.data_union.v_obj as *const $Obj;
                let this = Self::from_data(ObjectArc::from_raw(ptr));
                data.type_index = TypeIndex::kTVMFFINone as i32;
                data.data_union.v_int64 = 0;
                this
            }

            unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
                if Self::check_any_strict(data) {
                    Ok(Self::copy_from_any_view_after_check(data))
                } else {
                    Err(())
                }
            }
        }

        tvm_ffi::impl_try_from_any!($Ref);

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
