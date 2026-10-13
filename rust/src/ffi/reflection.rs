//! The reflection bridge: reach the C++ reflected type methods without any
//! C++-side support.
//!
//! The bindings library registers its API as reflected *type methods*
//! (`refl::ObjectDef`), which live in `TVMFFIGetTypeInfo(index)->methods[]`
//! rather than in the global function table; constructors are stored in the
//! `__ffi_init__` type-attribute column. This module resolves
//! `"<type_key>.<method_name>"` names against those tables, returning the
//! same `Function` objects the Python bindings dispatch to.
//!
//! The lookups depend only on the `TVMFFITypeInfo`/`TVMFFIMethodInfo`/
//! `TVMFFITypeAttrColumn` C ABI structs, which are declared by tvm-ffi-sys at
//! the pinned revision and stable across the linked libtvm_ffi version.

use tvm_ffi::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIGetTypeAttrColumn, TVMFFIGetTypeInfo, TVMFFITypeKeyToIndex,
};
use tvm_ffi::{Any, Function};

use crate::error::{Error, Result};

/// The name constructors are registered under (a type attribute, not a
/// method).
const INIT_METHOD: &str = "__ffi_init__";

/// Whether `type_key` is registered in this process.
pub(crate) fn type_key_registered(type_key: &str) -> bool {
    type_index_of(type_key).is_ok()
}

/// Resolve a `"<type_key>.<method_name>"` name via the reflection tables.
pub(crate) fn reflected_method(name: &str) -> Result<Function> {
    let (type_key, method_name) = name
        .rsplit_once('.')
        .ok_or_else(|| Error::Library(format!("`{name}` is not a \"<type_key>.<method>\" name")))?;
    let type_index = type_index_of(type_key)?;
    if method_name == INIT_METHOD {
        constructor(type_index, type_key)
    } else {
        method(type_index, type_key, method_name)
    }
}

fn type_index_of(type_key: &str) -> Result<i32> {
    // Safety: `key` borrows `type_key`, which outlives the call below.
    let key = unsafe { TVMFFIByteArray::from_str(type_key) };
    let mut index: i32 = -1;
    let ret = unsafe { TVMFFITypeKeyToIndex(&key, &mut index) };
    if ret != 0 {
        return Err(Error::Library(format!(
            "type key `{type_key}` is not registered; the bindings library must be loaded first"
        )));
    }
    Ok(index)
}

fn method(type_index: i32, type_key: &str, method_name: &str) -> Result<Function> {
    unsafe {
        let info = TVMFFIGetTypeInfo(type_index);
        assert!(!info.is_null(), "TVMFFIGetTypeInfo returned null");
        let methods =
            std::slice::from_raw_parts((*info).methods, (*info).num_methods.max(0) as usize);
        for entry in methods {
            if byte_array_bytes(&entry.name) == method_name.as_bytes() {
                return function_from_any(&entry.method).ok_or_else(|| {
                    Error::Library(format!(
                        "reflected entry `{type_key}.{method_name}` is not a function"
                    ))
                });
            }
        }
        Err(Error::Library(format!(
            "type `{type_key}` has no reflected method `{method_name}`"
        )))
    }
}

fn constructor(type_index: i32, type_key: &str) -> Result<Function> {
    unsafe {
        // Safety: `attr_name` borrows a 'static string.
        let attr_name = TVMFFIByteArray::from_str(INIT_METHOD);
        let column = TVMFFIGetTypeAttrColumn(&attr_name);
        let entry = if column.is_null() {
            None
        } else {
            let column = &*column;
            let offset = type_index - column.begin_index;
            (0..column.size)
                .contains(&offset)
                .then(|| &*column.data.add(offset as usize))
        };
        entry
            .and_then(|any| function_from_any(any))
            .ok_or_else(|| Error::Library(format!("type `{type_key}` has no constructor")))
    }
}

/// Borrow the `Function` held by a reflection-table `Any` (bumping its
/// refcount; the table entry itself is immortal once registered).
unsafe fn function_from_any(raw: &TVMFFIAny) -> Option<Function> {
    let tmp = std::mem::ManuallyDrop::new(Any::from_raw_ffi_any(std::ptr::read(raw)));
    tmp.try_as::<Function>()
}

fn byte_array_bytes(array: &TVMFFIByteArray) -> &[u8] {
    unsafe { std::slice::from_raw_parts(array.data, array.size) }
}
