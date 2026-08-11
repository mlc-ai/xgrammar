//! Helpers around the untyped object handles the bindings hold.
//!
//! Every xgrammar object crossing the FFI is held as a plain
//! [`tvm_ffi::ObjectRef`] strong reference — exactly like the Python
//! bindings, whose `XGRObject` wraps an untyped handle too. The Rust side
//! never inspects the objects; the C++ side type-checks every call against
//! the object header, so a wrong handle surfaces as a clean FFI error rather
//! than undefined behavior.

use tvm_ffi::object::ObjectRef;
use tvm_ffi::tvm_ffi_sys::TVMFFIObject;
use tvm_ffi::{Any, Object, ObjectArc, ObjectRefCore};

/// The raw object address, for identity checks and `Debug` formatting.
pub(crate) fn handle_ptr(handle: &ObjectRef) -> *const Object {
    unsafe { ObjectArc::as_raw(<ObjectRef as ObjectRefCore>::data(handle)) }
}

/// Pointer identity, the analogue of Python's `handle.same_as`.
pub(crate) fn same_handle(a: &ObjectRef, b: &ObjectRef) -> bool {
    std::ptr::eq(handle_ptr(a), handle_ptr(b))
}

/// An owned `Any` carrying `handle` under its *runtime* type tag.
///
/// `AnyView::from(&ObjectRef)` tags the value with the static `ffi.Object`
/// index. C++ parameters typed `ObjectRef` re-check the object header, so
/// that tag is fine for direct arguments — but elements of heterogeneous
/// `Array<Any>` arguments are dispatched by tag on the C++ side (e.g. the
/// Grammar-or-source values of `from_lark`), which needs the runtime index.
pub(crate) fn object_arg(handle: &ObjectRef) -> Any {
    unsafe {
        let arc = <ObjectRef as ObjectRefCore>::data(handle);
        let ptr = ObjectArc::as_raw(arc) as *mut TVMFFIObject;
        let mut raw = tvm_ffi::tvm_ffi_sys::TVMFFIAny::new();
        raw.type_index = (*ptr).type_index;
        raw.data_union.v_obj = ptr;
        // The Any owns its own strong reference.
        std::mem::forget(arc.clone());
        Any::from_raw_ffi_any(raw)
    }
}
