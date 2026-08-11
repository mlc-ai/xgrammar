//! Helpers around the untyped object handles the bindings hold.
//!
//! Every xgrammar object crossing the FFI is held as a plain
//! [`tvm_ffi::object::ObjectRef`] strong reference — exactly like the Python
//! bindings, whose `XGRObject` wraps an untyped handle too. The Rust side
//! never inspects the objects; the C++ side type-checks every call against
//! the object header, so a wrong handle surfaces as a clean FFI error rather
//! than undefined behavior.

use tvm_ffi::object::ObjectRef;
use tvm_ffi::{Object, ObjectArc, ObjectRefCore};

/// The raw object address, for identity checks and `Debug` formatting.
pub(crate) fn handle_ptr(handle: &ObjectRef) -> *const Object {
    unsafe { ObjectArc::as_raw(<ObjectRef as ObjectRefCore>::data(handle)) }
}

/// Pointer identity, the analogue of Python's `handle.same_as`.
pub(crate) fn same_handle(a: &ObjectRef, b: &ObjectRef) -> bool {
    std::ptr::eq(handle_ptr(a), handle_ptr(b))
}
