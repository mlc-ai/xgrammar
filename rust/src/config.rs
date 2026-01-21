use autocxx::c_int;

use crate::ffi::xgrammar::{
    GetMaxRecursionDepth as FFIGetMaxRecursionDepth,
    GetSerializationVersion as FFIGetSerializationVersion,
    SetMaxRecursionDepth as FFISetMaxRecursionDepth,
};

/// Get the serialization version number. The current version is "v4".
///
/// Returns
/// -------
/// serialization_version : str
///     The serialization version number.
pub fn get_serialization_version() -> String {
    FFIGetSerializationVersion().to_string()
}

/// Get the maximum allowed recursion depth. The depth is shared per process.
///
/// The maximum recursion depth is determined in the following order:
///
/// 1. Manually set via :py:func:`set_max_recursion_depth`
/// 2. `XGRAMMAR_MAX_RECURSION_DEPTH` environment variable (if set and is a valid integer <= 1,000,000)
/// 3. Default value of 10,000
///
/// Returns
/// -------
/// max_recursion_depth : int
///     The maximum allowed recursion depth.
pub fn get_max_recursion_depth() -> i32 {
    FFIGetMaxRecursionDepth().0
}

/// Set the maximum allowed recursion depth. The depth is shared per process.
/// This method is thread-safe.
///
/// Parameters
/// ----------
/// max_recursion_depth : int
///     The maximum allowed recursion depth.
pub fn set_max_recursion_depth(max_recursion_depth: i32) {
    FFISetMaxRecursionDepth(c_int(max_recursion_depth))
}
