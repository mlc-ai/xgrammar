//! Global configuration, mirroring the Python `xgrammar` module-level
//! functions.

use tvm_ffi::String as FfiString;

use crate::error::Result;
use crate::ffi::{ffi_call, ret};

/// The maximum recursion depth used by xgrammar's recursive algorithms
/// (grammar compilation, matching, ...).
pub fn get_max_recursion_depth() -> Result<usize> {
    let any = ffi_call!("xgrammar.tvm_ffi_binding.config.get_max_recursion_depth")?;
    Ok(ret::<i64>(any)? as usize)
}

/// Set the maximum recursion depth. The initial value can also be set via the
/// `XGRAMMAR_MAX_RECURSION_DEPTH` environment variable.
pub fn set_max_recursion_depth(depth: usize) -> Result<()> {
    ffi_call!(
        "xgrammar.tvm_ffi_binding.config.set_max_recursion_depth",
        depth as i64,
    )?;
    Ok(())
}

/// The serialization format version used by `serialize_json` /
/// `deserialize_json` across the library.
pub fn get_serialization_version() -> Result<String> {
    let any = ffi_call!("xgrammar.tvm_ffi_binding.config.get_serialization_version")?;
    Ok(ret::<FfiString>(any)?.as_str().to_string())
}

/// RAII guard that temporarily overrides the maximum recursion depth, the
/// analogue of Python's `with xgr.max_recursion_depth(...)`.
///
/// The previous depth is restored when the guard is dropped.
#[must_use = "the depth is restored as soon as the guard is dropped"]
pub struct MaxRecursionDepthGuard {
    saved: usize,
}

/// Temporarily override the maximum recursion depth until the returned guard
/// is dropped.
pub fn max_recursion_depth(depth: usize) -> Result<MaxRecursionDepthGuard> {
    let saved = get_max_recursion_depth()?;
    set_max_recursion_depth(depth)?;
    Ok(MaxRecursionDepthGuard { saved })
}

impl Drop for MaxRecursionDepthGuard {
    fn drop(&mut self) {
        // Best effort: restoring can only fail if the library disappeared.
        let _ = set_max_recursion_depth(self.saved);
    }
}
