//! Common utility functions for the xgrammar crate.

use std::ffi::c_char;

/// Convert a byte slice pointer to c_char pointer.
/// On platforms where c_char is u8 (e.g., ARM64 Linux), this is a direct cast.
#[cfg(not(any(
    all(target_os = "windows", target_arch = "x86_64"),
    all(target_os = "windows", target_arch = "x86"),
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "linux", target_arch = "x86"),
    all(target_os = "macos", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
)))]
#[inline]
pub fn bytes_as_c_char_ptr(bytes: &[u8]) -> *const c_char {
    bytes.as_ptr() as *const c_char
}

/// Convert a byte slice pointer to c_char pointer.
/// On platforms where c_char is i8 (e.g., x86_64, macOS), this requires a cast.
#[cfg(any(
    all(target_os = "windows", target_arch = "x86_64"),
    all(target_os = "windows", target_arch = "x86"),
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "linux", target_arch = "x86"),
    all(target_os = "macos", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
#[inline]
pub fn bytes_as_c_char_ptr(bytes: &[u8]) -> *const c_char {
    bytes.as_ptr() as *const i8
}
