//! Modular build system for xgrammar-rs
//!
//! This module provides platform-specific build configuration and utilities.

// Allow unused items as they're available for future use or platform-specific code
#![allow(dead_code)]

pub mod autocxx;
pub mod cmake;
pub mod common;
pub mod submodules;

#[cfg(target_os = "windows")]
pub mod windows;

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "linux")]
pub mod linux;

use std::path::PathBuf;

/// Build context containing all paths and configuration needed for the build.
#[derive(Debug, Clone)]
pub struct BuildContext {
    pub manifest_dir: PathBuf,
    pub xgrammar_src_dir: PathBuf,
    pub out_dir: PathBuf,

    pub src_include_dir: PathBuf,
    pub xgrammar_include_dir: PathBuf,
    pub dlpack_include_dir: PathBuf,
    pub picojson_include_dir: PathBuf,

    pub target: String,
    pub host: String,
    pub profile: String,
}

impl BuildContext {
    /// Returns true if building for MSVC target
    pub fn is_msvc(&self) -> bool {
        self.target.contains("msvc")
    }

    /// Returns true if building for Windows
    pub fn is_windows(&self) -> bool {
        self.target.contains("windows")
    }

    /// Returns true if building for macOS
    pub fn is_macos(&self) -> bool {
        self.target.contains("apple-darwin")
    }

    /// Returns true if building for iOS
    pub fn is_ios(&self) -> bool {
        self.target.contains("apple-ios")
    }

    /// Returns true if building for Linux
    pub fn is_linux(&self) -> bool {
        self.target.contains("linux")
    }

    /// Returns true if building for ARM64/AArch64
    pub fn is_aarch64(&self) -> bool {
        self.target.contains("aarch64")
    }

    /// Returns true if building for x86_64
    pub fn is_x86_64(&self) -> bool {
        self.target.contains("x86_64")
    }

    /// Returns true if this is a debug build
    pub fn is_debug(&self) -> bool {
        self.profile == "debug"
    }

    /// Returns true if this is a release build
    pub fn is_release(&self) -> bool {
        self.profile == "release"
    }

    /// Get the architecture string for the current target
    pub fn arch(&self) -> &str {
        if self.is_aarch64() {
            "aarch64"
        } else if self.is_x86_64() {
            "x86_64"
        } else if self.target.contains("i686") {
            "i686"
        } else {
            "unknown"
        }
    }
}
