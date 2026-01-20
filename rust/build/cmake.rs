//! CMake build configuration and execution

use std::{
    env,
    fs::{self, create_dir_all},
    path::{Path, PathBuf},
};

use cmake::Config as CMakeConfig;

use super::common::find_xgrammar_lib_dir;
use super::BuildContext;

/// Clear the CMake build directory if the source directory has changed
fn maybe_clear_cmake_build_dir(build_dir: &Path, source_dir: &Path) {
    let cache = build_dir.join("CMakeCache.txt");
    let Ok(contents) = fs::read_to_string(&cache) else {
        return;
    };
    let src = source_dir
        .canonicalize()
        .unwrap_or_else(|_| source_dir.to_path_buf());
    for line in contents.lines() {
        if !line.starts_with("CMAKE_HOME_DIRECTORY") {
            continue;
        }
        let needs_cleanup = line
            .split('=')
            .next_back()
            .and_then(|cmake_home| fs::canonicalize(cmake_home).ok())
            .map(|cmake_home| cmake_home != src)
            .unwrap_or(true);
        if needs_cleanup {
            let _ = fs::remove_dir_all(build_dir);
        }
        break;
    }
}

/// Get the MSVC runtime library setting based on build profile
/// Uses static CRT by default because dependencies like esaxx-rs use static CRT.
/// NOTE: Build with RUSTFLAGS="-C target-feature=+crt-static" to ensure all native code matches.
fn get_msvc_runtime_library(ctx: &BuildContext) -> &'static str {
    let use_static = env::var("XGRAMMAR_RS_STATIC_CRT")
        .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(true); // Static CRT required to match esaxx-rs and other dependencies

    match (ctx.is_debug(), use_static) {
        (true, true) => "MultiThreadedDebug",      // /MTd
        (true, false) => "MultiThreadedDebugDLL",  // /MDd
        (false, true) => "MultiThreaded",          // /MT
        (false, false) => "MultiThreadedDLL",      // /MD
    }
}

/// Build the XGrammar C++ library using CMake
pub fn build_xgrammar_cmake(ctx: &BuildContext) -> PathBuf {
    let cmake_build_dir = ctx.out_dir.join("build");
    maybe_clear_cmake_build_dir(&cmake_build_dir, &ctx.xgrammar_src_dir);
    create_dir_all(&cmake_build_dir).ok();

    // Determine build profile - sync with Cargo's profile
    let build_profile = if ctx.is_debug() { "Debug" } else { "Release" };

    let config_cmake_path = cmake_build_dir.join("config.cmake");
    std::fs::write(
        &config_cmake_path,
        format!(
            "set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF)\n\
             set(XGRAMMAR_BUILD_CXX_TESTS OFF)\n\
             set(XGRAMMAR_ENABLE_CPPTRACE OFF)\n\
             set(CMAKE_BUILD_TYPE {})\n",
            build_profile
        ),
    )
    .expect("Failed to write config.cmake");

    let mut cmake_config = CMakeConfig::new(&ctx.xgrammar_src_dir);
    cmake_config.out_dir(&ctx.out_dir);
    cmake_config.define("XGRAMMAR_BUILD_PYTHON_BINDINGS", "OFF");
    cmake_config.define("XGRAMMAR_BUILD_CXX_TESTS", "OFF");
    cmake_config.define("XGRAMMAR_ENABLE_CPPTRACE", "OFF");
    cmake_config.define("CMAKE_CXX_STANDARD", "17");
    cmake_config.define("CMAKE_CXX_STANDARD_REQUIRED", "ON");
    cmake_config.define("CMAKE_CXX_EXTENSIONS", "OFF");

    // Disable LTO to avoid linking issues with Rust on some platforms
    cmake_config.define("CMAKE_INTERPROCEDURAL_OPTIMIZATION", "OFF");

    if ctx.is_msvc() {
        let runtime_lib = get_msvc_runtime_library(ctx);
        cmake_config.define("CMAKE_MSVC_RUNTIME_LIBRARY", runtime_lib);
        cmake_config.cxxflag("/EHsc");
    } else {
        cmake_config.cflag("-fno-lto");
        cmake_config.cxxflag("-fno-lto");
    }

    cmake_config.profile(build_profile);

    // Platform-specific configuration
    if ctx.is_macos() {
        let arch = if ctx.is_aarch64() { "arm64" } else { "x86_64" };
        cmake_config.define("CMAKE_OSX_ARCHITECTURES", arch);
    } else if ctx.is_ios() {
        let is_sim = ctx.target.contains("apple-ios-sim") || ctx.target.contains("x86_64-apple-ios");
        let arch = if ctx.is_aarch64() { "arm64" } else { "x86_64" };
        let sysroot = if is_sim { "iphonesimulator" } else { "iphoneos" };
        cmake_config.define("CMAKE_OSX_ARCHITECTURES", arch);
        cmake_config.define("CMAKE_OSX_SYSROOT", sysroot);
        if let Ok(dep_target) = env::var("IPHONEOS_DEPLOYMENT_TARGET") {
            cmake_config.define("CMAKE_OSX_DEPLOYMENT_TARGET", dep_target);
        }
    }

    cmake_config.build_target("xgrammar").build()
}

/// Link the XGrammar static library
pub fn link_xgrammar_static(ctx: &BuildContext, destination_path: &Path) {
    let cmake_build_dir = ctx.out_dir.join("build");
    let lib_search_dir = find_xgrammar_lib_dir(&cmake_build_dir)
        .or_else(|| find_xgrammar_lib_dir(destination_path))
        .unwrap_or_else(|| destination_path.join("lib"));
    println!("cargo:rustc-link-search=native={}", lib_search_dir.display());
    println!("cargo:rustc-link-lib=static=xgrammar");
}
