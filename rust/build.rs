use std::{
    env,
    fs::{self, copy, create_dir_all},
    path::{Path, PathBuf},
    process::Command,
};

use cmake::Config as CMakeConfig;
use walkdir::WalkDir;

// ============================================================================
// Submodule pins (needed for crates.io builds where git submodules are not vendored)
// ============================================================================
//
// IMPORTANT:
// - Cargo packaging does NOT include git submodule contents by default.
// - When users install from crates.io, `3rdparty/<submodule>` directories will be missing.
// - Pins are stored in `rust/submodules.toml` (update via
//   `bash scripts/update_rust_submodules.sh`).

// ============================================================================
// Helper Functions
// ============================================================================

fn abs_path<P: AsRef<Path>>(p: P) -> PathBuf {
    if p.as_ref().is_absolute() {
        p.as_ref().to_path_buf()
    } else {
        env::current_dir().expect("current_dir failed").join(p)
    }
}

fn looks_like_xgrammar_repo_root(dir: &Path) -> bool {
    // xgrammar repo root should have at least:
    // - CMakeLists.txt (for cmake::Config::new())
    // - include/ (public headers)
    // - cpp/ (implementation sources)
    dir.join("CMakeLists.txt").exists()
        && dir.join("include").exists()
        && dir.join("cpp").exists()
}

fn is_truthy_env(name: &str) -> bool {
    let Ok(v) = env::var(name) else {
        return false;
    };
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn cargo_offline() -> bool {
    is_truthy_env("CARGO_NET_OFFLINE") || is_truthy_env("XGRAMMAR_RS_OFFLINE")
}

fn submodule_cache_dir(out_dir: &Path) -> PathBuf {
    // User override
    if let Ok(p) = env::var("XGRAMMAR_RS_CACHE_DIR") {
        return abs_path(p);
    }

    // Prefer Cargo home if available (global cache across projects)
    if let Ok(p) = env::var("CARGO_HOME") {
        return abs_path(p).join("xgrammar-rs-cache");
    }

    // OS-specific user cache locations
    if let Ok(p) = env::var("HOME") {
        return PathBuf::from(p).join(".cache/xgrammar-rs");
    }
    if let Ok(p) = env::var("LOCALAPPDATA") {
        return PathBuf::from(p).join("xgrammar-rs");
    }

    // Fallback: per-build cache
    out_dir.join("xgrammar-rs-cache")
}

fn run_checked(mut cmd: Command, what: &str) {
    let output = cmd.output().unwrap_or_else(|e| {
        panic!("Failed to run {}: {}", what, e);
    });
    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "{} failed (exit={:?})\n--- stdout ---\n{}\n--- stderr ---\n{}\n",
            what,
            output.status.code(),
            stdout,
            stderr
        );
    }
}

fn submodules_toml_path(manifest_dir: &Path) -> PathBuf {
    if let Ok(p) = env::var("XGRAMMAR_RS_SUBMODULES_TOML") {
        return abs_path(p);
    }
    manifest_dir.join("rust/submodules.toml")
}

fn read_pinned_submodule(submodules_toml: &Path, name: &str) -> (String, String) {
    let contents = fs::read_to_string(submodules_toml).unwrap_or_else(|e| {
        panic!(
            "Failed to read submodule pins at {}: {}.\n\
             Run `bash scripts/update_rust_submodules.sh` to regenerate it.",
            submodules_toml.display(),
            e
        )
    });

    let mut in_section = false;
    let mut url: Option<String> = None;
    let mut rev: Option<String> = None;

    for raw in contents.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            // Example: [submodules.dlpack]
            let header = &line[1..line.len() - 1];
            in_section = header.trim() == format!("submodules.{}", name);
            continue;
        }
        if !in_section {
            continue;
        }

        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let key = k.trim();
        let mut val = v.trim();
        if val.starts_with('"') && val.ends_with('"') && val.len() >= 2 {
            val = &val[1..val.len() - 1];
        }
        match key {
            "url" => url = Some(val.to_string()),
            "rev" => rev = Some(val.to_string()),
            _ => {}
        }
        if url.is_some() && rev.is_some() {
            break;
        }
    }

    let url = url.unwrap_or_else(|| {
        panic!(
            "Missing `url` for submodule '{}' in {}",
            name,
            submodules_toml.display()
        )
    });
    let rev = rev.unwrap_or_else(|| {
        panic!(
            "Missing `rev` for submodule '{}' in {}",
            name,
            submodules_toml.display()
        )
    });

    (url, rev)
}

fn copy_dir_recursive_filtered(
    src: &Path,
    dst: &Path,
    should_skip: impl Fn(&Path) -> bool,
) {
    for entry in WalkDir::new(src).into_iter().filter_map(Result::ok) {
        let p = entry.path();
        let rel = p.strip_prefix(src).expect("strip_prefix failed");
        if should_skip(rel) {
            continue;
        }
        let out_path = dst.join(rel);
        if entry.file_type().is_dir() {
            create_dir_all(&out_path).ok();
            continue;
        }
        if entry.file_type().is_file() {
            if let Some(parent) = out_path.parent() {
                create_dir_all(parent).ok();
            }
            let _ = fs::copy(p, &out_path);
        }
    }
}

fn ensure_git_checkout_cached(
    name: &str,
    url: &str,
    rev: &str,
    cache_dir: &Path,
) -> PathBuf {
    let checkout_dir = cache_dir.join(format!("{}-{}", name, rev));
    let marker = checkout_dir.join(".xgrammar_rs_fetched");
    if marker.exists() {
        return checkout_dir;
    }

    if checkout_dir.exists() {
        // Best-effort cleanup of partial checkouts.
        let _ = fs::remove_dir_all(&checkout_dir);
    }
    create_dir_all(cache_dir).expect("Failed to create cache dir");

    // Note: We intentionally do a full clone for reliability (the pinned commit might not
    // be fetchable shallowly in all server configs). These repos are small enough and are
    // cached across builds.
    run_checked(
        {
            let mut c = Command::new("git");
            c.arg("clone").arg(url).arg(&checkout_dir);
            c
        },
        &format!("git clone {} into cache", name),
    );
    run_checked(
        {
            let mut c = Command::new("git");
            c.arg("-C").arg(&checkout_dir).arg("checkout").arg(rev);
            c
        },
        &format!("git checkout {}@{}", name, rev),
    );

    // Marker indicates the cache entry is ready to use.
    let _ = fs::write(&marker, rev);
    checkout_dir
}

fn prepare_xgrammar_source_tree(
    xgrammar_repo_dir: &Path,
    out_dir: &Path,
    submodules_toml: &Path,
) -> PathBuf {
    // If submodules are present, build directly from the repo dir.
    let dlpack_header = xgrammar_repo_dir
        .join("3rdparty/dlpack/include/dlpack/dlpack.h");
    if dlpack_header.exists() {
        return xgrammar_repo_dir.to_path_buf();
    }

    // Missing submodule contents (common for crates.io installs) -> fetch into cache and
    // materialize a buildable source tree under OUT_DIR.
    if cargo_offline() {
        panic!(
            "Required git submodule `3rdparty/dlpack` is missing (expected {}). \
             Cargo is in offline mode. Either:\n\
             - build with network access, or\n\
             - build from a git checkout with submodules initialized, or\n\
             - set XGRAMMAR_SRC_DIR to an XGrammar repo root that already has submodules.",
            dlpack_header.display()
        );
    }

    let cache_dir = submodule_cache_dir(out_dir);
    println!(
        "cargo:warning=xgrammar-rs: dlpack submodule missing; fetching into cache at {}",
        cache_dir.display()
    );

    let (dlpack_url, dlpack_rev) =
        read_pinned_submodule(submodules_toml, "dlpack");
    let dlpack_checkout =
        ensure_git_checkout_cached("dlpack", &dlpack_url, &dlpack_rev, &cache_dir);

    let work_dir = out_dir.join("xgrammar-src");
    if work_dir.exists() {
        let _ = fs::remove_dir_all(&work_dir);
    }

    // Copy the minimal set of XGrammar sources needed for the CMake build.
    // This avoids mutating Cargo's registry sources while still satisfying the
    // repo's CMakeLists.txt assumptions about `3rdparty/` layout.
    let to_copy = [
        "CMakeLists.txt",
        "cmake",
        "cpp",
        "include",
        "3rdparty/picojson",
    ];
    for rel in to_copy {
        let src = xgrammar_repo_dir.join(rel);
        let dst = work_dir.join(rel);
        if src.is_dir() {
            copy_dir_recursive_filtered(&src, &dst, |_| false);
        } else if src.is_file() {
            if let Some(parent) = dst.parent() {
                create_dir_all(parent).ok();
            }
            let _ = fs::copy(&src, &dst);
        }
    }

    // Materialize dlpack into the expected submodule path.
    let dlpack_dst = work_dir.join("3rdparty/dlpack");
    copy_dir_recursive_filtered(&dlpack_checkout, &dlpack_dst, |rel| {
        // Don't copy git metadata into the build tree.
        rel.components().any(|c| c.as_os_str() == ".git")
    });

    let dlpack_header_work = work_dir
        .join("3rdparty/dlpack/include/dlpack/dlpack.h");
    if !dlpack_header_work.exists() {
        panic!(
            "Fetched dlpack but the expected header was not found at {}",
            dlpack_header_work.display()
        );
    }

    work_dir
}

fn find_xgrammar_lib_dir(root: &Path) -> Option<PathBuf> {
    let static_candidates = [
        "libxgrammar.a", // Unix/macOS static
        "xgrammar.lib",  // Windows static
    ];

    for entry in
        WalkDir::new(root).max_depth(6).into_iter().filter_map(Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }

        let name = entry.file_name().to_string_lossy();
        if static_candidates.iter().any(|c| name == *c) {
            return entry.path().parent().map(|p| p.to_path_buf());
        }
    }

    None
}

fn strip_autocxx_generated_doc_comments(out_dir: &Path) {
    // The autocxx/bindgen generated Rust file may contain doc comments copied from C/C++
    // headers (e.g. Doxygen `\brief`). These leak into docs.rs and can also trigger
    // rustdoc warnings (broken intra-doc links, invalid HTML tags). We strip all `#[doc = ...]`
    // attributes from generated bindings to keep public docs clean; Rust-side wrappers and
    // re-exports provide their own documentation.
    let debug = env::var("XGRAMMAR_RS_DEBUG_DOCSTRIP").is_ok();
    let rs_dir = out_dir.join("autocxx-build-dir/rs");
    if debug {
        println!("cargo:warning=docstrip: scanning {}", rs_dir.display());
    }
    let Ok(rd) = std::fs::read_dir(&rs_dir) else {
        if debug {
            println!("cargo:warning=docstrip: rs dir missing");
        }
        return;
    };
    let entries: Vec<_> = rd.flatten().collect();
    if debug {
        let mut names: Vec<String> = entries
            .iter()
            .filter_map(|e| e.file_name().into_string().ok())
            .collect();
        names.sort();
        println!("cargo:warning=docstrip: entries={}", names.join(", "));
    }
    for entry in entries {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if !file_name.starts_with("autocxx-") || !file_name.ends_with("-gen.rs")
        {
            continue;
        }
        let Ok(contents) = std::fs::read_to_string(&path) else {
            if debug {
                println!(
                    "cargo:warning=docstrip: failed to read {}",
                    path.display()
                );
            }
            continue;
        };
        if debug {
            let count = contents.matches("#[doc =").count();
            println!(
                "cargo:warning=docstrip: {} contains {} #[doc =] lines",
                file_name, count
            );
        }
        let mut changed = false;
        let mut removed = 0usize;
        let mut out = String::with_capacity(contents.len());
        for line in contents.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("#[doc =") {
                changed = true;
                removed += 1;
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
        if changed {
            if debug {
                println!(
                    "cargo:warning=docstrip: {} removed {} doc lines",
                    file_name, removed
                );
            }
            let _ = std::fs::write(&path, out);
        }
    }
}

#[cfg(target_os = "windows")]
fn find_libclang_windows() -> Option<PathBuf> {
    let vswhere = PathBuf::from(
        r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe",
    );

    let mut candidates: Vec<PathBuf> = Vec::new();

    // 1) Try vswhere to locate VS with LLVM Clang component
    if vswhere.exists() {
        let args = [
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Llvm.Clang",
            "-property",
            "installationPath",
        ];

        if let Ok(out) = Command::new(&vswhere).args(args).output() {
            if out.status.success() {
                let stdout = String::from_utf8_lossy(&out.stdout);
                for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
                    let base = PathBuf::from(line.trim());
                    candidates.push(base.join(r"VC\Tools\Llvm\x64\bin"));
                    candidates.push(base.join(r"VC\Tools\Llvm\bin"));
                }
            }
        }
    }

    // 2) Common fallback locations (VS 2022 editions)
    for edition in ["Community", "Professional", "Enterprise"] {
        candidates.push(PathBuf::from(format!(
            r"C:\Program Files\Microsoft Visual Studio\2022\{}\VC\Tools\Llvm\x64\bin",
            edition
        )));
        candidates.push(PathBuf::from(format!(
            r"C:\Program Files\Microsoft Visual Studio\2022\{}\VC\Tools\Llvm\bin",
            edition
        )));
    }

    // 3) Standalone LLVM installation
    candidates.push(PathBuf::from(r"C:\Program Files\LLVM\bin"));

    // Return the first directory that contains libclang.dll
    for dir in candidates {
        if dir.join("libclang.dll").exists() {
            return Some(dir);
        }
    }

    None
}

#[cfg(not(target_os = "windows"))]
fn find_libclang_windows() -> Option<PathBuf> {
    None
}

// ============================================================================
// Main Build Script
// ============================================================================

fn main() {
    // ========================================================================
    // Step 1: Configure libclang (Windows-specific)
    // ========================================================================
    if env::var("LIBCLANG_PATH").is_err() {
        if cfg!(target_os = "windows") {
            if let Some(dir) = find_libclang_windows() {
                // Make available to this build script and to downstream rustc invocations
                unsafe {
                    env::set_var("LIBCLANG_PATH", &dir);
                }
                println!("cargo:rustc-env=LIBCLANG_PATH={}", dir.display());
            }
        }
    }

    // ========================================================================
    // Step 2: Locate XGrammar source and set up paths
    // ========================================================================

    let manifest_dir = abs_path(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );

    // Re-run build script if these env vars change.
    println!("cargo:rerun-if-env-changed=XGRAMMAR_SRC_DIR");
    println!("cargo:rerun-if-env-changed=XGRAMMAR_RS_CACHE_DIR");
    println!("cargo:rerun-if-env-changed=XGRAMMAR_RS_SUBMODULES_TOML");
    println!("cargo:rerun-if-env-changed=XGRAMMAR_RS_OFFLINE");
    println!("cargo:rerun-if-env-changed=CARGO_NET_OFFLINE");
    println!("cargo:rerun-if-env-changed=CARGO_HOME");

    // Use the official XGrammar sources from this repository by default.
    // Allow overriding explicitly via XGRAMMAR_SRC_DIR for custom setups.
    let xgrammar_repo_dir = if let Ok(p) = env::var("XGRAMMAR_SRC_DIR") {
        abs_path(p)
    } else {
        manifest_dir.clone()
    };
    if !looks_like_xgrammar_repo_root(&xgrammar_repo_dir) {
        panic!(
            "XGrammar source dir does not look like the official repo root: {} \
             (expected CMakeLists.txt + include/ + cpp/). \
             Set XGRAMMAR_SRC_DIR to override.",
            xgrammar_repo_dir.display()
        );
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let submodules_toml = submodules_toml_path(&manifest_dir);
    println!("cargo:rerun-if-changed={}", submodules_toml.display());

    // If git submodules are missing (common for crates.io), materialize a buildable
    // source tree under OUT_DIR with fetched/cached submodules.
    let xgrammar_src_dir = prepare_xgrammar_source_tree(
        &xgrammar_repo_dir,
        &out_dir,
        &submodules_toml,
    );

    let xgrammar_include_dir = xgrammar_src_dir.join("include");
    let dlpack_include_dir = xgrammar_src_dir.join("3rdparty/dlpack/include");
    let picojson_include_dir = xgrammar_src_dir.join("3rdparty/picojson");
    let src_include_dir = manifest_dir.join("rust/src");

    // Track changes in the official source tree (not the OUT_DIR materialization).
    println!(
        "cargo:rerun-if-changed={}",
        xgrammar_repo_dir.join("include").display()
    );
    println!("cargo:rerun-if-changed={}/cpp", xgrammar_repo_dir.display());
    println!(
        "cargo:rerun-if-changed={}/3rdparty",
        xgrammar_repo_dir.display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        xgrammar_repo_dir.join("CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        xgrammar_repo_dir.join(".gitmodules").display()
    );

    // ========================================================================
    // Step 3: Configure and build XGrammar C++ library with CMake
    // ========================================================================
    let cmake_build_dir = out_dir.join("build");
    create_dir_all(&cmake_build_dir).ok();
    let config_cmake_path = cmake_build_dir.join("config.cmake");
    std::fs::write(
        &config_cmake_path,
        "set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF)\n\
         set(XGRAMMAR_BUILD_CXX_TESTS OFF)\n\
         set(XGRAMMAR_ENABLE_CPPTRACE OFF)\n\
         set(CMAKE_BUILD_TYPE RelWithDebInfo)\n",
    )
    .expect("Failed to write config.cmake");

    let mut cmake_config = CMakeConfig::new(&xgrammar_src_dir);
    cmake_config.out_dir(&out_dir);
    cmake_config.define("XGRAMMAR_BUILD_PYTHON_BINDINGS", "OFF");
    cmake_config.define("XGRAMMAR_BUILD_CXX_TESTS", "OFF");
    cmake_config.define("XGRAMMAR_ENABLE_CPPTRACE", "OFF");
    cmake_config.define("CMAKE_CXX_STANDARD", "17");
    cmake_config.define("CMAKE_CXX_STANDARD_REQUIRED", "ON");
    cmake_config.define("CMAKE_CXX_EXTENSIONS", "OFF");

    // Disable LTO to avoid linking issues with Rust on some platforms
    cmake_config.define("CMAKE_INTERPROCEDURAL_OPTIMIZATION", "OFF");

    // Platform-specific compiler flags
    let target = env::var("TARGET").unwrap_or_default();
    let is_msvc = target.contains("msvc");
    if !is_msvc {
        cmake_config.cflag("-fno-lto");
        cmake_config.cxxflag("-fno-lto");
    } else {
        // Ensure correct exception semantics for C++ code generated/used via autocxx/cxx
        cmake_config.cxxflag("/EHsc");
    }

    let build_profile =
        match env::var("PROFILE").unwrap_or_else(|_| "release".into()).as_str()
        {
            "debug" => "Debug",
            "release" => "Release",
            other => {
                eprintln!(
                    "Unknown cargo PROFILE '{}' -> using RelWithDebInfo",
                    other
                );
                "RelWithDebInfo"
            },
        };
    cmake_config.profile(build_profile);

    // Apple platform-specific configuration
    if let Ok(target) = env::var("TARGET") {
        if target.contains("apple-darwin") {
            let arch = if target.contains("aarch64") {
                "arm64"
            } else {
                "x86_64"
            };
            cmake_config.define("CMAKE_OSX_ARCHITECTURES", arch);
        } else if target.contains("apple-ios")
            || target.contains("apple-ios-sim")
        {
            let is_sim = target.contains("apple-ios-sim")
                || target.contains("x86_64-apple-ios");
            let arch = if target.contains("aarch64") {
                "arm64"
            } else {
                "x86_64"
            };
            let sysroot = if is_sim {
                "iphonesimulator"
            } else {
                "iphoneos"
            };
            cmake_config.define("CMAKE_OSX_ARCHITECTURES", arch);
            cmake_config.define("CMAKE_OSX_SYSROOT", sysroot);
            if let Ok(dep_target) = env::var("IPHONEOS_DEPLOYMENT_TARGET") {
                cmake_config.define("CMAKE_OSX_DEPLOYMENT_TARGET", dep_target);
            }
        }
    }

    let destination_path = cmake_config.build_target("xgrammar").build();

    // ========================================================================
    // Step 4: Link the built XGrammar library
    // ========================================================================

    let cmake_build_dir = out_dir.join("build");
    let lib_search_dir = find_xgrammar_lib_dir(&cmake_build_dir)
        .or_else(|| find_xgrammar_lib_dir(&destination_path))
        .unwrap_or_else(|| destination_path.join("lib"));
    println!("cargo:rustc-link-search=native={}", lib_search_dir.display());
    println!("cargo:rustc-link-lib=static=xgrammar");

    // ========================================================================
    // Step 5: Generate and compile Rust/C++ bindings with autocxx
    // ========================================================================

    println!("cargo:rerun-if-changed=rust/src/lib.rs");

    // Prepare extra clang args for autocxx
    let mut extra_clang_args = vec!["-std=c++17".to_string()];

    // Platform-specific clang args for autocxx
    let target = env::var("TARGET").unwrap_or_default();

    // Windows: explicitly set the target to avoid ARM NEON header issues
    if target.contains("windows") {
        if target.contains("aarch64") {
            extra_clang_args
                .push("--target=aarch64-pc-windows-msvc".to_string());
        } else if target.contains("x86_64") {
            extra_clang_args
                .push("--target=x86_64-pc-windows-msvc".to_string());
        }
    }

    // iOS Simulator: set correct target triple and sysroot for C++ headers
    if target.contains("apple-ios-sim") || target.contains("x86_64-apple-ios") {
        let arch = if target.contains("aarch64") {
            "arm64"
        } else {
            "x86_64"
        };
        let version = env::var("IPHONEOS_DEPLOYMENT_TARGET")
            .unwrap_or_else(|_| "17.0".into());
        extra_clang_args
            .push(format!("--target={}-apple-ios{}-simulator", arch, version));
        if let Ok(sdkroot) = env::var("SDKROOT") {
            extra_clang_args.push(format!("-isysroot{}", sdkroot));
        }
    }

    let extra_clang_args_refs: Vec<&str> =
        extra_clang_args.iter().map(|s| s.as_str()).collect();

    // Build the autocxx bridge
    let mut autocxx_builder = autocxx_build::Builder::new(
        "rust/src/lib.rs",
        &[
            &src_include_dir,
            &xgrammar_include_dir,
            &dlpack_include_dir,
            &picojson_include_dir,
            // Allow `#include "cpp/..."` for internal XGrammar headers used in tests/utilities.
            &xgrammar_src_dir,
        ],
    )
    .extra_clang_args(&extra_clang_args_refs) // for libclang parsing
    .build()
    .expect("autocxx build failed");

    autocxx_builder
        .flag_if_supported("-std=c++17")
        .flag_if_supported("/std:c++17")
        .flag_if_supported("/EHsc")
        .include(&src_include_dir)
        .include(&xgrammar_include_dir)
        .include(&dlpack_include_dir)
        .include(&picojson_include_dir)
        .include(&xgrammar_src_dir)
        .include(&manifest_dir);

    autocxx_builder.compile("xgrammar_rs_bridge");

    // ========================================================================
    // Step 6: Copy headers for generated Rust code
    // ========================================================================

    let rs_dir = out_dir.join("autocxx-build-dir/rs");
    // 1) autocxxgen_ffi.h
    let gen_include_dir = out_dir.join("autocxx-build-dir/include");
    let _ = copy(
        gen_include_dir.join("autocxxgen_ffi.h"),
        rs_dir.join("autocxxgen_ffi.h"),
    );
    // 2) xgrammar/xgrammar.h
    let rs_xgrammar_dir = rs_dir.join("xgrammar");
    create_dir_all(&rs_xgrammar_dir).ok();
    let _ = copy(
        xgrammar_include_dir.join("xgrammar/xgrammar.h"),
        rs_xgrammar_dir.join("xgrammar.h"),
    );
    // 3) dlpack/dlpack.h
    let rs_dlpack_dir = rs_dir.join("dlpack");
    create_dir_all(&rs_dlpack_dir).ok();
    let _ = copy(
        dlpack_include_dir.join("dlpack/dlpack.h"),
        rs_dlpack_dir.join("dlpack.h"),
    );

    // ========================================================================
    // Step 7: Format generated bindings (optional)
    // ========================================================================
    let gen_rs =
        out_dir.join("autocxx-build-dir/rs/autocxx-ffi-default-gen.rs");
    if gen_rs.exists() {
        match Command::new("rustfmt").arg(&gen_rs).status() {
            Ok(status) => {
                if !status.success() {
                    eprintln!(
                        "rustfmt returned non-zero status on {}",
                        gen_rs.display()
                    );
                }
            },
            Err(err) => {
                eprintln!("rustfmt not executed: {}", err);
            },
        }
    }

    // Clean up doc comments in generated Rust bindings so docs.rs doesn't show Doxygen markup.
    // Run this at the end to ensure it applies after any generation/formatting steps.
    strip_autocxx_generated_doc_comments(&out_dir);
}
