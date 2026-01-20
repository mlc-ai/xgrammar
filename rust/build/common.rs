//! Common utilities shared across all platforms

use std::{
    env,
    fs::{self, create_dir_all},
    path::{Path, PathBuf},
    process::Command,
};

use walkdir::WalkDir;

use super::BuildContext;

/// Convert a potentially relative path to an absolute path
pub fn abs_path<P: AsRef<Path>>(p: P) -> PathBuf {
    if p.as_ref().is_absolute() {
        p.as_ref().to_path_buf()
    } else {
        env::current_dir().expect("current_dir failed").join(p)
    }
}

/// Check if a directory looks like the XGrammar repo root
pub fn looks_like_xgrammar_repo_root(dir: &Path) -> bool {
    dir.join("CMakeLists.txt").exists()
        && dir.join("include").exists()
        && dir.join("cpp").exists()
}

/// Check if an environment variable is set to a truthy value
pub fn is_truthy_env(name: &str) -> bool {
    env::var(name)
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Check if cargo is in offline mode
pub fn cargo_offline() -> bool {
    is_truthy_env("CARGO_NET_OFFLINE") || is_truthy_env("XGRAMMAR_RS_OFFLINE")
}

/// Get the path to the submodule cache directory
pub fn submodule_cache_dir(out_dir: &Path) -> PathBuf {
    if let Ok(p) = env::var("XGRAMMAR_RS_CACHE_DIR") {
        return abs_path(p);
    }

    if let Ok(p) = env::var("CARGO_HOME") {
        return abs_path(p).join("xgrammar-rs-cache");
    }

    if let Ok(p) = env::var("HOME") {
        return PathBuf::from(p).join(".cache/xgrammar-rs");
    }
    if let Ok(p) = env::var("LOCALAPPDATA") {
        return PathBuf::from(p).join("xgrammar-rs");
    }

    out_dir.join("xgrammar-rs-cache")
}

/// Run a command and panic if it fails
pub fn run_checked(mut cmd: Command, what: &str) {
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

/// Get the path to the submodules.toml file
pub fn submodules_toml_path(manifest_dir: &Path) -> PathBuf {
    if let Ok(p) = env::var("XGRAMMAR_RS_SUBMODULES_TOML") {
        return abs_path(p);
    }
    manifest_dir.join("rust/submodules.toml")
}

/// Copy a directory recursively with a filter function
pub fn copy_dir_recursive_filtered(
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

/// Find the xgrammar library directory in the build output
pub fn find_xgrammar_lib_dir(root: &Path) -> Option<PathBuf> {
    let static_candidates = ["libxgrammar.a", "xgrammar.lib"];

    WalkDir::new(root)
        .max_depth(6)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .find(|entry| {
            let name = entry.file_name().to_string_lossy();
            static_candidates.iter().any(|c| name == *c)
        })
        .and_then(|entry| entry.path().parent().map(|p| p.to_path_buf()))
}

/// Collect the build context from environment variables
pub fn collect_build_context() -> BuildContext {
    println!("cargo:rerun-if-env-changed=XGRAMMAR_SRC_DIR");
    println!("cargo:rerun-if-env-changed=XGRAMMAR_RS_CACHE_DIR");
    println!("cargo:rerun-if-env-changed=XGRAMMAR_RS_SUBMODULES_TOML");
    println!("cargo:rerun-if-env-changed=XGRAMMAR_RS_OFFLINE");
    println!("cargo:rerun-if-env-changed=CARGO_NET_OFFLINE");
    println!("cargo:rerun-if-env-changed=CARGO_HOME");

    let manifest_dir =
        abs_path(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

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

    let submodules_toml = submodules_toml_path(&manifest_dir);
    println!("cargo:rerun-if-changed={}", submodules_toml.display());

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

    let xgrammar_src_dir = super::submodules::prepare_xgrammar_source_tree(
        &xgrammar_repo_dir,
        &out_dir,
        &submodules_toml,
    );

    let xgrammar_include_dir = xgrammar_src_dir.join("include");
    let dlpack_include_dir = xgrammar_src_dir.join("3rdparty/dlpack/include");
    let picojson_include_dir = xgrammar_src_dir.join("3rdparty/picojson");
    let src_include_dir = manifest_dir.join("rust/src");

    let target = env::var("TARGET").unwrap_or_default();
    let host = env::var("HOST").unwrap_or_default();
    let profile = env::var("PROFILE").unwrap_or_else(|_| "release".into());

    BuildContext {
        manifest_dir,
        xgrammar_src_dir,
        out_dir,
        src_include_dir,
        xgrammar_include_dir,
        dlpack_include_dir,
        picojson_include_dir,
        target,
        host,
        profile,
    }
}
