//! Git submodule handling for fetching dependencies

use std::{
    fs::{self, create_dir_all},
    path::{Path, PathBuf},
    process::Command,
};

use super::common::{
    cargo_offline, copy_dir_recursive_filtered, run_checked, submodule_cache_dir,
};

/// Read pinned submodule information from submodules.toml
pub fn read_pinned_submodule(submodules_toml: &Path, name: &str) -> (String, String) {
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

/// Ensure a git checkout is cached locally
pub fn ensure_git_checkout_cached(
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

    let _ = fs::write(&marker, rev);
    checkout_dir
}

/// Prepare the XGrammar source tree, fetching submodules if necessary
pub fn prepare_xgrammar_source_tree(
    xgrammar_repo_dir: &Path,
    out_dir: &Path,
    submodules_toml: &Path,
) -> PathBuf {
    let dlpack_header = xgrammar_repo_dir.join("3rdparty/dlpack/include/dlpack/dlpack.h");
    if dlpack_header.exists() {
        return xgrammar_repo_dir.to_path_buf();
    }

    // crates.io sources may be missing submodules; fetch pinned dlpack into a cache and
    // materialize a buildable tree under OUT_DIR.
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

    let (dlpack_url, dlpack_rev) = read_pinned_submodule(submodules_toml, "dlpack");
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

    let dlpack_dst = work_dir.join("3rdparty/dlpack");
    copy_dir_recursive_filtered(&dlpack_checkout, &dlpack_dst, |rel| {
        rel.components().any(|c| c.as_os_str() == ".git")
    });

    let dlpack_header_work = work_dir.join("3rdparty/dlpack/include/dlpack/dlpack.h");
    if !dlpack_header_work.exists() {
        panic!(
            "Fetched dlpack but the expected header was not found at {}",
            dlpack_header_work.display()
        );
    }

    work_dir
}
