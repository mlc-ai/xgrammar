//! Autocxx bridge building and code generation

use std::{
    env,
    fs::{self, copy, create_dir_all},
    path::Path,
    process::Command,
};

use super::BuildContext;

/// Build the autocxx bridge for Rust/C++ interop
pub fn build_autocxx_bridge(ctx: &BuildContext) {
    println!("cargo:rerun-if-changed=rust/src/lib.rs");

    let mut extra_clang_args = vec!["-std=c++17".to_string()];

    // Windows: explicitly set the target to avoid ARM NEON header issues
    if ctx.is_windows() {
        if ctx.is_aarch64() {
            extra_clang_args.push("--target=aarch64-pc-windows-msvc".to_string());
        } else if ctx.is_x86_64() {
            extra_clang_args.push("--target=x86_64-pc-windows-msvc".to_string());
        }
    }

    // iOS Simulator: set correct target triple and sysroot for C++ headers
    if ctx.target.contains("apple-ios-sim") || ctx.target.contains("x86_64-apple-ios") {
        let arch = if ctx.is_aarch64() { "arm64" } else { "x86_64" };
        let version = env::var("IPHONEOS_DEPLOYMENT_TARGET").unwrap_or_else(|_| "17.0".into());
        extra_clang_args.push(format!("--target={}-apple-ios{}-simulator", arch, version));
        if let Ok(sdkroot) = env::var("SDKROOT") {
            extra_clang_args.push(format!("-isysroot{}", sdkroot));
        }
    }

    let extra_clang_args_refs: Vec<&str> = extra_clang_args.iter().map(|s| s.as_str()).collect();

    let mut autocxx_builder = autocxx_build::Builder::new(
        "rust/src/lib.rs",
        [
            &ctx.src_include_dir,
            &ctx.xgrammar_include_dir,
            &ctx.dlpack_include_dir,
            &ctx.picojson_include_dir,
            &ctx.xgrammar_src_dir,
        ],
    )
    .extra_clang_args(&extra_clang_args_refs)
    .build()
    .expect("autocxx build failed");

    autocxx_builder
        .flag_if_supported("-std=c++17")
        .flag_if_supported("/std:c++17")
        .flag_if_supported("/EHsc")
        .include(&ctx.src_include_dir)
        .include(&ctx.xgrammar_include_dir)
        .include(&ctx.dlpack_include_dir)
        .include(&ctx.picojson_include_dir)
        .include(&ctx.xgrammar_src_dir)
        .include(&ctx.manifest_dir);

    if ctx.is_msvc() {
        let use_static = env::var("XGRAMMAR_RS_STATIC_CRT")
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(true);
        autocxx_builder.static_crt(use_static);
    }

    autocxx_builder.compile("xgrammar_rs_bridge");
}

/// Copy headers needed for generated Rust code
pub fn copy_headers_for_generated_rust_code(ctx: &BuildContext) {
    let rs_dir = ctx.out_dir.join("autocxx-build-dir/rs");

    let gen_include_dir = ctx.out_dir.join("autocxx-build-dir/include");
    let _ = copy(
        gen_include_dir.join("autocxxgen_ffi.h"),
        rs_dir.join("autocxxgen_ffi.h"),
    );

    let rs_xgrammar_dir = rs_dir.join("xgrammar");
    create_dir_all(&rs_xgrammar_dir).ok();
    let _ = copy(
        ctx.xgrammar_include_dir.join("xgrammar/xgrammar.h"),
        rs_xgrammar_dir.join("xgrammar.h"),
    );

    let rs_dlpack_dir = rs_dir.join("dlpack");
    create_dir_all(&rs_dlpack_dir).ok();
    let _ = copy(
        ctx.dlpack_include_dir.join("dlpack/dlpack.h"),
        rs_dlpack_dir.join("dlpack.h"),
    );
}

/// Optionally format the generated bindings with rustfmt
pub fn format_generated_bindings_optional(out_dir: &Path) {
    let gen_rs = out_dir.join("autocxx-build-dir/rs/autocxx-ffi-default-gen.rs");
    if gen_rs.exists() {
        match Command::new("rustfmt").arg(&gen_rs).status() {
            Ok(status) => {
                if !status.success() {
                    eprintln!("rustfmt returned non-zero status on {}", gen_rs.display());
                }
            }
            Err(err) => {
                eprintln!("rustfmt not executed: {}", err);
            }
        }
    }
}

/// Strip autocxx-generated doc comments to keep public docs clean
pub fn strip_autocxx_generated_doc_comments(out_dir: &Path) {
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
    let Ok(rd) = fs::read_dir(&rs_dir) else {
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
        if !file_name.starts_with("autocxx-") || !file_name.ends_with("-gen.rs") {
            continue;
        }
        let Ok(contents) = fs::read_to_string(&path) else {
            if debug {
                println!("cargo:warning=docstrip: failed to read {}", path.display());
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
            let _ = fs::write(&path, out);
        }
    }
}
