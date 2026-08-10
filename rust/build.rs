use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let lib_dir = config_path("--libdir");
    let include_dir = config_path("--includedir");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let library_name = match target_os.as_str() {
        "windows" => "tvm_ffi.dll",
        "macos" => "libtvm_ffi.dylib",
        _ => "libtvm_ffi.so",
    };
    let library_path = lib_dir.join(library_name);

    // tvm-ffi-sys links libtvm_ffi dynamically. A dependency build script's
    // rpath flags do not propagate to the final Cargo binary, so downstream
    // applications otherwise fail in the platform loader before main(). This
    // bundled static shim resolves the small stable-C-ABI surface used by the
    // Rust crate with dlopen/LoadLibrary. Native static-library requirements do
    // propagate through rlibs, so the same mechanism reaches consumer binaries.
    let mut loader = cc::Build::new();
    loader
        .file("src/tvm_ffi_loader.c")
        .include(include_dir)
        .define(
            "TVM_FFI_LIBRARY_PATH",
            Some(c_string_literal(&library_path).as_str()),
        )
        .define(
            "TVM_FFI_LIBRARY_BASENAME",
            Some(c_string_literal(Path::new(library_name)).as_str()),
        );
    if target_os != "windows" {
        // The shim symbols satisfy Rust's references at static-link time, but
        // must not interpose the real runtime while dlopen runs its initializers.
        loader.flag_if_supported("-fvisibility=hidden");
    }
    loader.compile("xgrammar_tvm_ffi_loader");

    if target_os == "macos" {
        build_macos_link_stubs();
    }

    if target_os != "windows" && target_os != "macos" {
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=pthread");
    }
    println!("cargo:rerun-if-changed=src/tvm_ffi_loader.c");
    println!("cargo:rerun-if-changed=src/tvm_ffi_link_stub.c");
}

fn build_macos_link_stubs() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo did not set OUT_DIR"));
    let compiler = cc::Build::new().get_compiler();

    // tvm-ffi-sys emits -ltvm_ffi and -ltvm_ffi_testing. Unlike ELF linkers,
    // ld64 keeps unused dylibs by default, which would recreate the downstream
    // loader failure that the static shim avoids. Put dead-strippable stand-ins
    // first in the native search path; all actual TVM FFI calls are satisfied
    // by the static shim and resolved from the real runtime at execution time.
    for library_name in ["libtvm_ffi.dylib", "libtvm_ffi_testing.dylib"] {
        let output_path = out_dir.join(library_name);
        // The stub must NOT claim the real library's install name: dyld
        // matches dependent load commands against already-loaded images by
        // install name, and newer ld64 versions do not drop the (unused)
        // dead-strippable stub, so a stub named `@rpath/libtvm_ffi.dylib`
        // shadows the real runtime for every dylib dlopen'ed later —
        // `libxgrammar_bindings.dylib` then fails with "Symbol not found".
        // Give the stub a distinct identity; the bindings resolve the real
        // library through the image the static shim already dlopen'ed (and
        // through their own rpath).
        let stub_leaf = format!("xgrammar-link-stub-{library_name}");
        let install_name = format!("@rpath/{stub_leaf}");
        let status = compiler
            .to_command()
            .arg("-dynamiclib")
            .arg("src/tvm_ffi_link_stub.c")
            .arg("-Wl,-mark_dead_strippable_dylib")
            .arg(format!("-Wl,-install_name,{install_name}"))
            .arg("-o")
            .arg(&output_path)
            .status()
            .unwrap_or_else(|err| panic!("failed to build {library_name}: {err}"));
        assert!(status.success(), "failed to build {library_name}: {status}");
        // When the linker does keep the stub, dyld resolves its @rpath
        // reference by the install-name leaf, so the file must also exist
        // under that name (the `libtvm_ffi.dylib` spelling above is what
        // `-ltvm_ffi` finds at link time).
        std::fs::copy(&output_path, out_dir.join(&stub_leaf))
            .unwrap_or_else(|err| panic!("failed to copy {library_name} stub: {err}"));
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
}

fn config_path(option: &str) -> PathBuf {
    let output = Command::new("tvm-ffi-config")
        .arg(option)
        .output()
        .unwrap_or_else(|err| panic!("failed to run tvm-ffi-config {option}: {err}"));
    assert!(
        output.status.success(),
        "tvm-ffi-config {option} failed with status {}",
        output.status
    );
    let value = String::from_utf8(output.stdout)
        .unwrap_or_else(|err| panic!("tvm-ffi-config {option} returned non-UTF-8 output: {err}"));
    let path = PathBuf::from(value.trim());
    assert!(
        !path.as_os_str().is_empty(),
        "tvm-ffi-config {option} returned an empty path"
    );
    path
}

fn c_string_literal(path: &Path) -> String {
    let escaped = path
        .to_string_lossy()
        .replace('\\', "\\\\")
        .replace('"', "\\\"");
    format!("\"{escaped}\"")
}
