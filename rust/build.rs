use std::env;
use std::process::Command;

fn main() {
    // Every lookup below shells out to tools on PATH (tvm-ffi-config, python
    // with xgrammar). Editor tooling often runs cargo with a different
    // environment and would cache an empty result; rerunning on PATH changes
    // lets the next real build self-heal.
    println!("cargo:rerun-if-env-changed=PATH");

    // tvm-ffi-sys links libtvm_ffi dynamically but leaves runtime lookup to
    // the consumer. Bake an rpath to the tvm-ffi installation so binaries
    // built in this environment (tests, examples, downstream builds on the
    // same machine) start without loader configuration. Relocated deployments
    // point the platform loader path (e.g. LD_LIBRARY_PATH) at the
    // `tvm-ffi-config --libdir` directory instead.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if let Some(lib_dir) = tvm_ffi_libdir() {
        // Re-emit the link directives tvm-ffi-sys is responsible for: its
        // build script caches an *empty* output when it ever runs without
        // `tvm-ffi-config` on PATH under a rustdoc-ish environment (editor
        // tooling), and never reruns on environment changes. Duplicates are
        // harmless.
        println!("cargo:rustc-link-search=native={lib_dir}");
        println!("cargo:rustc-link-lib=dylib=tvm_ffi");
        if target_os != "windows" {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
        }
    }
    emit_python_package_dir();
}

fn tvm_ffi_libdir() -> Option<String> {
    let output = Command::new("tvm-ffi-config")
        .arg("--libdir")
        .output()
        .ok()?;
    if !output.status.success() {
        return None; // tvm-ffi-sys reports the actionable error
    }
    let dir = String::from_utf8(output.stdout).ok()?;
    let dir = dir.trim();
    (!dir.is_empty()).then(|| dir.to_string())
}

/// Locate an installed `xgrammar` Python package (a build-time snapshot). The
/// bindings library it bundles becomes one of the runtime load candidates, so
/// a `pip install xgrammar` is enough for consumer binaries to start without
/// any loading code.
fn emit_python_package_dir() {
    let package_dir = ["python3", "python"].iter().find_map(|python| {
        let output = Command::new(python)
            .args([
                "-c",
                "import xgrammar, os, sys; \
                 sys.stdout.write(os.path.dirname(os.path.abspath(xgrammar.__file__)))",
            ])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let dir = String::from_utf8(output.stdout).ok()?;
        let dir = dir.trim();
        (!dir.is_empty()).then(|| dir.to_string())
    });
    println!(
        "cargo:rustc-env=XGRAMMAR_PYTHON_PACKAGE_DIR={}",
        package_dir.unwrap_or_default()
    );
}
