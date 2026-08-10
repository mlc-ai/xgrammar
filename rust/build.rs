use std::process::Command;

/// `tvm-ffi-sys` links `libtvm_ffi.so` but only injects the library directory
/// into the compile-time environment, so binaries (including our tests) would
/// need `LD_LIBRARY_PATH` set by hand. Bake an rpath instead so `cargo test`
/// and `cargo run --example` work out of the box.
fn main() {
    let output = match Command::new("tvm-ffi-config").arg("--libdir").output() {
        Ok(output) if output.status.success() => output,
        _ => return, // tvm-ffi-sys will report the actionable error
    };
    let lib_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !lib_dir.is_empty() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
}
