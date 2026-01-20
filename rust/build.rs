//! Build script for xgrammar-rs
//!
//! This script handles:
//! - Building the XGrammar C++ library via CMake
//! - Generating Rust bindings via autocxx
//! - Platform-specific configuration (Windows ARM64/x64, macOS, Linux)
//!
//! ## Environment Variables
//!
//! - `XGRAMMAR_SRC_DIR`: Override the XGrammar source directory
//! - `XGRAMMAR_RS_CACHE_DIR`: Override the submodule cache directory
//! - `XGRAMMAR_RS_STATIC_CRT`: Use static CRT on Windows (default: true)
//! - `LIBCLANG_PATH`: Override libclang location

#[path = "build/mod.rs"]
mod build;

fn main() {
    // On Windows, check if required tools are available and print setup instructions if needed
    #[cfg(target_os = "windows")]
    {
        build::windows::configure_libclang_early();
        
        // Check if link.exe is on PATH - if not, warn with setup instructions
        if std::process::Command::new("link.exe")
            .arg("/?")
            .output()
            .is_err()
        {
            build::windows::print_path_setup_instructions();
        }
    }

    // Collect build context and prepare source tree
    let ctx = build::common::collect_build_context();

    // Build the C++ library via CMake
    let destination_path = build::cmake::build_xgrammar_cmake(&ctx);

    // Link the static library
    build::cmake::link_xgrammar_static(&ctx, &destination_path);

    // Build autocxx bridge
    build::autocxx::build_autocxx_bridge(&ctx);

    // Copy headers for generated code
    build::autocxx::copy_headers_for_generated_rust_code(&ctx);

    // Format and clean up generated bindings
    build::autocxx::format_generated_bindings_optional(&ctx.out_dir);
    build::autocxx::strip_autocxx_generated_doc_comments(&ctx.out_dir);
}
