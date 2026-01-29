//! Linux-specific build configuration

use super::BuildContext;

/// Configure Linux-specific build settings
pub fn configure_linux_build(_ctx: &BuildContext) {
    // Linux typically has libclang available through:
    // - System package manager (apt install libclang-dev, dnf install clang-devel, etc.)
    // - LLVM installation from llvm.org
    //
    // The libclang-sys crate handles most of the detection automatically on Linux.
    // If needed, users can set LIBCLANG_PATH manually.
}
