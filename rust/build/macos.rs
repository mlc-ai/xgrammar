//! macOS and iOS specific build configuration

use super::BuildContext;

/// Configure macOS/iOS specific build settings
pub fn configure_macos_build(_ctx: &BuildContext) {
    // macOS typically has Xcode command line tools installed
    // which provides clang/libclang automatically.
    //
    // If needed, we could add detection for:
    // - Xcode installation path
    // - Command Line Tools path
    // - Homebrew LLVM installation
}

/// Get the SDK path for iOS builds
pub fn get_ios_sdk_path(is_simulator: bool) -> Option<String> {
    use std::process::Command;

    let sdk = if is_simulator {
        "iphonesimulator"
    } else {
        "iphoneos"
    };

    let output = Command::new("xcrun")
        .args(["--sdk", sdk, "--show-sdk-path"])
        .output()
        .ok()?;

    if output.status.success() {
        String::from_utf8(output.stdout)
            .ok()
            .map(|s| s.trim().to_string())
    } else {
        None
    }
}
