// swift-tools-version: 6.0
//
// Package.swift — Swift Package Manager support for xgrammar.
//
// Adds the `XGrammar` library product so Swift projects can import xgrammar
// C++ source directly via SwiftPM, without a separate CMake build step.
//
// Usage in a consumer's Package.swift:
//   .package(url: "https://github.com/mlc-ai/xgrammar", from: "0.2.1"),
//   .product(name: "XGrammar", package: "xgrammar")

import PackageDescription

let package = Package(
    name: "xgrammar",
    platforms: [.macOS("14.0"), .iOS("17.0")],
    products: [
        .library(name: "XGrammar", targets: ["XGrammar"]),
    ],
    targets: [
        .target(
            name: "XGrammar",
            path: ".",
            exclude: [
                // Build configuration files
                "CMakeLists.txt",
                "cpp/CMakeLists.txt",
                // Non-source top-level directories
                "cmake",
                "docs",
                "examples",
                "python",
                "scripts",
                "site",
                "assets",
                // Test files
                "tests",
                // Web bindings
                "web",
                // Python / TVM bindings inside cpp/
                "cpp/tvm_ffi",
                // 3rdparty non-source / header-only dependencies
                "3rdparty/cpptrace",
                "3rdparty/googletest",
                "3rdparty/dlpack",
                "3rdparty/picojson",
            ],
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("cpp"),
                .headerSearchPath("3rdparty/dlpack/include"),
                .headerSearchPath("3rdparty/picojson"),
                .define("XGRAMMAR_ENABLE_LOG_DEBUG", to: "0"),
                .define("XGRAMMAR_ENABLE_CPPTRACE", to: "0"),
            ]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
