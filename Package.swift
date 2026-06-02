// swift-tools-version: 6.0
//
// Package.swift — Swift Package Manager support for xgrammar.
//
// Adds the `XGrammar` library product so Swift projects can import xgrammar
// C++ source directly via SwiftPM, without a separate CMake build step.
//
// Usage in a consumer's Package.swift:
//   .package(url: "https://github.com/mlc-ai/xgrammar", from: "0.1.30"),
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
                // 3rdparty non-source
                "3rdparty/cpptrace",
                "3rdparty/googletest",
                "3rdparty/dlpack/apps",
                "3rdparty/dlpack/cmake",
                "3rdparty/dlpack/contrib",
                "3rdparty/dlpack/docs",
                "3rdparty/dlpack/tests",
                "3rdparty/picojson/test_picojson.cpp",
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
