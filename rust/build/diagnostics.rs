use std::env;
use std::process::Command;

pub fn check_requirements() {
    println!("cargo:warning=Build Diagnostics:");
    
    // Check CMake
    if let Ok(output) = Command::new("cmake").arg("--version").output() {
        let version = String::from_utf8_lossy(&output.stdout);
        println!("cargo:warning=  CMake: Found ({})", version.lines().next().unwrap_or("unknown").trim());
    } else {
        println!("cargo:warning=  CMake: NOT FOUND");
    }

    // Check Ninja
    if let Ok(output) = Command::new("ninja").arg("--version").output() {
        let version = String::from_utf8_lossy(&output.stdout);
        println!("cargo:warning=  Ninja: Found ({})", version.trim());
    } else {
        println!("cargo:warning=  Ninja: NOT FOUND (CMake might use MSBuild)");
    }

    // Check Compiler (CC/CXX)
    if let Ok(cc) = env::var("CC") {
        println!("cargo:warning=  CC: {}", cc);
    }
    if let Ok(cxx) = env::var("CXX") {
        println!("cargo:warning=  CXX: {}", cxx);
    }

    // Check CUDA
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!("cargo:warning=  Feature: CUDA enabled");
        if let Ok(output) = Command::new("nvcc").arg("--version").output() {
             let version = String::from_utf8_lossy(&output.stdout);
             // nvcc version output is multi-line
             let version_line = version.lines().find(|l| l.contains("release")).unwrap_or("unknown").trim();
             println!("cargo:warning=  nvcc: Found ({})", version_line);
        } else {
             println!("cargo:warning=  nvcc: NOT FOUND (CMake might fail if not in PATH)");
        }
    }
}
