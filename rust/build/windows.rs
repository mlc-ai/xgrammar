use std::{env, path::PathBuf, process::Command};

#[derive(Debug, Clone)]
pub struct VsInstallation {
    pub path: PathBuf,
    pub version: String,
    pub display_name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowsArch {
    Arm64,
    X64,
    X86,
}

impl WindowsArch {
    pub fn detect_from_env() -> Self {
        let target = env::var("TARGET").unwrap_or_default();
        if target.contains("aarch64") {
            return Self::Arm64;
        }
        if target.contains("x86_64") {
            return Self::X64;
        }
        if target.contains("i686") || target.contains("i586") {
            return Self::X86;
        }

        let host = env::var("HOST").unwrap_or_default();
        if host.contains("aarch64") {
            Self::Arm64
        } else {
            Self::X64
        }
    }

    pub fn llvm_subdir(&self) -> &'static str {
        match self {
            Self::Arm64 => "ARM64",
            Self::X64 => "x64",
            Self::X86 => "x86",
        }
    }

    pub fn msvc_host_dir(&self) -> &'static str {
        match self {
            Self::Arm64 => "Hostarm64",
            Self::X64 => "Hostx64",
            Self::X86 => "Hostx86",
        }
    }

    pub fn vcvars_arg(&self) -> &'static str {
        match self {
            Self::Arm64 => "arm64",
            Self::X64 => "x64",
            Self::X86 => "x86",
        }
    }
}

pub fn find_vs_installations() -> Vec<VsInstallation> {
    let vswhere_paths = [
        PathBuf::from(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"),
        PathBuf::from(r"C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe"),
    ];

    let vswhere = vswhere_paths.iter().find(|p| p.exists());
    let Some(vswhere) = vswhere else {
        return Vec::new();
    };

    // Determine which VC tools component to require based on target architecture
    let arch = WindowsArch::detect_from_env();
    let vc_component = match arch {
        WindowsArch::Arm64 => "Microsoft.VisualStudio.Component.VC.Tools.ARM64",
        WindowsArch::X64 | WindowsArch::X86 => "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
    };

    let args = [
        "-latest",
        "-products",
        "*",
        "-prerelease",
        "-requires",
        vc_component,
        "-format",
        "text",
    ];

    let output = match Command::new(vswhere).args(&args).output() {
        Ok(out) if out.status.success() => out,
        _ => {
            // Fallback: try the other architecture's component
            let fallback_component = match arch {
                WindowsArch::Arm64 => "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                WindowsArch::X64 | WindowsArch::X86 => "Microsoft.VisualStudio.Component.VC.Tools.ARM64",
            };
            let args_fallback_arch = [
                "-latest",
                "-products",
                "*",
                "-prerelease",
                "-requires",
                fallback_component,
                "-format",
                "text",
            ];
            match Command::new(vswhere).args(&args_fallback_arch).output() {
                Ok(out) if out.status.success() => out,
                _ => {
                    // Final fallback: broader search without specific component
                    let args_fallback = ["-latest", "-products", "*", "-prerelease", "-format", "text"];
                    match Command::new(vswhere).args(args_fallback).output() {
                        Ok(out) if out.status.success() => out,
                        _ => return Vec::new(),
                    }
                }
            }
        }
    };


    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_vswhere_output(&stdout)
}

fn parse_vswhere_output(output: &str) -> Vec<VsInstallation> {
    let mut installations = Vec::new();
    let mut current_path: Option<PathBuf> = None;
    let mut current_version: Option<String> = None;
    let mut current_name: Option<String> = None;

    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            if let (Some(path), Some(version), Some(name)) =
                (current_path.take(), current_version.take(), current_name.take())
            {
                installations.push(VsInstallation {
                    path,
                    version,
                    display_name: name,
                });
            }
            continue;
        }

        if let Some((key, value)) = line.split_once(':') {
            let key = key.trim();
            let value = value.trim();
            match key {
                "installationPath" => current_path = Some(PathBuf::from(value)),
                "installationVersion" => current_version = Some(value.to_string()),
                "displayName" => current_name = Some(value.to_string()),
                _ => {}
            }
        }
    }

    if let (Some(path), Some(version), Some(name)) =
        (current_path, current_version, current_name)
    {
        installations.push(VsInstallation {
            path,
            version,
            display_name: name,
        });
    }

    installations
}

pub fn find_ninja_in_vs() -> Option<PathBuf> {
    for vs in find_vs_installations() {
        // Common locations for Ninja in VS
        let candidates = [
            r"Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe",
            r"Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\v1.10\ninja.exe",
        ];
        
        for candidate in candidates {
            let path = vs.path.join(candidate);
            if path.exists() {
                return Some(path);
            }
        }
    }
    None
}

pub fn find_libclang_for_arch(arch: WindowsArch) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    for vs in find_vs_installations() {
        candidates.push(vs.path.join(format!(r"VC\Tools\Llvm\{}\bin", arch.llvm_subdir())));
        candidates.push(vs.path.join(r"VC\Tools\Llvm\bin"));
    }

    let vs_years = ["18", "2026", "2025", "2024", "2023", "2022", "2019"];
    let editions = ["Community", "Professional", "Enterprise", "BuildTools"];

    for year in vs_years {
        for edition in editions {
            let base = PathBuf::from(format!(
                r"C:\Program Files\Microsoft Visual Studio\{}\{}",
                year, edition
            ));
            candidates.push(base.join(format!(r"VC\Tools\Llvm\{}\bin", arch.llvm_subdir())));
            candidates.push(base.join(r"VC\Tools\Llvm\x64\bin"));
            candidates.push(base.join(r"VC\Tools\Llvm\bin"));
        }
    }

    candidates.push(PathBuf::from(r"C:\Program Files\LLVM\bin"));

    candidates
        .into_iter()
        .find(|dir| dir.join("libclang.dll").exists())
}

pub fn configure_libclang_early() {
    let arch = WindowsArch::detect_from_env();
    configure_msvc_environment(arch);

    if env::var("LIBCLANG_PATH").is_ok() {
        return;
    }

    if let Some(libclang_dir) = find_libclang_for_arch(arch) {
        unsafe {
            env::set_var("LIBCLANG_PATH", &libclang_dir);
        }
        println!("cargo:rustc-env=LIBCLANG_PATH={}", libclang_dir.display());
    } else if arch == WindowsArch::Arm64 {
        if let Some(libclang_dir) = find_libclang_for_arch(WindowsArch::X64) {
            unsafe {
                env::set_var("LIBCLANG_PATH", &libclang_dir);
            }
            println!("cargo:rustc-env=LIBCLANG_PATH={}", libclang_dir.display());
        } else {
            print_missing_tools_message(arch);
        }
    } else {
        print_missing_tools_message(arch);
    }
}

fn print_missing_tools_message(arch: WindowsArch) {
    let arch_name = match arch {
        WindowsArch::Arm64 => "ARM64",
        WindowsArch::X64 => "x64",
        WindowsArch::X86 => "x86",
    };

    eprintln!();
    eprintln!("==============================================================================");
    eprintln!("ERROR: Could not find libclang.dll for {} Windows", arch_name);
    eprintln!("==============================================================================");
    eprintln!();
    eprintln!("xgrammar-rs requires LLVM/Clang tools to build. Please install one of:");
    eprintln!();
    eprintln!("  1. Visual Studio with C++ Clang tools:");
    eprintln!("     - Open Visual Studio Installer");
    eprintln!("     - Modify your installation");
    eprintln!("     - Under 'Individual components', install:");
    if arch == WindowsArch::Arm64 {
        eprintln!("       * 'C++ Clang tools for Windows (ARM64)'");
        eprintln!("       * 'MSBuild support for LLVM toolset (clang-cl) for ARM64'");
    } else {
        eprintln!("       * 'C++ Clang tools for Windows'");
        eprintln!("       * 'MSBuild support for LLVM toolset (clang-cl)'");
    }
    eprintln!();
    eprintln!("  2. Standalone LLVM installation:");
    eprintln!("     - Download from https://releases.llvm.org/");
    eprintln!("     - Install to C:\\Program Files\\LLVM");
    eprintln!();
    eprintln!("  3. Set LIBCLANG_PATH manually:");
    eprintln!("     $env:LIBCLANG_PATH = 'C:\\path\\to\\llvm\\bin'");
    eprintln!();
    eprintln!("==============================================================================");
    eprintln!();
}

pub fn print_path_setup_instructions() {
    let arch = WindowsArch::detect_from_env();
    let mut paths_to_add = Vec::new();

    if let Some((_vs_path, msvc_version_dir)) = find_msvc_tools_dir(arch) {
        let bin_dir = msvc_version_dir
            .join("bin")
            .join(arch.msvc_host_dir())
            .join(arch.vcvars_arg());
        if bin_dir.exists() {
            paths_to_add.push(bin_dir);
        }
    }

    if let Some(llvm_path) = find_libclang_for_arch(arch) {
        paths_to_add.push(llvm_path);
    }
    
    if !paths_to_add.is_empty() {
        let paths_str = paths_to_add
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(";");
        
        println!("cargo:warning=");
        println!("cargo:warning=================================================================================");
        println!("cargo:warning=xgrammar-rs: Add these paths to your system PATH (one-time setup):");
        println!("cargo:warning=");
        for path in &paths_to_add {
            println!("cargo:warning=  {}", path.display());
        }
        println!("cargo:warning=");
        println!("cargo:warning=PowerShell command (run once):");
        println!("cargo:warning=  [Environment]::SetEnvironmentVariable(\"PATH\", $env:PATH + \";{}\", \"User\")", paths_str);
        println!("cargo:warning=");
        println!("cargo:warning=Or run from Developer Command Prompt / Developer PowerShell for VS");
        println!("cargo:warning=================================================================================");
        println!("cargo:warning=");
    }
}

fn find_msvc_tools_dir(arch: WindowsArch) -> Option<(PathBuf, PathBuf)> {
    for vs in find_vs_installations() {
        let msvc_tools_dir = vs.path.join(r"VC\Tools\MSVC");
        if !msvc_tools_dir.exists() {
            continue;
        }

        let Ok(entries) = std::fs::read_dir(&msvc_tools_dir) else {
            continue;
        };

        let mut versions: Vec<_> = entries
            .flatten()
            .filter(|e| e.path().is_dir())
            .collect();
        versions.sort_by_key(|b| std::cmp::Reverse(b.file_name()));

        for entry in versions {
            let version_dir = entry.path();
            let bin_dir = version_dir
                .join("bin")
                .join(arch.msvc_host_dir())
                .join(arch.vcvars_arg());
            let cl_path = bin_dir.join("cl.exe");

            if cl_path.exists() {
                return Some((vs.path.clone(), version_dir));
            }
        }
    }
    None
}

fn find_windows_sdk() -> Option<(PathBuf, String)> {
    let sdk_roots = [
        PathBuf::from(r"C:\Program Files (x86)\Windows Kits\10"),
        PathBuf::from(r"C:\Program Files\Windows Kits\10"),
    ];

    for sdk_root in sdk_roots {
        let include_dir = sdk_root.join("Include");
        if !include_dir.exists() {
            continue;
        }

        let Ok(entries) = std::fs::read_dir(&include_dir) else {
            continue;
        };

        let mut versions: Vec<String> = entries
            .flatten()
            .filter(|e| e.path().is_dir())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| n.starts_with("10."))
            .collect();
        versions.sort();
        versions.reverse();

        if let Some(version) = versions.first() {
            return Some((sdk_root, version.clone()));
        }
    }
    None
}

pub fn configure_msvc_environment(arch: WindowsArch) {
    // If INCLUDE is set, we assume the MSVC environment is already set up (e.g. via vcvars)
    // We also check for Ninja in PATH, if not found, we try to add it from VS.
    let has_include = env::var("INCLUDE").is_ok();
    
    // Attempt to add Ninja to PATH if not present
    if std::process::Command::new("ninja").arg("--version").output().is_err() {
        if let Some(ninja_path) = find_ninja_in_vs() {
             if let Some(parent) = ninja_path.parent() {
                 let current_path = env::var("PATH").unwrap_or_default();
                 let new_path = format!("{};{}", parent.display(), current_path);
                 unsafe {
                     env::set_var("PATH", &new_path);
                 }
             }
        }
    }

    if has_include {
        return;
    }

    let Some((_vs_path, msvc_version_dir)) = find_msvc_tools_dir(arch) else {
        return;
    };

    let arch_dir = arch.vcvars_arg();
    let host_arch_dir = arch.msvc_host_dir();

    let bin_dir = msvc_version_dir
        .join("bin")
        .join(host_arch_dir)
        .join(arch_dir);
    
    let cl_path = bin_dir.join("cl.exe");
    if cl_path.exists() {
        unsafe {
            env::set_var("CC", &cl_path);
            env::set_var("CXX", &cl_path);
        }
    }

    let mut path_additions = Vec::new();
    path_additions.push(bin_dir);

    let mut include_paths = Vec::new();
    include_paths.push(msvc_version_dir.join("include"));
    include_paths.push(msvc_version_dir.join("ATLMFC").join("include"));

    let windows_sdk = find_windows_sdk();

    if let Some((sdk_root, sdk_version)) = &windows_sdk {
        let sdk_include = sdk_root.join("Include").join(sdk_version);
        include_paths.push(sdk_include.join("ucrt"));
        include_paths.push(sdk_include.join("shared"));
        include_paths.push(sdk_include.join("um"));
        include_paths.push(sdk_include.join("winrt"));
        include_paths.push(sdk_include.join("cppwinrt"));

        // Add SDK bin to PATH (rc.exe, mt.exe)
        let sdk_bin = sdk_root.join("bin").join(sdk_version).join(arch.vcvars_arg());
        if sdk_bin.exists() {
            path_additions.push(sdk_bin);
        }
    }

    let include_str: String = include_paths
        .iter()
        .filter(|p| p.exists())
        .map(|p| p.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(";");

    if !include_str.is_empty() {
        unsafe {
            env::set_var("INCLUDE", &include_str);
        }
    }

    let mut lib_paths = Vec::new();
    lib_paths.push(msvc_version_dir.join("lib").join(arch_dir));
    lib_paths.push(msvc_version_dir.join("ATLMFC").join("lib").join(arch_dir));

    if let Some((sdk_root, sdk_version)) = &windows_sdk {
        let sdk_lib = sdk_root.join("Lib").join(sdk_version);
        lib_paths.push(sdk_lib.join("ucrt").join(arch_dir));
        lib_paths.push(sdk_lib.join("um").join(arch_dir));
    }

    let lib_str: String = lib_paths
        .iter()
        .filter(|p| p.exists())
        .map(|p| p.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(";");

    if !lib_str.is_empty() {
        unsafe {
            env::set_var("LIB", &lib_str);
        }
    }

    let current_path = env::var("PATH").unwrap_or_default();
    let added_path = path_additions
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(";");
    
    let new_path = if added_path.is_empty() {
        current_path
    } else {
        format!("{};{}", added_path, current_path)
    };
    
    unsafe {
        env::set_var("PATH", &new_path);
    }
}
