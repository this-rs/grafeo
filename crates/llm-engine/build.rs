use std::env;
use std::path::PathBuf;

fn main() {
    let llama_dir = env::var("LLAMA_CPP_DIR")
        .unwrap_or_else(|_| "/Users/triviere/projects/ia/llama.cpp".to_string());
    let llama_path = PathBuf::from(&llama_dir);

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    // LLAMA_STATIC=1 → link .a instead of .dylib/.so/.dll (self-contained binary)
    let link_static = env::var("LLAMA_STATIC").is_ok();
    let link_kind = if link_static { "static" } else { "dylib" };

    // --- Determine build directory ---
    let default_build_subdir = match (target_os.as_str(), target_arch.as_str()) {
        ("windows", "x86_64") => "build-win-x64",
        ("linux", "x86_64") if link_static => "build-linux-x64-static",
        ("linux", "x86_64") => "build-linux-x64",
        _ if link_static => "build-static",
        _ => "build",
    };

    let build_dir = if let Ok(dir) = env::var("LLAMA_BUILD_DIR") {
        let p = PathBuf::from(&dir);
        if p.is_absolute() {
            p
        } else {
            llama_path.join(dir)
        }
    } else {
        llama_path.join(default_build_subdir)
    };

    // Static libs are in ggml/src/ and src/, dynamic in bin/
    if link_static {
        // Flat layout (deps/ directory from Docker builds): .a files at root
        if build_dir.join("libllama.a").exists() {
            println!("cargo:rustc-link-search=native={}", build_dir.display());
        } else {
            // Nested layout (cmake build tree): .a files in src/, ggml/src/, etc.
            for sub in &[
                "src", "ggml/src", "ggml/src/ggml-metal", "ggml/src/ggml-blas",
                "ggml/src/ggml-cpu", "ggml/src/ggml-cuda", "ggml/src/ggml-vulkan",
                "ggml/src/ggml-sycl",
            ] {
                let p = build_dir.join(sub);
                if p.exists() {
                    println!("cargo:rustc-link-search=native={}", p.display());
                }
            }
        }
    } else {
        let lib_dir = if let Ok(dir) = env::var("LLAMA_LIB_DIR") {
            PathBuf::from(dir)
        } else {
            build_dir.join("bin")
        };
        println!("cargo:rustc-link-search=native={}", lib_dir.display());

        // RPATH for runtime dylib resolution (Unix only)
        if target_os != "windows" {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        }
    }

    // --- Core libraries (always needed) ---
    // Link order matters for static: dependents before dependencies
    println!("cargo:rustc-link-lib={}=llama", link_kind);
    println!("cargo:rustc-link-lib={}=ggml", link_kind);
    println!("cargo:rustc-link-lib={}=ggml-cpu", link_kind);
    println!("cargo:rustc-link-lib={}=ggml-base", link_kind);

    // --- Platform-specific backends ---

    // Metal (macOS aarch64 only)
    if target_os == "macos" && target_arch != "x86_64" {
        println!("cargo:rustc-link-lib={}=ggml-metal", link_kind);
    }

    // BLAS (macOS via Accelerate; Linux/Windows opt-in via LLAMA_BLAS=1)
    if target_os == "macos" || env::var("LLAMA_BLAS").is_ok() {
        println!("cargo:rustc-link-lib={}=ggml-blas", link_kind);
    }

    // CUDA (NVIDIA GPU, opt-in via LLAMA_CUDA=1)
    if env::var("LLAMA_CUDA").is_ok() {
        println!("cargo:rustc-link-lib={}=ggml-cuda", link_kind);
        // CUDA runtime libraries
        println!("cargo:rustc-link-lib=dylib=cuda");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cublasLt");
        // Standard CUDA lib paths
        for cuda_path in &["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"] {
            let p = PathBuf::from(cuda_path);
            if p.exists() {
                println!("cargo:rustc-link-search=native={}", p.display());
            }
        }
    }

    // Vulkan (universal GPU, opt-in via LLAMA_VULKAN=1)
    if env::var("LLAMA_VULKAN").is_ok() {
        println!("cargo:rustc-link-lib={}=ggml-vulkan", link_kind);
        println!("cargo:rustc-link-lib=dylib=vulkan");
    }

    // SYCL (Intel GPU via oneAPI, opt-in via LLAMA_SYCL=1)
    if env::var("LLAMA_SYCL").is_ok() {
        println!("cargo:rustc-link-lib={}=ggml-sycl", link_kind);
        println!("cargo:rustc-link-lib=dylib=sycl");
        println!("cargo:rustc-link-lib=dylib=OpenCL");
        println!("cargo:rustc-link-lib=dylib=mkl_core");
        println!("cargo:rustc-link-lib=dylib=mkl_sycl_blas");
        println!("cargo:rustc-link-lib=dylib=mkl_intel_ilp64");
        println!("cargo:rustc-link-lib=dylib=mkl_tbb_thread");
        // Intel compiler runtime: SVML (vectorized math) + irc (fast memcpy/memset)
        // Required when llama.cpp is compiled with icpx (Intel oneAPI C++ compiler)
        println!("cargo:rustc-link-lib=dylib=svml");
        println!("cargo:rustc-link-lib=dylib=irc");
        println!("cargo:rustc-link-lib=dylib=imf");
        // oneDNN (Deep Neural Network Library) — used by ggml-sycl for matmul ops
        println!("cargo:rustc-link-lib=dylib=dnnl");
        // oneAPI lib paths
        for sycl_path in &[
            "/opt/intel/oneapi/compiler/latest/lib",
            "/opt/intel/oneapi/mkl/latest/lib",
            "/opt/intel/oneapi/tbb/latest/lib",
        ] {
            let p = PathBuf::from(sycl_path);
            if p.exists() {
                println!("cargo:rustc-link-search=native={}", p.display());
            }
        }
    }

    // --- System libraries ---

    // macOS frameworks
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        if target_arch != "x86_64" {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
    }

    // C++ standard library
    match target_os.as_str() {
        "macos" => {
            if link_static {
                // Static link: still need dynamic c++ (no static libc++ on macOS)
                println!("cargo:rustc-link-lib=dylib=c++");
            } else {
                println!("cargo:rustc-link-lib=dylib=c++");
            }
        }
        "linux" => {
            let host_os = env::consts::OS;
            if env::var("LLAMA_LIBCXX").is_ok() {
                // llama.cpp compiled with clang -stdlib=libc++ (Docker full builds
                // for CPU/Vulkan/CUDA variants). Symbols are std::__1::* (libc++ ABI).
                println!("cargo:rustc-link-lib=c++");
            } else if host_os == "linux" {
                // Native Linux build with libstdc++ ABI (e.g. SYCL via icpx).
                // Force dynamic stdc++ when LLAMA_STDCPP_DYNAMIC=1 (Docker full builds
                // on Ubuntu 24.04+ where libstdc++.a may not be available)
                let force_dynamic = env::var("LLAMA_STDCPP_DYNAMIC").is_ok();
                if link_static && !force_dynamic {
                    println!("cargo:rustc-link-lib=static=stdc++");
                } else {
                    println!("cargo:rustc-link-lib=stdc++");
                }
            } else {
                // Cross-compile from macOS via zig: llama.cpp was compiled with
                // zig (libc++ ABI = std::__1::*), so link libc++ not stdc++.
                println!("cargo:rustc-link-lib=c++");
            }
        }
        "windows" => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
        _ => {}
    }

    // --- Bindgen ---
    let header = llama_path.join("include/llama.h");
    assert!(
        header.exists(),
        "llama.h not found at {}. Set LLAMA_CPP_DIR to your fork.",
        header.display()
    );

    let mut builder = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .clang_arg(format!("-I{}", llama_path.join("include").display()))
        .clang_arg(format!("-I{}", llama_path.join("ggml/include").display()))
        .allowlist_function("llama_.*")
        .allowlist_function("ggml_.*")
        .allowlist_type("llama_.*")
        .allowlist_type("ggml_.*")
        .allowlist_var("LLAMA_.*")
        .allowlist_var("GGML_.*")
        .rustified_enum("llama_.*")
        .rustified_enum("ggml_.*")
        .layout_tests(false);

    // Cross-compilation: provide system headers for bindgen's clang
    match (target_os.as_str(), target_arch.as_str()) {
        ("windows", "x86_64") => {
            builder = builder.clang_arg("--target=x86_64-pc-windows-gnu");
            // MinGW sysroot
            let sysroot = env::var("MINGW_SYSROOT").unwrap_or_else(|_| {
                std::process::Command::new("x86_64-w64-mingw32-gcc")
                    .arg("--print-sysroot")
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default()
            });
            if !sysroot.is_empty() {
                let inc = PathBuf::from(&sysroot).join("x86_64-w64-mingw32/include");
                if inc.exists() {
                    builder = builder.clang_arg(format!("-isystem{}", inc.display()));
                }
            }
        }
        ("linux", "x86_64") => {
            // Host-mode bindings (LP64 compatible). Provide macOS SDK headers.
            if cfg!(target_os = "macos") {
                if let Some(sdk) = xcrun_sdk_path() {
                    let inc = PathBuf::from(&sdk).join("usr/include");
                    if inc.exists() {
                        builder = builder.clang_arg(format!("-isystem{}", inc.display()));
                    }
                }
            }
        }
        _ => {}
    }

    let bindings = builder
        .generate()
        .expect("Failed to generate bindings for llama.h");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Failed to write bindings.rs");

    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_LIB_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_STATIC");
    println!("cargo:rerun-if-env-changed=LLAMA_BLAS");
    println!("cargo:rerun-if-env-changed=LLAMA_CUDA");
    println!("cargo:rerun-if-env-changed=LLAMA_VULKAN");
    println!("cargo:rerun-if-env-changed=LLAMA_SYCL");
    println!("cargo:rerun-if-env-changed=LLAMA_STDCPP_DYNAMIC");
    println!("cargo:rerun-if-env-changed=LLAMA_LIBCXX");
    println!("cargo:rerun-if-env-changed=MINGW_SYSROOT");
}

fn xcrun_sdk_path() -> Option<String> {
    std::process::Command::new("xcrun")
        .args(["--show-sdk-path"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}
