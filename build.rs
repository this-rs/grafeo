use std::env;
use std::path::PathBuf;

fn main() {
    // Our fork of llama.cpp with llama_set_attn_mask
    let llama_dir = env::var("LLAMA_CPP_DIR")
        .unwrap_or_else(|_| "/Users/triviere/projects/ia/llama.cpp".to_string());
    let llama_path = PathBuf::from(&llama_dir);

    // Cross-compile aware: use CARGO_CFG_TARGET_* instead of #[cfg(...)]
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    // --- Link search paths ---
    let build_subdir = env::var("LLAMA_BUILD_DIR").unwrap_or_else(|_| "build".to_string());
    let lib_dir = llama_path.join(&build_subdir).join("bin");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Also check build/src for some cmake configurations
    let lib_dir_src = llama_path.join(&build_subdir).join("src");
    if lib_dir_src.exists() {
        println!("cargo:rustc-link-search=native={}", lib_dir_src.display());
    }

    // --- Link libraries ---
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    println!("cargo:rustc-link-lib=dylib=ggml-cpu");

    // ggml-metal only on aarch64 macOS (Metal not available on x86_64 macOS)
    if target_os == "macos" && target_arch != "x86_64" {
        println!("cargo:rustc-link-lib=dylib=ggml-metal");
    }

    if target_os != "windows" {
        println!("cargo:rustc-link-lib=dylib=ggml-blas");
    }

    // macOS frameworks
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        // Metal frameworks only when Metal backend is linked (not x86_64)
        if target_arch != "x86_64" {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
    }

    // C++ standard library
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if target_os == "linux" {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // --- RPATH for runtime dylib resolution ---
    if target_os != "windows" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }

    // --- Bindgen ---
    let header = llama_path.join("include/llama.h");
    assert!(
        header.exists(),
        "llama.h not found at {}. Set LLAMA_CPP_DIR to your fork.",
        header.display()
    );

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .clang_arg(format!("-I{}", llama_path.join("include").display()))
        .clang_arg(format!("-I{}", llama_path.join("ggml/include").display()))
        // Only generate bindings for llama_ and ggml_ prefixed symbols
        .allowlist_function("llama_.*")
        .allowlist_function("ggml_.*")
        .allowlist_type("llama_.*")
        .allowlist_type("ggml_.*")
        .allowlist_var("LLAMA_.*")
        .allowlist_var("GGML_.*")
        // Generate proper enums
        .rustified_enum("llama_.*")
        .rustified_enum("ggml_.*")
        // Size/align tests can fail across platforms, skip them
        .layout_tests(false)
        .generate()
        .expect("Failed to generate bindings for llama.h");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Failed to write bindings.rs");

    // Rebuild if header changes
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_BUILD_DIR");
}
