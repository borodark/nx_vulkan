// build.rs — compile the C++ shim + spirit's Vulkan backend, link Vulkan.
//
// Path-deps spirit at ~/projects/learn_erl/spirit (sibling of
// nx_vulkan/). Future v1.0 vendors the spirit sources or pulls them
// from a release artifact. For now: assume the spirit checkout is
// at SPIRIT_DIR (default = ../../../spirit relative to this crate).

use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // crate is at nx_vulkan/native/nx_vulkan_native/ ; spirit is at
    // nx_vulkan/../spirit/ ; the relative path is three "up" levels.
    let nx_vulkan_root = crate_dir.parent().unwrap().parent().unwrap();
    let default_spirit = nx_vulkan_root.parent().unwrap().join("spirit");
    let spirit = env::var("SPIRIT_DIR")
        .map(PathBuf::from)
        .unwrap_or(default_spirit);

    let shim_c = nx_vulkan_root.join("c_src");

    println!("cargo:rerun-if-changed={}", shim_c.join("nx_vulkan_shim.cpp").display());
    println!("cargo:rerun-if-changed={}", shim_c.join("nx_vulkan_shim.h").display());
    println!("cargo:rerun-if-changed={}", spirit.join("core/src/engine/Backend_par_vulkan.cpp").display());
    println!("cargo:rerun-if-env-changed=SPIRIT_DIR");

    cc::Build::new()
        .cpp(true)
        .std("c++14")
        .flag_if_supported("-Wno-unused-result")
        .flag_if_supported("-Wno-unused-parameter")
        .define("SPIRIT_USE_VULKAN", None)
        .include(shim_c.clone())
        .include(spirit.join("core/include"))
        .file(shim_c.join("nx_vulkan_shim.cpp"))
        .file(spirit.join("core/src/engine/Backend_par_vulkan.cpp"))
        .compile("nx_vulkan_shim");

    // Link Vulkan loader. On Linux this is libvulkan.so; on FreeBSD
    // the same; on macOS, MoltenVK provides libvulkan.dylib.
    println!("cargo:rustc-link-lib=dylib=vulkan");

    // Copy Spirit's pre-compiled SPIR-V shaders into the priv/shaders
    // directory so Elixir-side code can resolve them via
    // :code.priv_dir(:nx_vulkan). On every build (cheap; small files).
    let priv_shaders = nx_vulkan_root.join("priv").join("shaders");
    let _ = std::fs::create_dir_all(&priv_shaders);
    let spirit_shaders = spirit.join("shaders");
    if spirit_shaders.exists() {
        for entry in std::fs::read_dir(&spirit_shaders).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("spv") {
                let dst = priv_shaders.join(path.file_name().unwrap());
                let _ = std::fs::copy(&path, &dst);
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}
