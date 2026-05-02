// build.rs — compile the C++ shim + vendored Spirit Vulkan backend,
// link Vulkan.
//
// As of 2026-05-02 the Spirit Vulkan backend is vendored under
// nx_vulkan/c_src/spirit/ (see c_src/spirit/VENDOR.md for the
// pinned upstream commit). This build script no longer requires a
// sibling Spirit checkout — hex.pm installs work out of the box.
//
// SPIRIT_DIR is honored as a development override: when set,
// shaders from $SPIRIT_DIR/shaders/ are copied into priv/shaders/
// at build time, letting mac-248's freshly-compiled shaders flow
// in without an explicit vendor refresh. The C++ backend itself is
// always taken from the vendored copy; refresh it explicitly via
// the procedure documented in c_src/spirit/VENDOR.md.

use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // crate is at nx_vulkan/native/nx_vulkan_native/ ; root is two up.
    let nx_vulkan_root = crate_dir.parent().unwrap().parent().unwrap();

    let shim_c = nx_vulkan_root.join("c_src");
    let vendored_spirit = shim_c.join("spirit");

    println!("cargo:rerun-if-changed={}", shim_c.join("nx_vulkan_shim.cpp").display());
    println!("cargo:rerun-if-changed={}", shim_c.join("nx_vulkan_shim.h").display());
    println!(
        "cargo:rerun-if-changed={}",
        vendored_spirit.join("src/engine/Backend_par_vulkan.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        vendored_spirit.join("include/engine/Backend_par_vulkan.hpp").display()
    );
    println!("cargo:rerun-if-env-changed=SPIRIT_DIR");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++14")
        .flag_if_supported("-Wno-unused-result")
        .flag_if_supported("-Wno-unused-parameter")
        .define("SPIRIT_USE_VULKAN", None)
        .include(shim_c.clone())
        .include(vendored_spirit.join("include"))
        .file(shim_c.join("nx_vulkan_shim.cpp"))
        .file(vendored_spirit.join("src/engine/Backend_par_vulkan.cpp"));

    // FreeBSD + macOS install Vulkan headers under /usr/local/include
    // (FreeBSD: pkg install vulkan-headers; macOS: brew install vulkan-headers).
    // Linux's clang searches /usr/include automatically; FreeBSD's
    // does not search /usr/local/include without being asked.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "freebsd" || target_os == "macos" {
        build.include("/usr/local/include");
        println!("cargo:rustc-link-search=native=/usr/local/lib");
    }

    build.compile("nx_vulkan_shim");

    // Link Vulkan loader. On Linux this is libvulkan.so; on FreeBSD
    // the same; on macOS, MoltenVK provides libvulkan.dylib.
    println!("cargo:rustc-link-lib=dylib=vulkan");

    // Shader handling:
    //
    // - Default (hex.pm install, no SPIRIT_DIR): the vendored
    //   priv/shaders/*.spv are already in place; do nothing.
    //
    // - SPIRIT_DIR set (developer with a Spirit checkout, typically
    //   mac-248 iterating on shaders): refresh priv/shaders/*.spv
    //   from $SPIRIT_DIR/shaders/ on every build.
    let priv_shaders = nx_vulkan_root.join("priv").join("shaders");
    let _ = std::fs::create_dir_all(&priv_shaders);

    if let Ok(spirit_dir) = env::var("SPIRIT_DIR") {
        let spirit_shaders = PathBuf::from(spirit_dir).join("shaders");
        if spirit_shaders.exists() {
            // Watch the directory itself so a new .spv appearing triggers a rerun.
            // Per-file rerun-if-changed only fires on modifications to existing
            // files; without the dir watch, fresh shaders need a `cargo clean`.
            println!("cargo:rerun-if-changed={}", spirit_shaders.display());
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
}
