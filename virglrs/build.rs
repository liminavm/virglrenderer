// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! Run the venus protocol generator into `OUT_DIR`.
//!
//! The generated Rust is never checked in. Generated code in the tree is code someone edits by
//! hand, and the next regeneration eats the edit -- see CLAUDE.md. It costs a python3-with-mako at
//! build time, which the C tree already required of anyone building venus at all.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let generator = manifest.join("venus-gen");
    let protocol = manifest.parent().unwrap().join("subprojects/venus-protocol-1.0");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("venus");

    for dep in ["gen.py", "rustgen.py", "templates"] {
        println!("cargo::rerun-if-changed={}", generator.join(dep).display());
    }
    println!("cargo::rerun-if-changed={}", protocol.join("vkxml.py").display());
    println!("cargo::rerun-if-changed={}", protocol.join("vn_protocol.py").display());
    println!("cargo::rerun-if-changed={}", protocol.join("xmls").display());

    let status = Command::new("python3")
        .arg(generator.join("gen.py"))
        .arg("--outdir")
        .arg(&out)
        .arg("--protocol")
        .arg(&protocol)
        .status()
        .expect("python3 must be on PATH to build the venus protocol");
    assert!(status.success(), "venus-gen failed");

    gl_bindings(&manifest);
    vrend_formats(&manifest);
    link_vulkan_loader();
    link_egl();

    #[cfg(feature = "reply-oracle")]
    reply_oracle(&manifest, &protocol, &out);
}

/// Run the GLES/EGL binding generator into `OUT_DIR/gl`, from the vendored Khronos registries.
fn gl_bindings(manifest: &std::path::Path) {
    let generator = manifest.join("gl-gen");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("gl");
    println!("cargo::rerun-if-changed={}", generator.join("gen.py").display());
    println!("cargo::rerun-if-changed={}", generator.join("registry").display());
    let status = Command::new("python3")
        .arg(generator.join("gen.py"))
        .arg("--outdir")
        .arg(&out)
        .status()
        .expect("python3 must be on PATH to build the GL bindings");
    assert!(status.success(), "gl-gen failed");
}

/// Run the classic renderer's format generator into `OUT_DIR/vrend`.
///
/// The wire numbering is read from `src/virgl_hw.h`, the header the guest's copy is a copy of,
/// and the format descriptions from the gallium `u_format.yaml` the C's own table is generated
/// from -- one copy of each (`vrend-gen/README.md`).
fn vrend_formats(manifest: &std::path::Path) {
    let generator = manifest.join("vrend-gen");
    let tree = manifest.parent().unwrap();
    let virgl_hw = tree.join("src/virgl_hw.h");
    let gallium = tree.join("src/gallium/auxiliary/util");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("vrend");
    for dep in ["gen.py", "gl_formats.py"] {
        println!("cargo::rerun-if-changed={}", generator.join(dep).display());
    }
    println!("cargo::rerun-if-changed={}", virgl_hw.display());
    for dep in ["u_format.yaml", "u_format_parse.py"] {
        println!("cargo::rerun-if-changed={}", gallium.join(dep).display());
    }
    let status = Command::new("python3")
        .arg(generator.join("gen.py"))
        .arg("--outdir")
        .arg(&out)
        .arg("--virgl-hw")
        .arg(&virgl_hw)
        .arg("--gallium")
        .arg(&gallium)
        .status()
        .expect("python3 must be on PATH to build the format tables");
    assert!(status.success(), "vrend-gen failed");
}

/// Link Mesa's libEGL, for the same reason the Vulkan loader is linked rather than dlopened.
///
/// It is the one GL-side library this crate links: every GLES and EGL entry point is resolved
/// through `eglGetProcAddress`, so libGLESv2 is never named. The C reaches the same library
/// through epoxy's bare-soname dlopen, which is what costs the worker its
/// `DYLD_FALLBACK_LIBRARY_PATH` and the entitlement to keep it.
///
/// Found through pkg-config (`egl`), or `EGL_LIB_DIR`, or the zink-on-KosmicKrisp prefix limina
/// itself defaults to (`MESA_PREFIX`), in that order.
fn link_egl() {
    println!("cargo::rerun-if-env-changed=EGL_LIB_DIR");
    println!("cargo::rerun-if-env-changed=MESA_PREFIX");

    let dir = if let Ok(dir) = std::env::var("EGL_LIB_DIR") {
        Some(dir)
    } else {
        let out = Command::new("pkg-config").args(["--libs-only-L", "egl"]).output();
        out.ok().filter(|o| o.status.success()).and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .split_whitespace()
                .find_map(|f| f.strip_prefix("-L").map(str::to_string))
        })
    };
    let dir = dir.unwrap_or_else(|| {
        let prefix = std::env::var("MESA_PREFIX")
            .unwrap_or_else(|_| "/Volumes/mesa-cs/zink-kk-prefix".to_string());
        format!("{prefix}/lib")
    });
    assert!(
        std::path::Path::new(&dir).join("libEGL.dylib").exists(),
        "no libEGL.dylib under {dir}; point EGL_LIB_DIR or MESA_PREFIX at a Mesa prefix"
    );
    println!("cargo::rustc-link-search=native={dir}");
    println!("cargo::rustc-link-lib=dylib=EGL");
}

/// Link the Khronos loader.
///
/// Linked, not dlopened, and the reason is limina's: the worker is codesigned, so the hardened
/// runtime strips `DYLD_*`, and the loader's directory is not on dyld's default search path -- a
/// bare-name dlopen finds nothing and venus enumerates zero GPUs. `src/vulkan.rs` carries the
/// rest of the argument.
///
/// The path comes from pkg-config rather than a constant, because a hardcoded Cellar path is
/// wrong on the next loader upgrade and wrong for anyone who did not install it the same way.
fn link_vulkan_loader() {
    println!("cargo::rerun-if-env-changed=VULKAN_LOADER_LIB_DIR");

    if let Ok(dir) = std::env::var("VULKAN_LOADER_LIB_DIR") {
        println!("cargo::rustc-link-search=native={dir}");
        rpath(&dir);
    } else {
        let out = Command::new("pkg-config")
            .args(["--libs-only-L", "vulkan"])
            .output()
            .expect("pkg-config must be on PATH to find the Vulkan loader");
        assert!(
            out.status.success(),
            "pkg-config found no vulkan; install the loader or set VULKAN_LOADER_LIB_DIR"
        );
        for flag in String::from_utf8_lossy(&out.stdout).split_whitespace() {
            if let Some(dir) = flag.strip_prefix("-L") {
                println!("cargo::rustc-link-search=native={dir}");
                rpath(dir);
            }
        }
    }

    println!("cargo::rustc-link-lib=dylib=vulkan");
}

/// Give the dylib (and the test binaries) an rpath at `dir`.
///
/// zink is not linked to the loader: it dlopens `@rpath/libvulkan.1.dylib`, and `@rpath` is
/// resolved against the images on the load path -- of which this library is one. Without it the
/// classic renderer's GL comes up with no Vulkan under it, and the C's answer to that is the
/// worker's `DYLD_FALLBACK_LIBRARY_PATH`, which this crate's tests do not inherit.
fn rpath(dir: &str) {
    println!("cargo::rustc-link-arg=-Wl,-rpath,{dir}");
}

/// Build venus-protocol's own C renderer encoder for the tests to diff against.
///
/// The same generator, run the way the C tree runs it. Its output is the ground truth for reply
/// encoding precisely because it is not ours: every venus guest in existence decodes it.
///
/// It brings the layout oracle with it. The reply differential is what *needs* the two sides'
/// structs to have one layout -- it hands a Rust pointer to a C encoder -- so the check that they
/// do belongs behind the same switch, where the headers to ask are already on the include path.
#[cfg(feature = "reply-oracle")]
fn reply_oracle(manifest: &std::path::Path, protocol: &std::path::Path, out: &std::path::Path) {
    let c_out = out.join("c");
    std::fs::create_dir_all(&c_out).expect("create the oracle output directory");

    // meson stamps this with a `vcs_tag`; the generator only wants a file to paste at the top of
    // each header, and a build-id in a test artifact is noise.
    let banner = c_out.join("banner");
    std::fs::write(&banner, "/* Generated by venus-protocol for the virglrs reply oracle. */\n")
        .expect("write the oracle banner");

    let status = Command::new("python3")
        .arg(protocol.join("vn_protocol.py"))
        .arg("--outdir")
        .arg(&c_out)
        .arg("--banner")
        .arg(&banner)
        .arg("--renderer")
        .status()
        .expect("python3 must be on PATH to build the reply oracle");
    assert!(status.success(), "vn_protocol.py --renderer failed");

    let oracle = manifest.join("tests/oracle");
    println!("cargo::rerun-if-changed={}", oracle.display());

    cc::Build::new()
        .file(out.join("reply_oracle.c"))
        .file(out.join("layout_oracle.c"))
        .include(&c_out)
        .include(&oracle)
        .include(protocol.join("include"))
        .flag("-std=c11")
        // The generated encoder is written for a compiler that will inline it all away; its
        // unused-parameter and pointer-arith warnings are noise we do not act on.
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-pointer-arith")
        .flag_if_supported("-Wno-missing-field-initializers")
        // From the generated dispatch functions, which the oracle pulls in but never calls.
        .flag_if_supported("-Wno-uninitialized-const-pointer")
        .compile("vn_reply_oracle");
}
