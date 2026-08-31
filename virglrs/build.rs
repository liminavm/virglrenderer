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
}
