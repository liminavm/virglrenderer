// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! `VIRGLRS_DEBUG`: what the classic renderer prints beyond its errors. A comma-separated list
//! of switches, read once; the C's is `VREND_DEBUG`, and the switches keep its names where they
//! print the same thing.

use std::sync::OnceLock;

/// One switch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Switch {
    /// Every shader as it is created: the TGSI parsed from the guest's text, printed back, and
    /// the GLSL it became -- the C's `shader` switch, in the C's format, so the two logs diff.
    Shader,
}

impl Switch {
    const ALL: &'static [(Switch, &'static str)] = &[(Switch::Shader, "shader")];
}

fn switches() -> &'static [Switch] {
    static SWITCHES: OnceLock<Vec<Switch>> = OnceLock::new();
    SWITCHES.get_or_init(|| {
        let Ok(spec) = std::env::var("VIRGLRS_DEBUG") else {
            return Vec::new();
        };
        spec.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .filter_map(|name| {
                let found = Switch::ALL.iter().find(|(_, n)| *n == name).map(|(s, _)| *s);
                if found.is_none() {
                    eprintln!("[virglrs] VIRGLRS_DEBUG: no switch named {name:?}");
                }
                found
            })
            .collect()
    })
}

/// Whether `switch` was asked for.
pub fn enabled(switch: Switch) -> bool {
    switches().contains(&switch)
}
