// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The reply encoder's differential test: encode every recorded command's reply twice -- once with
//! the generated Rust, once with venus-protocol's own generated C -- and compare the bytes.
//!
//! The wire round trip covers the request path for free, because the recorded bytes are mesa's
//! encoder output. No recording contains a reply: both replay entry points call
//! `vkr_replay_strip_reply`, so a replayed stream never asks for one. That leaves the 326
//! per-command reply wrappers with no witness at all, and they are generated -- one template
//! mistake is 326 identical bugs, each of which reaches a guest as plausible garbage rather than
//! as an error.
//!
//! Ground truth is the C renderer encoder because that is what every venus guest in existence
//! decodes. Both sides encode the *same* `vn_command_*`: it is `#[repr(C)]`, so the C reads the
//! memory the Rust decoder filled rather than a second construction of it.
//!
//! What this does not reach is the populated-out path. A recorded command carries the guest's
//! request, so its output members are null or zero and the reply wrappers take their count-query
//! branch. That is the half the corpus can witness; the filled branch needs synthesised arguments.

// The reader is shared with the replayer, which uses more of it than this does.
#[allow(dead_code)]
#[path = "corpus.rs"]
mod corpus;

use std::cell::Cell;
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::process::ExitCode;

use bumpalo::Bump;
use virglrenderer::venus::cs::{AllOfIt, Decoder, Encoder, IdentityObjects};
use virglrenderer::venus::proto::serialize::{vn_command_name, vn_reply_oracle_args};
use virglrenderer::venus::proto::types::{VkCommandTypeEXT, VkFlags};

extern "C" {
    /// venus-protocol's `vn_encode_<command>_reply`, dispatched by command type. C has one return
    /// slot, so the two failures come back as sentinels; nothing above [`CReply::from_raw`] sees
    /// them as numbers.
    fn vn_oracle_reply(cmd: i32, buf: *mut c_void, cap: usize, args: *const c_void) -> usize;
}

/// What the C encoder did, with the sentinels resolved at the boundary they arrive on.
///
/// The two failures pull in opposite directions -- one wants a retry, the other must never be
/// retried -- so leaving them as one `usize::MAX` would make a generator mismatch look like an
/// oversized reply and spin until the size ceiling. The type is what keeps them apart.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CReply {
    Wrote(usize),
    /// The reply did not fit in `cap`. Nothing was written that can be compared.
    Overran,
    /// This encoder has no arm for the command type, though the Rust side had one: the two
    /// generators disagree about the command list, which no buffer size will fix.
    NoArm,
}

impl CReply {
    const OVERRAN: usize = usize::MAX;
    const NO_ARM: usize = usize::MAX - 1;

    fn from_raw(raw: usize) -> Self {
        match raw {
            Self::OVERRAN => CReply::Overran,
            Self::NO_ARM => CReply::NoArm,
            n => CReply::Wrote(n),
        }
    }
}

/// Why one command failed. Ordered by how much it tells you: a decode that never ran says nothing
/// about the reply, and a mismatch is the only outcome where both encoders ran and disagreed.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Outcome {
    Ok,
    Unknown,
    Poisoned,
    NoArm,
    Mismatch,
    WrongSize,
}

#[derive(Default)]
struct Tally {
    counts: BTreeMap<(u32, Outcome), u64>,
    detail: BTreeMap<u32, String>,
}

impl Tally {
    fn note(&mut self, cmd: u32, outcome: Outcome, detail: impl FnOnce() -> String) {
        *self.counts.entry((cmd, outcome)).or_default() += 1;
        if outcome != Outcome::Ok {
            self.detail.entry(cmd).or_insert_with(detail);
        }
    }
}

fn name(cmd: u32) -> String {
    match vn_command_name(VkCommandTypeEXT(cmd as i32)) {
        Some(n) => n.to_string(),
        None => format!("command type {cmd}"),
    }
}

/// The first offset at which the two differ, and what each holds there.
fn first_divergence(want: &[u8], got: &[u8]) -> String {
    let at = want.iter().zip(got).position(|(a, b)| a != b).unwrap_or(want.len().min(got.len()));
    let show = |b: &[u8]| match b.get(at) {
        Some(v) => format!("{v:#04x}"),
        None => "end of buffer".to_string(),
    };
    format!(
        "byte {at} of {}: C wrote {}, we wrote {} ({} bytes back)",
        want.len(),
        show(want),
        show(got),
        got.len()
    )
}

/// How much bigger than the request a reply is assumed to be. A reply is usually smaller, but a
/// count query answered with an array is not bounded by the question that asked for it --
/// `vkEnumerateDeviceExtensionProperties` answers 8 words with a hundred 256-byte structs -- so an
/// overrun retries at a larger size rather than being reported as a mismatch.
const SLACK: usize = 4096;

/// The point at which a reply that still does not fit is a bug rather than a big answer.
const MAX_SLACK: usize = 64 << 20;

fn compare(wire: &[u8], tally: &mut Tally) {
    let mut slack = SLACK;
    while !compare_within(wire, slack, tally) {
        slack *= 8;
        assert!(slack <= MAX_SLACK, "a reply larger than {MAX_SLACK} bytes is not a reply");
    }
}

/// Encode the reply into `wire.len() + slack` bytes both ways. `false` means one of the two ran out
/// of room and nothing was recorded, so the caller should retry with more.
///
/// Both buffers start zeroed, so the padding a scalar sits in agrees without a tolerance: the Rust
/// encoder zeroes its padding deliberately and the C leaves whatever the buffer held.
fn compare_within(wire: &[u8], slack: usize, tally: &mut Tally) -> bool {
    let temp = Bump::new();
    let hard = Cell::new(false);
    let mut dec = Decoder::new(wire, &temp, &IdentityObjects, &hard);
    let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
    let _flags = dec.decode_scalar::<VkFlags>();
    let key = cmd.0 as u32;

    let cap = wire.len() + slack;
    let mut ours = vec![0u8; cap];
    let mut theirs = vec![0u8; cap];
    let mut reply = CReply::Overran;

    let mut enc = Encoder::new(&mut ours, &AllOfIt);
    let size = {
        let theirs = &mut theirs;
        let reply = &mut reply;
        // SAFETY: `args` points at the `vn_command_*` the decoder just filled, which is
        // `#[repr(C)]` and lives until this call returns; `theirs` is `cap` bytes we own and the
        // C writes no further than it is told.
        let mut also = |args: *const c_void| unsafe {
            *reply = CReply::from_raw(vn_oracle_reply(
                cmd.0,
                theirs.as_mut_ptr() as *mut c_void,
                cap,
                args,
            ));
        };
        let Some(size) = vn_reply_oracle_args(&mut dec, &mut enc, cmd, &mut also) else {
            tally.note(key, Outcome::Unknown, || "this protocol has no such command".into());
            return true;
        };
        size
    };
    if dec.fatal() {
        tally.note(key, Outcome::Poisoned, || {
            format!("decode poisoned the stream {} bytes into {}", dec.pos(), wire.len())
        });
        return true;
    }

    let written = match reply {
        CReply::Wrote(n) if !enc.fatal() => n,
        // Either encoder running out of room says nothing about the other; retry both at a size
        // that fits rather than diffing a truncated reply against a whole one.
        CReply::Wrote(_) | CReply::Overran => return false,
        CReply::NoArm => {
            tally.note(key, Outcome::NoArm, || {
                "the Rust generator has this command and venus-protocol's does not".into()
            });
            return true;
        }
    };

    let got = enc.written();
    if got != &theirs[..written] {
        tally.note(key, Outcome::Mismatch, || first_divergence(&theirs[..written], got));
        return true;
    }
    if size != written {
        tally.note(key, Outcome::WrongSize, || {
            format!("sizeof said {size} bytes, the encoders wrote {written}")
        });
        return true;
    }
    tally.note(key, Outcome::Ok, String::new);
    true
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: venus-reply-oracle <corpus.vkrc>");
        return ExitCode::from(2);
    };

    let blob = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{path}: {e}");
            return ExitCode::from(2);
        }
    };
    let corpus = match corpus::parse(&blob) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{path}: {e}");
            return ExitCode::from(2);
        }
    };

    let mut tally = Tally::default();
    // The prologue journals hold the commands that built the state the stream then uses, so they
    // are the same wire bytes and belong in the same comparison.
    for p in &corpus.prologues {
        for e in &p.entries {
            compare(&e.wire, &mut tally);
        }
    }
    for r in &corpus.records {
        if let corpus::Record::Cmd { wire, .. } = r {
            compare(wire, &mut tally);
        }
    }

    let mut ok = 0u64;
    let mut bad = 0u64;
    for ((cmd, outcome), n) in &tally.counts {
        if *outcome == Outcome::Ok {
            ok += n;
        } else {
            bad += n;
            println!("FAIL {:>8}  {:<52} {:?}", n, name(*cmd), outcome);
            if let Some(d) = tally.detail.get(cmd) {
                println!("            {d}");
            }
        }
    }
    println!("{ok} replies matched, {bad} did not");
    if bad == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
