// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The venus decoder's differential test: decode every recorded command with the Rust decoder,
//! encode it straight back, and compare against the bytes the guest sent.
//!
//! There is no C dump to diff against because there does not need to be one. The recorded wire IS
//! mesa's encoder output, unmodified, header included -- so a byte-identical re-encode is a diff
//! between the two implementations over every command in the corpus, for free. What it does not
//! cover is reply encoding, which no recording contains.
//!
//! Failure is per command, not per corpus: a stream that cannot be followed still reports what the
//! rest of it did, because a single unimplemented shape must not hide the hundred commands behind
//! it.

// The reader is shared with the replayer, which uses more of it than this does.
#[allow(dead_code)]
#[path = "corpus.rs"]
mod corpus;

use std::cell::Cell;
use std::collections::BTreeMap;
use std::process::ExitCode;

use bumpalo::Bump;
use virglrenderer::venus::cs::{AllOfIt, Decoder, Encoder, IdentityObjects};
use virglrenderer::venus::proto::serialize::{vn_command_name, vn_round_trip_args};
use virglrenderer::venus::proto::types::{VkCommandTypeEXT, VkFlags};

/// Why one command failed. Ordered by how much it tells you: an unknown command means the protocol
/// does not have it at all, a poisoned decode means a stub or a real rejection, and a mismatch
/// means both sides ran and disagreed -- the only outcome that needs a diff.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Outcome {
    Ok,
    Unknown,
    Poisoned,
    Short,
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
        "byte {at} of {}: guest sent {}, we wrote {} ({} bytes back)",
        want.len(),
        show(want),
        show(got),
        got.len()
    )
}

fn round_trip(wire: &[u8], tally: &mut Tally) {
    let temp = Bump::new();
    let hard = Cell::new(false);
    let mut dec = Decoder::new(wire, &temp, &IdentityObjects, &hard);
    let cmd = dec.decode_scalar::<VkCommandTypeEXT>();
    let flags = dec.decode_scalar::<VkFlags>();
    let key = cmd.0 as u32;

    let mut buf = vec![0u8; wire.len() + 4096];
    let mut enc = Encoder::new(&mut buf, &AllOfIt);
    enc.record_padding();
    let Some(size) = vn_round_trip_args(&mut dec, &mut enc, cmd, flags) else {
        tally.note(key, Outcome::Unknown, || "this protocol has no such command".into());
        return;
    };
    if dec.fatal() {
        tally.note(key, Outcome::Poisoned, || {
            format!("decode poisoned the stream {} bytes into {}", dec.pos(), wire.len())
        });
        return;
    }
    if dec.pos() != wire.len() {
        tally.note(key, Outcome::Short, || {
            format!("decode consumed {} of {} bytes", dec.pos(), wire.len())
        });
        return;
    }
    // Padding is the one place a faithful reproduction cannot match: mesa's encoder leaves those
    // bytes uninitialised, and this one zeroes them on purpose. Compare everything else.
    let mut got = enc.written().to_vec();
    for span in enc.padding() {
        if span.end <= got.len() && span.end <= wire.len() {
            got[span.clone()].copy_from_slice(&wire[span.clone()]);
        }
    }
    if got != wire {
        tally.note(key, Outcome::Mismatch, || first_divergence(wire, &got));
        return;
    }
    if size != wire.len() {
        tally.note(key, Outcome::WrongSize, || {
            format!("sizeof said {size} bytes, the command is {}", wire.len())
        });
        return;
    }
    tally.note(key, Outcome::Ok, String::new);
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: venus-roundtrip <corpus.vkrc>");
        return ExitCode::from(2);
    };

    let blob = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let parsed = match corpus::parse(&blob) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut tally = Tally::default();
    // The prologue journals hold the commands that built the state the stream then uses, so they
    // are the same wire bytes and belong in the same comparison.
    for p in &parsed.prologues {
        for e in &p.entries {
            round_trip(&e.wire, &mut tally);
        }
    }
    for r in &parsed.records {
        if let corpus::Record::Cmd { wire, .. } = r {
            round_trip(wire, &mut tally);
        }
    }

    let mut totals: BTreeMap<Outcome, u64> = BTreeMap::new();
    let mut failing: BTreeMap<u32, BTreeMap<Outcome, u64>> = BTreeMap::new();
    for (&(cmd, outcome), &n) in &tally.counts {
        *totals.entry(outcome).or_default() += n;
        if outcome != Outcome::Ok {
            *failing.entry(cmd).or_default().entry(outcome).or_default() += n;
        }
    }

    for (cmd, by_outcome) in &failing {
        for (outcome, n) in by_outcome {
            println!("{:<12} {:>7}  {}", format!("{outcome:?}"), n, name(*cmd));
        }
        println!("             {}", tally.detail[cmd]);
    }

    let commands: u64 = totals.values().sum();
    let ok = totals.get(&Outcome::Ok).copied().unwrap_or(0);
    println!();
    println!("{ok} of {commands} commands reproduced the wire exactly");
    for (outcome, n) in totals.iter().filter(|(o, _)| **o != Outcome::Ok) {
        println!("  {n} {outcome:?}");
    }
    if ok == commands { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
