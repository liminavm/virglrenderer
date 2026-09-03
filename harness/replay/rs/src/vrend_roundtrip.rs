// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The classic decoder's differential test: frame every recorded batch with the Rust decoder,
//! decode each command, encode it straight back, and compare against the dwords the guest sent.
//!
//! The corpus is a vrend trace dump (`src/vrend/vrend_trace.h`): every command the C dispatched
//! successfully, in order, with the batch boundaries it arrived in. The recorded dwords are
//! mesa's encoder output unmodified, so a dword-identical re-encode is a diff between the two
//! implementations over every command in the corpus -- and because the C recorded only what it
//! accepted, every refusal here is a command the C served and this decoder would not.
//!
//! Failure is per command, not per corpus: one unimplemented shape must not hide the thousand
//! commands behind it.

use std::collections::BTreeMap;
use std::process::ExitCode;

use virglrenderer::vrend::decode::{Batch, Header};
use virglrenderer::vrend::encode::encode;
use virglrenderer::vrend::proto::{Cmd, Refused};

const TRACE_MAGIC: u32 = 0x4c4d_5654;
const T_SUBMIT: u8 = 1;
const T_CMD: u8 = 2;
const REC_HDR: usize = 32;
const RES_EV: usize = 56;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Outcome {
    Ok,
    Refused,
    Mismatch,
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
    match Cmd::from_wire(cmd) {
        Some(c) => c.name().to_string(),
        None => format!("command {cmd}"),
    }
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

/// One recorded batch, as the guest's dwords and the number of commands the C dispatched from it.
struct Recorded {
    words: Vec<u32>,
    commands: usize,
}

/// Walks the dump and groups each context's commands by the submit that carried them.
fn batches(blob: &[u8]) -> Result<Vec<Recorded>, String> {
    if blob.len() < 64 || u32_at(blob, 0) != TRACE_MAGIC {
        return Err("not a vrend trace dump".into());
    }
    let res_n = u32_at(blob, 48) as usize;
    let mut p = 64 + res_n * RES_EV;
    let mut open: BTreeMap<u16, Recorded> = BTreeMap::new();
    let mut done = Vec::new();
    while p + REC_HDR <= blob.len() {
        let total = u32_at(blob, p) as usize;
        if total < REC_HDR || p + total > blob.len() {
            return Err(format!("record at {p} runs past the dump"));
        }
        let kind = blob[p + 4];
        let ctx = u16::from_le_bytes([blob[p + 6], blob[p + 7]]);
        let payload_len = u32_at(blob, p + 24) as usize;
        let aux = u32_at(blob, p + 28) as usize;
        let payload = &blob[p + REC_HDR + aux * 4..p + REC_HDR + aux * 4 + payload_len];
        match kind {
            T_SUBMIT => {
                if let Some(b) = open.remove(&ctx) {
                    done.push(b);
                }
            }
            T_CMD => {
                if !payload_len.is_multiple_of(4) {
                    return Err(format!("record at {p}: a command of {payload_len} bytes"));
                }
                let b =
                    open.entry(ctx).or_insert_with(|| Recorded { words: Vec::new(), commands: 0 });
                b.words.extend(payload.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)));
                b.commands += 1;
            }
            _ => {}
        }
        p += total;
    }
    done.extend(open.into_values());
    Ok(done)
}

fn first_divergence(want: &[u32], got: &[u32]) -> String {
    let at = want.iter().zip(got).position(|(a, b)| a != b).unwrap_or(want.len().min(got.len()));
    let show = |b: &[u32]| match b.get(at) {
        Some(v) => format!("{v:#010x}"),
        None => "end".to_string(),
    };
    format!(
        "dword {at} of {}: guest sent {}, we wrote {} ({} dwords back)",
        want.len(),
        show(want),
        show(got),
        got.len()
    )
}

fn round_trip(rec: &Recorded, tally: &mut Tally) -> usize {
    let mut batch = Batch::new(&rec.words);
    let mut seen = 0;
    loop {
        let from = batch.position();
        let Some(next) = batch.next() else { break };
        seen += 1;
        let header = Header::parse(rec.words[from]);
        match next {
            Ok(cmd) => {
                let want = &rec.words[from..batch.position()];
                let mut got = Vec::with_capacity(want.len());
                encode(&cmd, &mut got);
                if got == want {
                    tally.note(header.cmd, Outcome::Ok, String::new);
                } else {
                    tally.note(header.cmd, Outcome::Mismatch, || first_divergence(want, &got));
                }
            }
            Err(e) => {
                let key = match e {
                    Refused::UnknownCommand { cmd, .. } => cmd,
                    _ => header.cmd,
                };
                tally.note(key, Outcome::Refused, || e.to_string());
            }
        }
    }
    seen
}

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: vrend-roundtrip <vrend.bin>");
        return ExitCode::from(2);
    };
    let blob = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let batches = match batches(&blob) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut tally = Tally::default();
    let mut recorded = 0;
    let mut framed = 0;
    for b in &batches {
        recorded += b.commands;
        framed += round_trip(b, &mut tally);
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

    let ok = totals.get(&Outcome::Ok).copied().unwrap_or(0) as usize;
    println!();
    println!(
        "{ok} of {recorded} recorded commands reproduced the wire exactly, in {} batches",
        batches.len()
    );
    if framed != recorded {
        println!("  framed {framed} commands where the C dispatched {recorded}");
    }
    for (outcome, n) in totals.iter().filter(|(o, _)| **o != Outcome::Ok) {
        println!("  {n} {outcome:?}");
    }
    if ok == recorded && framed == recorded {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
