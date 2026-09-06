// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! Reader for a venus capture (`src/venus/vkr_record.h`, format v3) and for the `VKJR` journal
//! exports it carries as per-context prologues.
//!
//! The parser is total: every field is bounds-checked against the buffer, and a malformed corpus
//! yields an error naming the offset rather than a panic or a short read. A capture is a file on
//! disk that a crashing renderer wrote, so it is untrusted input in exactly the sense the project
//! tenets mean.

use std::fmt;

pub const MAGIC: u32 = 0x4352_4b56; // 'VKRC'
pub const JOURNAL_MAGIC: u32 = 0x524a_4b56; // 'VKJR'
pub const VERSION: u32 = 4;

pub const FLAG_TRUNC_FULL: u32 = 0x1;
pub const FLAG_TRUNC_FATAL: u32 = 0x2;

/// A context's identity for the life of a capture. The guest reuses context ids, so the id alone
/// names two unrelated contexts; the generation is what disambiguates them.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct CtxKey {
    pub id: u32,
    pub generation: u32,
}

impl fmt::Display for CtxKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ctx {}/gen {}", self.id, self.generation)
    }
}

/// A control-path event: something the VMM did through the public ABI, which the command stream
/// references but does not contain.
#[derive(Clone, Debug)]
pub enum Ctl {
    CtxCreate {
        ctx_id: u32,
        context_init: u32,
        name: String,
    },
    CtxDestroy {
        ctx_id: u32,
    },
    CreateBlob {
        res_handle: u32,
        ctx_id: u32,
        blob_mem: u32,
        blob_flags: u32,
        blob_id: u64,
        size: u64,
        num_iovs: u32,
    },
    /// Arrived as a host fd from outside the renderer; a replayer cannot reconstruct it.
    ImportBlob {
        res_handle: u32,
        fd_type: u32,
        size: u64,
    },
    AttachResource {
        ctx_id: u32,
        res_handle: u32,
    },
    DetachResource {
        ctx_id: u32,
        res_handle: u32,
    },
    ResourceUnref {
        res_handle: u32,
    },
}

/// One stream record. `Cmd` payloads are venus wire bytes; the replay ABI strips the reply bit in
/// place, so they are handed over as owned, mutable buffers.
#[derive(Clone, Debug)]
pub enum Record {
    Cmd { tick: u64, ctx: CtxKey, ring_id: u64, cmd_type: u32, wire: Vec<u8> },
    Ctl { tick: u64, ctx: CtxKey, event: Ctl },
}

impl Record {
    /// Execution-order stamp. The file is in append order, which is NOT execution order across
    /// threads; see the ordering rule in vkr_record.h.
    pub fn tick(&self) -> u64 {
        match self {
            Record::Cmd { tick, .. } | Record::Ctl { tick, .. } => *tick,
        }
    }
}

impl Record {
    pub fn seq_ctx(&self) -> CtxKey {
        match self {
            Record::Cmd { ctx, .. } | Record::Ctl { ctx, .. } => *ctx,
        }
    }
}

/// One entry of a `VKJR` journal export, as the prologue carries it.
#[derive(Clone, Debug)]
pub struct JournalEntry {
    pub seq: u64,
    pub cmd_type: u32,
    pub klass: u8,
    /// Non-zero means the entry belongs to a ring and must replay through `replay_ring_cmd`.
    pub ring_key: u64,
    pub wire: Vec<u8>,
}

pub struct Prologue {
    pub ctx: CtxKey,
    pub entries: Vec<JournalEntry>,
}

pub struct Corpus {
    pub flags: u32,
    pub prologues: Vec<Prologue>,
    /// Sorted into execution order by `tick`. The file's own order is append order, which
    /// inverts a dependency whenever the guest acts on a command's mid-dispatch reply before the
    /// recording thread reaches the append lock.
    pub records: Vec<Record>,
}

pub type Result<T> = std::result::Result<T, String>;

/// A bounds-checked cursor. Every read that would run past the end names the offset it failed at,
/// because "the corpus is short" is a diagnosis and "index out of range" is not.
struct Cursor<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, at: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.at.checked_add(n).ok_or("offset overflow")?;
        if end > self.buf.len() {
            return Err(format!(
                "want {} bytes at offset {}, only {} remain",
                n,
                self.at,
                self.buf.len().saturating_sub(self.at)
            ));
        }
        let s = &self.buf[self.at..end];
        self.at = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// Payloads are padded to 4; the padding is part of the record, not of the payload.
    fn payload(&mut self, size: usize) -> Result<&'a [u8]> {
        let body = self.take(size)?;
        self.take(((size + 3) & !3) - size)?;
        Ok(body)
    }

    fn done(&self) -> bool {
        self.at >= self.buf.len()
    }
}

fn parse_ctl(op: u32, p: &[u8]) -> Result<Ctl> {
    let mut c = Cursor::new(p);
    Ok(match op {
        1 => {
            let (ctx_id, context_init, nlen) = (c.u32()?, c.u32()?, c.u32()? as usize);
            let _pad = c.u32()?;
            // The ABI's nlen is the caller's buffer, not the string: libkrun passes a fixed
            // 64-byte field. The name is diagnostic only, so trim at the first NUL.
            let raw = c.take(nlen)?;
            let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
            let name = String::from_utf8_lossy(&raw[..end]).into_owned();
            Ctl::CtxCreate { ctx_id, context_init, name }
        }
        2 => Ctl::CtxDestroy { ctx_id: c.u32()? },
        3 => Ctl::CreateBlob {
            res_handle: c.u32()?,
            ctx_id: c.u32()?,
            blob_mem: c.u32()?,
            blob_flags: c.u32()?,
            blob_id: c.u64()?,
            size: c.u64()?,
            num_iovs: c.u32()?,
        },
        4 => Ctl::ImportBlob { res_handle: c.u32()?, fd_type: c.u32()?, size: c.u64()? },
        5 => Ctl::AttachResource { ctx_id: c.u32()?, res_handle: c.u32()? },
        6 => Ctl::DetachResource { ctx_id: c.u32()?, res_handle: c.u32()? },
        7 => Ctl::ResourceUnref { res_handle: c.u32()? },
        _ => return Err(format!("unknown control op {op}")),
    })
}

pub fn parse_journal(blob: &[u8], ctx: CtxKey) -> Result<Vec<JournalEntry>> {
    if blob.is_empty() {
        return Ok(Vec::new());
    }
    let mut c = Cursor::new(blob);
    let magic = c.u32()?;
    if magic != JOURNAL_MAGIC {
        return Err(format!("{ctx}: prologue magic {magic:#x}, expected 'VKJR'"));
    }
    let _version = c.u32()?;
    let count = c.u32()?;
    let _reserved = c.u32()?;

    let mut out = Vec::with_capacity(count as usize);
    for i in 0..count {
        let seq = c.u64()?;
        let cmd_type = c.u32()?;
        let klass = c.take(4)?[0]; // u8 klass + 3 bytes of pad
        let ring_key = c.u64()?;
        let size = c.u32()? as usize;
        let wire = c.payload(size).map_err(|e| format!("{ctx}: journal entry {i}: {e}"))?.to_vec();
        out.push(JournalEntry { seq, cmd_type, klass, ring_key, wire });
    }
    Ok(out)
}

pub fn parse(blob: &[u8]) -> Result<Corpus> {
    let mut c = Cursor::new(blob);
    let magic = c.u32()?;
    if magic != MAGIC {
        return Err(format!("magic {magic:#x}, expected {MAGIC:#x} ('VKRC')"));
    }
    let version = c.u32()?;
    if version != VERSION {
        return Err(format!("format version {version}, this replayer reads {VERSION}"));
    }
    let flags = c.u32()?;
    let ctx_count = c.u32()?;
    let record_count = c.u64()?;
    let prologue_bytes = c.u64()? as usize;
    let stream_bytes = c.u64()? as usize;

    let prologue_buf = c.take(prologue_bytes)?;
    let stream_buf = c.take(stream_bytes)?;
    if !c.done() {
        return Err(format!(
            "{} trailing bytes: the sections disagree with the file",
            blob.len() - c.at
        ));
    }

    let mut prologues = Vec::with_capacity(ctx_count as usize);
    let mut pc = Cursor::new(prologue_buf);
    while !pc.done() {
        let ctx = CtxKey { id: pc.u32()?, generation: pc.u32()? };
        let size = pc.u64()? as usize;
        let entries = parse_journal(pc.payload(size)?, ctx)?;
        prologues.push(Prologue { ctx, entries });
    }
    if prologues.len() != ctx_count as usize {
        return Err(format!(
            "header claims {} prologues, the section holds {}",
            ctx_count,
            prologues.len()
        ));
    }

    let mut records = Vec::with_capacity(record_count as usize);
    let mut sc = Cursor::new(stream_buf);
    let mut expect_seq = 0u64;
    while !sc.done() {
        let seq = sc.u64()?;
        // Anchored at 0 and contiguous. The recorder is a prefix recorder: a gap means records
        // were lost, and a replay of a stream with a hole in it is a replay of something that
        // never ran.
        if seq != expect_seq {
            return Err(format!("record {expect_seq} carries seq {seq}: the stream has a gap"));
        }
        expect_seq += 1;

        let tick = sc.u64()?;
        let ring_id = sc.u64()?;
        let ctx = CtxKey { id: sc.u32()?, generation: sc.u32()? };
        let kind = sc.u32()?;
        let op = sc.u32()?;
        let size = sc.u32()? as usize;
        let _reserved = sc.u32()?;
        let payload = sc.payload(size).map_err(|e| format!("seq {seq}: {e}"))?;

        records.push(match kind {
            0 => Record::Cmd { tick, ctx, ring_id, cmd_type: op, wire: payload.to_vec() },
            1 => Record::Ctl {
                tick,
                ctx,
                event: parse_ctl(op, payload).map_err(|e| format!("seq {seq}: {e}"))?,
            },
            k => return Err(format!("seq {seq}: unknown record kind {k}")),
        });
    }
    if records.len() != record_count as usize {
        return Err(format!(
            "header claims {} records, the stream holds {}",
            record_count,
            records.len()
        ));
    }

    // Into execution order. Stable, so two records that somehow share a tick keep append order
    // rather than swapping arbitrarily -- and a duplicate tick is a recorder bug, so say it.
    records.sort_by_key(Record::tick);
    if let Some(w) = records.windows(2).find(|w| w[0].tick() == w[1].tick()) {
        return Err(format!(
            "two records share tick {}: the execution clock is broken",
            w[0].tick()
        ));
    }

    Ok(Corpus { flags, prologues, records })
}
