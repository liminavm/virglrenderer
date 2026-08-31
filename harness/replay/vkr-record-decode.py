#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright © 2026 the limina authors
#
# Decode a venus full-stream capture (LIMINA_VKR_RECORD, src/venus/vkr_record.[ch]).
#
# The capture is a prologue plus a stream: one vkr_journal export per context, describing the
# state that context's stream starts from, followed by the wire bytes of every command dispatched
# after that point. Both sections are needed to replay; this tool exists so a capture can be
# checked for plausibility BEFORE a replayer is written against it, which is the only way to tell
# a broken recorder from a broken replayer later.
#
#   vkr-record-decode.py <file>              header, per-context and per-ring summary
#   vkr-record-decode.py <file> --list       every stream record, one per line
#   vkr-record-decode.py <file> --types      command-type histogram
#   vkr-record-decode.py <file> --prologue   the journal entries of each context's prologue
#   vkr-record-decode.py <file> --check      structural checks only; exit 1 on any failure
import collections
import struct
import sys

RECORD_MAGIC = 0x43524B56  # 'VKRC'
JOURNAL_MAGIC = 0x524A4B56  # 'VKJR'

FLAG_TRUNC_FULL = 0x1
FLAG_TRUNC_FATAL = 0x2

# u32 magic, version, flags, ctx_count + u64 record_count, prologue_bytes, stream_bytes
HEADER_SIZE = 40
RECORD_HEADER_SIZE = 32   # u64 seq, ring_id + u32 ctx_id, cmd_type, size, pad
PROLOGUE_HEADER_SIZE = 16  # u32 ctx_id, pad + u64 size
JOURNAL_ENTRY_SIZE = 28    # u64 seq + u32 cmd_type + u8 klass + pad[3] + u64 ring_key + u32 size


def align4(n):
    return (n + 3) & ~3


class Corpus:
    def __init__(self, blob):
        self.blob = blob
        if len(blob) < HEADER_SIZE:
            raise ValueError(f"file is {len(blob)} bytes; a header alone is {HEADER_SIZE}")
        (magic, self.version, self.flags, self.ctx_count, self.record_count,
         self.prologue_bytes, self.stream_bytes) = struct.unpack_from("<IIIIQQQ", blob, 0)
        if magic != RECORD_MAGIC:
            raise ValueError(f"bad magic {magic:#x}, expected {RECORD_MAGIC:#x} ('VKRC')")
        self.prologue_off = HEADER_SIZE
        self.stream_off = HEADER_SIZE + self.prologue_bytes

    @property
    def truncated(self):
        why = []
        if self.flags & FLAG_TRUNC_FULL:
            why.append("hit the cap")
        if self.flags & FLAG_TRUNC_FATAL:
            why.append("a decode went fatal")
        return why

    def prologues(self):
        """(ctx_id, journal_blob) per context."""
        p, end = self.prologue_off, self.prologue_off + self.prologue_bytes
        while p + PROLOGUE_HEADER_SIZE <= end:
            ctx_id, _pad, size = struct.unpack_from("<IIQ", self.blob, p)
            p += PROLOGUE_HEADER_SIZE
            yield ctx_id, self.blob[p:p + size]
            p += align4(size)

    def records(self):
        """(seq, ring_id, ctx_id, cmd_type, payload) in stream order."""
        p, end = self.stream_off, self.stream_off + self.stream_bytes
        while p + RECORD_HEADER_SIZE <= end:
            seq, ring_id, ctx_id, cmd_type, size, _pad = struct.unpack_from("<QQIIII", self.blob, p)
            p += RECORD_HEADER_SIZE
            yield seq, ring_id, ctx_id, cmd_type, self.blob[p:p + size]
            p += align4(size)


def journal_entries(blob):
    """(seq, cmd_type, klass, ring_key, payload) from a 'VKJR' export."""
    if len(blob) < 16:
        return
    magic, _version, count, _reserved = struct.unpack_from("<IIII", blob, 0)
    if magic != JOURNAL_MAGIC:
        raise ValueError(f"prologue is not a journal export (magic {magic:#x})")
    p = 16
    for _ in range(count):
        if p + JOURNAL_ENTRY_SIZE > len(blob):
            raise ValueError("journal export ends mid-entry")
        seq, cmd_type, klass = struct.unpack_from("<QIB", blob, p)
        ring_key, size = struct.unpack_from("<QI", blob, p + 16)
        p += JOURNAL_ENTRY_SIZE
        yield seq, cmd_type, klass, ring_key, blob[p:p + size]
        p += align4(size)


def summarize(c):
    print(f"version {c.version}, {c.record_count} records, {c.ctx_count} context prologues")
    print(f"  prologue {c.prologue_bytes} bytes, stream {c.stream_bytes} bytes")
    if c.truncated:
        print(f"  TRUNCATED: {', '.join(c.truncated)} — this is a valid PREFIX, not a window")

    for ctx_id, blob in c.prologues():
        n = sum(1 for _ in journal_entries(blob))
        print(f"  context {ctx_id}: prologue {len(blob)} bytes, {n} journal entries")

    per_ctx = collections.Counter()
    per_ring = collections.Counter()
    for _seq, ring_id, ctx_id, _cmd, _pay in c.records():
        per_ctx[ctx_id] += 1
        per_ring[(ctx_id, ring_id)] += 1
    for ctx_id, n in sorted(per_ctx.items()):
        print(f"  context {ctx_id}: {n} stream records")
    for (ctx_id, ring_id), n in sorted(per_ring.items()):
        where = "context decoder" if ring_id == 0 else f"ring {ring_id:#x}"
        print(f"    ctx {ctx_id} / {where}: {n}")


def check(c):
    """Structural checks. Each failure is something that makes the corpus unreplayable, so they
    are worth running before a capture is pinned as a fixture."""
    bad = []

    total = HEADER_SIZE + c.prologue_bytes + c.stream_bytes
    if total != len(c.blob):
        bad.append(f"section sizes ({total}) disagree with the file ({len(c.blob)})")

    prologue_ctxs = set()
    n_prologue = 0
    for ctx_id, blob in c.prologues():
        prologue_ctxs.add(ctx_id)
        n_prologue += 1
        try:
            list(journal_entries(blob))
        except ValueError as e:
            bad.append(f"context {ctx_id}: {e}")
    if n_prologue != c.ctx_count:
        bad.append(f"header claims {c.ctx_count} prologues, found {n_prologue}")

    # The stream must be totally ordered by seq: the recorder assigns the sequence number inside
    # the same critical section that appends the bytes precisely so this holds, and a replayer
    # trusts it. A gap or an inversion means the append path lost its lock discipline.
    n = 0
    prev = None
    for seq, _ring, ctx_id, _cmd, payload in c.records():
        n += 1
        # Anchored at 0, not merely consecutive: the recorder's counter starts there, so a
        # stream whose first record is seq 1 lost a record before anything else was written —
        # which a pairwise check alone reads as perfectly contiguous.
        if prev is None and seq != 0:
            bad.append(f"stream starts at seq {seq}, not 0: records were lost at the head")
        elif prev is not None and seq != prev + 1:
            bad.append(f"seq {seq} follows {prev}: the stream is not contiguous")
        prev = seq
        if ctx_id not in prologue_ctxs:
            bad.append(f"seq {seq}: context {ctx_id} has no prologue")
            prologue_ctxs.add(ctx_id)  # report once
        if len(payload) < 4:
            bad.append(f"seq {seq}: payload {len(payload)} bytes cannot hold a command type")
    if n != c.record_count:
        bad.append(f"header claims {c.record_count} records, found {n}")

    for line in bad:
        print(f"FAIL: {line}")
    if not bad:
        print(f"ok: {n} records, {n_prologue} prologues, contiguous and well-formed")
    return not bad


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip() if __doc__ else "usage: vkr-record-decode.py <file> [mode]")
        return 2
    path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else None
    with open(path, "rb") as f:
        c = Corpus(f.read())

    if mode == "--check":
        return 0 if check(c) else 1
    if mode == "--list":
        for seq, ring_id, ctx_id, cmd_type, payload in c.records():
            where = "ctx" if ring_id == 0 else f"ring {ring_id:#x}"
            print(f"{seq:8d} ctx={ctx_id} {where:>20} type={cmd_type:5d} {len(payload):6d} bytes")
        return 0
    if mode == "--types":
        hist = collections.Counter(cmd for _s, _r, _c, cmd, _p in c.records())
        for cmd, n in hist.most_common():
            print(f"{n:8d}  cmd_type {cmd}")
        return 0
    if mode == "--prologue":
        for ctx_id, blob in c.prologues():
            print(f"== context {ctx_id} ({len(blob)} bytes)")
            for seq, cmd_type, klass, ring_key, payload in journal_entries(blob):
                ring = f" ring={ring_key:#x}" if ring_key else ""
                print(f"  {seq:8d} type={cmd_type:5d} klass={klass}{ring} {len(payload):6d} bytes")
        return 0

    summarize(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
