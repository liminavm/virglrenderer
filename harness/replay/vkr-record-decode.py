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
#
# A stream carries two kinds of record: venus wire commands, and the host control-path events
# (blob creates, attaches, unrefs) that build the resources those commands name. Both share one
# sequence, and the interleaving is the dependency order -- see vkr_record.h.
import collections
import struct
import sys

RECORD_MAGIC = 0x43524B56  # 'VKRC'
JOURNAL_MAGIC = 0x524A4B56  # 'VKJR'

FLAG_TRUNC_FULL = 0x1
FLAG_TRUNC_FATAL = 0x2

# u32 magic, version, flags, ctx_count + u64 record_count, prologue_bytes, stream_bytes
HEADER_SIZE = 40
# u64 seq, ring_id + u32 ctx_id, generation, kind, op, size, reserved
RECORD_HEADER_SIZE = 40
PROLOGUE_HEADER_SIZE = 16  # u32 ctx_id, generation + u64 size
SUPPORTED_VERSION = 3

KIND_CMD = 0
KIND_CTL = 1

CTL_NAMES = {
    1: "ctx_create",
    2: "ctx_destroy",
    3: "create_blob",
    4: "import_blob",
    5: "attach_resource",
    6: "detach_resource",
    7: "resource_unref",
}


def ctl_describe(op, payload):
    """Render a control record's payload per the layouts in vkr_record.h."""
    name = CTL_NAMES.get(op, f"ctl-{op}")
    try:
        if op == 1:
            ctx_id, init, nlen, _pad = struct.unpack_from("<IIII", payload, 0)
            # The ABI's nlen is the caller's buffer, not the string: libkrun passes a
            # fixed 64-byte field. Trim at the first NUL for display.
            raw = payload[16:16 + nlen].split(b"\x00", 1)[0]
            who = raw.decode("utf-8", "replace")
            return f"{name} ctx={ctx_id} init={init:#x} name={who!r}"
        if op == 2:
            return f"{name} ctx={struct.unpack_from('<I', payload, 0)[0]}"
        if op == 3:
            res, ctx_id, mem, flags, blob_id, size, niov, _p = struct.unpack_from(
                "<IIIIQQII", payload, 0)
            return (f"{name} res={res} ctx={ctx_id} mem={mem} flags={flags:#x} "
                    f"blob_id={blob_id} size={size} iovs={niov}")
        if op == 4:
            res, fd_type, size = struct.unpack_from("<IIQ", payload, 0)
            return f"{name} res={res} fd_type={fd_type} size={size}"
        if op in (5, 6):
            ctx_id, res = struct.unpack_from("<II", payload, 0)
            return f"{name} ctx={ctx_id} res={res}"
        if op == 7:
            return f"{name} res={struct.unpack_from('<I', payload, 0)[0]}"
    except struct.error:
        return f"{name} TRUNCATED ({len(payload)} bytes)"
    return f"{name} ({len(payload)} bytes)"
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
        if self.version != SUPPORTED_VERSION:
            raise ValueError(f"format version {self.version}, this tool reads {SUPPORTED_VERSION}")
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
        """(ctx_id, generation, journal_blob) per context."""
        p, end = self.prologue_off, self.prologue_off + self.prologue_bytes
        while p + PROLOGUE_HEADER_SIZE <= end:
            ctx_id, generation, size = struct.unpack_from("<IIQ", self.blob, p)
            p += PROLOGUE_HEADER_SIZE
            yield ctx_id, generation, self.blob[p:p + size]
            p += align4(size)

    def records(self):
        """(seq, ring_id, ctx_id, generation, kind, op, payload) in stream order."""
        p, end = self.stream_off, self.stream_off + self.stream_bytes
        while p + RECORD_HEADER_SIZE <= end:
            seq, ring_id, ctx_id, gen, kind, op, size, _rsv = struct.unpack_from(
                "<QQIIIIII", self.blob, p)
            p += RECORD_HEADER_SIZE
            yield seq, ring_id, ctx_id, gen, kind, op, self.blob[p:p + size]
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

    # Contexts are named (ctx_id, generation) throughout: the guest reuses ids, so a summary
    # keyed on the id alone reports two unrelated contexts as one busy one.
    for ctx_id, gen, blob in c.prologues():
        n = sum(1 for _ in journal_entries(blob))
        print(f"  context {ctx_id} gen {gen}: prologue {len(blob)} bytes, {n} journal entries")

    per_ctx = collections.Counter()
    per_ring = collections.Counter()
    per_ctl = collections.Counter()
    for _seq, ring_id, ctx_id, gen, kind, op, _pay in c.records():
        if kind == KIND_CTL:
            per_ctl[op] += 1
            continue
        per_ctx[(ctx_id, gen)] += 1
        per_ring[(ctx_id, gen, ring_id)] += 1
    for (ctx_id, gen), n in sorted(per_ctx.items()):
        print(f"  context {ctx_id} gen {gen}: {n} stream records")
    for (ctx_id, gen, ring_id), n in sorted(per_ring.items()):
        where = "context decoder" if ring_id == 0 else f"ring {ring_id:#x}"
        print(f"    ctx {ctx_id} gen {gen} / {where}: {n}")
    if per_ctl:
        print(f"  control-path events ({sum(per_ctl.values())}):")
        for op, n in sorted(per_ctl.items()):
            print(f"    {CTL_NAMES.get(op, f'ctl-{op}'):>16}: {n}")


def check(c):
    """Structural checks. Each failure is something that makes the corpus unreplayable, so they
    are worth running before a capture is pinned as a fixture."""
    bad = []

    total = HEADER_SIZE + c.prologue_bytes + c.stream_bytes
    if total != len(c.blob):
        bad.append(f"section sizes ({total}) disagree with the file ({len(c.blob)})")

    prologue_ctxs = set()
    n_prologue = 0
    for ctx_id, gen, blob in c.prologues():
        if (ctx_id, gen) in prologue_ctxs:
            bad.append(f"context {ctx_id} gen {gen} has two prologues")
        prologue_ctxs.add((ctx_id, gen))
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
    for seq, _ring, ctx_id, gen, kind, op, payload in c.records():
        n += 1
        # Anchored at 0, not merely consecutive: the recorder's counter starts there, so a
        # stream whose first record is seq 1 lost a record before anything else was written —
        # which a pairwise check alone reads as perfectly contiguous.
        if prev is None and seq != 0:
            bad.append(f"stream starts at seq {seq}, not 0: records were lost at the head")
        elif prev is not None and seq != prev + 1:
            bad.append(f"seq {seq} follows {prev}: the stream is not contiguous")
        prev = seq
        if kind == KIND_CTL:
            # A control record needs no prologue: it names a context only to place itself in that
            # context's history, and some (a resource unref) name none at all.
            if op not in CTL_NAMES:
                bad.append(f"seq {seq}: unknown control op {op}")
            elif ctl_describe(op, payload).endswith(f"TRUNCATED ({len(payload)} bytes)"):
                bad.append(f"seq {seq}: {CTL_NAMES[op]} payload is {len(payload)} bytes")
            continue
        if kind != KIND_CMD:
            bad.append(f"seq {seq}: unknown record kind {kind}")
            continue
        if (ctx_id, gen) not in prologue_ctxs:
            bad.append(f"seq {seq}: context {ctx_id} gen {gen} has no prologue")
            prologue_ctxs.add((ctx_id, gen))  # report once
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
        for seq, ring_id, ctx_id, gen, kind, op, payload in c.records():
            if kind == KIND_CTL:
                print(f"{seq:8d} ctx={ctx_id}/{gen} {'CTL':>20} {ctl_describe(op, payload)}")
                continue
            where = "ctx" if ring_id == 0 else f"ring {ring_id:#x}"
            print(f"{seq:8d} ctx={ctx_id}/{gen} {where:>20} type={op:5d} "
                  f"{len(payload):6d} bytes")
        return 0
    if mode == "--types":
        hist = collections.Counter(op for _s, _r, _c, _g, k, op, _p in c.records()
                                   if k == KIND_CMD)
        for cmd, n in hist.most_common():
            print(f"{n:8d}  cmd_type {cmd}")
        return 0
    if mode == "--prologue":
        for ctx_id, gen, blob in c.prologues():
            print(f"== context {ctx_id} gen {gen} ({len(blob)} bytes)")
            for seq, cmd_type, klass, ring_key, payload in journal_entries(blob):
                ring = f" ring={ring_key:#x}" if ring_key else ""
                print(f"  {seq:8d} type={cmd_type:5d} klass={klass}{ring} {len(payload):6d} bytes")
        return 0

    summarize(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
