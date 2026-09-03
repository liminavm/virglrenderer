#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright © 2026 the limina authors
"""Break the renderer on purpose, and report which breakages the tests notice.

A passing suite says nothing about what it would catch. This says it directly: each entry below
is a one-line edit that makes the renderer wrong in a way a guest would see, applied to a clean
tree, tested, and reverted. `RED` is the suite catching it. `SURVIVED` is a hole, named.

    harness/sabotage/sweep.py [pattern ...]

Entries are matched by substring against their name; with none, every entry runs. Each is applied
alone, so one sabotage never masks another.

The edits are exact string replacements and every one asserts it matched, so an entry whose target
has been refactored away fails loudly instead of quietly testing nothing -- a sweep that reports
`RED` for an edit it never made is worse than no sweep.
"""

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
RS = ROOT / 'virglrs'

# (name, path relative to the repo root, what to replace, what with, cargo test filter)
SABOTAGES = [
    (
        'an array accessor hands its handler one element fewer',
        'virglrs/venus-gen/rustgen.py',
        "unsafe { cs::%s((%s) as usize, val.%s as *%s _) }'\n                    % (call, n, f, 'mut' if mutable else 'const')",
        "unsafe { cs::%s(((%s) as usize).saturating_sub(1), val.%s as *%s _) }'\n                    % (call, n, f, 'mut' if mutable else 'const')",
        'witness',
    ),
    (
        'an array accessor hands its handler one element more',
        'virglrs/venus-gen/rustgen.py',
        "unsafe { cs::%s((%s) as usize, val.%s as *%s _) }'\n                    % (call, n, f, 'mut' if mutable else 'const')",
        "unsafe { cs::%s((%s) as usize + 1, val.%s as *%s _) }'\n                    % (call, n, f, 'mut' if mutable else 'const')",
        'witness',
    ),
    (
        'a reply is committed before the command that produced it is judged',
        'virglrs/src/venus/ring.rs',
        '        self.pos += bytes.len();\n        Ok(())',
        '        Ok(())',
        '',
    ),
    (
        'a reply that does not fit is written anyway',
        'virglrs/src/venus/ring.rs',
        '        if bytes.len() > remaining {\n            return Err(ReplyOverflow { wanted: bytes.len(), remaining });\n        }',
        '        let bytes = &bytes[..bytes.len().min(remaining)];',
        '',
    ),
    (
        'a seek past the reply window is clamped instead of refused',
        'virglrs/src/venus/ring.rs',
        '        if pos > self.window.size() {\n            return false;\n        }\n        self.pos = pos;',
        '        self.pos = pos.min(self.window.size());',
        '',
    ),
    (
        'a run of pool allocations is filed one object out of step',
        'virglrs/src/venus/driver.rs',
        '            out.iter().copied().zip(ids.iter().copied()).filter(|(h, _)| h.host().0 != 0),',
        '            out.iter().copied().zip(ids.iter().skip(1).copied()).filter(|(h, _)| h.host().0 != 0),',
        '',
    ),
    (
        'a refused run of pool allocations leaves its ids plain missing',
        'virglrs/src/venus/context.rs',
        '            eprintln!("[virglrs] vkAllocateCommandBuffers refused by the driver");\n            self.ghost_ids(ids);',
        '            eprintln!("[virglrs] vkAllocateCommandBuffers refused by the driver");',
        '',
    ),
    (
        'a short enumeration ghosts the devices it did answer for',
        'virglrs/src/venus/context.rs',
        '        self.ghost_ids(&ids[got as usize..]);',
        '        self.ghost_ids(ids);',
        '',
    ),
    (
        'a short enumeration reports the length the guest asked for',
        'virglrs/src/venus/context.rs',
        '        if let Some(count) = args.pPhysicalDeviceCount_mut() {\n            *count = got;\n        }',
        '        let asked = ids.len() as u32;\n        if let Some(count) = args.pPhysicalDeviceCount_mut() {\n            *count = asked;\n        }',
        '',
    ),
    (
        'extensions are learned for slots the driver never filled',
        'virglrs/src/venus/context.rs',
        '        for pd in out.iter().take(got as usize) {',
        '        for pd in out.iter() {',
        '',
    ),
    (
        'a shader whose code is not a whole number of words is passed on',
        'virglrs/src/venus/context.rs',
        '        if info.codeSize % 4 != 0 {',
        '        if false {',
        '',
    ),
    (
        'constants are pushed without the bytes that are the command',
        'virglrs/src/venus/context.rs',
        '        let Some(values) = args.pValues() else {\n            self.reject = Some("pushed constants without saying what they are");\n            return;\n        };',
        '        let values = args.pValues().unwrap_or(&[]);',
        '',
    ),
    (
        'an empty submit is turned away instead of signalling its fence',
        'virglrs/src/venus/context.rs',
        '        let submits = args.pSubmits();\n        let Some(ret) = self.driver.queue_submit(args.queue, submits, args.fence) else {',
        '        let submits = args.pSubmits();\n        if submits.is_empty() {\n            return;\n        }\n        let Some(ret) = self.driver.queue_submit(args.queue, submits, args.fence) else {',
        '',
    ),
    (
        "a bind sends the dynamic offsets' count with the descriptor sets",
        'virglrs/src/venus/context.rs',
        '            sets,\n            offsets,\n        );\n        self.recorded(done);',
        '            sets,\n            &offsets[..offsets.len().min(sets.len())],\n        );\n        self.recorded(done);',
        '',
    ),
    (
        'a failed pipeline run leaks the pipelines it did make',
        'virglrs/src/venus/driver.rs',
        "            unsafe { (d.fns.vkDestroyPipeline())(device, *survivor, ptr(alloc)) };\n            // The guest's reply must not carry a handle that is now gone.\n            *survivor = VkPipeline(0);",
        '            *survivor = VkPipeline(0);',
        '',
    ),
    (
        'a failed pipeline run leaves destroyed handles in the reply',
        'virglrs/src/venus/driver.rs',
        "            unsafe { (d.fns.vkDestroyPipeline())(device, *survivor, ptr(alloc)) };\n            // The guest's reply must not carry a handle that is now gone.\n            *survivor = VkPipeline(0);",
        '            unsafe { (d.fns.vkDestroyPipeline())(device, *survivor, ptr(alloc)) };',
        '',
    ),
    (
        'the census reports storage a guest only borrowed',
        'virglrs/src/venus/driver.rs',
        '            .filter(|(_, a)| a.censused())',
        '            .filter(|(_, a)| a.censused() || true)',
        '',
    ),
    (
        'an import is billed as though it were fresh storage',
        'virglrs/src/venus/driver.rs',
        '            Backing::Imported => None,\n        };\n        let charge = match charge.transpose() {',
        '            Backing::Imported => Some(self.account.try_charge("device memory", size)),\n        };\n        let charge = match charge.transpose() {',
        '',
    ),
    (
        'the cap is consulted and then ignored',
        'virglrs/src/venus/budget.rs',
        '        if let Some(cap) = self.cap\n            && live.saturating_add(size) > cap\n        {',
        '        if let Some(cap) = self.cap\n            && false\n        {',
        '',
    ),
    (
        'a budget refusal leaves the context running',
        'virglrs/src/venus/driver.rs',
        'return Err(NoMemory::OverBudget { stop: self.account.kills_context() });',
        'return Err(NoMemory::OverBudget { stop: false });',
        '',
    ),
    (
        'a scanout is charged at the size the guest asked for, not the surface it got',
        'virglrs/src/venus/driver.rs',
        'Backing::Scanout(s) => Some(self.account.try_charge("IOSurface", s.alloc_size())),',
        'Backing::Scanout(_) => Some(self.account.try_charge("IOSurface", size)),',
        '',
    ),
    (
        'a budget refusal is reported to the guest and to nobody else',
        'virglrs/src/venus/context.rs',
        '''            if let driver::NoMemory::OverBudget { stop: true } = e {
                self.reject = Some("the host memory budget refused this allocation");
            }
''',
        '',
        '',
    ),
    (
        'an image-to-buffer copy hands the driver no regions',
        'virglrs/src/venus/driver.rs',
        '''            (d.vkCmdCopyImageToBuffer())(
                cb,
                src,
                layout,
                dst,
                regions.len() as u32,
                regions.as_ptr(),
            )''',
        '''            (d.vkCmdCopyImageToBuffer())(cb, src, layout, dst, 0, regions.as_ptr())''',
        '',
    ),
    (
        'an image-to-buffer copy never reaches the driver at all',
        'virglrs/src/venus/context.rs',
        '''        let done = self.driver.cmd_copy_image_to_buffer(
            args.commandBuffer,
            args.srcImage,
            args.srcImageLayout,
            args.dstBuffer,
            regions,
        );''',
        '''        let done = Some(());''',
        '',
    ),
    (
        'a monitored ring is registered but never stamped',
        'virglrs/src/venus/monitor.rs',
        '''                Some(status) => {
                    status.set_bits(STATUS_ALIVE);
                    true
                }''',
        '''                Some(_) => true,''',
        '',
    ),
    (
        'a ring asking to be monitored is quietly not monitored',
        'virglrs/src/venus/context.rs',
        '''        if let Some(want) = monitor_period(info) {''',
        '''        if let Some(want) = None::<Option<u32>> {''',
        '',
    ),
    (
        'a reporting period of zero is given a default instead of being refused',
        'virglrs/src/venus/context.rs',
        '''    Some(Some(m.maxReportingPeriodMicroseconds).filter(|&us| us != 0))''',
        '''    Some(Some(m.maxReportingPeriodMicroseconds).filter(|&us| us != 0).or(Some(3_000_000)))''',
        '',
    ),
    (
        'a shortened reporting period does not wake the sleeping monitor',
        'virglrs/src/venus/monitor.rs',
        '''            self.shared.wake.notify_one();
        }
    }
}''',
        '''        }
    }
}''',
        '',
    ),
    (
        'a ring blocks on a virtqueue seqno without telling the waiter',
        'virglrs/src/venus/ring_thread.rs',
        '''    state.blocked_on_vq = Some(seqno);''',
        '''    state.blocked_on_vq = None;''',
        '',
    ),
    (
        'a ring wait sleeps without ever checking whether it is the only thing that could end it',
        'virglrs/src/venus/ring_thread.rs',
        '''            if let Some(want) = self.park.stalled_on() {''',
        '''            if let Some(want) = None::<u64> {''',
        '',
    ),
    (
        'a stalled ring is judged on two values that were never true together',
        'virglrs/src/venus/ring_thread.rs',
        '''        state.blocked_on_vq.filter(|&want| state.vq_seqno < want)''',
        '''        state.blocked_on_vq''',
        '',
    ),
    (
        'a ring wait past everything the guest wrote is waited through',
        'virglrs/src/venus/ring_thread.rs',
        '''            if head == tail && !seqno_ge(tail, self.seqno) {''',
        '''            if false {''',
        '',
    ),
    (
        'a suspended batch reports the wait command as already run',
        'virglrs/src/venus/context.rs',
        '''            suspended = Some((at, on));''',
        '''            suspended = Some((dec.pos(), on));''',
        '',
    ),
    (
        'a virtqueue wait suspends even when the seqno is already published',
        'virglrs/src/venus/context.rs',
        '''        if published < args.seqno {
            self.wait = Some(Wait::Virtqueue(args.seqno));
        }''',
        '''        let _ = published;
        self.wait = Some(Wait::Virtqueue(args.seqno));''',
        '',
    ),
    (
        'a ring seqno wider than a ring position is truncated instead of refused',
        'virglrs/src/venus/context.rs',
        '''        let Ok(seqno) = u32::try_from(args.seqno) else {
            self.reject = Some("waited on a ring seqno too large to be a position in a ring");
            return;
        };''',
        '''        let seqno = args.seqno as u32;''',
        '',
    ),
    (
        'a ring extra write reaches past the extra region',
        'virglrs/src/venus/ring.rs',
        '''        if end > self.extra.begin() + self.extra.size() {
            return false;
        }''',
        '''''',
        '',
    ),
    (
        'a virtqueue seqno submitted before the ring started is dropped',
        'virglrs/src/venus/ring_thread.rs',
        '''        state: Mutex::new(ParkState { vq_seqno: ring.virtqueue_seqno, ..ParkState::default() }),''',
        '''        state: Mutex::new(ParkState::default()),''',
        '',
    ),
    (
        'a transport command is served on whichever stream it arrives on',
        'virglrs/src/venus/context.rs',
        '''        let Some(id) = self.current_ring else {
            self.reject = Some("waited on a virtqueue seqno from the context's own stream");
            return;
        };''',
        '''        let id = self.current_ring.unwrap_or(RingId(7));''',
        '',
    ),
    (
        'an executed stream is not bounds-checked against the resource holding it',
        'virglrs/src/venus/context.rs',
        """        let end = s.offset.checked_add(s.size);
        if end.is_none_or(|end| end > map.len()) {""",
        """        let end = s.offset.checked_add(s.size);
        if false {""",
        '',
    ),
    (
        'an executed stream may execute streams of its own, as deep as the guest likes',
        'virglrs/src/venus/context.rs',
        """            if depth > 0 {
                poison(
                    fatal,
                    id,
                    &dec,
                    cmd,
                    "executes command streams from inside a command stream it is already executing",
                );
                break;
            }""",
        """""",
        '',
    ),
    (
        'a per-stream reply position is ignored',
        'virglrs/src/venus/context.rs',
        """        if let Some(&pos) = exec.reply_positions.as_ref().map(|p| &p[i]) {""",
        """        if let Some(&pos) = None::<&usize> {""",
        '',
    ),
    (
        'a reply position outside the window is clamped rather than refused',
        'virglrs/src/venus/context.rs',
        """            if !stream.seek(pos) {""",
        """            let pos = pos.min(stream.window().size());
            if !stream.seek(pos) {""",
        '',
    ),
    (
        'an empty stream skips before its reply position is honoured',
        'virglrs/src/venus/context.rs',
        """        if let Some(&pos) = exec.reply_positions.as_ref().map(|p| &p[i]) {""",
        """        if s.size == 0 {
            continue;
        }
        if let Some(&pos) = exec.reply_positions.as_ref().map(|p| &p[i]) {""",
        '',
    ),
    (
        'reply positions are accepted with no window for them to be positions in',
        'virglrs/src/venus/context.rs',
        """            Some(_) if self.reply.is_none() => {
                self.reject =
                    Some("executed command streams with reply positions and no reply stream");
                return;
            }""",
        """""",
        '',
    ),
    (
        'only the fd half of the emulated external memory is advertised',
        'virglrs/src/venus/driver.rs',
        """            out.extend(EMULATED_ON_THE_HOST.iter().filter_map(|n| extension_properties(n)));""",
        """            out.extend(
                EMULATED_ON_THE_HOST
                    .iter()
                    .take(1)
                    .filter_map(|n| extension_properties(n)),
            );""",
        '',
    ),
    (
        'the emulated external memory is advertised on drivers that emulate nothing',
        'virglrs/src/venus/driver.rs',
        """        if self.supports(pd, "VK_EXT_external_memory_metal")
            && !self.supports(pd, "VK_KHR_external_memory_fd")
        {""",
        """        {""",
        '',
    ),
    (
        'a transport wait inside an executed stream suspends a batch it cannot resume',
        'virglrs/src/venus/context.rs',
        """            if depth > 0 {
                poison(
                    fatal,
                    id,
                    &dec,
                    cmd,
                    "suspends the batch, and a command stream being executed has nowhere to suspend to",
                );
                break;
            }""",
        """""",
        '',
    ),
]

# Not here, and deliberately: "a ring-seqno wake is never sent". Deleting any single
# `wait_ring.changed()` leaves every witness green, and that is the design rather than a hole. The
# waiter's stuck-log timeout doubles as a poll, so no individual wake is load-bearing for
# correctness -- what a missing one costs is latency: the wait ends at half a second instead of at
# microseconds, and prints a line the C's own comment calls a frame stutter, on a path that runs
# per exported frame sync fd. A test for that is a stopwatch, and the two arrangements it needs are
# mutually exclusive: the head must advance while the waiter is already asleep, but a waiter that
# suspends before the guest has written is refused outright by the drained-and-short guard, which
# is correct and is itself under test. An entry that can only be caught by winning a race would
# report a hole on a loaded machine and coverage on a quiet one.

# Not here, and deliberately: "a free forgets to credit the ledger". There is no such line to
# break. A charge is a value held by the record of what it paid for, so crediting is that record
# going away -- the edit would have to delete the field, which is a different change. An entry
# that cannot be written because the bug cannot be written is the design working.


def run(cmd, cwd=RS, timeout=None):
    """Run a command, killing the whole process group if it outstays `timeout`.

    The group, not the child: `cargo test` spawns the test binary, and a sabotage that deadlocks
    leaves that binary wedged forever. Killing only cargo would orphan it, and the next run would
    contend with a process holding the same shared memory.

    A timeout is reported, never raised. It is a legitimate verdict here -- see `main`.
    """
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        return SimpleNamespace(returncode=124, stdout='', stderr='', timed_out=True)
    return SimpleNamespace(returncode=proc.returncode, stdout=out, stderr=err, timed_out=False)


def main():
    patterns = sys.argv[1:]
    chosen = [s for s in SABOTAGES if not patterns or any(p in s[0] for p in patterns)]
    if not chosen:
        sys.exit('no sabotage matches %r' % patterns)

    dirty = run(['git', 'status', '--porcelain'], cwd=ROOT).stdout.strip()
    if dirty:
        sys.exit('the tree has uncommitted changes; sweep would restore over them:\n' + dirty)

    started = time.monotonic()
    baseline = run(['cargo', 'test'])
    if baseline.returncode != 0:
        sys.exit('the tests do not pass before any sabotage; fix that first')
    # Derived from the clean run rather than fixed, so a slow machine is not called a hang and a
    # fast one still catches a wedge quickly. The floor covers a rebuild after each edit.
    budget = max(180.0, (time.monotonic() - started) * 8)

    holes = []
    for name, rel, old, new, filt in chosen:
        path = ROOT / rel
        original = path.read_text()
        assert old in original, 'sabotage %r no longer matches %s' % (name, rel)
        path.write_text(original.replace(old, new, 1))
        try:
            r = run(['cargo', 'test'] + ([filt] if filt else []), timeout=budget)
        finally:
            path.write_text(original)
        if r.timed_out:
            # Caught, and in the loudest way there is. Several of these sabotages delete a
            # deadlock guard, and a missing deadlock guard does not produce a wrong answer -- it
            # produces a wait that never ends. A sweep with no clock of its own hangs here rather
            # than reporting it, which is the instrument failing at exactly the cases it was
            # extended to cover.
            print('RED       %-58s the suite hung: nothing on that path ends on its own' % name)
            continue
        if r.returncode == 0:
            holes.append(name)
            print('SURVIVED  %s' % name)
            continue
        # Which test noticed, not how many failed. `cargo test` lists the failures only when the
        # run finishes; a panic that aborts the binary ends it early, and then the name is in the
        # panic line instead. Reporting a count from whichever of those happened to be there is
        # how a sweep comes to claim coverage it cannot point at.
        named = re.findall(r"^    (\S+::\S+)$", r.stdout, re.M)
        if not named:
            named = re.findall(r"^thread '(\S+::\S+)'", r.stdout + r.stderr, re.M)[:1]
        witness = ', '.join(sorted(set(named))[:2]) if named else 'the test binary failed'
        more = len(set(named)) - 2
        print('RED       %-58s %s%s' % (name, witness, ' +%d more' % more if more > 0 else ''))

    print('\n%d of %d caught' % (len(chosen) - len(holes), len(chosen)))
    return 1 if holes else 0


if __name__ == '__main__':
    sys.exit(main())
