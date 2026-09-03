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

import re
import subprocess
import sys
from pathlib import Path

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
]

# Not here, and deliberately: "a free forgets to credit the ledger". There is no such line to
# break. A charge is a value held by the record of what it paid for, so crediting is that record
# going away -- the edit would have to delete the field, which is a different change. An entry
# that cannot be written because the bug cannot be written is the design working.


def run(cmd, cwd=RS):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def main():
    patterns = sys.argv[1:]
    chosen = [s for s in SABOTAGES if not patterns or any(p in s[0] for p in patterns)]
    if not chosen:
        sys.exit('no sabotage matches %r' % patterns)

    dirty = run(['git', 'status', '--porcelain'], cwd=ROOT).stdout.strip()
    if dirty:
        sys.exit('the tree has uncommitted changes; sweep would restore over them:\n' + dirty)

    baseline = run(['cargo', 'test'])
    if baseline.returncode != 0:
        sys.exit('the tests do not pass before any sabotage; fix that first')

    holes = []
    for name, rel, old, new, filt in chosen:
        path = ROOT / rel
        original = path.read_text()
        assert old in original, 'sabotage %r no longer matches %s' % (name, rel)
        path.write_text(original.replace(old, new, 1))
        try:
            r = run(['cargo', 'test'] + ([filt] if filt else []))
        finally:
            path.write_text(original)
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
