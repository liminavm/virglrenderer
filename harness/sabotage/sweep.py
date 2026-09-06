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
`RED` for an edit it never made is worse than no sweep. Every target is checked up front, before
anything is built, so a refactor costs one message naming all of them.
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
        'an unserved command is counted and then continues, as though the host had done it',
        'virglrs/src/venus/context.rs',
        """        *self.todo.seen.entry(cmd.0).or_default() += 1;
        self.reject = Some("is not a command this build serves");""",
        """        *self.todo.seen.entry(cmd.0).or_default() += 1;""",
        'unserved_command',
    ),
    (
        'a query read-back is handed to the driver with results that run past the room the guest offered',
        'virglrs/src/venus/driver.rs',
        """        if facts.bytes_for(count, stride, flags)? > out.len() as u64 {
            return Err(QueryRefused::OutOfRoom);
        }""",
        """        facts.bytes_for(count, stride, flags)?;""",
        'query',
    ),
    (
        'a query read-back names queries past the end of the pool',
        'virglrs/src/venus/driver.rs',
        """        let facts = self.query_facts(pool)?;
        facts.holds(first, count)?;
        if facts.bytes_for""",
        """        let facts = self.query_facts(pool)?;
        if facts.bytes_for""",
        'query',
    ),
    (
        'a host-side query pool reset is handed to the driver past the end of the pool, and the driver zeroes host memory there',
        'virglrs/src/venus/driver.rs',
        """        self.query_facts(pool)?.holds(first, count)?;
        // SAFETY: a device in this table, a pool recorded on it, and a range of queries the pool
        // holds -- which is what the driver writes host memory at.""",
        """        self.query_facts(pool)?;
        // SAFETY: a device in this table, a pool recorded on it, and a range of queries the pool
        // holds -- which is what the driver writes host memory at.""",
        'query',
    ),
    (
        'a recorded query begin is handed to the driver past the end of the pool',
        'virglrs/src/venus/driver.rs',
        """        let (d, facts) = self.query_recorder(cb, pool)?;
        facts.holds(query, 1)?;
        // SAFETY: as above, and a query the pool holds.
        unsafe { (d.vkCmdBeginQuery())(cb, pool, query, flags) };""",
        """        let (d, _facts) = self.query_recorder(cb, pool)?;
        // SAFETY: as above, and a query the pool holds.
        unsafe { (d.vkCmdBeginQuery())(cb, pool, query, flags) };""",
        'query',
    ),
    (
        'a recorded query-pool result copy is handed to the driver past the end of the pool',
        'virglrs/src/venus/driver.rs',
        """        let (d, facts) = self.query_recorder(cb, pool)?;
        facts.holds(first, count)?;
        // SAFETY: as above, and a range of queries the pool holds.
        unsafe {
            (d.vkCmdCopyQueryPoolResults())""",
        """        let (d, _facts) = self.query_recorder(cb, pool)?;
        // SAFETY: as above, and a range of queries the pool holds.
        unsafe {
            (d.vkCmdCopyQueryPoolResults())""",
        'query',
    ),
    (
        'a query kind this host advertises through an extension is sized as one nobody can size, so its read-back is refused',
        'virglrs/src/venus/driver.rs',
        """            | VkQueryType::VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT
            | VkQueryType::VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT""",
        """            | VkQueryType::VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT""",
        'query',
    ),
    (
        "a query read-back's status word is not counted in the room it needs",
        'virglrs/src/venus/driver.rs',
        """            + u64::from(has(VkQueryResultFlagBits::VK_QUERY_RESULT_WITH_AVAILABILITY_BIT))
            + u64::from(has(VkQueryResultFlagBits::VK_QUERY_RESULT_WITH_STATUS_BIT_KHR));""",
        """            + u64::from(has(VkQueryResultFlagBits::VK_QUERY_RESULT_WITH_AVAILABILITY_BIT));""",
        'query',
    ),
    (
        'a destroyed query pool keeps its record, so a recycled handle is measured against a previous life',
        'virglrs/src/venus/context.rs',
        """        self.driver.forget_query_pool(args.queryPool);
        self.driver.destroy_object(""",
        """        self.driver.destroy_object(""",
        'query_results',
    ),
    (
        "a query pool the device's teardown took keeps its record",
        'virglrs/src/venus/driver.rs',
        """                VkObjectType::VK_OBJECT_TYPE_QUERY_POOL => {
                    self.forget_query_pool(VkQueryPool::from_host(handle));
                }""",
        """                VkObjectType::VK_OBJECT_TYPE_QUERY_POOL => {}""",
        'query',
    ),
    (
        'a query command hands the driver its arguments in the wrong order',
        'virglrs/src/venus/context.rs',
        """        let done = self.driver.cmd_copy_query_pool_results(
            args.commandBuffer,
            args.queryPool,
            args.firstQuery,
            args.queryCount,
            args.dstBuffer,
            args.dstOffset,
            args.stride,""",
        """        let done = self.driver.cmd_copy_query_pool_results(
            args.commandBuffer,
            args.queryPool,
            args.firstQuery,
            args.queryCount,
            args.dstBuffer,
            args.stride,
            args.dstOffset,""",
        'query_results',
    ),
    (
        'a ghost absorbs a command the guest is waiting on, and the guest reads a stale reply slot as its answer',
        'virglrs/src/venus/context.rs',
        """                if wants_reply {
                    poison(
                        id,
                        &dec,
                        cmd,
                        &format!("wanted a reply, and names object {} the host refused", ghost.0),
                    );
                    break;
                }
""",
        '',
        'a_ghost_absorbs_a_command',
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
        'a capture larger than the allocation is clamped to fit instead of refused',
        'virglrs/src/venus/driver.rs',
        """        if src.len() as u64 > record.size {
            return Err(MemoryError::LargerThanAllocation);
        }""",
        """        let src = &src[..src.len().min(record.size as usize)];""",
        'a_capture_goes_back_in_by_the_route_it_came_out_of',
    ),
    # The classic contents blob. Only the codec is reachable from a unit test: a no-op restore is
    # caught by the replay gate's scrub, and "a level that could not be read back is written as
    # zeros" is caught by the composite corpora, whose planar decode targets a texture transfer
    # cannot move -- 41 skipped levels on --ctx 10,11 and 6 on --ctx 8,9. Zeros in their place
    # would change the entry count and the all-zero count, both of which the gate prints.
    (
        'a truncated content blob is parsed as far as it goes instead of refused',
        'virglrs/src/vrend/content.rs',
        """        if blob.len() - at < size {
            return Err(Malformed::Truncated);
        }""",
        """        let size = size.min(blob.len() - at);""",
        'a_blob_that_is_not_one_is_refused',
    ),
    (
        "a scanout's restore goes through vkMapMemory like any other allocation",
        'virglrs/src/venus/driver.rs',
        """        if let Some(surface) = record.surface() {
            return Ok(surface.write_from(src));
        }
""",
        """""",
        'a_capture_goes_back_in_by_the_route_it_came_out_of',
    ),
    (
        'a blob is attributed to whichever context holds its id now',
        'virglrs/src/renderer.rs',
        '                    if from.ctx == key =>',
        '                    if from.ctx.id() == key.id() =>',
        'a_reused_context_id',
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
        """            (Some(bytes), _, _) => Backing::Imported(bytes),""",
        """            (Some(bytes), _, _) => {
                std::mem::forget(self.admit("device memory", size)?);
                Backing::Imported(bytes)
            }""",
        '',
    ),
    (
        'the cap is consulted and then ignored',
        'virglrs/src/venus/budget.rs',
        """        if let Some(cap) = self.budget.cap
            && live.saturating_add(size) > cap
        {""",
        """        if let Some(cap) = self.budget.cap
            && false
        {""",
        '',
    ),
    (
        'a budget refusal leaves the context running',
        'virglrs/src/venus/driver.rs',
        """            NoMemory::OverBudget { stop: self.account.kills_context() }""",
        """            NoMemory::OverBudget { stop: false }""",
        '',
    ),
    (
        'a scanout is charged at the size the guest asked for, not the surface it got',
        'virglrs/src/venus/driver.rs',
        """                let charge = self.admit("IOSurface", surface.alloc_size())?;""",
        """                let charge = self.admit("IOSurface", size)?;""",
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
        "a ring's shared memory is reachable by any context that guesses its handle",
        'virglrs/src/renderer.rs',
        """        if !res.attached.contains(&ctx) {
            eprintln!(
                "[virglrs] ctx {}: resource {handle:?} is not attached to this context",
                ctx.get(),
            );
            return None;
        }
        if let Some(map) = res.shm() {
            return Some(ResourceBytes::Host(Arc::clone(map)));
        }""",
        """        if let Some(map) = res.shm() {
            return Some(ResourceBytes::Host(Arc::clone(map)));
        }
        if !res.attached.contains(&ctx) {
            eprintln!(
                "[virglrs] ctx {}: resource {handle:?} is not attached to this context",
                ctx.get(),
            );
            return None;
        }""",
        'a_context_reaches_the_resources_the_guest_attached_to_it',
    ),
    (
        'the gate on what the guest attached to a context is dropped',
        'virglrs/src/renderer.rs',
        """        if !res.attached.contains(&ctx) {""",
        """        if false {""",
        '',
    ),
    (
        'a scanout export lends no share, so its storage stays trapped in one context',
        'virglrs/src/venus/driver.rs',
        """        let share = storage.clone();""",
        """        let share = match storage {
            Storage::Linear(s) => Storage::Linear(Arc::clone(s)),
            Storage::Texture(_) => return Err(ExportError::NotMappable),
        };""",
        'memory_is_published_once_and_leaves_the_census_when_it_is',
    ),
    (
        "a context's destroy uncounts what still outlives it",
        'virglrs/src/venus/budget.rs',
        """        slot.live.drain_into(&mut ledger.shared);""",
        """        drop(slot);""",
        'a_charge_that_outlives_its_context_stays_counted',
    ),
    (
        'the cap stops counting storage the moment its context is gone',
        'virglrs/src/venus/budget.rs',
        """        self.ctxs.values().map(|s| s.live.bytes()).sum::<u64>() + self.shared.bytes()""",
        """        self.ctxs.values().map(|s| s.live.bytes()).sum::<u64>()""",
        'a_charge_that_outlives_its_context_stays_counted',
    ),
    (
        'a late credit lands on whichever context holds the id now',
        'virglrs/src/venus/budget.rs',
        """        self.ctxs.get_mut(&ctx.id()).filter(|s| s.ctx == ctx).map(|s| &mut s.live)""",
        """        self.ctxs.get_mut(&ctx.id()).map(|s| &mut s.live)""",
        'a_late_credit_never_lands_on_the_next_context_with_the_same_id',
    ),
    (
        'an image the guest shares keeps the opaque tiling it asked for',
        'virglrs/src/venus/driver.rs',
        """        info.tiling = VkImageTiling::VK_IMAGE_TILING_LINEAR;
""",
        """""",
        '',
    ),
    (
        'an opaque image gets a surface the driver will never write into',
        'virglrs/src/venus/driver.rs',
        """        if !matches!(
            facts.tiling,
            VkImageTiling::VK_IMAGE_TILING_LINEAR
                | VkImageTiling::VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT
        ) {""",
        """        if false {""",
        '',
    ),
    (
        'host-visible memory is left to the driver again, so a blob over it has nothing to hold',
        'virglrs/src/venus/driver.rs',
        """            && props.is_some_and(|p| p.0 & HOST_VISIBLE_BIT != 0)""",
        """            && false""",
        'host_addressable_memory_is_backed_by_pages_this_renderer_minted',
    ),
    (
        'an allocation is published twice, and neither resource can learn of the other',
        'virglrs/src/venus/driver.rs',
        """        *published = true;""",
        """        *published = false;""",
        'memory_is_published_once_and_leaves_the_census_when_it_is',
    ),
    (
        'a blob is published larger than the storage it was minted from',
        'virglrs/src/venus/driver.rs',
        """        if blob_size > len {
            return Err(ExportError::LargerThanAllocation);
        }""",
        """""",
        'memory_is_published_once_and_leaves_the_census_when_it_is',
    ),
    (
        'minted pages lend no share, so the buffer stays trapped in one context',
        'virglrs/src/venus/driver.rs',
        """        let share = storage.clone();""",
        """        let share = match storage {
            Storage::Texture(s) => Storage::Texture(Arc::clone(s)),
            Storage::Linear(_) => return Err(ExportError::NotMappable),
        };""",
        '',
    ),
    (
        'freeing a descriptor set is refused again, and every GTK client dies a few frames in',
        'virglrs/src/venus/context.rs',
        """    fn vkFreeDescriptorSets(&mut self, args: &mut vn_command_vkFreeDescriptorSets<'_>) {
        let sets = self.array_or_empty(args.pDescriptorSets());""",
        """    fn vkFreeDescriptorSets(&mut self, args: &mut vn_command_vkFreeDescriptorSets<'_>) {
        self.unsupported(VkCommandTypeEXT::VK_COMMAND_TYPE_vkFreeDescriptorSets_EXT);
        if true { return; }
        let sets = self.array_or_empty(args.pDescriptorSets());""",
        '',
    ),
    (
        'a free is handed to the driver under whatever pool the guest named',
        'virglrs/src/venus/driver.rs',
        """        if !self.pools.all_from(pool, objects) {
            return Err(FreeRefused::NotFromThisPool);
        }""",
        """        if !self.pools.is_open(pool) {
            return Err(FreeRefused::NotFromThisPool);
        }""",
        'freeing_descriptor_sets_releases_them_from_their_pool',
    ),
    (
        'saving the pipeline cache is refused again, and every GTK client dies after its first pipeline',
        'virglrs/src/venus/context.rs',
        """    fn vkGetPipelineCacheData(&mut self, args: &mut vn_command_vkGetPipelineCacheData<'_>) {
        let device = args.device;""",
        """    fn vkGetPipelineCacheData(&mut self, args: &mut vn_command_vkGetPipelineCacheData<'_>) {
        self.unsupported(VkCommandTypeEXT::VK_COMMAND_TYPE_vkGetPipelineCacheData_EXT);
        if true { return; }
        let device = args.device;""",
        '',
    ),
    (
        'an out blob the guest offers room for is decoded as absent, and its handler writes nowhere',
        'virglrs/venus-gen/rustgen.py',
        """                hit = ['let n = dec.decode_array_size(%s) as usize;' % shape[1],
                       'let Some(a) = dec.alloc_temp_array::<u8>(n) else { return };',
                       '%s = a.as_mut_ptr() as %s _;' % (m, ptr)]""",
        """                hit = ['let n = dec.decode_array_size(%s) as usize;' % shape[1],
                       '%s = %s;' % (m, null)]""",
        'an_out_blob_is_room',
    ),
    (
        'the count call of an out blob is held to the garbage size beside it, and every count call is poisoned',
        'virglrs/venus-gen/rustgen.py',
        """        if count is not None and not var.is_optional() and var.can_validate():
            miss = ['dec.decode_array_size(%s);' % count]""",
        """        if count is not None:
            miss = ['dec.decode_array_size(%s);' % count]""",
        'an_out_blob_is_room',
    ),
    (
        "an out blob's slice is sized by the length member the handler rewrites, not the room",
        'virglrs/venus-gen/rustgen.py',
        """                    rows.append((f, 'u8', '(val.room_%s) as u64' % f, True))""",
        """                    rows.append((f, 'u8', shape[1], True))""",
        'an_out_blob_is_bounded_by_the_room',
    ),
    (
        'a reply encodes as many out-blob bytes as the handler claims, past the room',
        'virglrs/venus-gen/rustgen.py',
        """                        '    assert!(n <= %s, "%s wrote {n} bytes of %s into room for {}", %s);'
                        % (room, ty.name, var.name, room),""",
        """                        '    let _ = %s;' % room,""",
        'an_out_blob_is_bounded_by_the_room',
    ),
    (
        'the fill call of the pipeline cache data reports the room offered, not the bytes written',
        'virglrs/src/venus/context.rs',
        """        let asked = self.driver.pipeline_cache_data(device, cache, Some(out));
        match asked {
            Ok((n, ret)) => {""",
        """        let room = out.len();
        let asked = self.driver.pipeline_cache_data(device, cache, Some(out));
        match asked {
            Ok((_, ret)) => {
                let n = room;""",
        '',
    ),
    (
        'merging pipeline caches is refused again',
        'virglrs/src/venus/context.rs',
        """    fn vkMergePipelineCaches(&mut self, args: &mut vn_command_vkMergePipelineCaches<'_>) {
        let srcs = args.pSrcCaches();""",
        """    fn vkMergePipelineCaches(&mut self, args: &mut vn_command_vkMergePipelineCaches<'_>) {
        self.unsupported(VkCommandTypeEXT::VK_COMMAND_TYPE_vkMergePipelineCaches_EXT);
        if true { return; }
        let srcs = args.pSrcCaches();""",
        '',
    ),
    (
        'the image copy the overview blur asks for is refused again',
        'virglrs/src/venus/context.rs',
        """    fn vkCmdCopyImage(&mut self, args: &mut vn_command_vkCmdCopyImage<'_>) {
        let regions = args.pRegions();
        let done = self.driver.cmd_copy_image(
            args.commandBuffer,
            args.srcImage,
            args.srcImageLayout,
            args.dstImage,
            args.dstImageLayout,
            regions,
        );
        self.recorded(done);
    }

""",
        """""",
        '',
    ),
    (
        "an image copy hands each image the other one's layout",
        'virglrs/src/venus/context.rs',
        """    fn vkCmdCopyImage(&mut self, args: &mut vn_command_vkCmdCopyImage<'_>) {
        let regions = args.pRegions();
        let done = self.driver.cmd_copy_image(
            args.commandBuffer,
            args.srcImage,
            args.srcImageLayout,
            args.dstImage,
            args.dstImageLayout,
            regions,
        );
        self.recorded(done);
    }

""",
        """    fn vkCmdCopyImage(&mut self, args: &mut vn_command_vkCmdCopyImage<'_>) {
        let regions = args.pRegions();
        let done = self.driver.cmd_copy_image(
            args.commandBuffer,
            args.srcImage,
            args.dstImageLayout,
            args.dstImage,
            args.srcImageLayout,
            regions,
        );
        self.recorded(done);
    }

""",
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
    (
        'a classic blend factor is read from the bit beside it',
        'virglrs/src/vrend/decode.rs',
        "Some(RtBlendEq { rgb: eq(1, 4, 9)?, alpha: eq(14, 17, 22)? })",
        "Some(RtBlendEq { rgb: eq(1, 5, 9)?, alpha: eq(14, 17, 22)? })",
        'vrend',
    ),
    (
        'a refused classic command does not end its batch',
        'virglrs/src/vrend/decode.rs',
        "self.at = if decoded.is_ok() { end } else { self.words.len() };",
        "self.at = end;",
        'vrend',
    ),
    (
        'a classic slot range past the per-stage array is bound',
        'virglrs/src/vrend/decode.rs',
        "if count > max || start as usize > max - count {",
        "if count > max {",
        'vrend',
    ),
    (
        'a classic shader continuation is taken for a new shader',
        'virglrs/src/vrend/decode.rs',
        "let chunk = if offlen >> 31 != 0 {",
        "let chunk = if offlen >> 31 == 2 {",
        'vrend',
    ),
    (
        'an inline write hands its bytes over starting one dword early',
        'virglrs/src/vrend/decode.rs',
        "Command::ResourceInlineWrite { transfer: w.transfer()?, data: w.tail(12) }",
        "Command::ResourceInlineWrite { transfer: w.transfer()?, data: w.tail(11) }",
        'vrend',
    ),
    (
        'a classic resource lends nothing, so a venus context cannot import one',
        'virglrs/src/renderer.rs',
        'Some(held) => Some(ResourceBytes::Shared(Storage::lent(Arc::clone(held)))),',
        'Some(_) => None,',
        'a_classic_resource_lends',
    ),
    (
        'the sample-count ceiling is ignored and the host maximum advertised anyway',
        'virglrs/src/vrend/caps.rs',
        'Some(c) if max_samples > c => {',
        'Some(c) if max_samples > c && false => {',
        'sample_ceiling',
    ),
    (
        'a ceiling of zero is read as no ceiling, which is what it least means',
        'virglrs/src/vrend/caps.rs',
        'Ok(n) => Some(n.max(1)),',
        'Ok(0) => None,\n        Ok(n) => Some(n),',
        'sample_ceiling',
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

    # Every target, before any building. The per-entry assert below is the guarantee that an edit
    # was made; this is what makes a refactor cost one message naming all of them rather than a
    # build's wait followed by an abort on the first, and another on the next.
    stale = [(n, r) for n, r, old, _, _ in SABOTAGES if old not in (ROOT / r).read_text()]
    if stale:
        sys.exit(
            'sabotage targets no longer in the tree -- fix or retire each:\n'
            + '\n'.join('  %s\n    %s' % (n, r) for n, r in stale)
        )

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
