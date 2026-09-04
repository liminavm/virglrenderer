// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The classic wire's trust boundary: dwords in, [`Command`]s out, and everything the guest could
//! have got wrong refused here.
//!
//! The rules are `vrend_decode.c`'s, which is what the pinned scores were recorded through: each
//! command's allowed lengths, each array's bound, each stage index's range. Where the C reads a
//! field it never checks and crashes on later (a blend factor into an `assert(0)` switch, a
//! format into a table it indexes unchecked), the check is here instead, and a stream the C would
//! have died on is a stream this refuses. Where the C is looser than the wire -- ignoring a
//! trailing partial element, ignoring dwords past the ones it reads -- this is exact, because a
//! decoder that ignores bytes cannot reproduce them and the differential gate would be blind
//! there.
//!
//! A refusal ends the batch. The commands before it ran; the rest are never framed, because a
//! guest that mis-framed one command has forfeited the claim that the next header is a header.
//! What the caller does with the context afterwards is its business -- the C poisons it.

use super::pipe::slots::*;
use super::pipe::*;
use super::proto::*;
use crate::ids::{BlobId, ResourceHandle};

/// Gallium's limits, from `p_state.h`, as the C decoder applies them.
const MAX_STREAMOUT_TARGETS: usize = 16;
/// `VIRGL_GBM_MAX_PLANES`.
const MAX_PLANES: usize = 4;
/// `VREND_VIDEO_BUFFER_PLANE_NUM`.
const MAX_VIDEO_PLANES: usize = 3;

/// The dwords the wire's header is decoded from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Header {
    pub cmd: u32,
    pub obj: u32,
    pub len: usize,
}

impl Header {
    pub fn parse(word: u32) -> Header {
        Header { cmd: word & 0xff, obj: (word >> 8) & 0xff, len: (word >> 16) as usize }
    }
}

/// One batch of commands, framed and decoded one at a time. Yields nothing after a refusal.
pub struct Batch<'a> {
    words: &'a [u32],
    at: usize,
}

impl<'a> Batch<'a> {
    pub fn new(words: &'a [u32]) -> Batch<'a> {
        Batch { words, at: 0 }
    }

    /// Dwords consumed so far, refused command included.
    pub fn position(&self) -> usize {
        self.at
    }
}

impl<'a> Iterator for Batch<'a> {
    type Item = Result<Command<'a>, Refused>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.at >= self.words.len() {
            return None;
        }
        let at = self.at;
        let header = Header::parse(self.words[at]);
        let Some(cmd) = Cmd::from_wire(header.cmd) else {
            self.at = self.words.len();
            return Some(Err(Refused::UnknownCommand { at, cmd: header.cmd }));
        };
        let left = self.words.len() - at - 1;
        if header.len > left {
            self.at = self.words.len();
            return Some(Err(Refused::Overrun { at, cmd, len: header.len, left }));
        }
        let end = at + 1 + header.len;
        let words = &self.words[at..end];
        let decoded = decode(cmd, header.obj, words);
        self.at = if decoded.is_ok() { end } else { self.words.len() };
        Some(decoded)
    }
}

/// One command's dwords, header at index 0 so that every index below reads as the
/// `VIRGL_*` define it implements.
struct Words<'a> {
    cmd: Cmd,
    w: &'a [u32],
}

impl<'a> Words<'a> {
    /// The payload length: dwords after the header.
    fn len(&self) -> usize {
        self.w.len() - 1
    }

    fn exact(&self, n: usize) -> Result<(), Refused> {
        if self.len() == n { Ok(()) } else { Err(self.length()) }
    }

    fn at_least(&self, n: usize) -> Result<(), Refused> {
        if self.len() >= n { Ok(()) } else { Err(self.length()) }
    }

    fn length(&self) -> Refused {
        Refused::Length { cmd: self.cmd, len: self.len() }
    }

    fn refuse(&self, field: &'static str, value: u32) -> Refused {
        Refused::Field { cmd: self.cmd, field, value }
    }

    fn u(&self, i: usize) -> u32 {
        self.w[i]
    }

    fn i(&self, i: usize) -> i32 {
        self.w[i] as i32
    }

    fn f(&self, i: usize) -> f32 {
        f32::from_bits(self.w[i])
    }

    fn tail(&self, from: usize) -> &'a [u32] {
        &self.w[from..]
    }

    /// A value parsed from a whole dword or a masked piece of one; `field` names it in the
    /// refusal.
    fn parse<T>(
        &self,
        field: &'static str,
        value: u32,
        parse: impl FnOnce(u32) -> Option<T>,
    ) -> Result<T, Refused> {
        parse(value).ok_or_else(|| self.refuse(field, value))
    }

    fn resource(&self, field: &'static str, i: usize) -> Result<ResourceHandle, Refused> {
        self.parse(field, self.u(i), ResourceHandle::new)
    }

    fn resource_or_none(&self, i: usize) -> Option<ResourceHandle> {
        ResourceHandle::new(self.u(i))
    }

    fn object(&self, field: &'static str, i: usize) -> Result<ObjectHandle, Refused> {
        self.parse(field, self.u(i), ObjectHandle::new)
    }

    fn object_or_none(&self, i: usize) -> Option<ObjectHandle> {
        ObjectHandle::new(self.u(i))
    }

    fn stage(&self, i: usize) -> Result<ShaderStage, Refused> {
        self.parse("shader stage", self.u(i), ShaderStage::from_wire)
    }

    fn format(&self, field: &'static str, i: usize) -> Result<Format, Refused> {
        self.parse(field, self.u(i), Format::from_wire)
    }

    fn region(&self, from: usize) -> Box3 {
        Box3 {
            x: self.i(from),
            y: self.i(from + 1),
            z: self.i(from + 2),
            width: self.i(from + 3),
            height: self.i(from + 4),
            depth: self.i(from + 5),
        }
    }

    /// `VIRGL_RESOURCE_IW_*`: the transfer prologue at dwords 1..=11.
    fn transfer(&self) -> Result<Transfer, Refused> {
        Ok(Transfer {
            resource: self.resource("resource", 1)?,
            level: self.u(2),
            usage: self.u(3),
            stride: self.u(4),
            layer_stride: self.u(5),
            region: self.region(6),
        })
    }

    /// A stage plus a slot range against a per-stage array of `max` slots: the C's
    /// `num > max || start > max - num` shape.
    fn slots(&self, start: u32, count: usize, max: usize) -> Result<(), Refused> {
        if count > max || start as usize > max - count {
            return Err(self.refuse("start slot", start));
        }
        Ok(())
    }

    /// The count of `each`-dword elements after `after` leading dwords, refusing a length that
    /// is not whole elements.
    fn elements(&self, after: usize, each: usize) -> Result<usize, Refused> {
        let body = self.len().checked_sub(after).ok_or_else(|| self.length())?;
        if body % each != 0 {
            return Err(self.length());
        }
        Ok(body / each)
    }
}

fn bit(word: u32, at: u32) -> bool {
    (word >> at) & 1 != 0
}

/// Decode one framed command: `words[0]` is its header, `obj` the header's object byte.
pub fn decode<'a>(cmd: Cmd, obj: u32, words: &'a [u32]) -> Result<Command<'a>, Refused> {
    let w = Words { cmd, w: words };
    Ok(match cmd {
        Cmd::Nop => Command::Nop,
        Cmd::CreateObject => create_object(&w, obj)?,
        Cmd::BindObject => {
            w.exact(1)?;
            let kind = w.parse("object type", obj, ObjectType::from_wire)?;
            if !matches!(
                kind,
                ObjectType::Blend
                    | ObjectType::Dsa
                    | ObjectType::Rasterizer
                    | ObjectType::VertexElements
            ) {
                return Err(w.refuse("object type", obj));
            }
            Command::BindObject { kind, handle: w.object_or_none(1) }
        }
        Cmd::DestroyObject => {
            w.exact(1)?;
            Command::DestroyObject {
                kind: w.parse("object type", obj, ObjectType::from_wire)?,
                handle: w.object("handle", 1)?,
            }
        }
        Cmd::SetViewportState => {
            let n = w.elements(1, 6)?;
            let start_slot = w.u(1);
            w.slots(start_slot, n, MAX_VIEWPORTS)?;
            let viewports = (0..n)
                .map(|v| {
                    let at = 2 + v * 6;
                    Viewport {
                        scale: [w.f(at), w.f(at + 1), w.f(at + 2)],
                        translate: [w.f(at + 3), w.f(at + 4), w.f(at + 5)],
                    }
                })
                .collect();
            Command::SetViewportState { start_slot, viewports }
        }
        Cmd::SetFramebufferState => {
            w.at_least(2)?;
            let nr_cbufs = w.u(1) as usize;
            if w.len() != 2 + nr_cbufs {
                return Err(w.length());
            }
            if nr_cbufs > MAX_COLOR_BUFS {
                return Err(w.refuse("colour buffer count", w.u(1)));
            }
            Command::SetFramebufferState {
                zsurf: w.object_or_none(2),
                cbufs: (0..nr_cbufs).map(|i| w.object_or_none(3 + i)).collect(),
            }
        }
        Cmd::SetVertexBuffers => {
            let n = w.elements(0, 3)?;
            if n > MAX_ATTRIBS {
                return Err(w.refuse("vertex buffer count", n as u32));
            }
            Command::SetVertexBuffers(
                (0..n)
                    .map(|i| VertexBuffer {
                        stride: w.u(1 + i * 3),
                        offset: w.u(2 + i * 3),
                        resource: w.resource_or_none(3 + i * 3),
                    })
                    .collect(),
            )
        }
        Cmd::Clear => {
            w.exact(8)?;
            Command::Clear {
                buffers: w.u(1),
                color: [w.u(2), w.u(3), w.u(4), w.u(5)],
                depth: f64::from_bits(u64::from(w.u(6)) | (u64::from(w.u(7)) << 32)),
                stencil: w.u(8),
            }
        }
        Cmd::DrawVbo => draw_vbo(&w)?,
        Cmd::ResourceInlineWrite => {
            w.at_least(12)?;
            Command::ResourceInlineWrite { transfer: w.transfer()?, data: w.tail(12) }
        }
        Cmd::SetSamplerViews => {
            w.at_least(2)?;
            let stage = w.stage(1)?;
            let start_slot = w.u(2);
            let n = w.len() - 2;
            w.slots(start_slot, n, MAX_SHADER_SAMPLER_VIEWS)?;
            Command::SetSamplerViews {
                stage,
                start_slot,
                views: (0..n).map(|i| w.object_or_none(3 + i)).collect(),
            }
        }
        Cmd::SetIndexBuffer => match w.len() {
            1 => {
                if w.u(1) != 0 {
                    return Err(w.refuse("index buffer without a size", w.u(1)));
                }
                Command::SetIndexBuffer(None)
            }
            3 => Command::SetIndexBuffer(Some(IndexBuffer {
                resource: w.resource("index buffer", 1)?,
                index_type: w.parse("index size", w.u(2), IndexType::from_wire)?,
                offset: w.u(3),
            })),
            _ => return Err(w.length()),
        },
        Cmd::SetConstantBuffer => {
            w.at_least(2)?;
            Command::SetConstantBuffer { stage: w.stage(1)?, index: w.u(2), data: w.tail(3) }
        }
        Cmd::SetStencilRef => {
            w.exact(1)?;
            let v = w.u(1);
            if v >> 16 != 0 {
                return Err(w.refuse("stencil ref", v));
            }
            Command::SetStencilRef { front: v as u8, back: (v >> 8) as u8 }
        }
        Cmd::SetBlendColor => {
            w.exact(4)?;
            Command::SetBlendColor([w.f(1), w.f(2), w.f(3), w.f(4)])
        }
        Cmd::SetScissorState => {
            let n = w.elements(1, 2)?;
            let start_slot = w.u(1);
            w.slots(start_slot, n, MAX_VIEWPORTS)?;
            Command::SetScissorState {
                start_slot,
                scissors: (0..n).map(|s| scissor(w.u(2 + s * 2), w.u(3 + s * 2))).collect(),
            }
        }
        Cmd::Blit => blit(&w)?,
        Cmd::ResourceCopyRegion => {
            w.exact(13)?;
            Command::ResourceCopyRegion {
                dst: w.resource("destination", 1)?,
                dst_level: w.u(2),
                dst_x: w.u(3),
                dst_y: w.u(4),
                dst_z: w.u(5),
                src: w.resource("source", 6)?,
                src_level: w.u(7),
                src_region: w.region(8),
            }
        }
        Cmd::BindSamplerStates => {
            w.at_least(2)?;
            let stage = w.stage(1)?;
            let start_slot = w.u(2);
            let n = w.len() - 2;
            w.slots(start_slot, n, MAX_SAMPLERS)?;
            Command::BindSamplerStates {
                stage,
                start_slot,
                states: (0..n).map(|i| w.object_or_none(3 + i)).collect(),
            }
        }
        Cmd::BeginQuery => {
            w.exact(1)?;
            Command::BeginQuery(w.object("query", 1)?)
        }
        Cmd::EndQuery => {
            w.exact(1)?;
            Command::EndQuery(w.object("query", 1)?)
        }
        Cmd::GetQueryResult => {
            w.exact(2)?;
            Command::GetQueryResult { query: w.object("query", 1)?, wait: flag(&w, "wait", 2)? }
        }
        Cmd::SetPolygonStipple => {
            w.exact(32)?;
            Command::SetPolygonStipple(std::array::from_fn(|i| w.u(1 + i)))
        }
        Cmd::SetClipState => {
            w.exact(32)?;
            Command::SetClipState(std::array::from_fn(|p| {
                std::array::from_fn(|c| w.f(1 + p * 4 + c))
            }))
        }
        Cmd::SetSampleMask => {
            w.exact(1)?;
            Command::SetSampleMask(w.u(1))
        }
        Cmd::SetStreamoutTargets => {
            w.at_least(1)?;
            let n = w.len() - 1;
            if n > MAX_STREAMOUT_TARGETS {
                return Err(w.refuse("target count", n as u32));
            }
            Command::SetStreamoutTargets {
                append_bitmask: w.u(1),
                targets: (0..n).map(|i| w.object_or_none(2 + i)).collect(),
            }
        }
        Cmd::SetRenderCondition => {
            w.exact(3)?;
            Command::SetRenderCondition {
                query: w.object_or_none(1),
                condition: flag(&w, "condition", 2)?,
                mode: w.parse("mode", w.u(3), RenderCondMode::from_wire)?,
            }
        }
        Cmd::SetUniformBuffer => {
            w.exact(5)?;
            let index = w.u(2);
            if index as usize >= MAX_CONSTANT_BUFFERS {
                return Err(w.refuse("index", index));
            }
            Command::SetUniformBuffer {
                stage: w.stage(1)?,
                index,
                offset: w.u(3),
                length: w.u(4),
                resource: w.resource_or_none(5),
            }
        }
        Cmd::SetSubCtx => {
            w.exact(1)?;
            Command::SetSubCtx(SubContextId(w.u(1)))
        }
        Cmd::CreateSubCtx => {
            w.exact(1)?;
            Command::CreateSubCtx(SubContextId(w.u(1)))
        }
        Cmd::DestroySubCtx => {
            w.exact(1)?;
            Command::DestroySubCtx(SubContextId(w.u(1)))
        }
        Cmd::BindShader => {
            w.exact(2)?;
            Command::BindShader { handle: w.object_or_none(1), stage: w.stage(2)? }
        }
        Cmd::SetTessState => {
            w.exact(6)?;
            Command::SetTessState(std::array::from_fn(|i| w.f(1 + i)))
        }
        Cmd::SetMinSamples => {
            w.exact(1)?;
            Command::SetMinSamples(w.u(1))
        }
        Cmd::SetShaderBuffers => {
            let n = w.elements(2, 3)?;
            let stage = w.stage(1)?;
            let start_slot = w.u(2);
            if n > 0 {
                w.slots(start_slot, n, MAX_SHADER_BUFFERS)?;
            }
            Command::SetShaderBuffers {
                stage,
                start_slot,
                buffers: (0..n).map(|i| shader_buffer(&w, 3 + i * 3)).collect(),
            }
        }
        Cmd::SetShaderImages => {
            let n = w.elements(2, 5)?;
            let stage = w.stage(1)?;
            let start_slot = w.u(2);
            if n > 0 {
                w.slots(start_slot, n, MAX_SHADER_IMAGES)?;
            }
            let images = (0..n)
                .map(|i| -> Result<Option<ShaderImage>, Refused> {
                    let at = 3 + i * 5;
                    let Some(resource) = w.resource_or_none(at + 4) else {
                        // An unbind is all zeros; anything else in its dwords would not
                        // reproduce.
                        if w.w[at..at + 4].iter().any(|&d| d != 0) {
                            return Err(w.refuse("unbound image", w.u(at + 1)));
                        }
                        return Ok(None);
                    };
                    Ok(Some(ShaderImage {
                        format: w.format("image format", at)?,
                        access: w.parse("image access", w.u(at + 1), ImageAccess::from_wire)?,
                        layer_offset: w.u(at + 2),
                        level_size: w.u(at + 3),
                        resource,
                    }))
                })
                .collect::<Result<_, _>>()?;
            Command::SetShaderImages { stage, start_slot, images }
        }
        Cmd::MemoryBarrier => {
            w.exact(1)?;
            Command::MemoryBarrier(w.u(1))
        }
        Cmd::LaunchGrid => {
            w.exact(8)?;
            Command::LaunchGrid {
                block: [w.u(1), w.u(2), w.u(3)],
                grid: [w.u(4), w.u(5), w.u(6)],
                indirect: w.resource_or_none(7),
                indirect_offset: w.u(8),
            }
        }
        Cmd::SetFramebufferStateNoAttach => {
            w.exact(2)?;
            let wh = w.u(1);
            let ls = w.u(2);
            if ls >> 24 != 0 {
                return Err(w.refuse("samples", ls));
            }
            Command::SetFramebufferStateNoAttach {
                width: wh as u16,
                height: (wh >> 16) as u16,
                layers: ls as u16,
                samples: (ls >> 16) as u8,
            }
        }
        Cmd::TextureBarrier => {
            w.exact(1)?;
            Command::TextureBarrier(w.u(1))
        }
        Cmd::SetAtomicBuffers => {
            w.at_least(2)?;
            let n = w.elements(1, 3)?;
            let start_slot = w.u(1);
            if n > 0 {
                w.slots(start_slot, n, MAX_HW_ATOMIC_BUFFERS)?;
            }
            Command::SetAtomicBuffers {
                start_slot,
                buffers: (0..n).map(|i| shader_buffer(&w, 2 + i * 3)).collect(),
            }
        }
        Cmd::SetDebugFlags => {
            w.at_least(2)?;
            Command::SetDebugFlags(w.tail(1))
        }
        Cmd::GetQueryResultQbo => {
            w.exact(6)?;
            Command::GetQueryResultQbo {
                query: w.object("query", 1)?,
                buffer: w.resource("buffer", 2)?,
                wait: flag(&w, "wait", 3)?,
                result_type: w.parse("result type", w.u(4), QueryValueType::from_wire)?,
                offset: w.u(5),
                index: w.i(6),
            }
        }
        Cmd::Transfer3d => {
            w.exact(13)?;
            Command::Transfer3d {
                transfer: w.transfer()?,
                offset: w.u(12),
                direction: w.parse("direction", w.u(13), TransferDirection::from_wire)?,
            }
        }
        Cmd::EndTransfers => Command::EndTransfers(w.tail(1)),
        Cmd::CopyTransfer3d => {
            w.exact(14)?;
            let flags = w.u(14);
            if flags & !0b11 != 0 {
                return Err(w.refuse("flags", flags));
            }
            Command::CopyTransfer3d {
                direction: if bit(flags, 1) {
                    CopyDirection::FromHost
                } else {
                    CopyDirection::ToHost
                },
                transfer: w.transfer()?,
                staging: w.resource("staging", 12)?,
                staging_offset: w.u(13),
                synchronized: bit(flags, 0),
            }
        }
        Cmd::SetTweaks => {
            w.exact(2)?;
            Command::SetTweaks { id: w.u(1), value: w.u(2) }
        }
        Cmd::ClearTexture => {
            w.exact(12)?;
            Command::ClearTexture {
                resource: w.resource("texture", 1)?,
                level: w.u(2),
                region: w.region(3),
                data: [w.u(9), w.u(10), w.u(11), w.u(12)],
            }
        }
        Cmd::PipeResourceCreate => {
            w.exact(11)?;
            Command::PipeResourceCreate {
                target: w.parse("target", w.u(1), TextureTarget::from_wire)?,
                format: w.format("format", 2)?,
                bind: w.u(3),
                width: w.u(4),
                height: w.u(5),
                depth: w.u(6),
                array_size: w.u(7),
                last_level: w.u(8),
                nr_samples: w.u(9),
                flags: w.u(10),
                blob_id: BlobId(u64::from(w.u(11))),
            }
        }
        Cmd::PipeResourceSetType => {
            let n = w.elements(8, 2)?;
            if n == 0 || n > MAX_PLANES {
                return Err(w.refuse("plane count", n as u32));
            }
            Command::PipeResourceSetType {
                resource: w.resource("resource", 1)?,
                format: w.format("format", 2)?,
                bind: w.u(3),
                width: w.u(4),
                height: w.u(5),
                usage: w.u(6),
                modifier: u64::from(w.u(7)) | (u64::from(w.u(8)) << 32),
                planes: (0..n)
                    .map(|p| Plane { stride: w.u(9 + p * 2), offset: w.u(10 + p * 2) })
                    .collect(),
            }
        }
        Cmd::GetMemoryInfo => {
            w.exact(1)?;
            Command::GetMemoryInfo(w.resource("resource", 1)?)
        }
        Cmd::SendStringMarker => {
            w.at_least(2)?;
            let len = w.u(1);
            let text = w.tail(2);
            if len as usize > text.len() * 4 {
                return Err(w.refuse("string length", len));
            }
            Command::SendStringMarker { len, text }
        }
        Cmd::LinkShader => {
            w.exact(6)?;
            Command::LinkShader(std::array::from_fn(|s| w.object_or_none(1 + s)))
        }
        Cmd::CreateVideoCodec => {
            let max_references = match w.len() {
                7 => None,
                8 => Some(w.u(8)),
                _ => return Err(w.length()),
            };
            Command::CreateVideoCodec(VideoCodec {
                handle: VideoCodecHandle(w.u(1)),
                profile: w.u(2),
                entrypoint: w.u(3),
                chroma_format: w.u(4),
                level: w.u(5),
                width: w.u(6),
                height: w.u(7),
                max_references,
            })
        }
        Cmd::DestroyVideoCodec => {
            w.exact(1)?;
            Command::DestroyVideoCodec(VideoCodecHandle(w.u(1)))
        }
        Cmd::CreateVideoBuffer => {
            w.at_least(5)?;
            let n = w.len() - 4;
            if n > MAX_VIDEO_PLANES {
                return Err(w.refuse("plane count", n as u32));
            }
            Command::CreateVideoBuffer {
                handle: VideoBufferHandle(w.u(1)),
                format: w.u(2),
                width: w.u(3),
                height: w.u(4),
                planes: (0..n).map(|p| w.resource("plane", 5 + p)).collect::<Result<_, _>>()?,
            }
        }
        Cmd::DestroyVideoBuffer => {
            w.exact(1)?;
            Command::DestroyVideoBuffer(VideoBufferHandle(w.u(1)))
        }
        Cmd::BeginFrame => {
            w.exact(2)?;
            Command::BeginFrame {
                codec: VideoCodecHandle(w.u(1)),
                target: VideoBufferHandle(w.u(2)),
            }
        }
        Cmd::DecodeMacroblock => Command::DecodeMacroblock(w.tail(1)),
        Cmd::DecodeBitstream => {
            w.exact(5)?;
            Command::DecodeBitstream {
                codec: VideoCodecHandle(w.u(1)),
                target: VideoBufferHandle(w.u(2)),
                descriptor: w.resource("descriptor", 3)?,
                buffer: w.resource("bitstream", 4)?,
                buffer_size: w.u(5),
            }
        }
        Cmd::EncodeBitstream => {
            w.exact(5)?;
            Command::EncodeBitstream {
                codec: VideoCodecHandle(w.u(1)),
                source: VideoBufferHandle(w.u(2)),
                destination: w.resource("destination", 3)?,
                descriptor: w.resource("descriptor", 4)?,
                feedback: w.resource("feedback", 5)?,
            }
        }
        Cmd::EndFrame => {
            w.exact(2)?;
            Command::EndFrame { codec: VideoCodecHandle(w.u(1)), target: VideoBufferHandle(w.u(2)) }
        }
        Cmd::ClearSurface => {
            w.exact(10)?;
            let s0 = w.u(1);
            if s0 >> 4 != 0 {
                return Err(w.refuse("flags", s0));
            }
            Command::ClearSurface {
                render_condition_enable: bit(s0, 0),
                buffers: ((s0 >> 1) & 0x7) as u8,
                surface: w.object("surface", 2)?,
                color: [w.u(3), w.u(4), w.u(5), w.u(6)],
                dst_x: w.u(7),
                dst_y: w.u(8),
                width: w.u(9),
                height: w.u(10),
            }
        }
        Cmd::GetPipeResourceLayout => {
            w.exact(2)?;
            Command::GetPipeResourceLayout {
                out: w.resource("out", 1)?,
                target: w.resource("target", 2)?,
            }
        }
    })
}

/// A dword the C reads as `!!x`: only 0 and 1 reproduce, so only those are accepted.
fn flag(w: &Words, field: &'static str, i: usize) -> Result<bool, Refused> {
    match w.u(i) {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(w.refuse(field, other)),
    }
}

fn scissor(minxy: u32, maxxy: u32) -> Scissor {
    Scissor {
        minx: minxy as u16,
        miny: (minxy >> 16) as u16,
        maxx: maxxy as u16,
        maxy: (maxxy >> 16) as u16,
    }
}

fn shader_buffer(w: &Words, at: usize) -> ShaderBuffer {
    ShaderBuffer { offset: w.u(at), length: w.u(at + 1), resource: w.resource_or_none(at + 2) }
}

fn draw_vbo(w: &Words) -> Result<Command<'static>, Refused> {
    let (tess, indirect) = match w.len() {
        12 => (false, false),
        14 => (true, false),
        20 => (true, true),
        _ => return Err(w.length()),
    };
    let tess = tess.then(|| TessDraw { vertices_per_patch: w.u(13), drawid: w.u(14) });
    let indirect = if indirect {
        Some(IndirectDraw {
            resource: w.resource("indirect buffer", 15)?,
            offset: w.u(16),
            stride: w.u(17),
            draw_count: w.u(18),
            draw_count_offset: w.u(19),
            draw_count_resource: w.resource_or_none(20),
        })
    } else {
        None
    };
    Ok(Command::DrawVbo(Draw {
        start: w.u(1),
        count: w.u(2),
        mode: w.parse("mode", w.u(3), PrimType::from_wire)?,
        indexed: flag(w, "indexed", 4)?,
        instance_count: w.u(5),
        index_bias: w.i(6),
        start_instance: w.u(7),
        primitive_restart: flag(w, "primitive restart", 8)?,
        restart_index: w.u(9),
        min_index: w.u(10),
        max_index: w.u(11),
        count_from_so: w.object_or_none(12),
        tess,
        indirect,
    }))
}

fn blit(w: &Words) -> Result<Command<'static>, Refused> {
    w.exact(21)?;
    let s0 = w.u(1);
    if s0 >> 13 != 0 {
        return Err(w.refuse("flags", s0));
    }
    let target = |name, at: usize| -> Result<BlitTarget, Refused> {
        Ok(BlitTarget {
            resource: w.resource(name, at)?,
            level: w.u(at + 1),
            format: w.format("format", at + 2)?,
            region: w.region(at + 3),
        })
    };
    Ok(Command::Blit(Blit {
        mask: s0 as u8,
        filter: w.parse("filter", (s0 >> 8) & 0x3, TexFilter::from_wire)?,
        scissor_enable: bit(s0, 10),
        render_condition_enable: bit(s0, 11),
        alpha_blend: bit(s0, 12),
        scissor: scissor(w.u(2), w.u(3)),
        dst: target("destination", 4)?,
        src: target("source", 13)?,
    }))
}

fn create_object<'a>(w: &Words<'a>, obj: u32) -> Result<Command<'a>, Refused> {
    w.at_least(1)?;
    let kind = w.parse("object type", obj, ObjectType::from_wire)?;
    let handle = w.object("handle", 1)?;
    let object = match kind {
        ObjectType::Blend => Object::Blend(blend(w)?),
        ObjectType::Dsa => Object::Dsa(dsa(w)?),
        ObjectType::Rasterizer => Object::Rasterizer(rasterizer(w)?),
        ObjectType::Shader => Object::Shader(shader(w)?),
        ObjectType::VertexElements => {
            let n = w.elements(1, 4)?;
            if n > MAX_ATTRIBS {
                return Err(w.refuse("element count", n as u32));
            }
            let elements = (0..n)
                .map(|i| -> Result<VertexElement, Refused> {
                    let at = 2 + i * 4;
                    let vertex_buffer_index = w.u(at + 2);
                    if vertex_buffer_index as usize >= MAX_ATTRIBS {
                        return Err(w.refuse("vertex buffer index", vertex_buffer_index));
                    }
                    Ok(VertexElement {
                        src_offset: w.u(at),
                        instance_divisor: w.u(at + 1),
                        vertex_buffer_index,
                        src_format: w.format("source format", at + 3)?,
                    })
                })
                .collect::<Result<_, _>>()?;
            Object::VertexElements(elements)
        }
        ObjectType::SamplerView => {
            w.exact(6)?;
            let fd = w.u(3);
            let format = w.parse("format", fd & 0xff_ffff, Format::from_wire)?;
            if format == Format::NONE {
                return Err(w.refuse("format", fd));
            }
            let sw = w.u(6);
            if sw >> 12 != 0 {
                return Err(w.refuse("swizzle", sw));
            }
            let mut swizzle = [Swizzle::X; 4];
            for (i, s) in swizzle.iter_mut().enumerate() {
                *s = w.parse("swizzle", (sw >> (3 * i)) & 0x7, Swizzle::from_wire)?;
            }
            Object::SamplerView(SamplerView {
                resource: w.resource("resource", 2)?,
                format,
                target: w.parse("target", fd >> 24, TextureTarget::from_wire)?,
                first_element_or_layers: w.u(4),
                last_element_or_levels: w.u(5),
                swizzle,
            })
        }
        ObjectType::SamplerState => Object::SamplerState(sampler_state(w)?),
        ObjectType::Surface => {
            w.exact(5)?;
            Object::Surface(surface(w, 0)?)
        }
        ObjectType::MsaaSurface => {
            w.exact(6)?;
            let samples = w.u(6);
            if samples == 0 {
                return Err(w.refuse("sample count", samples));
            }
            Object::Surface(surface(w, samples)?)
        }
        ObjectType::Query => {
            w.exact(4)?;
            let ti = w.u(2);
            Object::Query(QueryCreate {
                kind: w.parse("query type", ti & 0xffff, QueryType::from_wire)?,
                index: (ti >> 16) as u16,
                offset: w.u(3),
                resource: w.resource("resource", 4)?,
            })
        }
        ObjectType::StreamoutTarget => {
            w.exact(4)?;
            Object::StreamoutTarget(StreamoutTarget {
                resource: w.resource("resource", 2)?,
                buffer_offset: w.u(3),
                buffer_size: w.u(4),
            })
        }
    };
    Ok(Command::CreateObject { handle, object })
}

fn blend(w: &Words) -> Result<BlendState, Refused> {
    w.exact(11)?;
    let s0 = w.u(2);
    if s0 >> 5 != 0 {
        return Err(w.refuse("blend flags", s0));
    }
    let s1 = w.u(3);
    if s1 >> 4 != 0 {
        return Err(w.refuse("logic op", s1));
    }
    let mut rt = [RtBlend { equation: None, colormask: 0 }; 8];
    for (i, target) in rt.iter_mut().enumerate() {
        let s2 = w.u(4 + i);
        if s2 >> 31 != 0 {
            return Err(w.refuse("render target blend", s2));
        }
        let colormask = ((s2 >> 27) & 0xf) as u8;
        let equation = if bit(s2, 0) {
            let eq = |func_at: u32, src_at: u32, dst_at: u32| -> Result<BlendEq, Refused> {
                Ok(BlendEq {
                    func: w.parse("blend func", (s2 >> func_at) & 0x7, BlendFunc::from_wire)?,
                    src: w.parse("blend factor", (s2 >> src_at) & 0x1f, BlendFactor::from_wire)?,
                    dst: w.parse("blend factor", (s2 >> dst_at) & 0x1f, BlendFactor::from_wire)?,
                })
            };
            Some(RtBlendEq { rgb: eq(1, 4, 9)?, alpha: eq(14, 17, 22)? })
        } else {
            if s2 & 0x07ff_fffe != 0 {
                return Err(w.refuse("disabled render target blend", s2));
            }
            None
        };
        *target = RtBlend { equation, colormask };
    }
    Ok(BlendState {
        independent_blend_enable: bit(s0, 0),
        logicop_enable: bit(s0, 1),
        dither: bit(s0, 2),
        alpha_to_coverage: bit(s0, 3),
        alpha_to_one: bit(s0, 4),
        logicop_func: w.parse("logic op", s1, LogicOp::from_wire)?,
        rt,
    })
}

fn dsa(w: &Words) -> Result<DepthStencilAlpha, Refused> {
    w.exact(5)?;
    let s0 = w.u(2);
    if s0 & !0x0f1f != 0 {
        return Err(w.refuse("depth/alpha flags", s0));
    }
    let face = |i: usize| -> Result<StencilFace, Refused> {
        let s = w.u(3 + i);
        if s >> 29 != 0 {
            return Err(w.refuse("stencil flags", s));
        }
        Ok(StencilFace {
            enabled: bit(s, 0),
            func: w.parse("stencil func", (s >> 1) & 0x7, CompareFunc::from_wire)?,
            fail_op: w.parse("stencil op", (s >> 4) & 0x7, StencilOp::from_wire)?,
            zpass_op: w.parse("stencil op", (s >> 7) & 0x7, StencilOp::from_wire)?,
            zfail_op: w.parse("stencil op", (s >> 10) & 0x7, StencilOp::from_wire)?,
            valuemask: ((s >> 13) & 0xff) as u8,
            writemask: ((s >> 21) & 0xff) as u8,
        })
    };
    Ok(DepthStencilAlpha {
        depth: DepthState {
            enabled: bit(s0, 0),
            writemask: bit(s0, 1),
            func: w.parse("depth func", (s0 >> 2) & 0x7, CompareFunc::from_wire)?,
        },
        alpha: AlphaState {
            enabled: bit(s0, 8),
            func: w.parse("alpha func", (s0 >> 9) & 0x7, CompareFunc::from_wire)?,
            ref_value: w.f(5),
        },
        stencil: [face(0)?, face(1)?],
    })
}

fn rasterizer(w: &Words) -> Result<RasterizerState, Refused> {
    w.exact(9)?;
    let s0 = w.u(2);
    let s3 = w.u(5);
    Ok(RasterizerState {
        flatshade: bit(s0, 0),
        depth_clip: bit(s0, 1),
        clip_halfz: bit(s0, 2),
        rasterizer_discard: bit(s0, 3),
        flatshade_first: bit(s0, 4),
        light_twoside: bit(s0, 5),
        sprite_coord_mode: bit(s0, 6),
        point_quad_rasterization: bit(s0, 7),
        cull_face: w.parse("cull face", (s0 >> 8) & 0x3, CullFace::from_wire)?,
        fill_front: w.parse("fill mode", (s0 >> 10) & 0x3, FillMode::from_wire)?,
        fill_back: w.parse("fill mode", (s0 >> 12) & 0x3, FillMode::from_wire)?,
        scissor: bit(s0, 14),
        front_ccw: bit(s0, 15),
        clamp_vertex_color: bit(s0, 16),
        clamp_fragment_color: bit(s0, 17),
        offset_line: bit(s0, 18),
        offset_point: bit(s0, 19),
        offset_tri: bit(s0, 20),
        poly_smooth: bit(s0, 21),
        poly_stipple_enable: bit(s0, 22),
        point_smooth: bit(s0, 23),
        point_size_per_vertex: bit(s0, 24),
        multisample: bit(s0, 25),
        line_smooth: bit(s0, 26),
        line_stipple_enable: bit(s0, 27),
        line_last_pixel: bit(s0, 28),
        half_pixel_center: bit(s0, 29),
        bottom_edge_rule: bit(s0, 30),
        force_persample_interp: bit(s0, 31),
        point_size: w.f(3),
        sprite_coord_enable: w.u(4),
        line_stipple_pattern: s3 as u16,
        line_stipple_factor: (s3 >> 16) as u8,
        clip_plane_enable: (s3 >> 24) as u8,
        line_width: w.f(6),
        offset_units: w.f(7),
        offset_scale: w.f(8),
        offset_clamp: w.f(9),
    })
}

fn sampler_state(w: &Words) -> Result<SamplerState, Refused> {
    w.exact(9)?;
    let s0 = w.u(2);
    if s0 >> 25 != 0 {
        return Err(w.refuse("sampler flags", s0));
    }
    Ok(SamplerState {
        wrap_s: w.parse("wrap", s0 & 0x7, TexWrap::from_wire)?,
        wrap_t: w.parse("wrap", (s0 >> 3) & 0x7, TexWrap::from_wire)?,
        wrap_r: w.parse("wrap", (s0 >> 6) & 0x7, TexWrap::from_wire)?,
        min_img_filter: w.parse("filter", (s0 >> 9) & 0x1, TexFilter::from_wire)?,
        min_mip_filter: w.parse("mip filter", (s0 >> 11) & 0x3, MipFilter::from_wire)?,
        mag_img_filter: w.parse("filter", (s0 >> 13) & 0x1, TexFilter::from_wire)?,
        compare_mode: bit(s0, 15),
        compare_func: w.parse("compare func", (s0 >> 16) & 0x7, CompareFunc::from_wire)?,
        seamless_cube_map: bit(s0, 19),
        max_anisotropy: ((s0 >> 20) & 0x1f) as u8,
        lod_bias: w.f(3),
        min_lod: w.f(4),
        max_lod: w.f(5),
        border_color: [w.u(6), w.u(7), w.u(8), w.u(9)],
    })
}

fn surface(w: &Words, samples: u32) -> Result<Surface, Refused> {
    let val1 = w.u(5);
    // The C's rule, applied to every surface: the high half of dword 5 must not be below the low
    // half. For a texture that is "last layer >= first layer"; for a buffer it is a check on the
    // halves of the last element, which is meaningless but is what the scores were recorded
    // through.
    if (val1 >> 16) < (val1 & 0xffff) {
        return Err(w.refuse("layers", val1));
    }
    Ok(Surface {
        resource: w.resource("resource", 2)?,
        format: w.format("format", 3)?,
        first_element_or_level: w.u(4),
        last_element_or_layers: val1,
        samples,
    })
}

fn shader<'a>(w: &Words<'a>) -> Result<ShaderCreate<'a>, Refused> {
    w.at_least(5)?;
    let stage = w.stage(2)?;
    let offlen = w.u(3);
    let chunk = if offlen >> 31 != 0 {
        ShaderChunk::Continuation { offset: offlen & 0x7fff_ffff }
    } else {
        ShaderChunk::New { total_bytes: offlen }
    };
    let num_tokens = w.u(4);
    let (kind, text_at) = if stage == ShaderStage::Compute {
        (ShaderKind::Compute { req_local_mem: w.u(5) }, 6)
    } else {
        let nso = w.u(5) as usize;
        if nso > MAX_SO_OUTPUTS {
            return Err(w.refuse("stream output count", nso as u32));
        }
        let mut stream_output = StreamOutput::default();
        if nso > 0 {
            w.at_least(5 + 4 + 2 * nso)?;
            stream_output.stride = [w.u(6), w.u(7), w.u(8), w.u(9)];
            for i in 0..nso {
                let o = w.u(10 + i * 2);
                let s = w.u(11 + i * 2);
                if s >> 2 != 0 {
                    return Err(w.refuse("stream", s));
                }
                stream_output.outputs.push(SoOutput {
                    register_index: o as u8,
                    start_component: ((o >> 8) & 0x3) as u8,
                    num_components: ((o >> 10) & 0x7) as u8,
                    output_buffer: w.parse(
                        "output buffer",
                        (o >> 13) & 0x7,
                        SoBuffer::from_wire,
                    )?,
                    dst_offset: (o >> 16) as u16,
                    stream: s as u8,
                });
            }
        }
        (ShaderKind::Graphics { stream_output }, 6 + if nso > 0 { 4 + 2 * nso } else { 0 })
    };
    Ok(ShaderCreate { stage, chunk, num_tokens, kind, text: w.tail(text_at) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vrend::encode::encode;

    fn r(n: u32) -> ResourceHandle {
        ResourceHandle::new(n).unwrap()
    }

    fn o(n: u32) -> ObjectHandle {
        ObjectHandle::new(n).unwrap()
    }

    fn fmt(n: u32) -> Format {
        Format::from_wire(n).unwrap()
    }

    fn region() -> Box3 {
        Box3 { x: 1, y: 2, z: 3, width: -4, height: 5, depth: 6 }
    }

    fn transfer() -> Transfer {
        Transfer {
            resource: r(7),
            level: 1,
            usage: 2,
            stride: 640,
            layer_stride: 0,
            region: region(),
        }
    }

    fn blend_eq() -> BlendEq {
        BlendEq { func: BlendFunc::Max, src: BlendFactor::InvSrc1Alpha, dst: BlendFactor::One }
    }

    fn stencil_face(enabled: bool) -> StencilFace {
        StencilFace {
            enabled,
            func: CompareFunc::GreaterEqual,
            fail_op: StencilOp::IncrWrap,
            zpass_op: StencilOp::Invert,
            zfail_op: StencilOp::Replace,
            valuemask: 0xa5,
            writemask: 0x5a,
        }
    }

    fn shader<'a>(stage: ShaderStage, text: &'a [u32]) -> ShaderCreate<'a> {
        let kind = if stage == ShaderStage::Compute {
            ShaderKind::Compute { req_local_mem: 4096 }
        } else {
            ShaderKind::Graphics {
                stream_output: StreamOutput {
                    stride: [16, 0, 8, 0],
                    outputs: vec![
                        SoOutput {
                            register_index: 3,
                            start_component: 1,
                            num_components: 2,
                            output_buffer: SoBuffer::from_wire(2).unwrap(),
                            dst_offset: 4,
                            stream: 1,
                        },
                        SoOutput {
                            register_index: 0,
                            start_component: 0,
                            num_components: 4,
                            output_buffer: SoBuffer::from_wire(0).unwrap(),
                            dst_offset: 0,
                            stream: 0,
                        },
                    ],
                },
            }
        };
        ShaderCreate {
            stage,
            chunk: ShaderChunk::New { total_bytes: text.len() as u32 * 4 },
            num_tokens: 12,
            kind,
            text,
        }
    }

    /// One of every command, with values that would show a swapped field.
    fn one_of_each<'a>(text: &'a [u32], slack: &'a [u32]) -> Vec<Command<'a>> {
        let stages = [ShaderStage::Vertex, ShaderStage::Fragment, ShaderStage::Compute];
        let mut rt = [RtBlend { equation: None, colormask: 0xf }; 8];
        rt[0].equation = Some(RtBlendEq { rgb: blend_eq(), alpha: blend_eq() });
        rt[3].colormask = 0x5;
        let mut cmds = vec![
            Command::Nop,
            Command::CreateObject {
                handle: o(1),
                object: Object::Blend(BlendState {
                    independent_blend_enable: true,
                    logicop_enable: false,
                    dither: true,
                    alpha_to_coverage: false,
                    alpha_to_one: true,
                    logicop_func: LogicOp::Xor,
                    rt,
                }),
            },
            Command::CreateObject {
                handle: o(2),
                object: Object::Dsa(DepthStencilAlpha {
                    depth: DepthState { enabled: true, writemask: false, func: CompareFunc::Less },
                    alpha: AlphaState {
                        enabled: true,
                        func: CompareFunc::NotEqual,
                        ref_value: 0.5,
                    },
                    stencil: [stencil_face(true), stencil_face(false)],
                }),
            },
            Command::CreateObject {
                handle: o(3),
                object: Object::Rasterizer(RasterizerState {
                    flatshade: true,
                    depth_clip: false,
                    clip_halfz: true,
                    rasterizer_discard: false,
                    flatshade_first: true,
                    light_twoside: false,
                    sprite_coord_mode: true,
                    point_quad_rasterization: false,
                    cull_face: CullFace::Back,
                    fill_front: FillMode::Line,
                    fill_back: FillMode::Point,
                    scissor: true,
                    front_ccw: false,
                    clamp_vertex_color: true,
                    clamp_fragment_color: false,
                    offset_line: true,
                    offset_point: false,
                    offset_tri: true,
                    poly_smooth: false,
                    poly_stipple_enable: true,
                    point_smooth: false,
                    point_size_per_vertex: true,
                    multisample: false,
                    line_smooth: true,
                    line_stipple_enable: false,
                    line_last_pixel: true,
                    half_pixel_center: false,
                    bottom_edge_rule: true,
                    force_persample_interp: true,
                    point_size: 2.5,
                    sprite_coord_enable: 0x81,
                    line_stipple_pattern: 0xf0f0,
                    line_stipple_factor: 3,
                    clip_plane_enable: 0x21,
                    line_width: 1.5,
                    offset_units: -1.0,
                    offset_scale: 2.0,
                    offset_clamp: 0.25,
                }),
            },
            Command::CreateObject {
                handle: o(5),
                object: Object::VertexElements(vec![
                    VertexElement {
                        src_offset: 12,
                        instance_divisor: 1,
                        vertex_buffer_index: 2,
                        src_format: fmt(67),
                    },
                    VertexElement {
                        src_offset: 0,
                        instance_divisor: 0,
                        vertex_buffer_index: 31,
                        src_format: fmt(1),
                    },
                ]),
            },
            Command::CreateObject {
                handle: o(6),
                object: Object::SamplerView(SamplerView {
                    resource: r(9),
                    format: fmt(67),
                    target: TextureTarget::Array2d,
                    first_element_or_layers: 0x0003_0001,
                    last_element_or_levels: 0x0002_0000,
                    swizzle: [Swizzle::Z, Swizzle::One, Swizzle::X, Swizzle::Zero],
                }),
            },
            Command::CreateObject {
                handle: o(7),
                object: Object::SamplerState(SamplerState {
                    wrap_s: TexWrap::MirrorClampToBorder,
                    wrap_t: TexWrap::ClampToEdge,
                    wrap_r: TexWrap::Repeat,
                    min_img_filter: TexFilter::Linear,
                    min_mip_filter: MipFilter::None,
                    mag_img_filter: TexFilter::Nearest,
                    compare_mode: true,
                    compare_func: CompareFunc::LessEqual,
                    seamless_cube_map: true,
                    max_anisotropy: 16,
                    lod_bias: -0.5,
                    min_lod: 1.0,
                    max_lod: 7.0,
                    border_color: [1, 2, 3, 4],
                }),
            },
            Command::CreateObject {
                handle: o(8),
                object: Object::Surface(Surface {
                    resource: r(9),
                    format: fmt(20),
                    first_element_or_level: 2,
                    last_element_or_layers: 0x0005_0003,
                    samples: 0,
                }),
            },
            Command::CreateObject {
                handle: o(11),
                object: Object::Surface(Surface {
                    resource: r(9),
                    format: fmt(20),
                    first_element_or_level: 0,
                    last_element_or_layers: 0,
                    samples: 4,
                }),
            },
            Command::CreateObject {
                handle: o(9),
                object: Object::Query(QueryCreate {
                    kind: QueryType::PipelineStatistics,
                    index: 5,
                    offset: 64,
                    resource: r(12),
                }),
            },
            Command::CreateObject {
                handle: o(10),
                object: Object::StreamoutTarget(StreamoutTarget {
                    resource: r(13),
                    buffer_offset: 256,
                    buffer_size: 1024,
                }),
            },
            Command::BindObject { kind: ObjectType::Rasterizer, handle: Some(o(3)) },
            Command::BindObject { kind: ObjectType::Blend, handle: None },
            Command::DestroyObject { kind: ObjectType::SamplerView, handle: o(6) },
            Command::SetViewportState {
                start_slot: 1,
                viewports: vec![
                    Viewport { scale: [320.0, -240.0, 0.5], translate: [320.0, 240.0, 0.5] },
                    Viewport { scale: [1.0, 2.0, 3.0], translate: [4.0, 5.0, 6.0] },
                ],
            },
            Command::SetFramebufferState { zsurf: Some(o(8)), cbufs: vec![None, Some(o(11))] },
            Command::SetFramebufferState { zsurf: None, cbufs: vec![] },
            Command::SetVertexBuffers(vec![
                VertexBuffer { stride: 16, offset: 32, resource: Some(r(7)) },
                VertexBuffer { stride: 0, offset: 0, resource: None },
            ]),
            Command::SetVertexBuffers(vec![]),
            Command::Clear { buffers: 0x7, color: [1, 2, 3, 4], depth: 0.75, stencil: 0x80 },
            Command::DrawVbo(Draw {
                start: 3,
                count: 60,
                mode: PrimType::TriangleStrip,
                indexed: true,
                instance_count: 2,
                index_bias: -5,
                start_instance: 1,
                primitive_restart: true,
                restart_index: 0xffff,
                min_index: 0,
                max_index: 99,
                count_from_so: Some(o(10)),
                tess: None,
                indirect: None,
            }),
            Command::DrawVbo(Draw {
                start: 0,
                count: 9,
                mode: PrimType::Patches,
                indexed: false,
                instance_count: 1,
                index_bias: 0,
                start_instance: 0,
                primitive_restart: false,
                restart_index: 0,
                min_index: 0,
                max_index: 8,
                count_from_so: None,
                tess: Some(TessDraw { vertices_per_patch: 3, drawid: 7 }),
                indirect: None,
            }),
            Command::DrawVbo(Draw {
                start: 0,
                count: 0,
                mode: PrimType::Points,
                indexed: false,
                instance_count: 0,
                index_bias: 0,
                start_instance: 0,
                primitive_restart: false,
                restart_index: 0,
                min_index: 0,
                max_index: 0,
                count_from_so: None,
                tess: Some(TessDraw { vertices_per_patch: 0, drawid: 0 }),
                indirect: Some(IndirectDraw {
                    resource: r(20),
                    offset: 16,
                    stride: 20,
                    draw_count: 4,
                    draw_count_offset: 8,
                    draw_count_resource: Some(r(21)),
                }),
            }),
            Command::ResourceInlineWrite { transfer: transfer(), data: text },
            Command::SetSamplerViews {
                stage: ShaderStage::Fragment,
                start_slot: 2,
                views: vec![Some(o(6)), None, Some(o(6))],
            },
            Command::SetIndexBuffer(None),
            Command::SetIndexBuffer(Some(IndexBuffer {
                resource: r(7),
                index_type: IndexType::U16,
                offset: 8,
            })),
            Command::SetConstantBuffer { stage: ShaderStage::Vertex, index: 0, data: text },
            Command::SetConstantBuffer { stage: ShaderStage::Geometry, index: 1, data: &[] },
            Command::SetStencilRef { front: 0x12, back: 0x34 },
            Command::SetBlendColor([0.1, 0.2, 0.3, 0.4]),
            Command::SetScissorState {
                start_slot: 0,
                scissors: vec![Scissor { minx: 1, miny: 2, maxx: 3, maxy: 4 }],
            },
            Command::Blit(Blit {
                mask: 0xf,
                filter: TexFilter::Linear,
                scissor_enable: true,
                render_condition_enable: false,
                alpha_blend: true,
                scissor: Scissor { minx: 5, miny: 6, maxx: 7, maxy: 8 },
                dst: BlitTarget { resource: r(30), level: 1, format: fmt(67), region: region() },
                src: BlitTarget { resource: r(31), level: 0, format: fmt(1), region: region() },
            }),
            Command::ResourceCopyRegion {
                dst: r(30),
                dst_level: 1,
                dst_x: 2,
                dst_y: 3,
                dst_z: 4,
                src: r(31),
                src_level: 5,
                src_region: region(),
            },
            Command::BindSamplerStates {
                stage: ShaderStage::Fragment,
                start_slot: 1,
                states: vec![Some(o(7)), None],
            },
            Command::BeginQuery(o(9)),
            Command::EndQuery(o(9)),
            Command::GetQueryResult { query: o(9), wait: true },
            Command::SetPolygonStipple(std::array::from_fn(|i| i as u32 * 0x0101_0101)),
            Command::SetClipState(std::array::from_fn(|p| {
                std::array::from_fn(|c| (p * 4 + c) as f32 * 0.5)
            })),
            Command::SetSampleMask(0xff),
            Command::SetStreamoutTargets { append_bitmask: 0x1, targets: vec![Some(o(10)), None] },
            Command::SetRenderCondition {
                query: Some(o(9)),
                condition: true,
                mode: RenderCondMode::ByRegionNoWait,
            },
            Command::SetUniformBuffer {
                stage: ShaderStage::TessEval,
                index: 3,
                offset: 256,
                length: 512,
                resource: Some(r(7)),
            },
            Command::SetSubCtx(SubContextId(2)),
            Command::CreateSubCtx(SubContextId(3)),
            Command::DestroySubCtx(SubContextId(3)),
            Command::BindShader { handle: Some(o(4)), stage: ShaderStage::Fragment },
            Command::SetTessState([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            Command::SetMinSamples(2),
            Command::SetShaderBuffers {
                stage: ShaderStage::Compute,
                start_slot: 1,
                buffers: vec![ShaderBuffer { offset: 0, length: 64, resource: Some(r(40)) }],
            },
            Command::SetShaderBuffers {
                stage: ShaderStage::Compute,
                start_slot: 0,
                buffers: vec![],
            },
            Command::SetShaderImages {
                stage: ShaderStage::Fragment,
                start_slot: 0,
                images: vec![
                    Some(ShaderImage {
                        format: fmt(67),
                        access: ImageAccess::ReadWrite,
                        layer_offset: 1,
                        level_size: 2,
                        resource: r(9),
                    }),
                    None,
                ],
            },
            Command::MemoryBarrier(0x3),
            Command::LaunchGrid {
                block: [8, 8, 1],
                grid: [4, 2, 1],
                indirect: Some(r(20)),
                indirect_offset: 12,
            },
            Command::SetFramebufferStateNoAttach { width: 640, height: 480, layers: 2, samples: 4 },
            Command::TextureBarrier(0x1),
            Command::SetAtomicBuffers {
                start_slot: 2,
                buffers: vec![ShaderBuffer { offset: 4, length: 8, resource: None }],
            },
            Command::SetDebugFlags(text),
            Command::GetQueryResultQbo {
                query: o(9),
                buffer: r(12),
                wait: false,
                result_type: QueryValueType::U64,
                offset: 16,
                index: -1,
            },
            Command::Transfer3d {
                transfer: transfer(),
                offset: 4096,
                direction: TransferDirection::FromHost,
            },
            Command::EndTransfers(slack),
            Command::CopyTransfer3d {
                direction: CopyDirection::FromHost,
                transfer: transfer(),
                staging: r(50),
                staging_offset: 128,
                synchronized: true,
            },
            Command::CopyTransfer3d {
                direction: CopyDirection::ToHost,
                transfer: transfer(),
                staging: r(50),
                staging_offset: 0,
                synchronized: false,
            },
            Command::SetTweaks { id: 1, value: 2 },
            Command::ClearTexture {
                resource: r(9),
                level: 1,
                region: region(),
                data: [5, 6, 7, 8],
            },
            Command::PipeResourceCreate {
                target: TextureTarget::Texture2d,
                format: fmt(67),
                bind: 0x2,
                width: 64,
                height: 32,
                depth: 1,
                array_size: 1,
                last_level: 0,
                nr_samples: 0,
                flags: 0,
                blob_id: BlobId(77),
            },
            Command::PipeResourceSetType {
                resource: r(60),
                format: fmt(67),
                bind: 0x2,
                width: 64,
                height: 32,
                usage: 1,
                modifier: 0x0100_0000_0000_0005,
                planes: vec![Plane { stride: 256, offset: 0 }, Plane { stride: 128, offset: 8192 }],
            },
            Command::GetMemoryInfo(r(61)),
            Command::SendStringMarker { len: 5, text: &[0x6c6c_6548, 0x6f] },
            Command::LinkShader([Some(o(4)), Some(o(14)), None, None, None, None]),
            Command::CreateVideoCodec(VideoCodec {
                handle: VideoCodecHandle(1),
                profile: 2,
                entrypoint: 3,
                chroma_format: 4,
                level: 5,
                width: 6,
                height: 7,
                max_references: Some(8),
            }),
            Command::CreateVideoCodec(VideoCodec {
                handle: VideoCodecHandle(1),
                profile: 2,
                entrypoint: 3,
                chroma_format: 4,
                level: 5,
                width: 6,
                height: 7,
                max_references: None,
            }),
            Command::DestroyVideoCodec(VideoCodecHandle(1)),
            Command::CreateVideoBuffer {
                handle: VideoBufferHandle(2),
                format: 3,
                width: 4,
                height: 5,
                planes: vec![r(70), r(71)],
            },
            Command::DestroyVideoBuffer(VideoBufferHandle(2)),
            Command::BeginFrame { codec: VideoCodecHandle(1), target: VideoBufferHandle(2) },
            Command::DecodeMacroblock(slack),
            Command::DecodeBitstream {
                codec: VideoCodecHandle(1),
                target: VideoBufferHandle(2),
                descriptor: r(72),
                buffer: r(73),
                buffer_size: 4096,
            },
            Command::EncodeBitstream {
                codec: VideoCodecHandle(1),
                source: VideoBufferHandle(2),
                destination: r(74),
                descriptor: r(75),
                feedback: r(76),
            },
            Command::EndFrame { codec: VideoCodecHandle(1), target: VideoBufferHandle(2) },
            Command::ClearSurface {
                render_condition_enable: true,
                buffers: 0x5,
                surface: o(8),
                color: [9, 8, 7, 6],
                dst_x: 1,
                dst_y: 2,
                width: 3,
                height: 4,
            },
            Command::GetPipeResourceLayout { out: r(80), target: r(81) },
        ];
        for stage in stages {
            cmds.push(Command::CreateObject {
                handle: o(4),
                object: Object::Shader(shader(stage, text)),
            });
        }
        cmds.push(Command::CreateObject {
            handle: o(4),
            object: Object::Shader(ShaderCreate {
                stage: ShaderStage::Vertex,
                chunk: ShaderChunk::Continuation { offset: 4000 },
                num_tokens: 0,
                kind: ShaderKind::Graphics { stream_output: StreamOutput::default() },
                text,
            }),
        });
        cmds
    }

    #[test]
    fn every_command_shape_round_trips() {
        let text = [0x4c47_5354, 0x3120, 0x00];
        let slack = [0xdead_beef; 5];
        let cmds = one_of_each(&text, &slack);
        let kinds: std::collections::BTreeSet<u32> = cmds.iter().map(|c| c.kind().wire()).collect();
        assert_eq!(kinds.len(), Cmd::ALL.len(), "one of every command");

        let mut wire = Vec::new();
        for c in &cmds {
            encode(c, &mut wire);
        }
        let decoded: Vec<Command> =
            Batch::new(&wire).map(|c| c.expect("a command we wrote")).collect();
        assert_eq!(decoded, cmds);

        let mut again = Vec::new();
        for c in &decoded {
            encode(c, &mut again);
        }
        assert_eq!(again, wire);
    }

    /// The layouts, from the header's defines rather than from the encoder, so that an encoder
    /// and decoder agreeing on the wrong bit cannot pass. Every field is written at its shift,
    /// zero ones included, so that the word reads as the define.
    #[test]
    #[allow(clippy::identity_op)]
    fn the_layout_is_the_headers() {
        // VIRGL_OBJ_BLEND_S2: enable<<0, rgb_func<<1, rgb_src<<4, rgb_dst<<9, alpha_func<<14,
        // alpha_src<<17, alpha_dst<<22, colormask<<27.
        let s2 =
            1 | (4 << 1) | (26 << 4) | (1 << 9) | (3 << 14) | (17 << 17) | (2 << 22) | (0xa << 27);
        let mut w = vec![0x0000_0101 | (11 << 16), 1, 0b10101, 6, s2];
        w.extend([0x8 << 27; 7]);
        let Command::CreateObject { handle, object: Object::Blend(b) } =
            decode(Cmd::CreateObject, 1, &w).unwrap()
        else {
            panic!()
        };
        assert_eq!(handle, o(1));
        assert!(b.independent_blend_enable && b.dither && b.alpha_to_one);
        assert!(!b.logicop_enable && !b.alpha_to_coverage);
        assert_eq!(b.logicop_func, LogicOp::Xor);
        let eq = b.rt[0].equation.unwrap();
        assert_eq!(
            eq.rgb,
            BlendEq { func: BlendFunc::Max, src: BlendFactor::InvSrc1Alpha, dst: BlendFactor::One }
        );
        assert_eq!(
            eq.alpha,
            BlendEq { func: BlendFunc::Min, src: BlendFactor::Zero, dst: BlendFactor::SrcColor }
        );
        assert_eq!(b.rt[0].colormask, 0xa);
        assert_eq!(b.rt[7], RtBlend { equation: None, colormask: 0x8 });

        // VIRGL_OBJ_DSA: S0 depth enable<<0, writemask<<1, func<<2, alpha enable<<8, func<<9;
        // S1/S2 stencil enable<<0, func<<1, fail<<4, zpass<<7, zfail<<10, valuemask<<13,
        // writemask<<21; then the alpha ref as a float.
        let s0 = 1 | (0 << 1) | (1 << 2) | (1 << 8) | (5 << 9);
        let s1 = 1 | (6 << 1) | (5 << 4) | (7 << 7) | (2 << 10) | (0xa5 << 13) | (0x5a << 21);
        let w = [0x0000_0301 | (5 << 16), 2, s0, s1, 0, 0.5f32.to_bits()];
        let Command::CreateObject { object: Object::Dsa(d), .. } =
            decode(Cmd::CreateObject, 3, &w).unwrap()
        else {
            panic!()
        };
        assert_eq!(
            d.depth,
            DepthState { enabled: true, writemask: false, func: CompareFunc::Less }
        );
        assert_eq!(
            d.alpha,
            AlphaState { enabled: true, func: CompareFunc::NotEqual, ref_value: 0.5 }
        );
        assert_eq!(d.stencil[0], stencil_face(true));
        assert!(!d.stencil[1].enabled);

        // VIRGL_OBJ_SAMPLER_STATE_S0: wrap s/t/r at 0/3/6, min img<<9, mip<<11, mag<<13,
        // compare mode<<15, compare func<<16, seamless<<19, anisotropy<<20.
        let s0 = 7
            | (2 << 3)
            | (0 << 6)
            | (1 << 9)
            | (2 << 11)
            | (0 << 13)
            | (1 << 15)
            | (3 << 16)
            | (1 << 19)
            | (16 << 20);
        let w = [
            0x0000_0701 | (9 << 16),
            7,
            s0,
            (-0.5f32).to_bits(),
            1.0f32.to_bits(),
            7.0f32.to_bits(),
            1,
            2,
            3,
            4,
        ];
        let Command::CreateObject { object: Object::SamplerState(s), .. } =
            decode(Cmd::CreateObject, 7, &w).unwrap()
        else {
            panic!()
        };
        assert_eq!(
            (s.wrap_s, s.wrap_t, s.wrap_r),
            (TexWrap::MirrorClampToBorder, TexWrap::ClampToEdge, TexWrap::Repeat)
        );
        assert_eq!(
            (s.min_img_filter, s.min_mip_filter, s.mag_img_filter),
            (TexFilter::Linear, MipFilter::None, TexFilter::Nearest)
        );
        assert!(s.compare_mode && s.seamless_cube_map);
        assert_eq!((s.compare_func, s.max_anisotropy), (CompareFunc::LessEqual, 16));
        assert_eq!((s.lod_bias, s.min_lod, s.max_lod), (-0.5, 1.0, 7.0));

        // VIRGL_OBJ_SAMPLER_VIEW_FORMAT: format in the low 24 bits, target above; swizzle three
        // bits per channel from R.
        let w = [
            0x0000_0601 | (6 << 16),
            6,
            9,
            67 | (7 << 24),
            0x0003_0001,
            0x0002_0000,
            2 | (5 << 3) | (0 << 6) | (4 << 9),
        ];
        let Command::CreateObject { object: Object::SamplerView(v), .. } =
            decode(Cmd::CreateObject, 6, &w).unwrap()
        else {
            panic!()
        };
        assert_eq!((v.format, v.target), (fmt(67), TextureTarget::Array2d));
        assert_eq!(v.swizzle, [Swizzle::Z, Swizzle::One, Swizzle::X, Swizzle::Zero]);

        // VIRGL_CMD_BLIT_S0: mask<<0, filter<<8, scissor<<10, render condition<<11, alpha<<12.
        let mut w = vec![
            16 | (21 << 16),
            0xf | (1 << 8) | (1 << 10) | (1 << 12),
            5 | (6 << 16),
            7 | (8 << 16),
            30,
            1,
            67,
        ];
        w.extend([1, 2, 3, -4i32 as u32, 5, 6]);
        w.extend([31, 0, 1]);
        w.extend([1, 2, 3, -4i32 as u32, 5, 6]);
        let Command::Blit(b) = decode(Cmd::Blit, 0, &w).unwrap() else { panic!() };
        assert_eq!((b.mask, b.filter), (0xf, TexFilter::Linear));
        assert!(b.scissor_enable && !b.render_condition_enable && b.alpha_blend);
        assert_eq!(b.scissor, Scissor { minx: 5, miny: 6, maxx: 7, maxy: 8 });
        assert_eq!(b.dst.region.width, -4);
        assert_eq!(b.src.resource, r(31));

        // A shader header with two stream outputs: type, offset, tokens, nso, four strides, then
        // (output, stream) pairs, then the text.
        let so0 = 3 | (1 << 8) | (2 << 10) | (2 << 13) | (4 << 16);
        let w =
            [1 | (4 << 8) | (14 << 16), 4, 1, 12, 12, 2, 16, 0, 8, 0, so0, 1, 0, 0, 0x4c47_5354];
        let Command::CreateObject { object: Object::Shader(s), .. } =
            decode(Cmd::CreateObject, 4, &w).unwrap()
        else {
            panic!()
        };
        assert_eq!(s.stage, ShaderStage::Fragment);
        assert_eq!(s.chunk, ShaderChunk::New { total_bytes: 12 });
        let ShaderKind::Graphics { stream_output } = &s.kind else { panic!() };
        assert_eq!(stream_output.stride, [16, 0, 8, 0]);
        assert_eq!(stream_output.outputs[0].dst_offset, 4);
        assert_eq!(stream_output.outputs[0].stream, 1);
        assert_eq!(s.text, &[0x4c47_5354]);

        // A continuation carries its offset under VIRGL_OBJ_SHADER_OFFSET_CONT.
        let w = [1 | (4 << 8) | (5 << 16), 4, 0, 4000 | (1 << 31), 0, 0];
        let Command::CreateObject { object: Object::Shader(s), .. } =
            decode(Cmd::CreateObject, 4, &w).unwrap()
        else {
            panic!()
        };
        assert_eq!(s.chunk, ShaderChunk::Continuation { offset: 4000 });
        assert!(s.text.is_empty());

        // COPY_TRANSFER3D: the staging handle at 12, its offset at 13, flags at 14 with
        // SYNCHRONIZED bit 0 and READ_FROM_HOST bit 1.
        let w = [45 | (14 << 16), 7, 1, 2, 640, 0, 1, 2, 3, -4i32 as u32, 5, 6, 50, 128, 0b11];
        let Command::CopyTransfer3d { direction, transfer, staging, staging_offset, synchronized } =
            decode(Cmd::CopyTransfer3d, 0, &w).unwrap()
        else {
            panic!()
        };
        assert_eq!(direction, CopyDirection::FromHost);
        assert!(synchronized);
        assert_eq!((staging, staging_offset), (r(50), 128));
        assert_eq!(transfer, self::transfer());

        // A clear's depth is a little-endian double across dwords 6 and 7.
        let d = 0.75f64.to_bits();
        let w = [7 | (8 << 16), 0x7, 1, 2, 3, 4, d as u32, (d >> 32) as u32, 0x80];
        assert_eq!(
            decode(Cmd::Clear, 0, &w).unwrap(),
            Command::Clear { buffers: 0x7, color: [1, 2, 3, 4], depth: 0.75, stencil: 0x80 }
        );
    }

    fn refused(words: &[u32]) -> Refused {
        let header = Header::parse(words[0]);
        decode(Cmd::from_wire(header.cmd).unwrap(), header.obj, words).expect_err("refused")
    }

    #[test]
    fn what_the_wire_cannot_say_is_refused() {
        // Lengths the command does not come in.
        assert!(matches!(
            refused(&[8 | (13 << 16), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            Refused::Length { cmd: Cmd::DrawVbo, len: 13 }
        ));
        assert!(matches!(
            refused(&[4 | (8 << 16), 0, 0, 0, 0, 0, 0, 0, 0]),
            Refused::Length { cmd: Cmd::SetViewportState, .. }
        ));
        assert!(matches!(
            refused(&[34 | (4 << 16), 0, 0, 0, 0]),
            Refused::Length { cmd: Cmd::SetShaderBuffers, .. }
        ));
        assert!(matches!(
            refused(&[5 | (3 << 16), 2, 0, 0]),
            Refused::Length { cmd: Cmd::SetFramebufferState, .. }
        ));
        // A stage that is not one.
        assert!(matches!(
            refused(&[10 | (2 << 16), 6, 0]),
            Refused::Field { field: "shader stage", value: 6, .. }
        ));
        // Slots past the per-stage array.
        assert!(matches!(
            refused(&[10 | (3 << 16), 1, 128, 0]),
            Refused::Field { field: "start slot", .. }
        ));
        assert!(matches!(
            refused(&[18 | (3 << 16), 1, 32, 0]),
            Refused::Field { field: "start slot", .. }
        ));
        assert!(matches!(
            refused(&[15 | (3 << 16), 16, 0, 0]),
            Refused::Field { field: "start slot", .. }
        ));
        // A handle that must name something.
        assert!(matches!(
            refused(&[1 | (1 << 8) | (11 << 16), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            Refused::Field { field: "handle", value: 0, .. }
        ));
        assert!(matches!(refused(&[19 | (1 << 16), 0]), Refused::Field { field: "query", .. }));
        assert!(matches!(
            refused(&[43 | (13 << 16), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
            Refused::Field { field: "resource", .. }
        ));
        // A blend factor that is not one, on a target that blends; and non-zero factor bits on
        // one that does not, which the decoder could not reproduce.
        assert!(matches!(
            refused(&[1 | (1 << 8) | (11 << 16), 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
            Refused::Field { field: "blend factor", value: 0, .. }
        ));
        assert!(matches!(
            refused(&[1 | (1 << 8) | (11 << 16), 1, 0, 0, 1 << 4, 0, 0, 0, 0, 0, 0, 0]),
            Refused::Field { field: "disabled render target blend", .. }
        ));
        // A format past the table; a sampler view with no format; a swizzle that is not one.
        assert!(matches!(
            refused(&[
                16 | (21 << 16),
                0,
                0,
                0,
                1,
                0,
                482,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ]),
            Refused::Field { field: "format", value: 482, .. }
        ));
        assert!(matches!(
            refused(&[1 | (6 << 8) | (6 << 16), 1, 9, 2 << 24, 0, 0, 0]),
            Refused::Field { field: "format", .. }
        ));
        assert!(matches!(
            refused(&[1 | (6 << 8) | (6 << 16), 1, 9, 67, 0, 0, 6]),
            Refused::Field { field: "swizzle", value: 6, .. }
        ));
        // An index buffer with a handle but no size, and one with a size but no handle; and
        // a width that is not one, two or four bytes, which the C drew as four while sizing its
        // bounds check by the value as sent.
        assert!(matches!(refused(&[11 | (1 << 16), 7]), Refused::Field { .. }));
        assert!(matches!(refused(&[11 | (3 << 16), 0, 2, 0]), Refused::Field { .. }));
        assert!(matches!(
            refused(&[11 | (3 << 16), 7, 3, 0]),
            Refused::Field { field: "index size", value: 3, .. }
        ));
        // An image with no access, which the C refused only at the draw, abandoning the rest of
        // the stage's images with it.
        assert!(matches!(
            refused(&[35 | (7 << 16), 1, 0, 67, 0, 0, 0, 9]),
            Refused::Field { field: "image access", value: 0, .. }
        ));
        // A stream output naming a buffer past the four the strides describe: three wire bits
        // for a four-entry table, which the C indexed as sent.
        let mut so = vec![1 | (4 << 8) | (14 << 16), 5, 0, 8, 1, 1, 16, 0, 0, 0, 4 << 13, 0, 0, 0];
        so.push(0);
        assert!(matches!(refused(&so), Refused::Field { field: "output buffer", value: 4, .. }));
        // A draw whose flags are not 0 or 1 would not reproduce through `!!`.
        let mut draw = vec![8 | (12 << 16), 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(matches!(refused(&draw), Refused::Field { field: "indexed", value: 2, .. }));
        draw[4] = 0;
        draw[3] = 15;
        assert!(matches!(refused(&draw), Refused::Field { field: "mode", value: 15, .. }));
        // Only one, two, three or four planes.
        assert!(matches!(
            refused(&[49 | (8 << 16), 1, 0, 0, 0, 0, 0, 0, 0]),
            Refused::Field { field: "plane count", value: 0, .. }
        ));
        // A string marker longer than its dwords.
        assert!(matches!(
            refused(&[51 | (2 << 16), 5, 0]),
            Refused::Field { field: "string length", .. }
        ));
        // A bind of something that is not bindable.
        assert!(matches!(
            refused(&[2 | (8 << 8) | (1 << 16), 1]),
            Refused::Field { field: "object type", value: 8, .. }
        ));
    }

    #[test]
    fn a_refusal_ends_the_batch() {
        let mut wire = Vec::new();
        encode(&Command::SetSampleMask(1), &mut wire);
        encode(&Command::SetMinSamples(2), &mut wire);
        let bad_at = wire.len();
        wire.push(24 | (2 << 16)); // SET_SAMPLE_MASK does not come in two dwords
        wire.extend([0, 0]);
        encode(&Command::SetSampleMask(3), &mut wire);

        let mut batch = Batch::new(&wire);
        assert_eq!(batch.next().unwrap().unwrap(), Command::SetSampleMask(1));
        assert_eq!(batch.next().unwrap().unwrap(), Command::SetMinSamples(2));
        assert_eq!(batch.position(), bad_at);
        assert_eq!(
            batch.next().unwrap().unwrap_err(),
            Refused::Length { cmd: Cmd::SetSampleMask, len: 2 }
        );
        assert!(batch.next().is_none(), "nothing after a refusal is framed");

        // A header that runs past the batch, and a command byte the protocol lacks.
        let wire = [24 | (1 << 16), 1, 24 | (5 << 16), 0];
        let mut batch = Batch::new(&wire);
        assert!(batch.next().unwrap().is_ok());
        assert_eq!(
            batch.next().unwrap().unwrap_err(),
            Refused::Overrun { at: 2, cmd: Cmd::SetSampleMask, len: 5, left: 1 }
        );
        assert!(batch.next().is_none());
        let wire = [64, 24 | (1 << 16), 1];
        let mut batch = Batch::new(&wire);
        assert_eq!(batch.next().unwrap().unwrap_err(), Refused::UnknownCommand { at: 0, cmd: 64 });
        assert!(batch.next().is_none());

        // An empty batch is a batch of nothing.
        assert!(Batch::new(&[]).next().is_none());
    }

    #[test]
    fn end_transfers_keeps_its_slack() {
        // Mesa pads the transfer prologue out to its fixed size with END_TRANSFERS's length, and
        // the dwords under it are whatever was there. They are carried so that the stream
        // reproduces, and nothing reads them.
        let mut wire = vec![44 | (1023 << 16)];
        wire.extend((0..1023).map(|i| i * 3));
        let Command::EndTransfers(slack) = decode(Cmd::EndTransfers, 0, &wire).unwrap() else {
            panic!()
        };
        assert_eq!(slack, &wire[1..]);
        assert_eq!(decode(Cmd::EndTransfers, 0, &[44]).unwrap(), Command::EndTransfers(&[]));
    }
}
