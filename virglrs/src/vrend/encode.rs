// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! The classic wire, written. The inverse of [`decode`](super::decode): a [`Command`] back to the
//! dwords the guest's encoder (`virgl_encode.c` in mesa) produces for it, bit for bit.
//!
//! Nothing in the renderer sends classic commands; this exists for the differential gate -- a
//! recorded stream decoded and re-encoded must reproduce itself, which proves every field was
//! read from where it lives -- and for tests, which build their wire with it rather than by hand.

use super::proto::*;
use crate::ids::ResourceHandle;

/// Appends one command, header included, to `out`.
pub fn encode(cmd: &Command<'_>, out: &mut Vec<u32>) {
    let mut e = Encoder { out, start: 0 };
    e.start = e.out.len();
    e.out.push(0);
    let obj = match cmd {
        Command::CreateObject { object, .. } => object.kind().wire(),
        Command::BindObject { kind, .. } | Command::DestroyObject { kind, .. } => kind.wire(),
        _ => 0,
    };
    e.body(cmd);
    let len = e.out.len() - e.start - 1;
    let len = u16::try_from(len).expect("a command longer than the header can frame");
    e.out[e.start] = cmd.kind().wire() | (obj << 8) | (u32::from(len) << 16);
}

struct Encoder<'o> {
    out: &'o mut Vec<u32>,
    start: usize,
}

fn res(r: Option<ResourceHandle>) -> u32 {
    r.map_or(0, ResourceHandle::get)
}

fn obj(o: Option<ObjectHandle>) -> u32 {
    o.map_or(0, ObjectHandle::get)
}

impl Encoder<'_> {
    fn u(&mut self, v: u32) {
        self.out.push(v);
    }

    fn i(&mut self, v: i32) {
        self.out.push(v as u32);
    }

    fn f(&mut self, v: f32) {
        self.out.push(v.to_bits());
    }

    fn words(&mut self, w: &[u32]) {
        self.out.extend_from_slice(w);
    }

    fn region(&mut self, b: &Box3) {
        for v in [b.x, b.y, b.z, b.width, b.height, b.depth] {
            self.i(v);
        }
    }

    fn transfer(&mut self, t: &Transfer) {
        self.u(t.resource.get());
        self.u(t.level);
        self.u(t.usage);
        self.u(t.stride);
        self.u(t.layer_stride);
        self.region(&t.region);
    }

    fn scissor(&mut self, s: &Scissor) {
        self.u(u32::from(s.minx) | (u32::from(s.miny) << 16));
        self.u(u32::from(s.maxx) | (u32::from(s.maxy) << 16));
    }

    fn shader_buffer(&mut self, b: &ShaderBuffer) {
        self.u(b.offset);
        self.u(b.length);
        self.u(res(b.resource));
    }

    fn body(&mut self, cmd: &Command<'_>) {
        match cmd {
            Command::Nop => {}
            Command::CreateObject { handle, object } => {
                self.u(handle.get());
                self.object(object);
            }
            Command::BindObject { handle, .. } => self.u(obj(*handle)),
            Command::DestroyObject { handle, .. } => self.u(handle.get()),
            Command::SetViewportState { start_slot, viewports } => {
                self.u(*start_slot);
                for v in viewports {
                    for x in v.scale.iter().chain(&v.translate) {
                        self.f(*x);
                    }
                }
            }
            Command::SetFramebufferState { zsurf, cbufs } => {
                self.u(cbufs.len() as u32);
                self.u(obj(*zsurf));
                for c in cbufs {
                    self.u(obj(*c));
                }
            }
            Command::SetVertexBuffers(buffers) => {
                for b in buffers {
                    self.u(b.stride);
                    self.u(b.offset);
                    self.u(res(b.resource));
                }
            }
            Command::Clear { buffers, color, depth, stencil } => {
                self.u(*buffers);
                self.words(color);
                let d = depth.to_bits();
                self.u(d as u32);
                self.u((d >> 32) as u32);
                self.u(*stencil);
            }
            Command::DrawVbo(d) => {
                self.u(d.start);
                self.u(d.count);
                self.u(d.mode.wire());
                self.u(u32::from(d.indexed));
                self.u(d.instance_count);
                self.i(d.index_bias);
                self.u(d.start_instance);
                self.u(u32::from(d.primitive_restart));
                self.u(d.restart_index);
                self.u(d.min_index);
                self.u(d.max_index);
                self.u(obj(d.count_from_so));
                if let Some(t) = &d.tess {
                    self.u(t.vertices_per_patch);
                    self.u(t.drawid);
                }
                if let Some(i) = &d.indirect {
                    assert!(d.tess.is_some(), "an indirect draw carries the tessellation dwords");
                    self.u(i.resource.get());
                    self.u(i.offset);
                    self.u(i.stride);
                    self.u(i.draw_count);
                    self.u(i.draw_count_offset);
                    self.u(res(i.draw_count_resource));
                }
            }
            Command::ResourceInlineWrite { transfer, data } => {
                self.transfer(transfer);
                self.words(data);
            }
            Command::SetSamplerViews { stage, start_slot, views } => {
                self.u(stage.wire());
                self.u(*start_slot);
                for v in views {
                    self.u(obj(*v));
                }
            }
            Command::SetIndexBuffer(None) => self.u(0),
            Command::SetIndexBuffer(Some(ib)) => {
                self.u(ib.resource.get());
                self.u(ib.index_type.wire());
                self.u(ib.offset);
            }
            Command::SetConstantBuffer { stage, index, data } => {
                self.u(stage.wire());
                self.u(*index);
                self.words(data);
            }
            Command::SetStencilRef { front, back } => {
                self.u(u32::from(*front) | (u32::from(*back) << 8));
            }
            Command::SetBlendColor(c) => {
                for x in c {
                    self.f(*x);
                }
            }
            Command::SetScissorState { start_slot, scissors } => {
                self.u(*start_slot);
                for s in scissors {
                    self.scissor(s);
                }
            }
            Command::Blit(b) => {
                self.u(u32::from(b.mask)
                    | (b.filter.wire() << 8)
                    | (u32::from(b.scissor_enable) << 10)
                    | (u32::from(b.render_condition_enable) << 11)
                    | (u32::from(b.alpha_blend) << 12));
                self.scissor(&b.scissor);
                for t in [&b.dst, &b.src] {
                    self.u(t.resource.get());
                    self.u(t.level);
                    self.u(t.format.wire());
                    self.region(&t.region);
                }
            }
            Command::ResourceCopyRegion {
                dst,
                dst_level,
                dst_x,
                dst_y,
                dst_z,
                src,
                src_level,
                src_region,
            } => {
                self.u(dst.get());
                self.u(*dst_level);
                self.u(*dst_x);
                self.u(*dst_y);
                self.u(*dst_z);
                self.u(src.get());
                self.u(*src_level);
                self.region(src_region);
            }
            Command::BindSamplerStates { stage, start_slot, states } => {
                self.u(stage.wire());
                self.u(*start_slot);
                for s in states {
                    self.u(obj(*s));
                }
            }
            Command::BeginQuery(q) | Command::EndQuery(q) => self.u(q.get()),
            Command::GetQueryResult { query, wait } => {
                self.u(query.get());
                self.u(u32::from(*wait));
            }
            Command::SetPolygonStipple(p) => self.words(p),
            Command::SetClipState(planes) => {
                for p in planes {
                    for x in p {
                        self.f(*x);
                    }
                }
            }
            Command::SetSampleMask(m) | Command::SetMinSamples(m) => self.u(*m),
            Command::MemoryBarrier(f) | Command::TextureBarrier(f) => self.u(*f),
            Command::SetStreamoutTargets { append_bitmask, targets } => {
                self.u(*append_bitmask);
                for t in targets {
                    self.u(obj(*t));
                }
            }
            Command::SetRenderCondition { query, condition, mode } => {
                self.u(obj(*query));
                self.u(u32::from(*condition));
                self.u(mode.wire());
            }
            Command::SetUniformBuffer { stage, index, offset, length, resource } => {
                self.u(stage.wire());
                self.u(*index);
                self.u(*offset);
                self.u(*length);
                self.u(res(*resource));
            }
            Command::SetSubCtx(id) | Command::CreateSubCtx(id) | Command::DestroySubCtx(id) => {
                self.u(id.0);
            }
            Command::BindShader { handle, stage } => {
                self.u(obj(*handle));
                self.u(stage.wire());
            }
            Command::SetTessState(f) => {
                for x in f {
                    self.f(*x);
                }
            }
            Command::SetShaderBuffers { stage, start_slot, buffers } => {
                self.u(stage.wire());
                self.u(*start_slot);
                for b in buffers {
                    self.shader_buffer(b);
                }
            }
            Command::SetShaderImages { stage, start_slot, images } => {
                self.u(stage.wire());
                self.u(*start_slot);
                for i in images {
                    match i {
                        Some(i) => {
                            self.u(i.format.wire());
                            self.u(i.access.wire());
                            self.u(i.layer_offset);
                            self.u(i.level_size);
                            self.u(i.resource.get());
                        }
                        None => self.words(&[0; 5]),
                    }
                }
            }
            Command::LaunchGrid { block, grid, indirect, indirect_offset } => {
                self.words(block);
                self.words(grid);
                self.u(res(*indirect));
                self.u(*indirect_offset);
            }
            Command::SetFramebufferStateNoAttach { width, height, layers, samples } => {
                self.u(u32::from(*width) | (u32::from(*height) << 16));
                self.u(u32::from(*layers) | (u32::from(*samples) << 16));
            }
            Command::SetAtomicBuffers { start_slot, buffers } => {
                self.u(*start_slot);
                for b in buffers {
                    self.shader_buffer(b);
                }
            }
            Command::SetDebugFlags(w) | Command::DecodeMacroblock(w) | Command::EndTransfers(w) => {
                self.words(w)
            }
            Command::GetQueryResultQbo { query, buffer, wait, result_type, offset, index } => {
                self.u(query.get());
                self.u(buffer.get());
                self.u(u32::from(*wait));
                self.u(result_type.wire());
                self.u(*offset);
                self.i(*index);
            }
            Command::Transfer3d { transfer, offset, direction } => {
                self.transfer(transfer);
                self.u(*offset);
                self.u(direction.wire());
            }
            Command::CopyTransfer3d {
                direction,
                transfer,
                staging,
                staging_offset,
                synchronized,
            } => {
                self.transfer(transfer);
                self.u(staging.get());
                self.u(*staging_offset);
                self.u(u32::from(*synchronized)
                    | (u32::from(*direction == CopyDirection::FromHost) << 1));
            }
            Command::SetTweaks { id, value } => {
                self.u(*id);
                self.u(*value);
            }
            Command::ClearTexture { resource, level, region, data } => {
                self.u(resource.get());
                self.u(*level);
                self.region(region);
                self.words(data);
            }
            Command::PipeResourceCreate {
                target,
                format,
                bind,
                width,
                height,
                depth,
                array_size,
                last_level,
                nr_samples,
                flags,
                blob_id,
            } => {
                self.u(target.wire());
                self.u(format.wire());
                self.u(*bind);
                self.u(*width);
                self.u(*height);
                self.u(*depth);
                self.u(*array_size);
                self.u(*last_level);
                self.u(*nr_samples);
                self.u(*flags);
                self.u(u32::try_from(blob_id.0).expect("a classic blob id is 32 bits"));
            }
            Command::PipeResourceSetType {
                resource,
                format,
                bind,
                width,
                height,
                usage,
                modifier,
                planes,
            } => {
                self.u(resource.get());
                self.u(format.wire());
                self.u(*bind);
                self.u(*width);
                self.u(*height);
                self.u(*usage);
                self.u(*modifier as u32);
                self.u((*modifier >> 32) as u32);
                for p in planes {
                    self.u(p.stride);
                    self.u(p.offset);
                }
            }
            Command::GetMemoryInfo(r) => self.u(r.get()),
            Command::SendStringMarker { len, text } => {
                self.u(*len);
                self.words(text);
            }
            Command::LinkShader(handles) => {
                for h in handles {
                    self.u(obj(*h));
                }
            }
            Command::CreateVideoCodec(c) => {
                self.u(c.handle.0);
                self.u(c.profile);
                self.u(c.entrypoint);
                self.u(c.chroma_format);
                self.u(c.level);
                self.u(c.width);
                self.u(c.height);
                if let Some(m) = c.max_references {
                    self.u(m);
                }
            }
            Command::DestroyVideoCodec(h) => self.u(h.0),
            Command::CreateVideoBuffer { handle, format, width, height, planes } => {
                self.u(handle.0);
                self.u(*format);
                self.u(*width);
                self.u(*height);
                for p in planes {
                    self.u(p.get());
                }
            }
            Command::DestroyVideoBuffer(h) => self.u(h.0),
            Command::BeginFrame { codec, target } | Command::EndFrame { codec, target } => {
                self.u(codec.0);
                self.u(target.0);
            }
            Command::DecodeBitstream { codec, target, descriptor, buffer, buffer_size } => {
                self.u(codec.0);
                self.u(target.0);
                self.u(descriptor.get());
                self.u(buffer.get());
                self.u(*buffer_size);
            }
            Command::EncodeBitstream { codec, source, destination, descriptor, feedback } => {
                self.u(codec.0);
                self.u(source.0);
                self.u(destination.get());
                self.u(descriptor.get());
                self.u(feedback.get());
            }
            Command::ClearSurface {
                render_condition_enable,
                buffers,
                surface,
                color,
                dst_x,
                dst_y,
                width,
                height,
            } => {
                self.u(u32::from(*render_condition_enable) | (u32::from(*buffers) << 1));
                self.u(surface.get());
                self.words(color);
                self.u(*dst_x);
                self.u(*dst_y);
                self.u(*width);
                self.u(*height);
            }
            Command::GetPipeResourceLayout { out, target } => {
                self.u(out.get());
                self.u(target.get());
            }
        }
    }

    fn object(&mut self, object: &Object<'_>) {
        match object {
            Object::Blend(b) => {
                self.u(u32::from(b.independent_blend_enable)
                    | (u32::from(b.logicop_enable) << 1)
                    | (u32::from(b.dither) << 2)
                    | (u32::from(b.alpha_to_coverage) << 3)
                    | (u32::from(b.alpha_to_one) << 4));
                self.u(b.logicop_func.wire());
                for rt in &b.rt {
                    let mut s2 = u32::from(rt.colormask) << 27;
                    if let Some(eq) = &rt.equation {
                        s2 |= 1
                            | (eq.rgb.func.wire() << 1)
                            | (eq.rgb.src.wire() << 4)
                            | (eq.rgb.dst.wire() << 9)
                            | (eq.alpha.func.wire() << 14)
                            | (eq.alpha.src.wire() << 17)
                            | (eq.alpha.dst.wire() << 22);
                    }
                    self.u(s2);
                }
            }
            Object::Dsa(d) => {
                self.u(u32::from(d.depth.enabled)
                    | (u32::from(d.depth.writemask) << 1)
                    | (d.depth.func.wire() << 2)
                    | (u32::from(d.alpha.enabled) << 8)
                    | (d.alpha.func.wire() << 9));
                for s in &d.stencil {
                    self.u(u32::from(s.enabled)
                        | (s.func.wire() << 1)
                        | (s.fail_op.wire() << 4)
                        | (s.zpass_op.wire() << 7)
                        | (s.zfail_op.wire() << 10)
                        | (u32::from(s.valuemask) << 13)
                        | (u32::from(s.writemask) << 21));
                }
                self.f(d.alpha.ref_value);
            }
            Object::Rasterizer(r) => {
                let bits = [
                    r.flatshade,
                    r.depth_clip,
                    r.clip_halfz,
                    r.rasterizer_discard,
                    r.flatshade_first,
                    r.light_twoside,
                    r.sprite_coord_mode,
                    r.point_quad_rasterization,
                ];
                let mut s0 = 0;
                for (i, b) in bits.iter().enumerate() {
                    s0 |= u32::from(*b) << i;
                }
                s0 |= (r.cull_face.wire() << 8)
                    | (r.fill_front.wire() << 10)
                    | (r.fill_back.wire() << 12);
                let bits = [
                    r.scissor,
                    r.front_ccw,
                    r.clamp_vertex_color,
                    r.clamp_fragment_color,
                    r.offset_line,
                    r.offset_point,
                    r.offset_tri,
                    r.poly_smooth,
                    r.poly_stipple_enable,
                    r.point_smooth,
                    r.point_size_per_vertex,
                    r.multisample,
                    r.line_smooth,
                    r.line_stipple_enable,
                    r.line_last_pixel,
                    r.half_pixel_center,
                    r.bottom_edge_rule,
                    r.force_persample_interp,
                ];
                for (i, b) in bits.iter().enumerate() {
                    s0 |= u32::from(*b) << (14 + i);
                }
                self.u(s0);
                self.f(r.point_size);
                self.u(r.sprite_coord_enable);
                self.u(u32::from(r.line_stipple_pattern)
                    | (u32::from(r.line_stipple_factor) << 16)
                    | (u32::from(r.clip_plane_enable) << 24));
                self.f(r.line_width);
                self.f(r.offset_units);
                self.f(r.offset_scale);
                self.f(r.offset_clamp);
            }
            Object::Shader(s) => {
                self.u(s.stage.wire());
                self.u(match s.chunk {
                    ShaderChunk::New { total_bytes } => total_bytes,
                    ShaderChunk::Continuation { offset } => offset | (1 << 31),
                });
                self.u(s.num_tokens);
                match &s.kind {
                    ShaderKind::Compute { req_local_mem } => self.u(*req_local_mem),
                    ShaderKind::Graphics { stream_output } => {
                        self.u(stream_output.outputs.len() as u32);
                        if !stream_output.outputs.is_empty() {
                            self.words(&stream_output.stride);
                            for o in &stream_output.outputs {
                                self.u(u32::from(o.register_index)
                                    | (u32::from(o.start_component) << 8)
                                    | (u32::from(o.num_components) << 10)
                                    | (o.output_buffer.wire() << 13)
                                    | (u32::from(o.dst_offset) << 16));
                                self.u(u32::from(o.stream));
                            }
                        }
                    }
                }
                self.words(s.text);
            }
            Object::VertexElements(elements) => {
                for e in elements {
                    self.u(e.src_offset);
                    self.u(e.instance_divisor);
                    self.u(e.vertex_buffer_index);
                    self.u(e.src_format.wire());
                }
            }
            Object::SamplerView(v) => {
                self.u(v.resource.get());
                self.u(v.format.wire() | (v.target.wire() << 24));
                self.u(v.first_element_or_layers);
                self.u(v.last_element_or_levels);
                let mut sw = 0;
                for (i, s) in v.swizzle.iter().enumerate() {
                    sw |= s.wire() << (3 * i);
                }
                self.u(sw);
            }
            Object::SamplerState(s) => {
                self.u(s.wrap_s.wire()
                    | (s.wrap_t.wire() << 3)
                    | (s.wrap_r.wire() << 6)
                    | (s.min_img_filter.wire() << 9)
                    | (s.min_mip_filter.wire() << 11)
                    | (s.mag_img_filter.wire() << 13)
                    | (u32::from(s.compare_mode) << 15)
                    | (s.compare_func.wire() << 16)
                    | (u32::from(s.seamless_cube_map) << 19)
                    | (u32::from(s.max_anisotropy) << 20));
                self.f(s.lod_bias);
                self.f(s.min_lod);
                self.f(s.max_lod);
                self.words(&s.border_color);
            }
            Object::Surface(s) => {
                self.u(s.resource.get());
                self.u(s.format.wire());
                self.u(s.first_element_or_level);
                self.u(s.last_element_or_layers);
                if s.samples != 0 {
                    self.u(s.samples);
                }
            }
            Object::Query(q) => {
                self.u(q.kind.wire() | (u32::from(q.index) << 16));
                self.u(q.offset);
                self.u(q.resource.get());
            }
            Object::StreamoutTarget(t) => {
                self.u(t.resource.get());
                self.u(t.buffer_offset);
                self.u(t.buffer_size);
            }
        }
    }
}
