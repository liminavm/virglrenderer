// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! `tgsi_dump.c`: a program printed back as text, in the C's exact spelling.
//!
//! This is what `VREND_DEBUG=shader` prints after "TGSI received:", and it is the first
//! differential of the shader port: the C's dump of a corpus shader and this dump of the same
//! text must be byte-identical, which settles that the parser read what the C read.

use super::*;
use crate::vrend::proto::Format;
use std::fmt::Write;

/// `%10.4f` / `%10.8f`, with printf's spellings for the values Rust spells differently.
fn cfloat(out: &mut String, v: f64, precision: usize) {
    if v.is_nan() {
        let s = if v.is_sign_negative() { "-nan" } else { "nan" };
        write!(out, "{s:>10}").expect("String");
    } else if v.is_infinite() {
        let s = if v < 0.0 { "-inf" } else { "inf" };
        write!(out, "{s:>10}").expect("String");
    } else {
        write!(out, "{v:>10.precision$}").expect("String");
    }
}

fn format_name(wire: u16) -> String {
    let name = Format::from_wire(u32::from(wire)).map_or("???", Format::name);
    format!("PIPE_FORMAT_{name}")
}

fn swizzle_name(s: u8) -> String {
    SWIZZLE_NAMES.get(usize::from(s)).map_or_else(|| s.to_string(), |n| (*n).to_string())
}

/// The register half a source and a destination share, for `_dump_register_src` and
/// `_dump_register_dst`, which print the same shape.
struct Register {
    file: File,
    dimension: bool,
    dim: Dimension,
    dim_ind: IndReg,
    indirect: bool,
    index: i16,
    ind: IndReg,
}

impl From<&Src> for Register {
    fn from(s: &Src) -> Register {
        Register {
            file: s.file,
            dimension: s.dimension,
            dim: s.dim,
            dim_ind: s.dim_ind,
            indirect: s.indirect,
            index: s.index,
            ind: s.ind,
        }
    }
}

impl From<&Dst> for Register {
    fn from(d: &Dst) -> Register {
        Register {
            file: d.file,
            dimension: d.dimension,
            dim: d.dim,
            dim_ind: d.dim_ind,
            indirect: d.indirect,
            index: d.index,
            ind: d.ind,
        }
    }
}

fn register(out: &mut String, r: Register) {
    out.push_str(r.file.name());
    if r.dimension {
        if r.dim.indirect {
            write!(
                out,
                "[{}[{}].{}",
                r.dim_ind.file.name(),
                r.dim_ind.index,
                swizzle_name(r.dim_ind.swizzle)
            )
            .expect("String");
            if r.dim.index != 0 {
                if r.dim.index > 0 {
                    out.push('+');
                }
                write!(out, "{}", r.dim.index).expect("String");
            }
            out.push(']');
            if r.dim_ind.array_id != 0 {
                write!(out, "({})", r.dim_ind.array_id).expect("String");
            }
        } else {
            write!(out, "[{}]", r.dim.index).expect("String");
        }
    }
    if r.indirect {
        write!(out, "[{}[{}].{}", r.ind.file.name(), r.ind.index, swizzle_name(r.ind.swizzle))
            .expect("String");
        if r.index != 0 {
            if r.index > 0 {
                out.push('+');
            }
            write!(out, "{}", r.index).expect("String");
        }
        out.push(']');
        if r.ind.array_id != 0 {
            write!(out, "({})", r.ind.array_id).expect("String");
        }
    } else {
        write!(out, "[{}]", r.index).expect("String");
    }
}

fn writemask(out: &mut String, mask: u8) {
    if mask != WRITEMASK_XYZW {
        out.push('.');
        for (bit, c) in
            [(WRITEMASK_X, 'x'), (WRITEMASK_Y, 'y'), (WRITEMASK_Z, 'z'), (WRITEMASK_W, 'w')]
        {
            if mask & bit != 0 {
                out.push(c);
            }
        }
    }
}

fn declaration(out: &mut String, processor: Processor, d: &Declaration) {
    let patch = matches!(
        d.semantic.name,
        Semantic::Patch | Semantic::TessInner | Semantic::TessOuter | Semantic::PrimId
    );
    out.push_str("DCL ");
    out.push_str(d.file.name());
    // Geometry shader inputs and non-patch tessellation inputs are two-dimensional.
    if d.file == File::Input
        && (processor == Processor::Geometry
            || (!patch && matches!(processor, Processor::TessCtrl | Processor::TessEval)))
    {
        out.push_str("[]");
    }
    // As are the non-patch outputs of a tessellation control shader.
    if d.file == File::Output && !patch && processor == Processor::TessCtrl {
        out.push_str("[]");
    }
    if let Some(dim) = d.dimension {
        write!(out, "[{dim}]").expect("String");
    }
    write!(out, "[{}", d.first).expect("String");
    if d.first != d.last {
        write!(out, "..{}", d.last).expect("String");
    }
    out.push(']');
    writemask(out, d.usage_mask);
    if let Some(a) = d.array {
        write!(out, ", ARRAY({a})").expect("String");
    }
    if d.local {
        out.push_str(", LOCAL");
    }
    if d.has_semantic {
        write!(out, ", {}", d.semantic.name.name()).expect("String");
        if d.semantic.index != 0
            || matches!(d.semantic.name, Semantic::TexCoord | Semantic::Generic)
        {
            write!(out, "[{}]", d.semantic.index).expect("String");
        }
    }
    match d.file {
        File::Image => {
            write!(out, ", {}, {}", d.image.resource.name(), format_name(d.image.format))
                .expect("String");
            if d.image.writable {
                out.push_str(", WR");
            }
            if d.image.raw {
                out.push_str(", RAW");
            }
        }
        File::Buffer => {
            if d.atomic {
                out.push_str(", ATOMIC");
            }
        }
        File::Memory => {
            write!(out, ", {}", d.mem_type.name()).expect("String");
        }
        File::SamplerView => {
            let rt = d.sampler_view.return_type;
            write!(out, ", {}, ", d.sampler_view.resource.name()).expect("String");
            if rt[1..].iter().all(|t| *t == rt[0]) {
                out.push_str(rt[0].name());
            } else {
                let names: Vec<&str> = rt.iter().map(|t| t.name()).collect();
                out.push_str(&names.join(", "));
            }
        }
        _ => {}
    }
    if d.has_interpolate {
        if processor == Processor::Fragment && d.file == File::Input {
            write!(out, ", {}", d.interp.interpolate.name()).expect("String");
        }
        if d.interp.location != Location::Center {
            write!(out, ", {}", d.interp.location.name()).expect("String");
        }
        if d.interp.cylindrical_wrap != 0 {
            out.push_str(", CYLWRAP_");
            for (bit, c) in [(1, 'X'), (2, 'Y'), (4, 'Z'), (8, 'W')] {
                if d.interp.cylindrical_wrap & bit != 0 {
                    out.push(c);
                }
            }
        }
    }
    if d.invariant {
        out.push_str(", INVARIANT");
    }
    out.push('\n');
}

fn property(out: &mut String, p: &PropertyToken) {
    write!(out, "PROPERTY {} ", p.name.name()).expect("String");
    let v = p.data as usize;
    match p.name {
        Property::GsInputPrim | Property::GsOutputPrim => {
            out.push_str(
                &Primitive::from_index(v).map_or_else(|| v.to_string(), |p| p.name().to_string()),
            );
        }
        Property::FsCoordOrigin => {
            out.push_str(
                &CoordOrigin::from_index(v).map_or_else(|| v.to_string(), |p| p.name().to_string()),
            );
        }
        Property::FsCoordPixelCenter => {
            out.push_str(
                &PixelCenter::from_index(v).map_or_else(|| v.to_string(), |p| p.name().to_string()),
            );
        }
        _ => write!(out, "{}", p.data as i32).expect("String"),
    }
    out.push('\n');
}

fn immediate(out: &mut String, index: u32, imm: &Immediate) {
    write!(out, "IMM[{index}] {} {{", imm.ty.name()).expect("String");
    let wide = |i: usize| u64::from(imm.data[i]) | u64::from(imm.data[i + 1]) << 32;
    let mut i = 0;
    while i < 4 {
        match imm.ty {
            ImmType::Float64 => {
                cfloat(out, f64::from_bits(wide(i)), 8);
                i += 1;
            }
            ImmType::Int64 => {
                write!(out, "{}", wide(i) as i64).expect("String");
                i += 1;
            }
            ImmType::Uint64 => {
                write!(out, "{}", wide(i)).expect("String");
                i += 1;
            }
            ImmType::Float32 => cfloat(out, f64::from(imm.float(i)), 4),
            ImmType::Uint32 => write!(out, "{}", imm.uint(i)).expect("String"),
            ImmType::Int32 => write!(out, "{}", imm.int(i)).expect("String"),
        }
        if i < 3 {
            out.push_str(", ");
        }
        i += 1;
    }
    out.push_str("}\n");
}

fn instruction(out: &mut String, index: u32, indent: &mut i32, inst: &Instruction) {
    let info = inst.opcode.info();
    write!(out, "{index:>3}: ").expect("String");
    *indent -= i32::from(info.pre_dedent);
    for _ in 0..(*indent).max(0) {
        out.push_str("  ");
    }
    *indent += i32::from(info.post_indent);
    out.push_str(info.mnemonic);
    if inst.saturate {
        out.push_str("_SAT");
    }
    let mut first = true;
    for d in inst.dsts() {
        if !first {
            out.push(',');
        }
        out.push(' ');
        register(out, d.into());
        writemask(out, d.writemask);
        first = false;
    }
    for s in inst.srcs() {
        if !first {
            out.push(',');
        }
        out.push(' ');
        if s.negate {
            out.push('-');
        }
        if s.absolute {
            out.push('|');
        }
        register(out, s.into());
        if s.swizzle != [SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W] {
            out.push('.');
            for c in s.swizzle {
                out.push_str(&swizzle_name(c));
            }
        }
        if s.absolute {
            out.push('|');
        }
        first = false;
    }
    if let Some(t) = inst.texture {
        if !(Opcode::Sample..=Opcode::Gather4).contains(&inst.opcode) {
            write!(out, ", {}", t.texture.name()).expect("String");
        }
        for off in &inst.tex_offsets[..usize::from(t.num_offsets)] {
            write!(out, ", {}[{}].", off.file.name(), off.index).expect("String");
            for c in off.swizzle {
                out.push_str(&swizzle_name(c));
            }
        }
    }
    if let Some(m) = inst.memory {
        for q in MemoryQualifier::ALL {
            if m.qualifier & (1 << (*q as u8)) != 0 {
                write!(out, ", {}", q.name()).expect("String");
            }
        }
        if m.texture != Texture::Buffer {
            write!(out, ", {}", m.texture.name()).expect("String");
        }
        if m.format != 0 {
            write!(out, ", {}", format_name(m.format)).expect("String");
        }
    }
    let labelled = matches!(
        inst.opcode,
        Opcode::If
            | Opcode::Uif
            | Opcode::Else
            | Opcode::Bgnloop
            | Opcode::Endloop
            | Opcode::Cal
            | Opcode::Bgnsub
    );
    if let Some(label) = inst.label.filter(|_| labelled) {
        write!(out, " :{label}").expect("String");
    }
    out.push('\n');
}

/// `tgsi_dump` with no flags: the program as the C prints it, one line per token, ending in a
/// newline.
pub fn dump(shader: &Shader) -> String {
    let mut out = String::new();
    out.push_str(shader.processor.name());
    out.push('\n');
    let (mut instno, mut immno, mut indent) = (0, 0, 0i32);
    for token in &shader.tokens {
        match token {
            Token::Declaration(d) => declaration(&mut out, shader.processor, d),
            Token::Property(p) => property(&mut out, p),
            Token::Immediate(i) => {
                immediate(&mut out, immno, i);
                immno += 1;
            }
            Token::Instruction(i) => {
                instruction(&mut out, instno, &mut indent, i);
                instno += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A program the C would print back exactly as written.
    const ROUND_TRIP: &str = "FRAG
PROPERTY FS_COORD_ORIGIN UPPER_LEFT
PROPERTY FS_COORD_PIXEL_CENTER HALF_INTEGER
DCL IN[0], GENERIC[0], PERSPECTIVE
DCL IN[1], COLOR[1], LINEAR, CENTROID
DCL OUT[0], COLOR
DCL SAMP[0]
DCL SVIEW[0], 2D, FLOAT
DCL CONST[1][0..3]
DCL TEMP[0..1], LOCAL
DCL TEMP[2..5], ARRAY(1), LOCAL
DCL ADDR[0]
IMM[0] FLT32 {    1.0000,     0.0000,    -2.5000,     0.5000}
IMM[1] INT32 {-1, 2, 3, 4}
IMM[2] FLT64 {1.50000000, -0.25000000}
  0: MOV TEMP[0], CONST[1][ADDR[0].x+2]
  1: MOV TEMP[1].xy, -|IN[0].yxzw|
  2: TEX TEMP[0], IN[0], SAMP[0], 2D, IMM[1].xyz
  3: IF TEMP[0].xxxx :0
  4:   MOV_SAT OUT[0], TEMP[ADDR[0].x-1](1)
  5:   BGNLOOP
  6:     BRK
  7:   ENDLOOP
  8: ELSE :0
  9:   MOV OUT[0], IMM[0]
 10: ENDIF
 11: END
";

    #[test]
    fn a_program_prints_back_as_it_was_written() {
        let s = text::parse(ROUND_TRIP.as_bytes(), 1000).expect("parses");
        assert_eq!(dump(&s), ROUND_TRIP);
    }

    #[test]
    fn a_geometry_shaders_inputs_print_their_empty_bracket() {
        let s = text::parse(
            "GEOM\nPROPERTY GS_INPUT_PRIMITIVE TRIANGLES\nDCL IN[][0], POSITION\n  0: END\n"
                .as_bytes(),
            100,
        )
        .expect("parses");
        assert_eq!(
            dump(&s),
            "GEOM\nPROPERTY GS_INPUT_PRIMITIVE TRIANGLES\nDCL IN[][0], POSITION\n  0: END\n"
        );
    }

    #[test]
    fn printfs_float_spellings() {
        let mut out = String::new();
        cfloat(&mut out, -1.0, 4);
        assert_eq!(out, "   -1.0000");
        out.clear();
        cfloat(&mut out, f64::NAN, 4);
        assert_eq!(out, "       nan");
    }
}
