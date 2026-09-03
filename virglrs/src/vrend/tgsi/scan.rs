// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! `tgsi_scan_shader`, reduced to what the translator reads and to what the scan refuses.
//!
//! The C's `tgsi_shader_info` counts and classifies most of a program; `vrend_shader.c` reads
//! three fields of it -- which register files are indexed indirectly, which are indexed
//! indirectly in their second dimension, and the processor -- and does its own counting for
//! everything else. The rest of the C scan matters here only for the programs it refuses, and
//! those refusals are reproduced in full: a program the C's scan rejects is rejected by the same
//! rule, before any translation.

use super::*;
use std::fmt;

/// `PIPE_MAX_SHADER_INPUTS` and `PIPE_MAX_SHADER_OUTPUTS`.
const MAX_SHADER_INPUTS: u32 = 80;
const MAX_SHADER_OUTPUTS: u32 = 80;
/// `PIPE_MAX_SAMPLERS`.
const MAX_SAMPLERS: u32 = 32;
/// `PIPE_MAX_CONSTANT_BUFFERS`.
const MAX_CONSTANT_BUFFERS: u32 = 32;

/// Why the scan said no, in the C's words.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Refusal(pub String);

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TGSI Error: {}", self.0)
    }
}

impl std::error::Error for Refusal {}

/// The scan's findings that the translator reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Info {
    pub processor: Processor,
    /// A bit per [`File`] ordinal: the files some register of which is indexed indirectly.
    pub indirect_files: u32,
    /// A bit per [`File`] ordinal: the files whose second dimension is indexed indirectly.
    pub dimension_indirect_files: u32,
}

impl Info {
    pub fn is_indirect(&self, file: File) -> bool {
        self.indirect_files & (1 << file as u8) != 0
    }

    pub fn is_dimension_indirect(&self, file: File) -> bool {
        self.dimension_indirect_files & (1 << file as u8) != 0
    }
}

fn refuse<T>(message: String) -> Result<T, Refusal> {
    Err(Refusal(message))
}

/// Scan `shader`, refusing what the C refuses.
pub fn scan(shader: &Shader) -> Result<Info, Refusal> {
    let mut info =
        Info { processor: shader.processor, indirect_files: 0, dimension_indirect_files: 0 };
    let mut input_array_first = [0u16; MAX_SHADER_INPUTS as usize];
    let mut num_inputs = 0u32;
    let mut num_outputs = 0u32;

    for token in &shader.tokens {
        match token {
            Token::Instruction(inst) => {
                if matches!(
                    inst.opcode,
                    Opcode::InterpCentroid | Opcode::InterpOffset | Opcode::InterpSample
                ) {
                    let src0 = &inst.src[0];
                    // The C reads the index as an `unsigned`, so a negative one is out of range.
                    let input = if src0.indirect && src0.ind.array_id != 0 {
                        let id = u32::from(src0.ind.array_id);
                        if id >= MAX_SHADER_INPUTS {
                            return refuse(format!("Indirect ArrayID {id} exeeds supported limit"));
                        }
                        u32::from(input_array_first[id as usize])
                    } else {
                        src0.index as u32
                    };
                    if input >= MAX_SHADER_INPUTS {
                        return refuse(format!("input {input} exeeds supported limit"));
                    }
                }
                for src in inst.srcs() {
                    if src.file == File::Input && !src.indirect {
                        let ind = i32::from(src.index);
                        if ind < 0 || ind >= MAX_SHADER_INPUTS as i32 {
                            return refuse(format!("input {ind} exeeds supported limit"));
                        }
                    }
                    if src.indirect {
                        info.indirect_files |= 1 << src.file as u8;
                    }
                    if src.dimension && src.dim.indirect {
                        info.dimension_indirect_files |= 1 << src.file as u8;
                    }
                    if src.file == File::Sampler {
                        if inst.texture.is_none() {
                            return refuse("unspecified sampler instruction texture".into());
                        }
                        if src.index as u32 >= MAX_SAMPLERS {
                            return refuse(format!("sampler ID {} out of range", src.index));
                        }
                    }
                }
                for dst in inst.dsts() {
                    if dst.indirect {
                        info.indirect_files |= 1 << dst.file as u8;
                    }
                    if dst.dimension && dst.dim.indirect {
                        info.dimension_indirect_files |= 1 << dst.file as u8;
                    }
                }
            }
            Token::Declaration(decl) => {
                if let Some(array_id) = decl.array {
                    let id = u32::from(array_id);
                    match decl.file {
                        File::Input => {
                            if id >= MAX_SHADER_INPUTS {
                                return refuse(format!(
                                    "input array ID {id} exeeds supported limit"
                                ));
                            }
                            input_array_first[id as usize] = decl.first;
                        }
                        File::Output if id >= MAX_SHADER_OUTPUTS => {
                            return refuse(format!("output array ID {id} exeeds supported limit"));
                        }
                        _ => {}
                    }
                }
                for reg in u32::from(decl.first)..=u32::from(decl.last) {
                    match decl.file {
                        File::Constant => {
                            let buffer = u32::from(decl.dimension.unwrap_or(0));
                            if buffer >= MAX_CONSTANT_BUFFERS {
                                return refuse(format!(
                                    "constant buffer id {buffer} exeeds supported limit"
                                ));
                            }
                        }
                        File::Input => {
                            if reg >= MAX_SHADER_INPUTS {
                                return refuse(format!(
                                    "input register {reg} exeeds supported limit"
                                ));
                            }
                            num_inputs += 1;
                            if num_inputs >= MAX_SHADER_INPUTS {
                                return refuse(format!(
                                    "mumber of inputs {num_inputs} exeeds supported limit"
                                ));
                            }
                        }
                        File::SystemValue => {
                            let index = u32::from(decl.first);
                            if index >= MAX_SHADER_INPUTS {
                                return refuse(format!(
                                    "system value {index} exeeds supported limit"
                                ));
                            }
                        }
                        File::Output => {
                            if reg >= MAX_SHADER_OUTPUTS {
                                return refuse(format!("output {reg} exeeds supported limit"));
                            }
                            num_outputs += 1;
                            if num_outputs >= MAX_SHADER_OUTPUTS {
                                return refuse(format!(
                                    "number of outputs {num_outputs} exeeds supported  limit"
                                ));
                            }
                        }
                        _ => {}
                    }
                }
            }
            Token::Immediate(_) | Token::Property(_) => {}
        }
    }
    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scanned(text: &str) -> Result<Info, Refusal> {
        scan(&text::parse(text.as_bytes(), 10000).expect("parses"))
    }

    #[test]
    fn indirect_files_are_noted_by_file() {
        let info = scanned(
            "VERT\nDCL IN[0]\nDCL OUT[0], POSITION\nDCL CONST[1][0..3]\nDCL TEMP[0..3]\nDCL ADDR[0]\n\
             0: MOV TEMP[ADDR[0].x], CONST[ADDR[0].x][1]\n1: MOV OUT[0], TEMP[0]\n2: END\n",
        )
        .expect("scans");
        assert_eq!(info.processor, Processor::Vertex);
        assert!(info.is_indirect(File::Temporary));
        assert!(!info.is_indirect(File::Constant));
        assert!(info.is_dimension_indirect(File::Constant));
        assert!(!info.is_dimension_indirect(File::Temporary));
    }

    #[test]
    fn the_scans_refusals() {
        let refused = |text: &str| scanned(text).expect_err("refused").0;
        assert_eq!(
            refused(
                "FRAG\nDCL SAMP[0]\nDCL IN[0], GENERIC[0]\nDCL OUT[0], COLOR\n0: MOV OUT[0], SAMP[0]\n1: END\n"
            ),
            "unspecified sampler instruction texture"
        );
        assert_eq!(
            refused(
                "FRAG\nDCL IN[0], GENERIC[0]\nDCL OUT[0], COLOR\n0: TEX OUT[0], IN[0], SAMP[40], 2D\n1: END\n"
            ),
            "sampler ID 40 out of range"
        );
        assert_eq!(
            refused("VERT\nDCL CONST[32][0]\n0: END\n"),
            "constant buffer id 32 exeeds supported limit"
        );
        assert_eq!(
            refused("VERT\nDCL IN[80]\n0: END\n"),
            "input register 80 exeeds supported limit"
        );
        assert_eq!(
            refused("VERT\nDCL IN[0..78]\nDCL IN[79]\n0: END\n"),
            "mumber of inputs 80 exeeds supported limit"
        );
        assert_eq!(
            refused("VERT\n0: MOV TEMP[0], IN[80]\n1: END\n"),
            "input 80 exeeds supported limit"
        );
        assert_eq!(
            refused("VERT\nDCL OUT[0..79]\n0: END\n"),
            "number of outputs 80 exeeds supported  limit"
        );
    }
}
