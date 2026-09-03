// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! `tgsi_text.c`: the text a guest sends, parsed into a [`Shader`].
//!
//! A faithful port, down to what it accepts and refuses: the same case-insensitive keyword
//! matching, the same optional pieces in the same order, the same silent truncations. The one
//! thing it does not reproduce is the C's diagnostic printing; an error here carries the message
//! and the line and column the C would have printed, for the log.
//!
//! The guest also says how many packed tokens its program takes (`num_tokens`), and the C
//! allocates that many plus ten and fails the parse when the program does not fit. That count is
//! reproduced from the typed program (`Shader::token_count`) and checked the same way, so the
//! boundary refuses exactly what the C refuses.

use super::info::Info;
use super::*;
use crate::vrend::proto::{FORMAT_MAX, Format};
use std::fmt;

/// Why the text was refused, with where in it the C would have pointed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Error {
    pub message: String,
    pub line: u32,
    pub column: u32,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TGSI asm error: {} [{} : {}]", self.message, self.line, self.column)
    }
}

impl std::error::Error for Error {}

fn is_alpha_underscore(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

fn is_digit(c: u8) -> bool {
    c.is_ascii_digit()
}

fn is_digit_alpha_underscore(c: u8) -> bool {
    is_digit(c) || is_alpha_underscore(c)
}

fn uprcase(c: u8) -> u8 {
    c.to_ascii_uppercase()
}

/// A cursor over the text. The text is what the guest sent up to its first NUL, and reading past
/// the end answers NUL, as the C's terminated buffer does.
#[derive(Clone, Copy)]
struct Cur<'a> {
    text: &'a [u8],
    pos: usize,
}

impl<'a> Cur<'a> {
    fn at(&self, off: usize) -> u8 {
        self.text.get(self.pos + off).copied().unwrap_or(0)
    }

    fn c(&self) -> u8 {
        self.at(0)
    }

    fn advance(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.text.len());
    }

    /// `str_match_no_case`: `s` (upper case) is a prefix of the text, case-insensitively.
    fn match_no_case(&mut self, s: &str) -> bool {
        let mut cur = *self;
        for &b in s.as_bytes() {
            if b != uprcase(cur.c()) {
                return false;
            }
            cur.advance(1);
        }
        *self = cur;
        true
    }

    /// `str_match_nocase_whole`: as above, and the word ends there.
    fn match_whole(&mut self, s: &str) -> bool {
        let mut cur = *self;
        if cur.match_no_case(s) && !is_digit_alpha_underscore(cur.c()) {
            *self = cur;
            return true;
        }
        false
    }

    /// `str_match_name_from_array`.
    fn match_name<T: Copy>(&mut self, names: &[T], name: impl Fn(T) -> &'static str) -> Option<T> {
        names.iter().copied().find(|&t| self.match_whole(name(t)))
    }

    fn eat_until_eol(&mut self) {
        while self.c() != 0 && self.c() != b'\n' {
            self.advance(1);
        }
    }

    fn eat_opt_white(&mut self) {
        while matches!(self.c(), b' ' | b'\t' | b'\n') {
            self.advance(1);
        }
    }

    /// `eat_white`: at least one whitespace character.
    fn eat_white(&mut self) -> bool {
        let start = self.pos;
        self.eat_opt_white();
        self.pos > start
    }

    /// `parse_uint`: decimal digits, wrapping as the C's `unsigned` does.
    fn parse_uint(&mut self) -> Option<u32> {
        if !is_digit(self.c()) {
            return None;
        }
        let mut v: u32 = 0;
        while is_digit(self.c()) {
            v = v.wrapping_mul(10).wrapping_add(u32::from(self.c() - b'0'));
            self.advance(1);
        }
        Some(v)
    }

    /// `parse_int`: an optional sign and digits, as the C computes it -- an unsigned parse
    /// reinterpreted, then negated.
    fn parse_int(&mut self) -> Option<i32> {
        let mut cur = *self;
        let neg = cur.c() == b'-';
        if matches!(cur.c(), b'+' | b'-') {
            cur.advance(1);
        }
        let v = cur.parse_uint()? as i32;
        *self = cur;
        Some(if neg { v.wrapping_neg() } else { v })
    }

    /// `parse_identifier`, into a buffer of 64 as the C's is.
    fn parse_identifier(&mut self) -> Option<String> {
        if !is_alpha_underscore(self.c()) {
            return None;
        }
        let mut cur = *self;
        let mut out = Vec::new();
        out.push(cur.c());
        cur.advance(1);
        while is_alpha_underscore(cur.c()) || is_digit(cur.c()) {
            if out.len() == 63 {
                return None;
            }
            out.push(cur.c());
            cur.advance(1);
        }
        *self = cur;
        Some(String::from_utf8(out).expect("ASCII"))
    }

    /// `strtoul(cur, NULL, 16)`: optional whitespace and sign, optional `0x`, hex digits;
    /// saturating on overflow as `strtoul` does. Consumes nothing -- the C skips a fixed width
    /// after.
    fn strtoul16(&self) -> u64 {
        let mut cur = *self;
        while matches!(cur.c(), b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c) {
            cur.advance(1);
        }
        let neg = cur.c() == b'-';
        if matches!(cur.c(), b'+' | b'-') {
            cur.advance(1);
        }
        if cur.c() == b'0' && uprcase(cur.at(1)) == b'X' && cur.at(2).is_ascii_hexdigit() {
            cur.advance(2);
        }
        let mut v: u64 = 0;
        let mut overflow = false;
        while cur.c().is_ascii_hexdigit() {
            let d = u64::from((cur.c() as char).to_digit(16).expect("hex"));
            match v.checked_mul(16).and_then(|v| v.checked_add(d)) {
                Some(n) => v = n,
                None => overflow = true,
            }
            cur.advance(1);
        }
        if overflow {
            return u64::MAX;
        }
        if neg { v.wrapping_neg() } else { v }
    }

    /// `skip_n_chars`: advance `n`, unless a NUL comes first.
    fn skip_n(&mut self, n: usize) -> bool {
        for i in 0..n {
            if self.at(i) == 0 {
                self.advance(i);
                return false;
            }
        }
        self.advance(n);
        true
    }

    /// `parse_float`.
    fn parse_float(&mut self) -> Option<f32> {
        let mut cur = *self;
        if cur.c() == b'0' && cur.at(1) == b'x' {
            let v = f32::from_bits(cur.strtoul16() as u32);
            if !cur.skip_n(10) {
                return None;
            }
            *self = cur;
            return Some(v);
        }
        let start = cur.pos;
        let (mut integral, mut fractional) = (false, false);
        if matches!(cur.c(), b'-' | b'+') {
            cur.advance(1);
        }
        if is_digit(cur.c()) {
            integral = true;
            while is_digit(cur.c()) {
                cur.advance(1);
            }
        }
        if cur.c() == b'.' {
            cur.advance(1);
            if is_digit(cur.c()) {
                fractional = true;
                while is_digit(cur.c()) {
                    cur.advance(1);
                }
            }
        }
        if !integral && !fractional {
            return None;
        }
        if uprcase(cur.c()) == b'E' {
            cur.advance(1);
            if matches!(cur.c(), b'-' | b'+') {
                cur.advance(1);
            }
            if is_digit(cur.c()) {
                while is_digit(cur.c()) {
                    cur.advance(1);
                }
            } else {
                return None;
            }
        }
        // `atof` over the same characters: a correctly rounded double, then a float.
        let s = std::str::from_utf8(&cur.text[start..cur.pos]).expect("ASCII");
        let v = s.parse::<f64>().unwrap_or(0.0) as f32;
        *self = cur;
        Some(v)
    }

    /// The prefix `strtod` would consume, as a decimal number (no hexadecimal, infinity or NaN
    /// spelling), and its value. `None` if it would consume nothing.
    fn strtod(&mut self) -> Option<f64> {
        let mut cur = *self;
        while matches!(cur.c(), b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c) {
            cur.advance(1);
        }
        let start = cur.pos;
        if matches!(cur.c(), b'-' | b'+') {
            cur.advance(1);
        }
        let mut digits = 0;
        while is_digit(cur.c()) {
            cur.advance(1);
            digits += 1;
        }
        if cur.c() == b'.' {
            let mut frac = cur;
            frac.advance(1);
            let mut n = 0;
            while is_digit(frac.c()) {
                frac.advance(1);
                n += 1;
            }
            if digits + n > 0 {
                cur = frac;
                digits += n;
            }
        }
        if digits == 0 {
            return None;
        }
        if uprcase(cur.c()) == b'E' {
            let mut exp = cur;
            exp.advance(1);
            if matches!(exp.c(), b'-' | b'+') {
                exp.advance(1);
            }
            if is_digit(exp.c()) {
                while is_digit(exp.c()) {
                    exp.advance(1);
                }
                cur = exp;
            }
        }
        let s = std::str::from_utf8(&cur.text[start..cur.pos]).expect("ASCII");
        let v = s.parse::<f64>().unwrap_or(0.0);
        *self = cur;
        Some(v)
    }

    /// `parse_double`: two words of a double.
    fn parse_double(&mut self) -> Option<[u32; 2]> {
        let mut cur = *self;
        if cur.c() == b'0' && cur.at(1) == b'x' {
            let lo = cur.strtoul16() as u32;
            if !cur.skip_n(11) {
                return None;
            }
            let hi = cur.strtoul16() as u32;
            if !cur.skip_n(11) {
                return None;
            }
            *self = cur;
            return Some([lo, hi]);
        }
        let v = cur.strtod()?;
        *self = cur;
        let bits = v.to_bits();
        Some([bits as u32, (bits >> 32) as u32])
    }

    /// `strtoll`/`strtoull` with base 0: whitespace, sign, then hexadecimal with `0x`, octal
    /// with a leading `0`, else decimal. `None` when nothing was consumed.
    fn strto64(&mut self, signed: bool) -> Option<u64> {
        let mut cur = *self;
        while matches!(cur.c(), b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c) {
            cur.advance(1);
        }
        let neg = cur.c() == b'-';
        if matches!(cur.c(), b'+' | b'-') {
            cur.advance(1);
        }
        let base = if cur.c() == b'0' && uprcase(cur.at(1)) == b'X' && cur.at(2).is_ascii_hexdigit()
        {
            cur.advance(2);
            16
        } else if cur.c() == b'0' {
            8
        } else {
            10
        };
        let mut v: u64 = 0;
        let mut any = false;
        let mut overflow = false;
        while let Some(d) = (cur.c() as char).to_digit(base) {
            any = true;
            match v.checked_mul(u64::from(base)).and_then(|v| v.checked_add(u64::from(d))) {
                Some(n) => v = n,
                None => overflow = true,
            }
            cur.advance(1);
        }
        if !any {
            return None;
        }
        *self = cur;
        if overflow {
            return Some(if signed {
                if neg { i64::MIN as u64 } else { i64::MAX as u64 }
            } else {
                u64::MAX
            });
        }
        if signed {
            let limit = if neg { 1u64 << 63 } else { (1u64 << 63) - 1 };
            if v > limit {
                return Some(if neg { i64::MIN as u64 } else { i64::MAX as u64 });
            }
        }
        Some(if neg { v.wrapping_neg() } else { v })
    }

    fn parse_int64(&mut self) -> Option<[u32; 2]> {
        let v = self.strto64(true)?;
        Some([v as u32, (v >> 32) as u32])
    }

    fn parse_uint64(&mut self) -> Option<[u32; 2]> {
        let v = self.strto64(false)?;
        Some([v as u32, (v >> 32) as u32])
    }
}

/// `parsed_bracket`: what sits between a register's brackets.
#[derive(Clone, Copy)]
struct Bracket {
    index: i32,
    ind_file: File,
    ind_index: i32,
    ind_comp: u8,
    ind_array: u32,
}

impl Default for Bracket {
    fn default() -> Bracket {
        Bracket { index: 0, ind_file: File::Null, ind_index: 0, ind_comp: 0, ind_array: 0 }
    }
}

impl Bracket {
    fn ind_reg(&self) -> IndReg {
        IndReg {
            file: self.ind_file,
            index: self.ind_index as i16,
            swizzle: self.ind_comp,
            array_id: (self.ind_array & 0x3ff) as u16,
        }
    }
}

/// What a declaration's register part names: the file, the range, and the second index of a
/// two-dimensional declaration.
struct DclRegister {
    file: File,
    range: (u32, u32),
    dimension: Option<u32>,
}

struct Parser<'a> {
    text: &'a [u8],
    cur: Cur<'a>,
    processor: Processor,
    /// Six bits in the C.
    implied_array_size: u32,
    num_immediates: u32,
    tokens: Vec<Token>,
    /// The packed tokens built so far, against `max_tokens`.
    count: u32,
    max_tokens: u32,
}

type R<T> = Result<T, Error>;

impl<'a> Parser<'a> {
    fn error<T>(&self, message: impl Into<String>) -> R<T> {
        let (mut line, mut column) = (1, 1);
        for &b in &self.text[..self.cur.pos] {
            if b == b'\n' {
                column = 1;
                line += 1;
            }
            column += 1;
        }
        Err(Error { message: message.into(), line, column })
    }

    /// The C's silent failure: a token that does not fit the buffer the guest sized.
    fn push(&mut self, token: Token) -> R<()> {
        self.count += token.token_count();
        if self.count > self.max_tokens {
            return Err(Error {
                message: format!(
                    "the program does not fit the {} tokens the guest declared",
                    self.max_tokens
                ),
                line: 0,
                column: 0,
            });
        }
        self.tokens.push(token);
        Ok(())
    }

    fn parse_header(&mut self) -> R<()> {
        let processor = if self.cur.match_whole("FRAG") {
            Processor::Fragment
        } else if self.cur.match_whole("VERT") {
            Processor::Vertex
        } else if self.cur.match_whole("GEOM") {
            Processor::Geometry
        } else if self.cur.match_whole("TESS_CTRL") {
            Processor::TessCtrl
        } else if self.cur.match_whole("TESS_EVAL") {
            Processor::TessEval
        } else if self.cur.match_whole("COMP") {
            Processor::Compute
        } else {
            return self.error("Unknown header");
        };
        self.count = 2;
        if self.count > self.max_tokens {
            return Err(Error { message: "no room for the header".into(), line: 0, column: 0 });
        }
        self.processor = processor;
        Ok(())
    }

    fn parse_label(&mut self) -> Option<u32> {
        let mut cur = self.cur;
        let val = cur.parse_uint()?;
        cur.eat_opt_white();
        if cur.c() == b':' {
            cur.advance(1);
            self.cur = cur;
            return Some(val);
        }
        None
    }

    fn parse_file(cur: &mut Cur<'a>) -> Option<File> {
        cur.match_name(File::ALL, File::name)
    }

    fn parse_opt_writemask(&mut self) -> R<u8> {
        let mut cur = self.cur;
        cur.eat_opt_white();
        if cur.c() != b'.' {
            return Ok(WRITEMASK_XYZW);
        }
        cur.advance(1);
        let mut mask = 0;
        cur.eat_opt_white();
        for (ch, bit) in
            [(b'X', WRITEMASK_X), (b'Y', WRITEMASK_Y), (b'Z', WRITEMASK_Z), (b'W', WRITEMASK_W)]
        {
            if uprcase(cur.c()) == ch {
                cur.advance(1);
                mask |= bit;
            }
        }
        if mask == 0 {
            return self.error("Writemask expected");
        }
        self.cur = cur;
        Ok(mask)
    }

    /// `parse_register_file_bracket`: `<file> [`.
    fn parse_register_file_bracket(&mut self) -> R<File> {
        let Some(file) = Self::parse_file(&mut self.cur) else {
            return self.error("Unknown register file");
        };
        self.cur.eat_opt_white();
        if self.cur.c() != b'[' {
            return self.error("Expected `['");
        }
        self.cur.advance(1);
        Ok(file)
    }

    /// `parse_register_1d`: `<file> [ <uint> ]`.
    fn parse_register_1d(&mut self) -> R<(File, i32)> {
        let file = self.parse_register_file_bracket()?;
        self.cur.eat_opt_white();
        let Some(index) = self.cur.parse_uint() else {
            return self.error("Expected literal unsigned integer");
        };
        self.cur.eat_opt_white();
        if self.cur.c() != b']' {
            return self.error("Expected `]'");
        }
        self.cur.advance(1);
        Ok((file, index as i32))
    }

    /// `parse_register_bracket`: what follows an opening bracket, through the close and an
    /// optional `(array)`.
    fn parse_register_bracket(&mut self) -> R<Bracket> {
        let mut b = Bracket::default();
        self.cur.eat_opt_white();
        let mut probe = self.cur;
        if Self::parse_file(&mut probe).is_some() {
            let (file, index) = self.parse_register_1d()?;
            b.ind_file = file;
            b.ind_index = index;
            self.cur.eat_opt_white();
            if self.cur.c() == b'.' {
                self.cur.advance(1);
                self.cur.eat_opt_white();
                b.ind_comp = match uprcase(self.cur.c()) {
                    b'X' => SWIZZLE_X,
                    b'Y' => SWIZZLE_Y,
                    b'Z' => SWIZZLE_Z,
                    b'W' => SWIZZLE_W,
                    _ => {
                        return self.error(
                            "Expected indirect register swizzle component `x', `y', `z' or `w'",
                        );
                    }
                };
                self.cur.advance(1);
                self.cur.eat_opt_white();
            }
            if matches!(self.cur.c(), b'+' | b'-') {
                if let Some(i) = self.cur.parse_int() {
                    b.index = i;
                }
            } else {
                b.index = 0;
            }
        } else {
            let Some(index) = self.cur.parse_int() else {
                return self.error("Expected literal integer");
            };
            b.index = index;
            b.ind_file = File::Null;
            b.ind_index = 0;
        }
        self.cur.eat_opt_white();
        if self.cur.c() != b']' {
            return self.error("Expected `]'");
        }
        self.cur.advance(1);
        if self.cur.c() == b'(' {
            self.cur.advance(1);
            self.cur.eat_opt_white();
            let Some(a) = self.cur.parse_uint() else {
                return self.error("Expected literal unsigned integer");
            };
            b.ind_array = a;
            self.cur.eat_opt_white();
            if self.cur.c() != b')' {
                return self.error("Expected `)'");
            }
            self.cur.advance(1);
        }
        Ok(b)
    }

    /// `parse_opt_register_src_bracket`: a second `[...]`, if there is one.
    fn parse_opt_bracket(&mut self) -> R<Option<Bracket>> {
        let mut cur = self.cur;
        cur.eat_opt_white();
        if cur.c() == b'[' {
            cur.advance(1);
            self.cur = cur;
            return Ok(Some(self.parse_register_bracket()?));
        }
        Ok(None)
    }

    /// `parse_register_src` and `parse_register_dst`, which are the same.
    fn parse_register(&mut self) -> R<(File, Bracket)> {
        let file = self.parse_register_file_bracket()?;
        let mut b = self.parse_register_bracket()?;
        if b.ind_file == File::Null {
            b.ind_comp = SWIZZLE_X;
        }
        Ok((file, b))
    }

    /// `parse_register_dcl_bracket`: `<uint> ]`, `<uint> .. <uint> ]`, or `]` for an implied
    /// range.
    fn parse_register_dcl_bracket(&mut self) -> R<(u32, u32)> {
        self.cur.eat_opt_white();
        let first = match self.cur.parse_uint() {
            Some(f) => f,
            None => {
                if self.cur.c() == b']' && self.implied_array_size != 0 {
                    self.cur.advance(1);
                    return Ok((0, self.implied_array_size - 1));
                }
                return self.error("Expected literal unsigned integer");
            }
        };
        self.cur.eat_opt_white();
        let last = if self.cur.c() == b'.' && self.cur.at(1) == b'.' {
            self.cur.advance(2);
            self.cur.eat_opt_white();
            let Some(l) = self.cur.parse_uint() else {
                return self.error("Expected literal integer");
            };
            self.cur.eat_opt_white();
            l
        } else {
            first
        };
        if self.cur.c() != b']' {
            return self.error("Expected `]' or `..'");
        }
        self.cur.advance(1);
        Ok((first, last))
    }

    /// `parse_register_dcl`: a file and one or two bracketed ranges.
    fn parse_register_dcl(&mut self) -> R<DclRegister> {
        let file = self.parse_register_file_bracket()?;
        let range = self.parse_register_dcl_bracket()?;
        let mut reg = DclRegister { file, range, dimension: None };
        let mut cur = self.cur;
        cur.eat_opt_white();
        if cur.c() == b'[' {
            let is_in = file == File::Input;
            let is_out = file == File::Output;
            cur.advance(1);
            self.cur = cur;
            let second = self.parse_register_dcl_bracket()?;
            // A geometry or tessellation shader's per-vertex arrays: the first bracket is the
            // primitive's size and only the second names registers.
            if (self.processor == Processor::Geometry && is_in)
                || (self.processor == Processor::TessEval && is_in)
                || (self.processor == Processor::TessCtrl && (is_in || is_out))
            {
                reg.range = second;
            } else {
                reg.dimension = Some(range.0);
                reg.range = second;
            }
        }
        Ok(reg)
    }

    fn parse_dst_operand(&mut self) -> R<Dst> {
        let (file, first) = self.parse_register()?;
        let second = self.parse_opt_bracket()?;
        let writemask = self.parse_opt_writemask()?;
        let mut dst = Dst { file, writemask, ..Dst::default() };
        let mut b = first;
        if let Some(second) = second {
            dst.dimension = true;
            dst.dim = Dimension { indirect: false, index: first.index as i16 };
            if first.ind_file != File::Null {
                dst.dim.indirect = true;
                dst.dim_ind = first.ind_reg();
            }
            b = second;
        }
        dst.index = b.index as i16;
        if b.ind_file != File::Null {
            dst.indirect = true;
            dst.ind = b.ind_reg();
        }
        Ok(dst)
    }

    /// `parse_optional_swizzle`: `.` and `components` letters.
    fn parse_optional_swizzle(&mut self, components: usize) -> R<Option<[u8; 4]>> {
        let mut cur = self.cur;
        cur.eat_opt_white();
        if cur.c() != b'.' {
            return Ok(None);
        }
        cur.advance(1);
        cur.eat_opt_white();
        let mut sw = [0u8; 4];
        for s in sw.iter_mut().take(components) {
            *s = match uprcase(cur.c()) {
                b'X' => SWIZZLE_X,
                b'Y' => SWIZZLE_Y,
                b'Z' => SWIZZLE_Z,
                b'W' => SWIZZLE_W,
                _ => return self.error("Expected register swizzle component `x', `y', `z' or `w'"),
            };
            cur.advance(1);
        }
        self.cur = cur;
        Ok(Some(sw))
    }

    fn parse_src_operand(&mut self) -> R<Src> {
        let mut src = Src::default();
        if self.cur.c() == b'-' {
            self.cur.advance(1);
            self.cur.eat_opt_white();
            src.negate = true;
        }
        if self.cur.c() == b'|' {
            self.cur.advance(1);
            self.cur.eat_opt_white();
            src.absolute = true;
        }
        let (file, first) = self.parse_register()?;
        let second = self.parse_opt_bracket()?;
        src.file = file;
        let mut b = first;
        if let Some(second) = second {
            src.dimension = true;
            src.dim = Dimension { indirect: false, index: first.index as i16 };
            if first.ind_file != File::Null {
                src.dim.indirect = true;
                src.dim_ind = first.ind_reg();
            }
            b = second;
        }
        src.index = b.index as i16;
        if b.ind_file != File::Null {
            src.indirect = true;
            src.ind = b.ind_reg();
        }
        if let Some(sw) = self.parse_optional_swizzle(4)? {
            src.swizzle = sw;
        }
        if src.absolute {
            self.cur.eat_opt_white();
            if self.cur.c() != b'|' {
                return self.error("Expected `|'");
            }
            self.cur.advance(1);
        }
        Ok(src)
    }

    fn parse_texoffset_operand(&mut self) -> R<TexOffset> {
        let (file, b) = self.parse_register()?;
        let mut off = TexOffset { file, index: b.index as i16, swizzle: [0; 3] };
        if let Some(sw) = self.parse_optional_swizzle(3)? {
            off.swizzle = [sw[0], sw[1], sw[2]];
        }
        Ok(off)
    }

    /// `match_inst`: the mnemonic, with its optional `_SAT` and `_PRECISE` suffixes.
    fn match_inst(cur: &mut Cur<'a>, info: &Info) -> Option<(bool, bool)> {
        let mut c = *cur;
        if c.match_whole(info.mnemonic) {
            *cur = c;
            return Some((false, false));
        }
        if c.match_no_case(info.mnemonic) {
            let (mut sat, mut precise) = (false, false);
            if c.match_no_case("_SAT") {
                *cur = c;
                sat = true;
            }
            if c.match_no_case("_PRECISE") {
                *cur = c;
                precise = true;
            }
            if !is_digit_alpha_underscore(c.c()) {
                return Some((sat, precise));
            }
        }
        None
    }

    fn parse_instruction(&mut self, has_label: bool) -> R<()> {
        self.cur.eat_opt_white();
        let mut found = None;
        for &op in Opcode::ALL {
            let mut cur = self.cur;
            let info = op.info();
            if let Some((sat, precise)) = Self::match_inst(&mut cur, info) {
                // An instruction with operands needs whitespace (or the end) after its mnemonic.
                let operands = info.num_dst + info.num_src + u8::from(info.is_tex) != 0;
                if !operands || cur.c() == 0 || cur.eat_white() {
                    self.cur = cur;
                    found = Some((op, sat, precise));
                    break;
                }
            }
        }
        let Some((opcode, saturate, precise)) = found else {
            return if has_label {
                self.error("Unknown opcode")
            } else {
                self.error("Expected `DCL', `IMM' or a label")
            };
        };
        let info = opcode.info();
        let mut inst = Instruction {
            opcode,
            saturate,
            precise,
            num_dst: info.num_dst,
            num_src: info.num_src,
            label: None,
            texture: None,
            memory: None,
            dst: [Dst::default(); MAX_DST],
            src: [Src::default(); MAX_SRC],
            tex_offsets: [TexOffset::default(); MAX_TEX_OFFSETS],
        };
        // The SAMPLE family takes no target argument but carries the texture word, for its
        // offsets.
        if (Opcode::Sample..=Opcode::Gather4).contains(&opcode) {
            inst.texture = Some(TextureInfo { texture: Texture::Unknown, num_offsets: 0 });
        }
        if ((Opcode::Load..=Opcode::Atomimax).contains(&opcode) && opcode != Opcode::Barrier)
            || opcode == Opcode::Resq
        {
            inst.memory = Some(MemoryInfo::default());
        }
        let operands =
            usize::from(info.num_dst) + usize::from(info.num_src) + usize::from(info.is_tex);
        for i in 0..operands {
            if i > 0 {
                self.cur.eat_opt_white();
                if self.cur.c() != b',' {
                    return self.error("Expected `,'");
                }
                self.cur.advance(1);
                self.cur.eat_opt_white();
            }
            if i < usize::from(info.num_dst) {
                inst.dst[i] = self.parse_dst_operand()?;
            } else if i < usize::from(info.num_dst) + usize::from(info.num_src) {
                inst.src[i - usize::from(info.num_dst)] = self.parse_src_operand()?;
            } else {
                let Some(t) = self.cur.match_name(Texture::ALL, Texture::name) else {
                    return self.error("Expected texture target");
                };
                inst.texture = Some(TextureInfo { texture: t, num_offsets: 0 });
            }
        }
        let mut cur = self.cur;
        cur.eat_opt_white();
        let mut offsets = 0;
        while inst.texture.is_some() && cur.c() == b',' && offsets < MAX_TEX_OFFSETS {
            cur.advance(1);
            cur.eat_opt_white();
            self.cur = cur;
            inst.tex_offsets[offsets] = self.parse_texoffset_operand()?;
            offsets += 1;
            cur = self.cur;
            cur.eat_opt_white();
        }
        if let Some(t) = inst.texture.as_mut() {
            t.num_offsets = offsets as u8;
        }
        let mut cur = self.cur;
        cur.eat_opt_white();
        while let Some(mem) = inst.memory.as_mut().filter(|_| cur.c() == b',') {
            cur.advance(1);
            cur.eat_opt_white();
            if let Some(q) = cur.match_name(MemoryQualifier::ALL, MemoryQualifier::name) {
                mem.qualifier |= 1 << (q as u8);
            } else if let Some(t) = cur.match_name(Texture::ALL, Texture::name) {
                mem.texture = t;
            } else if let Some(f) = match_format(&mut cur) {
                mem.format = f;
            } else {
                self.cur = cur;
                return self.error("Expected memory qualifier, texture target, or format\n");
            }
            self.cur = cur;
            cur.eat_opt_white();
        }
        let mut cur = self.cur;
        cur.eat_opt_white();
        if info.is_branch && cur.c() == b':' {
            cur.advance(1);
            cur.eat_opt_white();
            let Some(target) = cur.parse_uint() else {
                return self.error("Expected a label");
            };
            inst.label = Some(target & 0xff_ffff);
            self.cur = cur;
        }
        self.push(Token::Instruction(inst))
    }

    /// `parse_immediate_data`: `{a, b, c, d}`. Its verdict is ignored by the C's caller, so the
    /// error is only remembered for the cursor position it leaves behind.
    fn parse_immediate_data(&mut self, ty: ImmType, values: &mut [u32; 4]) -> R<()> {
        self.cur.eat_opt_white();
        if self.cur.c() != b'{' {
            return self.error("Expected `{'");
        }
        self.cur.advance(1);
        let mut i = 0;
        while i < 4 {
            self.cur.eat_opt_white();
            if i > 0 {
                if self.cur.c() != b',' {
                    return self.error("Expected `,'");
                }
                self.cur.advance(1);
                self.cur.eat_opt_white();
            }
            let ok = match ty {
                ImmType::Float64 | ImmType::Int64 | ImmType::Uint64 => {
                    let pair = match ty {
                        ImmType::Float64 => self.cur.parse_double(),
                        ImmType::Int64 => self.cur.parse_int64(),
                        _ => self.cur.parse_uint64(),
                    };
                    match pair {
                        Some([lo, hi]) => {
                            values[i] = lo;
                            if i + 1 < 4 {
                                values[i + 1] = hi;
                            }
                            true
                        }
                        None => false,
                    }
                }
                ImmType::Float32 => match self.cur.parse_float() {
                    Some(f) => {
                        values[i] = f.to_bits();
                        true
                    }
                    None => false,
                },
                ImmType::Uint32 => match self.cur.parse_uint() {
                    Some(u) => {
                        values[i] = u;
                        true
                    }
                    None => false,
                },
                ImmType::Int32 => match self.cur.parse_int() {
                    Some(v) => {
                        values[i] = v as u32;
                        true
                    }
                    None => false,
                },
            };
            if matches!(ty, ImmType::Float64 | ImmType::Int64 | ImmType::Uint64) {
                i += 1;
            }
            if !ok {
                return self.error("Expected immediate constant");
            }
            i += 1;
        }
        self.cur.eat_opt_white();
        if self.cur.c() != b'}' {
            return self.error("Expected `}'");
        }
        self.cur.advance(1);
        Ok(())
    }

    fn parse_immediate(&mut self) -> R<()> {
        if self.cur.c() == b'[' {
            self.cur.advance(1);
            self.cur.eat_opt_white();
            let Some(index) = self.cur.parse_uint() else {
                return self.error("Expected literal unsigned integer");
            };
            if index != self.num_immediates {
                return self.error("Immediates must be sorted");
            }
            self.cur.eat_opt_white();
            if self.cur.c() != b']' {
                return self.error("Expected `]'");
            }
            self.cur.advance(1);
        }
        if !self.cur.eat_white() {
            return self.error("Syntax error");
        }
        let Some(ty) = self.cur.match_name(ImmType::ALL, ImmType::name) else {
            return self.error("Expected immediate type");
        };
        // The C discards this verdict: a malformed tuple leaves the cursor where it failed and
        // the immediate is built from what was parsed.
        let mut data = [0u32; 4];
        self.parse_immediate_data(ty, &mut data).ok();
        self.push(Token::Immediate(Immediate { ty, data }))?;
        self.num_immediates += 1;
        Ok(())
    }

    fn parse_declaration(&mut self) -> R<()> {
        if !self.cur.eat_white() {
            return self.error("Syntax error");
        }
        let reg = self.parse_register_dcl()?;
        let file = reg.file;
        let writemask = self.parse_opt_writemask()?;
        let mut decl = Declaration {
            file,
            usage_mask: writemask,
            first: reg.range.0 as u16,
            last: reg.range.1 as u16,
            dimension: reg.dimension.map(|d| d as u16),
            ..Declaration::default()
        };
        // The C asserts this when building the token, and a build without the assert carries
        // the inverted range into every `last - first` downstream. It is a guest's mistake, so
        // it is refused here rather than either.
        if decl.first > decl.last {
            return Err(Error {
                message: "a declaration's range ends before it starts".into(),
                line: 0,
                column: 0,
            });
        }
        let is_vs_input = file == File::Input && self.processor == Processor::Vertex;

        let mut cur = self.cur;
        cur.eat_opt_white();
        if cur.c() == b',' {
            let mut cur2 = cur;
            cur2.advance(1);
            cur2.eat_opt_white();
            if cur2.match_whole("ARRAY") {
                if cur2.c() != b'(' {
                    self.cur = cur2;
                    return self.error("Expected `('");
                }
                cur2.advance(1);
                cur2.eat_opt_white();
                let Some(id) = cur2.parse_int() else {
                    self.cur = cur2;
                    return self.error("Expected `,'");
                };
                cur2.eat_opt_white();
                if cur2.c() != b')' {
                    self.cur = cur2;
                    return self.error("Expected `)'");
                }
                cur2.advance(1);
                decl.array = Some((id as u32 & 0x3ff) as u16);
                self.cur = cur2;
                cur = cur2;
            }
        }

        if cur.c() == b',' && !is_vs_input {
            cur.advance(1);
            cur.eat_opt_white();
            match file {
                File::Image => {
                    let Some(t) = cur.match_name(Texture::ALL, Texture::name) else {
                        self.cur = cur;
                        return self.error("Expected texture target");
                    };
                    decl.image.resource = t;
                    let mut cur2 = cur;
                    cur2.eat_opt_white();
                    while cur2.c() == b',' {
                        cur2.advance(1);
                        cur2.eat_opt_white();
                        if cur2.match_whole("RAW") {
                            decl.image.raw = true;
                        } else if cur2.match_whole("WR") {
                            decl.image.writable = true;
                        } else {
                            let Some(f) = match_format(&mut cur2) else {
                                break;
                            };
                            decl.image.format = f;
                        }
                        cur = cur2;
                        cur2.eat_opt_white();
                    }
                    self.cur = cur;
                }
                File::SamplerView => {
                    let Some(t) = cur.match_name(Texture::ALL, Texture::name) else {
                        self.cur = cur;
                        return self.error("Expected texture target");
                    };
                    decl.sampler_view.resource = t;
                    cur.eat_opt_white();
                    if cur.c() != b',' {
                        self.cur = cur;
                        return self.error("Expected `,'");
                    }
                    cur.advance(1);
                    cur.eat_opt_white();
                    let mut j = 0;
                    while j < 4 {
                        match cur.match_name(ReturnType::ALL, ReturnType::name) {
                            Some(rt) => {
                                decl.sampler_view.return_type[j] = rt;
                                let mut cur2 = cur;
                                cur2.eat_opt_white();
                                if cur2.c() == b',' {
                                    cur2.advance(1);
                                    cur2.eat_opt_white();
                                    cur = cur2;
                                    j += 1;
                                    continue;
                                }
                                break;
                            }
                            None => {
                                if j == 0 || j > 2 {
                                    self.cur = cur;
                                    return self.error("Expected type name");
                                }
                                break;
                            }
                        }
                    }
                    if j < 4 {
                        let x = decl.sampler_view.return_type[0];
                        decl.sampler_view.return_type = [x; 4];
                    }
                    self.cur = cur;
                }
                File::Buffer => {
                    if cur.match_whole("ATOMIC") {
                        decl.atomic = true;
                        self.cur = cur;
                    }
                }
                File::Memory => {
                    if let Some(m) = cur.match_name(MemoryType::ALL, MemoryType::name) {
                        decl.mem_type = m;
                        self.cur = cur;
                    }
                }
                _ => {
                    if cur.match_whole("LOCAL") {
                        decl.local = true;
                        self.cur = cur;
                    }
                    let mut cur = self.cur;
                    cur.eat_opt_white();
                    if cur.c() == b',' {
                        cur.advance(1);
                        cur.eat_opt_white();
                        if let Some(sem) = cur.match_name(Semantic::ALL, Semantic::name) {
                            let mut cur2 = cur;
                            cur2.eat_opt_white();
                            if cur2.c() == b'[' {
                                cur2.advance(1);
                                cur2.eat_opt_white();
                                let Some(index) = cur2.parse_uint() else {
                                    self.cur = cur2;
                                    return self.error("Expected literal integer");
                                };
                                cur2.eat_opt_white();
                                if cur2.c() != b']' {
                                    self.cur = cur2;
                                    return self.error("Expected `]'");
                                }
                                cur2.advance(1);
                                decl.semantic.index = index as u16;
                                cur = cur2;
                            }
                            decl.has_semantic = true;
                            decl.semantic.name = sem;
                            self.cur = cur;
                        }
                    }
                }
            }
        }

        let mut cur = self.cur;
        cur.eat_opt_white();
        if cur.c() == b',' && file == File::Output && self.processor == Processor::Geometry {
            cur.advance(1);
            cur.eat_opt_white();
            if cur.match_whole("STREAM") {
                cur.eat_opt_white();
                if cur.c() != b'(' {
                    self.cur = cur;
                    return self.error("Expected '('");
                }
                cur.advance(1);
                let mut stream = [0u32; 4];
                for (i, s) in stream.iter_mut().enumerate() {
                    cur.eat_opt_white();
                    let Some(v) = cur.parse_uint() else {
                        self.cur = cur;
                        return self.error("Expected literal integer");
                    };
                    *s = v;
                    cur.eat_opt_white();
                    if i < 3 {
                        if cur.c() != b',' {
                            self.cur = cur;
                            return self.error("Expected ','");
                        }
                        cur.advance(1);
                    }
                }
                if cur.c() != b')' {
                    self.cur = cur;
                    return self.error("Expected ')'");
                }
                cur.advance(1);
                decl.semantic.stream = stream.map(|s| (s & 3) as u8);
                self.cur = cur;
            }
        }

        let mut cur = self.cur;
        cur.eat_opt_white();
        if cur.c() == b',' && !is_vs_input {
            cur.advance(1);
            cur.eat_opt_white();
            if let Some(i) = cur.match_name(Interpolate::ALL, Interpolate::name) {
                decl.has_interpolate = true;
                decl.interp.interpolate = i;
                self.cur = cur;
            }
        }

        let mut cur = self.cur;
        cur.eat_opt_white();
        if cur.c() == b',' && !is_vs_input {
            cur.advance(1);
            cur.eat_opt_white();
            if let Some(l) = cur.match_name(Location::ALL, Location::name) {
                decl.interp.location = l;
                self.cur = cur;
            }
        }

        let mut cur = self.cur;
        cur.eat_opt_white();
        if cur.c() == b',' && !is_vs_input {
            cur.advance(1);
            cur.eat_opt_white();
            if cur.match_whole("INVARIANT") {
                decl.invariant = true;
                self.cur = cur;
            } else {
                let tail: String =
                    cur.text[cur.pos..].iter().take(10).map(|&b| b as char).collect();
                self.cur = cur;
                return self.error(format!(
                    "Expected semantic, interpolate attribute, or invariant \"{tail}...\" "
                ));
            }
        }

        self.push(Token::Declaration(decl))
    }

    fn parse_property(&mut self) -> R<()> {
        if !self.cur.eat_white() {
            return self.error("Syntax error");
        }
        let Some(id) = self.cur.parse_identifier() else {
            return self.error("Syntax error");
        };
        let name = Property::ALL.iter().copied().find(|p| p.name().eq_ignore_ascii_case(&id));
        let Some(name) = name else {
            // The C reports and *accepts*: the rest of the line is skipped.
            self.cur.eat_until_eol();
            eprintln!("[virglrs] TGSI asm error: Unknown property : '{id}'");
            return Ok(());
        };
        self.cur.eat_opt_white();
        let value = match name {
            Property::GsInputPrim | Property::GsOutputPrim => {
                let Some(p) = self.cur.match_name(Primitive::ALL, Primitive::name) else {
                    return self.error("Unknown primitive name as property!");
                };
                if name == Property::GsInputPrim && self.processor == Processor::Geometry {
                    self.implied_array_size = p.vertices_per_prim() & 0x3f;
                }
                p as u32
            }
            Property::FsCoordOrigin => {
                let Some(o) = self.cur.match_name(CoordOrigin::ALL, CoordOrigin::name) else {
                    return self.error(
                        "Unknown coord origin as property: must be UPPER_LEFT or LOWER_LEFT!",
                    );
                };
                o as u32
            }
            Property::FsCoordPixelCenter => {
                let Some(c) = self.cur.match_name(PixelCenter::ALL, PixelCenter::name) else {
                    return self.error(
                        "Unknown coord pixel center as property: must be HALF_INTEGER or INTEGER!",
                    );
                };
                c as u32
            }
            Property::NextShader => {
                let Some(p) = self.cur.match_name(Processor::ALL, Processor::name) else {
                    return self.error("Unknown next shader property value.");
                };
                p as u32
            }
            _ => {
                let Some(v) = self.cur.parse_uint() else {
                    return self.error("Expected unsigned integer as property!");
                };
                v
            }
        };
        self.push(Token::Property(PropertyToken { name, data: value }))
    }

    fn translate(&mut self) -> R<()> {
        self.cur.eat_opt_white();
        self.parse_header()?;
        if matches!(self.processor, Processor::TessCtrl | Processor::TessEval) {
            self.implied_array_size = 32;
        }
        while self.cur.c() != 0 {
            if !self.cur.eat_white() {
                return self.error("Syntax error");
            }
            if self.cur.c() == 0 {
                break;
            }
            if self.parse_label().is_some() {
                self.parse_instruction(true)?;
            } else if self.cur.match_whole("DCL") {
                self.parse_declaration()?;
            } else if self.cur.match_whole("IMM") {
                self.parse_immediate()?;
            } else if self.cur.match_whole("PROPERTY") {
                self.parse_property()?;
            } else {
                self.parse_instruction(false)?;
            }
        }
        Ok(())
    }
}

/// `str_match_format`: a `PIPE_FORMAT_*` name, by the wire number it stands for.
fn match_format(cur: &mut Cur<'_>) -> Option<u16> {
    for n in 0..FORMAT_MAX {
        let Some(desc) = Format::from_wire(n).and_then(Format::describe) else {
            continue;
        };
        let name = format!("PIPE_FORMAT_{}", desc.name);
        if cur.match_whole(&name) {
            return Some(n as u16);
        }
    }
    None
}

/// `tgsi_text_translate`: parse `text` into a program that fits `max_tokens` packed tokens.
///
/// `text` is read up to its first NUL; what the guest sends is NUL-terminated and the C's parser
/// stops there.
pub fn parse(text: &[u8], max_tokens: u32) -> Result<Shader, Error> {
    let end = text.iter().position(|&b| b == 0).unwrap_or(text.len());
    let text = &text[..end];
    let mut p = Parser {
        text,
        cur: Cur { text, pos: 0 },
        processor: Processor::Fragment,
        implied_array_size: 0,
        num_immediates: 0,
        tokens: Vec::new(),
        count: 0,
        max_tokens,
    };
    p.translate()?;
    Ok(Shader { processor: p.processor, tokens: p.tokens })
}

#[cfg(test)]
mod tests {
    use super::*;

    const VS: &str = "VERT
DCL IN[0]
DCL IN[1]
DCL OUT[0], POSITION
DCL OUT[1], GENERIC[0]
DCL CONST[0..3]
DCL TEMP[0..1]
IMM[0] FLT32 {0x3f800000, 0x00000000, 0x40000000, 0x3f000000}
  0: MUL TEMP[0], CONST[0], IN[0].xxxx
  1: MAD TEMP[1].xy, CONST[1].xyyy, IN[1].yyyy, TEMP[0].xyxx
  2: MOV OUT[0], TEMP[1]
  3: MOV_SAT OUT[1], -|IN[1].wzyx|
  4: END
";

    #[test]
    fn a_vertex_shader_parses_into_its_tokens() {
        let s = parse(VS.as_bytes(), 1000).expect("parses");
        assert_eq!(s.processor, Processor::Vertex);
        assert_eq!(s.declarations().count(), 6);
        assert_eq!(s.immediates().count(), 1);
        assert_eq!(s.instructions().count(), 5);
        let out1 = s.declarations().nth(3).unwrap();
        assert!(out1.has_semantic);
        assert_eq!(out1.semantic.name, Semantic::Generic);
        assert_eq!(out1.semantic.index, 0);
        let consts = s.declarations().nth(4).unwrap();
        assert_eq!((consts.first, consts.last), (0, 3));
        let imm = s.immediates().next().unwrap();
        assert_eq!(imm.float(0), 1.0);
        assert_eq!(imm.float(2), 2.0);
        assert_eq!(imm.float(3), 0.5);
        let mad = s.instructions().nth(1).unwrap();
        assert_eq!(mad.opcode, Opcode::Mad);
        assert_eq!(mad.dst[0].writemask, WRITEMASK_XY);
        assert_eq!(mad.src[0].swizzle, [SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y]);
        let sat = s.instructions().nth(3).unwrap();
        assert!(sat.saturate);
        assert!(sat.src[0].negate && sat.src[0].absolute);
        assert_eq!(sat.src[0].swizzle, [SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X]);
        // Header 2, six declarations of 2 (+1 for each semantic), one immediate of 5, five
        // instructions of 1 + operands.
        assert_eq!(s.token_count(), 2 + 14 + 5 + (4 + 5 + 3 + 3 + 1));
    }

    #[test]
    fn the_guests_token_count_bounds_the_program() {
        let n = parse(VS.as_bytes(), 1000).unwrap().token_count();
        assert!(parse(VS.as_bytes(), n).is_ok());
        assert!(parse(VS.as_bytes(), n - 1).is_err());
    }

    #[test]
    fn indirect_and_two_dimensional_registers() {
        let text = "FRAG
DCL IN[0], GENERIC[0], PERSPECTIVE, CENTROID
DCL CONST[1][0..7]
DCL TEMP[0], LOCAL
DCL ADDR[0]
DCL SAMP[0]
DCL SVIEW[0], 2D_ARRAY, FLOAT, UINT, SINT, SNORM
DCL OUT[0], COLOR
  0: MOV TEMP[0], CONST[1][ADDR[0].x+2](3)
  1: TEX OUT[0], IN[0], SAMP[0], 2D_ARRAY, IMM[0].xyz
  2: END
";
        let s = parse(text.as_bytes(), 1000).expect("parses");
        let d0 = s.declarations().next().unwrap();
        assert_eq!(d0.interp.interpolate, Interpolate::Perspective);
        assert_eq!(d0.interp.location, Location::Centroid);
        let d1 = s.declarations().nth(1).unwrap();
        assert_eq!(d1.dimension, Some(1));
        assert_eq!((d1.first, d1.last), (0, 7));
        assert!(s.declarations().nth(2).unwrap().local);
        let sv = s.declarations().nth(5).unwrap();
        assert_eq!(sv.sampler_view.resource, Texture::Array2d);
        // The C's loop replicates the first type over the rest unless a comma follows the
        // fourth; four types without one collapse to the first.
        assert_eq!(sv.sampler_view.return_type, [ReturnType::Float; 4]);
        let mov = s.instructions().next().unwrap();
        let src = mov.src[0];
        assert!(src.dimension && src.indirect);
        assert_eq!(src.dim.index, 1);
        assert_eq!(src.index, 2);
        assert_eq!(src.ind.file, File::Address);
        assert_eq!(src.ind.array_id, 3);
        let tex = s.instructions().nth(1).unwrap();
        let t = tex.texture.unwrap();
        assert_eq!(t.texture, Texture::Array2d);
        assert_eq!(t.num_offsets, 1);
        assert_eq!(tex.tex_offsets[0].file, File::Immediate);
    }

    #[test]
    fn four_return_types_survive_only_behind_a_trailing_comma() {
        // ... and the comma's whitespace is eaten with it, so only a declaration that ends the
        // program can carry four.
        let s = parse(
            "FRAG\n  0: END\nDCL SVIEW[0], CUBE, FLOAT, UINT, SINT, SNORM,\n".as_bytes(),
            100,
        )
        .expect("parses");
        assert_eq!(
            s.declarations().next().unwrap().sampler_view.return_type,
            [ReturnType::Float, ReturnType::Uint, ReturnType::Sint, ReturnType::Snorm]
        );
        assert!(
            parse(
                "FRAG\nDCL SVIEW[0], CUBE, FLOAT, UINT, SINT, SNORM,\n  0: END\n".as_bytes(),
                100
            )
            .is_err()
        );
    }

    #[test]
    fn what_the_c_refuses_is_refused_by_name() {
        let refused = |text: &str| parse(text.as_bytes(), 1000).expect_err("refused").message;
        assert_eq!(refused("PIXEL\n"), "Unknown header");
        assert_eq!(refused("VERT\nDCL FOO[0]\n"), "Unknown register file");
        assert_eq!(refused("VERT\n  0: FROB TEMP[0]\n"), "Unknown opcode");
        assert_eq!(refused("VERT\nMOV TEMP[0] TEMP[1]\n"), "Expected `,'");
        assert_eq!(refused("VERT\nIMM[1] FLT32 {0, 0, 0, 0}\n"), "Immediates must be sorted");
        assert_eq!(
            refused("FRAG\nDCL IN[0], GENERIC[0], BOGUS\n").split('"').next().unwrap().trim(),
            "Expected semantic, interpolate attribute, or invariant"
        );
        let e = parse("VERT\nDCL IN[0]\nDCL FOO".as_bytes(), 1000).expect_err("refused");
        assert_eq!((e.line, e.column), (3, 6));
    }

    #[test]
    fn an_unknown_property_is_skipped_as_the_c_skips_it() {
        let s = parse("FRAG\nPROPERTY FROBNICATE 1\nDCL OUT[0], COLOR\n  0: END\n".as_bytes(), 100)
            .expect("parses");
        assert_eq!(s.properties().count(), 0);
        assert_eq!(s.declarations().count(), 1);
    }

    #[test]
    fn a_geometry_shaders_input_prim_implies_its_array_size() {
        let s = parse(
            "GEOM\nPROPERTY GS_INPUT_PRIMITIVE TRIANGLES\nDCL IN[][0], POSITION\n  0: END\n"
                .as_bytes(),
            100,
        )
        .expect("parses");
        let d = s.declarations().next().unwrap();
        assert_eq!((d.first, d.last), (0, 0));
        assert_eq!(d.dimension, None);
        assert_eq!(s.property(Property::GsInputPrim), Some(Primitive::Triangles as u32));
    }

    #[test]
    fn immediates_of_every_type() {
        let text = "FRAG
IMM[0] FLT32 {1.5, -2.0e1, .25, 0x40490fdb}
IMM[1] INT32 {-1, 2, 3, 4}
IMM[2] UINT32 {4294967295, 0, 0, 0}
IMM[3] FLT64 {1.5, -0.25}
IMM[4] INT64 {-1, 0x10}
  0: END
";
        let s = parse(text.as_bytes(), 1000).expect("parses");
        let imms: Vec<_> = s.immediates().collect();
        assert_eq!(imms[0].float(0), 1.5);
        assert_eq!(imms[0].float(1), -20.0);
        assert_eq!(imms[0].float(2), 0.25);
        assert_eq!(imms[0].float(3), f32::from_bits(0x40490fdb));
        assert_eq!(imms[1].int(0), -1);
        assert_eq!(imms[2].uint(0), u32::MAX);
        let d = f64::from_bits(u64::from(imms[3].data[0]) | u64::from(imms[3].data[1]) << 32);
        assert_eq!(d, 1.5);
        let d = f64::from_bits(u64::from(imms[3].data[2]) | u64::from(imms[3].data[3]) << 32);
        assert_eq!(d, -0.25);
        let i = i64::from_ne_bytes(
            (u64::from(imms[4].data[0]) | u64::from(imms[4].data[1]) << 32).to_ne_bytes(),
        );
        assert_eq!(i, -1);
        assert_eq!(imms[4].data[2], 16);
    }
}
