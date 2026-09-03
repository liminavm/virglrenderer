// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! `tgsi_info.c`: the opcode table, and the types an opcode implies for its operands.

/// `tgsi_output_mode`: how an opcode produces its result.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OutputMode {
    /// No result.
    None,
    /// Each written channel is computed from the same channel of the sources.
    Componentwise,
    /// One value, written to every enabled channel.
    Replicate,
    /// What is computed depends on which channel is written.
    ChanDependent,
    /// Anything else (a texture fetch, say).
    Other,
}

/// `tgsi_opcode_info`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Info {
    pub mnemonic: &'static str,
    pub num_dst: u8,
    pub num_src: u8,
    pub is_tex: bool,
    pub is_branch: bool,
    /// How far a dump dedents before printing this opcode.
    pub pre_dedent: i8,
    /// How far a dump indents after it.
    pub post_indent: i8,
    pub output_mode: OutputMode,
}

/// `tgsi_opcode_type`: the type an opcode reads or writes its operands as.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OpType {
    /// `MOV` and `UCMP`, which carry whatever they are given.
    Untyped,
    Void,
    Unsigned,
    Signed,
    Float,
    Double,
    Unsigned64,
    Signed64,
}

macro_rules! opcodes {
    ($($name:ident = ($mn:literal, $nd:literal, $ns:literal, $tex:literal, $br:literal, $pre:literal, $post:literal, $mode:ident)),* $(,)?) => {
        /// `tgsi_opcode`, in the C's order (which the wire's text does not depend on, but the
        /// C's own range tests -- "SAMPLE through GATHER4", "LOAD through ATOMIMAX" -- do).
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        #[repr(u8)]
        pub enum Opcode { $($name),* }

        impl Opcode {
            pub const ALL: &'static [Opcode] = &[$(Opcode::$name),*];

            pub fn info(self) -> &'static Info {
                &TABLE[self as usize]
            }
        }

        const TABLE: [Info; Opcode::ALL.len()] = [
            $(Info {
                mnemonic: $mn,
                num_dst: $nd,
                num_src: $ns,
                is_tex: $tex != 0,
                is_branch: $br != 0,
                pre_dedent: $pre,
                post_indent: $post,
                output_mode: OutputMode::$mode,
            }),*
        ];
    };
}

opcodes! {
    Arl = ("ARL", 1, 1, 0, 0, 0, 0, Componentwise),
    Mov = ("MOV", 1, 1, 0, 0, 0, 0, Componentwise),
    Lit = ("LIT", 1, 1, 0, 0, 0, 0, ChanDependent),
    Rcp = ("RCP", 1, 1, 0, 0, 0, 0, Replicate),
    Rsq = ("RSQ", 1, 1, 0, 0, 0, 0, Replicate),
    Exp = ("EXP", 1, 1, 0, 0, 0, 0, ChanDependent),
    Log = ("LOG", 1, 1, 0, 0, 0, 0, ChanDependent),
    Mul = ("MUL", 1, 2, 0, 0, 0, 0, Componentwise),
    Add = ("ADD", 1, 2, 0, 0, 0, 0, Componentwise),
    Dp3 = ("DP3", 1, 2, 0, 0, 0, 0, Replicate),
    Dp4 = ("DP4", 1, 2, 0, 0, 0, 0, Replicate),
    Dst = ("DST", 1, 2, 0, 0, 0, 0, ChanDependent),
    Min = ("MIN", 1, 2, 0, 0, 0, 0, Componentwise),
    Max = ("MAX", 1, 2, 0, 0, 0, 0, Componentwise),
    Slt = ("SLT", 1, 2, 0, 0, 0, 0, Componentwise),
    Sge = ("SGE", 1, 2, 0, 0, 0, 0, Componentwise),
    Mad = ("MAD", 1, 3, 0, 0, 0, 0, Componentwise),
    Sub = ("SUB", 1, 2, 0, 0, 0, 0, Componentwise),
    Lrp = ("LRP", 1, 3, 0, 0, 0, 0, Componentwise),
    Fma = ("FMA", 1, 3, 0, 0, 0, 0, Componentwise),
    Sqrt = ("SQRT", 1, 1, 0, 0, 0, 0, Replicate),
    Frc = ("FRC", 1, 1, 0, 0, 0, 0, Componentwise),
    Flr = ("FLR", 1, 1, 0, 0, 0, 0, Componentwise),
    Round = ("ROUND", 1, 1, 0, 0, 0, 0, Componentwise),
    Ex2 = ("EX2", 1, 1, 0, 0, 0, 0, Replicate),
    Lg2 = ("LG2", 1, 1, 0, 0, 0, 0, Replicate),
    Pow = ("POW", 1, 2, 0, 0, 0, 0, Replicate),
    Xpd = ("XPD", 1, 2, 0, 0, 0, 0, Componentwise),
    Abs = ("ABS", 1, 1, 0, 0, 0, 0, Componentwise),
    Dph = ("DPH", 1, 2, 0, 0, 0, 0, Replicate),
    Cos = ("COS", 1, 1, 0, 0, 0, 0, Replicate),
    Ddx = ("DDX", 1, 1, 0, 0, 0, 0, Componentwise),
    Ddy = ("DDY", 1, 1, 0, 0, 0, 0, Componentwise),
    Kill = ("KILL", 0, 0, 0, 0, 0, 0, None),
    Pk2h = ("PK2H", 1, 1, 0, 0, 0, 0, Componentwise),
    Pk2us = ("PK2US", 1, 1, 0, 0, 0, 0, Componentwise),
    Pk4b = ("PK4B", 1, 1, 0, 0, 0, 0, Componentwise),
    Pk4ub = ("PK4UB", 1, 1, 0, 0, 0, 0, Componentwise),
    Seq = ("SEQ", 1, 2, 0, 0, 0, 0, Componentwise),
    Sgt = ("SGT", 1, 2, 0, 0, 0, 0, Componentwise),
    Sin = ("SIN", 1, 1, 0, 0, 0, 0, Replicate),
    Sle = ("SLE", 1, 2, 0, 0, 0, 0, Componentwise),
    Sne = ("SNE", 1, 2, 0, 0, 0, 0, Componentwise),
    Tex = ("TEX", 1, 2, 1, 0, 0, 0, Other),
    Txd = ("TXD", 1, 4, 1, 0, 0, 0, Other),
    Txp = ("TXP", 1, 2, 1, 0, 0, 0, Other),
    Up2h = ("UP2H", 1, 1, 0, 0, 0, 0, Componentwise),
    Up2us = ("UP2US", 1, 1, 0, 0, 0, 0, Componentwise),
    Up4b = ("UP4B", 1, 1, 0, 0, 0, 0, Componentwise),
    Up4ub = ("UP4UB", 1, 1, 0, 0, 0, 0, Componentwise),
    Arr = ("ARR", 1, 1, 0, 0, 0, 0, Componentwise),
    Cal = ("CAL", 0, 0, 0, 1, 0, 0, None),
    Ret = ("RET", 0, 0, 0, 0, 0, 0, None),
    Ssg = ("SSG", 1, 1, 0, 0, 0, 0, Componentwise),
    Cmp = ("CMP", 1, 3, 0, 0, 0, 0, Componentwise),
    Scs = ("SCS", 1, 1, 0, 0, 0, 0, ChanDependent),
    Txb = ("TXB", 1, 2, 1, 0, 0, 0, Other),
    Fbfetch = ("FBFETCH", 1, 1, 0, 0, 0, 0, Other),
    Div = ("DIV", 1, 2, 0, 0, 0, 0, Componentwise),
    Dp2 = ("DP2", 1, 2, 0, 0, 0, 0, Replicate),
    Txl = ("TXL", 1, 2, 1, 0, 0, 0, Other),
    Brk = ("BRK", 0, 0, 0, 0, 0, 0, None),
    If = ("IF", 0, 1, 0, 1, 0, 1, None),
    Uif = ("UIF", 0, 1, 0, 1, 0, 1, None),
    Else = ("ELSE", 0, 0, 0, 1, 1, 1, None),
    Endif = ("ENDIF", 0, 0, 0, 0, 1, 0, None),
    DdxFine = ("DDX_FINE", 1, 1, 0, 0, 0, 0, Componentwise),
    DdyFine = ("DDY_FINE", 1, 1, 0, 0, 0, 0, Componentwise),
    Ceil = ("CEIL", 1, 1, 0, 0, 0, 0, Componentwise),
    I2f = ("I2F", 1, 1, 0, 0, 0, 0, Componentwise),
    Not = ("NOT", 1, 1, 0, 0, 0, 0, Componentwise),
    Trunc = ("TRUNC", 1, 1, 0, 0, 0, 0, Componentwise),
    Shl = ("SHL", 1, 2, 0, 0, 0, 0, Componentwise),
    And = ("AND", 1, 2, 0, 0, 0, 0, Componentwise),
    Or = ("OR", 1, 2, 0, 0, 0, 0, Componentwise),
    Mod = ("MOD", 1, 2, 0, 0, 0, 0, Componentwise),
    Xor = ("XOR", 1, 2, 0, 0, 0, 0, Componentwise),
    Txf = ("TXF", 1, 2, 1, 0, 0, 0, Other),
    Txq = ("TXQ", 1, 2, 1, 0, 0, 0, Other),
    Cont = ("CONT", 0, 0, 0, 0, 0, 0, None),
    Emit = ("EMIT", 0, 1, 0, 0, 0, 0, None),
    Endprim = ("ENDPRIM", 0, 1, 0, 0, 0, 0, None),
    Bgnloop = ("BGNLOOP", 0, 0, 0, 1, 0, 1, None),
    Bgnsub = ("BGNSUB", 0, 0, 0, 0, 0, 1, None),
    Endloop = ("ENDLOOP", 0, 0, 0, 1, 1, 0, None),
    Endsub = ("ENDSUB", 0, 0, 0, 0, 1, 0, None),
    Txqs = ("TXQS", 1, 1, 1, 0, 0, 0, Other),
    Resq = ("RESQ", 1, 1, 0, 0, 0, 0, Other),
    Nop = ("NOP", 0, 0, 0, 0, 0, 0, None),
    Fseq = ("FSEQ", 1, 2, 0, 0, 0, 0, Componentwise),
    Fsge = ("FSGE", 1, 2, 0, 0, 0, 0, Componentwise),
    Fslt = ("FSLT", 1, 2, 0, 0, 0, 0, Componentwise),
    Fsne = ("FSNE", 1, 2, 0, 0, 0, 0, Componentwise),
    Membar = ("MEMBAR", 0, 1, 0, 0, 0, 0, Other),
    VoteAny = ("VOTE_ANY", 1, 1, 0, 0, 0, 0, Componentwise),
    VoteAll = ("VOTE_ALL", 1, 1, 0, 0, 0, 0, Componentwise),
    VoteEq = ("VOTE_EQ", 1, 1, 0, 0, 0, 0, Componentwise),
    KillIf = ("KILL_IF", 0, 1, 0, 0, 0, 0, None),
    End = ("END", 0, 0, 0, 0, 0, 0, None),
    Dfma = ("DFMA", 1, 3, 0, 0, 0, 0, Componentwise),
    F2i = ("F2I", 1, 1, 0, 0, 0, 0, Componentwise),
    Idiv = ("IDIV", 1, 2, 0, 0, 0, 0, Componentwise),
    Imax = ("IMAX", 1, 2, 0, 0, 0, 0, Componentwise),
    Imin = ("IMIN", 1, 2, 0, 0, 0, 0, Componentwise),
    Ineg = ("INEG", 1, 1, 0, 0, 0, 0, Componentwise),
    Isge = ("ISGE", 1, 2, 0, 0, 0, 0, Componentwise),
    Ishr = ("ISHR", 1, 2, 0, 0, 0, 0, Componentwise),
    Islt = ("ISLT", 1, 2, 0, 0, 0, 0, Componentwise),
    F2u = ("F2U", 1, 1, 0, 0, 0, 0, Componentwise),
    U2f = ("U2F", 1, 1, 0, 0, 0, 0, Componentwise),
    Uadd = ("UADD", 1, 2, 0, 0, 0, 0, Componentwise),
    Udiv = ("UDIV", 1, 2, 0, 0, 0, 0, Componentwise),
    Umad = ("UMAD", 1, 3, 0, 0, 0, 0, Componentwise),
    Umax = ("UMAX", 1, 2, 0, 0, 0, 0, Componentwise),
    Umin = ("UMIN", 1, 2, 0, 0, 0, 0, Componentwise),
    Umod = ("UMOD", 1, 2, 0, 0, 0, 0, Componentwise),
    Umul = ("UMUL", 1, 2, 0, 0, 0, 0, Componentwise),
    Useq = ("USEQ", 1, 2, 0, 0, 0, 0, Componentwise),
    Usge = ("USGE", 1, 2, 0, 0, 0, 0, Componentwise),
    Ushr = ("USHR", 1, 2, 0, 0, 0, 0, Componentwise),
    Uslt = ("USLT", 1, 2, 0, 0, 0, 0, Componentwise),
    Usne = ("USNE", 1, 2, 0, 0, 0, 0, Componentwise),
    Switch = ("SWITCH", 0, 1, 0, 0, 0, 0, None),
    Case = ("CASE", 0, 1, 0, 0, 0, 0, None),
    Default = ("DEFAULT", 0, 0, 0, 0, 0, 0, None),
    Endswitch = ("ENDSWITCH", 0, 0, 0, 0, 0, 0, None),
    Sample = ("SAMPLE", 1, 3, 0, 0, 0, 0, Other),
    SampleI = ("SAMPLE_I", 1, 2, 0, 0, 0, 0, Other),
    SampleIMs = ("SAMPLE_I_MS", 1, 3, 0, 0, 0, 0, Other),
    SampleB = ("SAMPLE_B", 1, 4, 0, 0, 0, 0, Other),
    SampleC = ("SAMPLE_C", 1, 4, 0, 0, 0, 0, Other),
    SampleCLz = ("SAMPLE_C_LZ", 1, 4, 0, 0, 0, 0, Other),
    SampleD = ("SAMPLE_D", 1, 5, 0, 0, 0, 0, Other),
    SampleL = ("SAMPLE_L", 1, 4, 0, 0, 0, 0, Other),
    Gather4 = ("GATHER4", 1, 3, 0, 0, 0, 0, Other),
    Sviewinfo = ("SVIEWINFO", 1, 2, 0, 0, 0, 0, Other),
    SamplePos = ("SAMPLE_POS", 1, 2, 0, 0, 0, 0, Other),
    SampleInfo = ("SAMPLE_INFO", 1, 2, 0, 0, 0, 0, Other),
    Uarl = ("UARL", 1, 1, 0, 0, 0, 0, Componentwise),
    Ucmp = ("UCMP", 1, 3, 0, 0, 0, 0, Componentwise),
    Iabs = ("IABS", 1, 1, 0, 0, 0, 0, Componentwise),
    Issg = ("ISSG", 1, 1, 0, 0, 0, 0, Componentwise),
    Load = ("LOAD", 1, 2, 0, 0, 0, 0, Other),
    Store = ("STORE", 1, 2, 0, 0, 0, 0, Other),
    Barrier = ("BARRIER", 0, 0, 0, 0, 0, 0, Other),
    Atomuadd = ("ATOMUADD", 1, 3, 0, 0, 0, 0, Other),
    Atomxchg = ("ATOMXCHG", 1, 3, 0, 0, 0, 0, Other),
    Atomcas = ("ATOMCAS", 1, 4, 0, 0, 0, 0, Other),
    Atomand = ("ATOMAND", 1, 3, 0, 0, 0, 0, Other),
    Atomor = ("ATOMOR", 1, 3, 0, 0, 0, 0, Other),
    Atomxor = ("ATOMXOR", 1, 3, 0, 0, 0, 0, Other),
    Atomumin = ("ATOMUMIN", 1, 3, 0, 0, 0, 0, Other),
    Atomumax = ("ATOMUMAX", 1, 3, 0, 0, 0, 0, Other),
    Atomimin = ("ATOMIMIN", 1, 3, 0, 0, 0, 0, Other),
    Atomimax = ("ATOMIMAX", 1, 3, 0, 0, 0, 0, Other),
    Tex2 = ("TEX2", 1, 3, 1, 0, 0, 0, Other),
    Txb2 = ("TXB2", 1, 3, 1, 0, 0, 0, Other),
    Txl2 = ("TXL2", 1, 3, 1, 0, 0, 0, Other),
    ImulHi = ("IMUL_HI", 1, 2, 0, 0, 0, 0, Componentwise),
    UmulHi = ("UMUL_HI", 1, 2, 0, 0, 0, 0, Componentwise),
    Tg4 = ("TG4", 1, 3, 1, 0, 0, 0, Other),
    Lodq = ("LODQ", 1, 2, 1, 0, 0, 0, Other),
    Ibfe = ("IBFE", 1, 3, 0, 0, 0, 0, Componentwise),
    Ubfe = ("UBFE", 1, 3, 0, 0, 0, 0, Componentwise),
    Bfi = ("BFI", 1, 4, 0, 0, 0, 0, Componentwise),
    Brev = ("BREV", 1, 1, 0, 0, 0, 0, Componentwise),
    Popc = ("POPC", 1, 1, 0, 0, 0, 0, Componentwise),
    Lsb = ("LSB", 1, 1, 0, 0, 0, 0, Componentwise),
    Imsb = ("IMSB", 1, 1, 0, 0, 0, 0, Componentwise),
    Umsb = ("UMSB", 1, 1, 0, 0, 0, 0, Componentwise),
    InterpCentroid = ("INTERP_CENTROID", 1, 1, 0, 0, 0, 0, Other),
    InterpSample = ("INTERP_SAMPLE", 1, 2, 0, 0, 0, 0, Other),
    InterpOffset = ("INTERP_OFFSET", 1, 2, 0, 0, 0, 0, Other),
    F2d = ("F2D", 1, 1, 0, 0, 0, 0, Componentwise),
    D2f = ("D2F", 1, 1, 0, 0, 0, 0, Componentwise),
    Dabs = ("DABS", 1, 1, 0, 0, 0, 0, Componentwise),
    Dneg = ("DNEG", 1, 1, 0, 0, 0, 0, Componentwise),
    Dadd = ("DADD", 1, 2, 0, 0, 0, 0, Componentwise),
    Dmul = ("DMUL", 1, 2, 0, 0, 0, 0, Componentwise),
    Dmax = ("DMAX", 1, 2, 0, 0, 0, 0, Componentwise),
    Dmin = ("DMIN", 1, 2, 0, 0, 0, 0, Componentwise),
    Dslt = ("DSLT", 1, 2, 0, 0, 0, 0, Componentwise),
    Dsge = ("DSGE", 1, 2, 0, 0, 0, 0, Componentwise),
    Dseq = ("DSEQ", 1, 2, 0, 0, 0, 0, Componentwise),
    Dsne = ("DSNE", 1, 2, 0, 0, 0, 0, Componentwise),
    Drcp = ("DRCP", 1, 1, 0, 0, 0, 0, Componentwise),
    Dsqrt = ("DSQRT", 1, 1, 0, 0, 0, 0, Componentwise),
    Dmad = ("DMAD", 1, 3, 0, 0, 0, 0, Componentwise),
    Dfrac = ("DFRAC", 1, 1, 0, 0, 0, 0, Componentwise),
    Dldexp = ("DLDEXP", 1, 2, 0, 0, 0, 0, Componentwise),
    Dfracexp = ("DFRACEXP", 2, 1, 0, 0, 0, 0, Componentwise),
    D2i = ("D2I", 1, 1, 0, 0, 0, 0, Componentwise),
    I2d = ("I2D", 1, 1, 0, 0, 0, 0, Componentwise),
    D2u = ("D2U", 1, 1, 0, 0, 0, 0, Componentwise),
    U2d = ("U2D", 1, 1, 0, 0, 0, 0, Componentwise),
    Drsq = ("DRSQ", 1, 1, 0, 0, 0, 0, Componentwise),
    Dtrunc = ("DTRUNC", 1, 1, 0, 0, 0, 0, Componentwise),
    Dceil = ("DCEIL", 1, 1, 0, 0, 0, 0, Componentwise),
    Dflr = ("DFLR", 1, 1, 0, 0, 0, 0, Componentwise),
    Dround = ("DROUND", 1, 1, 0, 0, 0, 0, Componentwise),
    Dssg = ("DSSG", 1, 1, 0, 0, 0, 0, Componentwise),
    Ddiv = ("DDIV", 1, 2, 0, 0, 0, 0, Componentwise),
    Clock = ("CLOCK", 1, 0, 0, 0, 0, 0, Other),
    I64abs = ("I64ABS", 1, 1, 0, 0, 0, 0, Componentwise),
    I64neg = ("I64NEG", 1, 1, 0, 0, 0, 0, Componentwise),
    I64ssg = ("I64SSG", 1, 1, 0, 0, 0, 0, Componentwise),
    I64slt = ("I64SLT", 1, 2, 0, 0, 0, 0, Componentwise),
    I64sge = ("I64SGE", 1, 2, 0, 0, 0, 0, Componentwise),
    I64min = ("I64MIN", 1, 2, 0, 0, 0, 0, Componentwise),
    I64max = ("I64MAX", 1, 2, 0, 0, 0, 0, Componentwise),
    I64shr = ("I64SHR", 1, 2, 0, 0, 0, 0, Componentwise),
    I64div = ("I64DIV", 1, 2, 0, 0, 0, 0, Componentwise),
    I64mod = ("I64MOD", 1, 2, 0, 0, 0, 0, Componentwise),
    F2i64 = ("F2I64", 1, 1, 0, 0, 0, 0, Componentwise),
    U2i64 = ("U2I64", 1, 1, 0, 0, 0, 0, Componentwise),
    I2i64 = ("I2I64", 1, 1, 0, 0, 0, 0, Componentwise),
    D2i64 = ("D2I64", 1, 1, 0, 0, 0, 0, Componentwise),
    I642f = ("I642F", 1, 1, 0, 0, 0, 0, Componentwise),
    I642d = ("I642D", 1, 1, 0, 0, 0, 0, Componentwise),
    U64add = ("U64ADD", 1, 2, 0, 0, 0, 0, Componentwise),
    U64mul = ("U64MUL", 1, 2, 0, 0, 0, 0, Componentwise),
    U64seq = ("U64SEQ", 1, 2, 0, 0, 0, 0, Componentwise),
    U64sne = ("U64SNE", 1, 2, 0, 0, 0, 0, Componentwise),
    U64slt = ("U64SLT", 1, 2, 0, 0, 0, 0, Componentwise),
    U64sge = ("U64SGE", 1, 2, 0, 0, 0, 0, Componentwise),
    U64min = ("U64MIN", 1, 2, 0, 0, 0, 0, Componentwise),
    U64max = ("U64MAX", 1, 2, 0, 0, 0, 0, Componentwise),
    U64shl = ("U64SHL", 1, 2, 0, 0, 0, 0, Componentwise),
    U64shr = ("U64SHR", 1, 2, 0, 0, 0, 0, Componentwise),
    U64div = ("U64DIV", 1, 2, 0, 0, 0, 0, Componentwise),
    U64mod = ("U64MOD", 1, 2, 0, 0, 0, 0, Componentwise),
    F2u64 = ("F2U64", 1, 1, 0, 0, 0, 0, Componentwise),
    D2u64 = ("D2U64", 1, 1, 0, 0, 0, 0, Componentwise),
    U642f = ("U642F", 1, 1, 0, 0, 0, 0, Componentwise),
    U642d = ("U642D", 1, 1, 0, 0, 0, 0, Componentwise),
}

impl Opcode {
    pub fn mnemonic(self) -> &'static str {
        self.info().mnemonic
    }

    /// `tgsi_opcode_infer_type`: the type of the destination.
    fn infer_type(self) -> OpType {
        use Opcode::*;
        match self {
            Mov | Ucmp => OpType::Untyped,
            Not | Shl | And | Or | Xor | Txq | Txqs | F2u | Udiv | Umad | Umax | Umin | Umod
            | Umul | Useq | Usge | Ushr | Uslt | Usne | Sviewinfo | UmulHi | Ubfe | Bfi | Brev
            | D2u | Clock | Uadd => OpType::Unsigned,
            Arl | Arr | Mod | F2i | Fseq | Fsge | Fslt | Fsne | Idiv | Imax | Imin | Ineg
            | Isge | Ishr | Islt | Uarl | Iabs | Issg | ImulHi | Ibfe | Imsb | Dseq | Dsge
            | Dslt | Dsne | D2i | Lsb | Popc | Umsb | U64seq | U64sne | U64slt | U64sge
            | I64slt | I64sge => OpType::Signed,
            Dadd | Dabs | Dfma | Dneg | Dmul | Dmax | Dmin | Drcp | Dsqrt | Dmad | Dldexp
            | Dfracexp | Dfrac | Drsq | Dtrunc | Dceil | Dflr | Dround | Dssg | Ddiv | F2d
            | I2d | U2d | U642d | I642d => OpType::Double,
            U64max | U64min | U64add | U64mul | U64div | U64mod | U64shl | U64shr | F2u64
            | D2u64 => OpType::Unsigned64,
            I64max | I64min | I64abs | I64ssg | I64neg | I64shr | I64div | I64mod | F2i64
            | U2i64 | I2i64 | D2i64 => OpType::Signed64,
            _ => OpType::Float,
        }
    }

    /// `tgsi_opcode_infer_src_type`.
    pub fn src_type(self) -> OpType {
        use Opcode::*;
        match self {
            Uif | Txf | U2f | U2d | Uadd | Switch | Case | SampleI | SampleIMs | UmulHi | Umsb
            | U2i64 | Membar => OpType::Unsigned,
            ImulHi | I2f | I2d | I2i64 => OpType::Signed,
            Arl | Arr | F2d | F2i | F2u | Fseq | Fsge | Fslt | Fsne | Ucmp | F2u64 | F2i64 => {
                OpType::Float
            }
            D2f | D2u | D2i | Dseq | Dsge | Dslt | Dsne | D2u64 | D2i64 => OpType::Double,
            U64seq | U64sne | U64slt | U64sge | U642f | U642d => OpType::Unsigned64,
            I64slt | I64sge | I642f | I642d => OpType::Signed64,
            _ => self.infer_type(),
        }
    }

    /// `tgsi_opcode_infer_dst_type`.
    pub fn dst_type(self) -> OpType {
        self.infer_type()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table is indexed by the enum's ordinal, so the two must agree everywhere -- a row
    /// out of place would answer another opcode's arity to the text parser.
    #[test]
    fn every_opcode_names_itself() {
        assert_eq!(Opcode::ALL.len(), 235);
        assert_eq!(Opcode::Arl as usize, 0);
        assert_eq!(Opcode::U642d as usize, 234);
        assert_eq!(Opcode::Tex.info().num_src, 2);
        assert!(Opcode::Tex.info().is_tex);
        assert_eq!(Opcode::SampleD.info().num_src, 5);
        assert_eq!(Opcode::Dfracexp.info().num_dst, 2);
        assert_eq!(Opcode::Else.info().pre_dedent, 1);
        assert_eq!(Opcode::Else.info().post_indent, 1);
        assert_eq!(Opcode::Mov.dst_type(), OpType::Untyped);
        assert_eq!(Opcode::Txf.src_type(), OpType::Unsigned);
        assert_eq!(Opcode::Txf.dst_type(), OpType::Float);
    }
}
