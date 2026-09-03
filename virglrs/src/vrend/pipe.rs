// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! Gallium's vocabulary, as the classic wire spells it.
//!
//! The guest's driver is a gallium driver, and every enumerated field on the classic wire is a
//! `pipe_*` value written as an integer. The C reads them back as integers and translates each to
//! GL at the point of use, through switches whose `default:` is an `assert(0)` -- so a guest that
//! writes a blend factor of 0 crashes the host. Here each vocabulary is an enum, parsed once at the
//! decoder and refused there, and the rest of vrend only ever sees a value that exists.
//!
//! Values are gallium's own (`p_defines.h`), because the wire is defined by them: renumbering one
//! would silently change what a guest's stream means.

/// An enum whose values are the wire's integers, parsed by `from_wire` and written by `wire`.
macro_rules! wire_enum {
    ($(#[$m:meta])* $name:ident { $($(#[$vm:meta])* $variant:ident = $val:literal),+ $(,)? }) => {
        $(#[$m])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        #[repr(u32)]
        pub enum $name { $($(#[$vm])* $variant = $val),+ }

        impl $name {
            pub const ALL: &'static [$name] = &[$($name::$variant),+];

            pub fn from_wire(raw: u32) -> Option<Self> {
                match raw {
                    $($val => Some(Self::$variant),)+
                    _ => None,
                }
            }

            pub fn wire(self) -> u32 {
                self as u32
            }

            pub fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => stringify!($variant),)+
                }
            }
        }
    };
}

pub(crate) use wire_enum;

wire_enum!(
    /// `pipe_shader_type`. Six stages, and the wire carries the stage as an index into
    /// per-stage state everywhere -- so this is also the index type of those arrays.
    ShaderStage {
        Vertex = 0,
        Fragment = 1,
        Geometry = 2,
        TessCtrl = 3,
        TessEval = 4,
        Compute = 5,
    }
);

impl ShaderStage {
    pub const COUNT: usize = 6;

    pub fn index(self) -> usize {
        self as usize
    }
}

wire_enum!(
    /// `pipe_texture_target`.
    TextureTarget {
        Buffer = 0,
        Texture1d = 1,
        Texture2d = 2,
        Texture3d = 3,
        Cube = 4,
        Rect = 5,
        Array1d = 6,
        Array2d = 7,
        CubeArray = 8,
    }
);

wire_enum!(
    /// `pipe_blend_func`.
    BlendFunc {
        Add = 0,
        Subtract = 1,
        ReverseSubtract = 2,
        Min = 3,
        Max = 4,
    }
);

wire_enum!(
    /// `pipe_blendfactor`. Gallium leaves 0 undefined and numbers the inverted factors from 17,
    /// so the five-bit field has holes; each is a refusal, not a factor.
    BlendFactor {
        One = 1,
        SrcColor = 2,
        SrcAlpha = 3,
        DstAlpha = 4,
        DstColor = 5,
        SrcAlphaSaturate = 6,
        ConstColor = 7,
        ConstAlpha = 8,
        Src1Color = 9,
        Src1Alpha = 10,
        Zero = 17,
        InvSrcColor = 18,
        InvSrcAlpha = 19,
        InvDstAlpha = 20,
        InvDstColor = 21,
        InvConstColor = 23,
        InvConstAlpha = 24,
        InvSrc1Color = 25,
        InvSrc1Alpha = 26,
    }
);

wire_enum!(
    /// `pipe_logicop`. All sixteen four-bit values are ops.
    LogicOp {
        Clear = 0,
        Nor = 1,
        AndInverted = 2,
        CopyInverted = 3,
        AndReverse = 4,
        Invert = 5,
        Xor = 6,
        Nand = 7,
        And = 8,
        Equiv = 9,
        Noop = 10,
        OrInverted = 11,
        Copy = 12,
        OrReverse = 13,
        Or = 14,
        Set = 15,
    }
);

wire_enum!(
    /// `pipe_compare_func`. All eight three-bit values are functions.
    CompareFunc {
        Never = 0,
        Less = 1,
        Equal = 2,
        LessEqual = 3,
        Greater = 4,
        NotEqual = 5,
        GreaterEqual = 6,
        Always = 7,
    }
);

wire_enum!(
    /// `pipe_stencil_op`. All eight three-bit values are ops.
    StencilOp {
        Keep = 0,
        Zero = 1,
        Replace = 2,
        Incr = 3,
        Decr = 4,
        IncrWrap = 5,
        DecrWrap = 6,
        Invert = 7,
    }
);

wire_enum!(
    /// `PIPE_FACE_*`, as the rasterizer's cull field.
    CullFace {
        None = 0,
        Front = 1,
        Back = 2,
        FrontAndBack = 3,
    }
);

wire_enum!(
    /// `PIPE_POLYGON_MODE_*`.
    FillMode {
        Fill = 0,
        Line = 1,
        Point = 2,
    }
);

wire_enum!(
    /// `pipe_tex_wrap`. All eight three-bit values are modes.
    TexWrap {
        Repeat = 0,
        Clamp = 1,
        ClampToEdge = 2,
        ClampToBorder = 3,
        MirrorRepeat = 4,
        MirrorClamp = 5,
        MirrorClampToEdge = 6,
        MirrorClampToBorder = 7,
    }
);

wire_enum!(
    /// `pipe_tex_filter`.
    TexFilter {
        Nearest = 0,
        Linear = 1,
    }
);

wire_enum!(
    /// `pipe_tex_mipfilter`.
    MipFilter {
        Nearest = 0,
        Linear = 1,
        None = 2,
    }
);

wire_enum!(
    /// `pipe_swizzle`, the six values a sampler view may select per channel. Gallium's `NONE`
    /// (6) is a driver-internal marker the wire never carries; the C refuses it too.
    Swizzle {
        X = 0,
        Y = 1,
        Z = 2,
        W = 3,
        Zero = 4,
        One = 5,
    }
);

wire_enum!(
    /// `pipe_prim_type`, the drawable subset: gallium's tessellation-spacing values share the
    /// enum but are never a draw's mode.
    PrimType {
        Points = 0,
        Lines = 1,
        LineLoop = 2,
        LineStrip = 3,
        Triangles = 4,
        TriangleStrip = 5,
        TriangleFan = 6,
        Quads = 7,
        QuadStrip = 8,
        Polygon = 9,
        LinesAdjacency = 10,
        LineStripAdjacency = 11,
        TrianglesAdjacency = 12,
        TriangleStripAdjacency = 13,
        Patches = 14,
    }
);

wire_enum!(
    /// `pipe_query_type`.
    QueryType {
        OcclusionCounter = 0,
        OcclusionPredicate = 1,
        Timestamp = 2,
        TimestampDisjoint = 3,
        TimeElapsed = 4,
        PrimitivesGenerated = 5,
        PrimitivesEmitted = 6,
        SoStatistics = 7,
        SoOverflowPredicate = 8,
        GpuFinished = 9,
        PipelineStatistics = 10,
        OcclusionPredicateConservative = 11,
        SoOverflowAnyPredicate = 12,
    }
);

wire_enum!(
    /// `pipe_query_value_type`: the width a query result is written to a buffer in.
    QueryValueType {
        I32 = 0,
        U32 = 1,
        I64 = 2,
        U64 = 3,
    }
);

wire_enum!(
    /// `pipe_render_cond_flag`.
    RenderCondMode {
        Wait = 0,
        NoWait = 1,
        ByRegionWait = 2,
        ByRegionNoWait = 3,
    }
);

wire_enum!(
    /// `VIRGL_TRANSFER_*`: which way a `TRANSFER3D` moves bytes.
    TransferDirection {
        ToHost = 1,
        FromHost = 2,
    }
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_value_round_trips_and_every_hole_is_refused() {
        for f in BlendFactor::ALL {
            assert_eq!(BlendFactor::from_wire(f.wire()), Some(*f));
        }
        for hole in [0, 11, 12, 13, 14, 15, 16, 22, 27, 28, 29, 30, 31] {
            assert_eq!(BlendFactor::from_wire(hole), None, "{hole} is not a factor");
        }
        assert_eq!(BlendFunc::from_wire(5), None);
        assert_eq!(FillMode::from_wire(3), None);
        assert_eq!(MipFilter::from_wire(3), None);
        assert_eq!(Swizzle::from_wire(6), None);
        assert_eq!(PrimType::from_wire(15), None);
        assert_eq!(QueryType::from_wire(13), None);
        assert_eq!(TransferDirection::from_wire(0), None);
        assert_eq!(ShaderStage::from_wire(6), None);
        assert_eq!(TextureTarget::from_wire(9), None);
    }
}
