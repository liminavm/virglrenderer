// Vulkan's types, as venus serializes them.
//
// Generated from the vk.xml pinned in the venus-protocol subproject, not taken from a binding
// crate. The wire format is defined by *that* vk.xml -- venus encodes a struct's members in the
// order and with the membership that version gives them -- so a binding generated from a
// different header revision would silently disagree with the guest.
//
// Enums are newtypes over their underlying scalar, never Rust `enum`s. The guest chooses the
// value, and a Rust enum holding a variant it has no discriminant for is undefined behaviour:
// the one thing this boundary must never do.

use core::ffi::c_void;
use core::marker::PhantomData;

use crate::venus::cs::{ObjectId, Scalar};

/// Vulkan's handles are pointer-sized on every target this renderer supports.
const _: () = assert!(size_of::<*const c_void>() == 8);

<%def name="newtype(name, repr)">\
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(transparent)]
pub struct ${name}(pub ${repr});

impl Scalar for ${name} {
    fn from_le_bytes(b: &[u8]) -> Self {
        Self(<${repr}>::from_le_bytes(b.try_into().expect("caller sized the slice")))
    }
    fn write_le(self, out: &mut [u8]) {
        out.copy_from_slice(&self.0.to_le_bytes());
    }
}
</%def>\
% for ty in GEN.supported_types[VkType.BASETYPE]:
<%
    inner = RUST.base_name(ty.typedef) if ty.typedef else 'u32'
%>\
${newtype(ty.name, inner)}\
% endfor

% for ty in GEN.supported_types[VkType.HANDLE]:
/// ${'Dispatchable' if ty.dispatchable else 'Non-dispatchable'} handle.
${newtype(ty.name, 'u64')}\
% endfor

% for ty in GEN.supported_types[VkType.ENUM]:
<% repr = RUST.enum_repr(ty) %>\
${newtype(ty.name, repr)}\
impl ${ty.name} {
%   for key, val in RUST.enum_values(ty):
    pub const ${key}: Self = Self(${val});
%   endfor
}

% endfor
% for ty in GEN.supported_types[VkType.BITMASK]:
<%
    repr = RUST.bitmask_repr(ty)
    bits = RUST.bits_of(ty)
%>\
${newtype(ty.name, repr)}\
%   if bits:
<%
    ## The bits of a VkFlags64 mask are already 64-bit, so the cast would be a no-op.
    cast = '' if RUST.enum_repr(bits) == repr else ' as %s' % repr
%>\
impl From<${bits.name}> for ${ty.name} {
    fn from(bit: ${bits.name}) -> Self {
        Self(bit.0${cast})
    }
}

%   endif
% endfor

% for ty in GEN.supported_types[VkType.FUNCPOINTER]:
pub type ${ty.name} = ${RUST.funcpointer(ty)};
% endfor

% for ty in GEN.supported_types[VkType.UNION]:
#[derive(Clone, Copy)]
#[repr(C)]
pub union ${ty.name} {
%   for name, rs in RUST.struct_fields(ty):
    pub ${name}: ${rs},
%   endfor
}

impl Default for ${ty.name} {
    fn default() -> Self {
        ${ty.name} { ${RUST.field_name(ty.variables[0].name)}: ${RUST.zero(ty.variables[0])} }
    }
}

% endfor
% for ty in GEN.supported_types[VkType.STRUCT]:
#[derive(Clone, Copy)]
#[repr(C)]
pub struct ${ty.name} {
%   for name, rs in RUST.struct_fields(ty):
    pub ${name}: ${rs},
%   endfor
}

impl Default for ${ty.name} {
    fn default() -> Self {
        ${ty.name} {
%   for var in ty.variables:
            ${RUST.field_name(var.name)}: ${RUST.zero(var)},
%   endfor
        }
    }
}

% endfor
/// Structs the renderer builds for the driver, which the wire never carries.
///
/// Same vk.xml, outside venus's own set -- see `RustGen.RENDERER_ONLY_STRUCTS`. No serializer and
/// no `Default`: nothing decodes one, and every field is set at the single site that builds it.
% for ty in RUST.renderer_only_structs():
#[derive(Clone, Copy)]
#[repr(C)]
pub struct ${ty.name} {
%   for name, rs in RUST.struct_fields(ty):
    pub ${name}: ${rs},
%   endfor
}

% endfor
/// A decoded command's arguments, and its reply where it has one.
///
/// `'a` is the decoder's: the wire bytes and the arena the command was decoded into, which both
/// outlive the submission. Every member that is a reference borrows from there, and the marker is
/// what carries the lifetime for the commands whose members are all scalars. It is a zero-sized
/// type at the end of a `#[repr(C)]` struct, so it costs no byte and moves no member -- which the
/// layout oracle is what actually checks.
% for ty in GEN.supported_types[VkType.COMMAND]:
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct vn_command_${ty.name}<'a> {
%   for vis, name, rs in RUST.command_params(ty):
    ${vis}${name}: ${rs},
%   endfor
    pub _marker: PhantomData<&'a ()>,
}

% endfor
