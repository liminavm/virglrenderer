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

use crate::venus::cs::{ObjectId, Scalar};

/// Vulkan's handles are pointer-sized on every target this renderer supports.
const _: () = assert!(size_of::<*const c_void>() == 8);

<%def name="newtype(name, repr)">\
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
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
pub type ${ty.name} = ${RUST.base_name(ty.typedef) if ty.typedef else 'VkFlags'};
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
/// A decoded command's arguments, and its reply where it has one.
% for ty in GEN.supported_types[VkType.COMMAND]:
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct vn_command_${ty.name} {
%   for name, rs in RUST.command_params(ty):
    pub ${name}: ${rs},
%   endfor
}

% endfor
