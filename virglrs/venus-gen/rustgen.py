# SPDX-License-Identifier: MIT
# Copyright © 2026 the limina authors
"""Rust backend for venus-protocol's model.

`vkxml.py` is a language-neutral model of vk.xml; `vn_protocol.py`'s `Gen` is a C backend on top of
it, and this is the Rust one. The split in the upstream tree is not along that line -- `Gen` emits C
statements as well as selecting types -- so this module deliberately reuses only `Gen`'s selection
(which types and commands venus serializes) and none of its emission.

Vulkan names are kept verbatim, `sType` and all. The generated Rust is meant to be diffable against
the generated C when a wire question comes up, and a renaming layer would cost that for nothing.
"""

from vkxml import VkType

# Rust keywords that a Vulkan member name can collide with (`VkDescriptorPoolSize.type`). Escaped
# as raw identifiers rather than renamed, so the Rust member keeps the name the spec gives it.
KEYWORDS = frozenset('''
as async await become box break const continue do dyn else enum extern false final fn for if impl
in let loop macro match mod move mut override priv pub ref return static struct trait true try type
typeof unsafe unsized use virtual where while yield
'''.split())

# The C primitives vk.xml uses, and their Rust equivalents. `size_t` is `usize` and `int` is
# `c_int`: both are the C ABI's, not a fixed width, because these types are handed to the Vulkan
# loader.
PRIMITIVES = {
    'void': 'core::ffi::c_void',
    'char': 'core::ffi::c_char',
    'int': 'core::ffi::c_int',
    'size_t': 'usize',
    'float': 'f32',
    'double': 'f64',
    'uint8_t': 'u8',
    'uint16_t': 'u16',
    'uint32_t': 'u32',
    'uint64_t': 'u64',
    'int8_t': 'i8',
    'int16_t': 'i16',
    'int32_t': 'i32',
    'int64_t': 'i64',
}

# Zero for each primitive, for the generated `Default`.
PRIMITIVE_ZERO = {
    'f32': '0.0',
    'f64': '0.0',
}


class RustGen:
    """Renders vk.xml's types as Rust. Holds no state beyond the model and the API constants."""

    def __init__(self, gen, constants):
        self.gen = gen
        # vk.xml's "API Constants" block, which the C generator never needs (it includes
        # vulkan.h) and which this one does: they are the static array dimensions.
        self.constants = constants

    # --- names ---

    def base_name(self, ty):
        """The Rust name of a type, ignoring any pointer or array decoration."""
        base = ty.base
        if base.category == VkType.DEFAULT:
            return PRIMITIVES[base.name]
        return base.name

    def field_type(self, var):
        """The Rust type of a struct member or command argument, decoration included."""
        return self._decorate(var.ty)

    def _decorate(self, ty):
        inner = self.base_name(ty)

        if ty.is_static_array():
            # vk.xml writes a 2-D array as the single dimension string "3][4".
            for dim in reversed(ty.static_array_size().split('][')):
                inner = '[%s; %s]' % (inner, self.dimension(dim))
            return inner

        # ref_quals is outermost-last, so build the pointer chain from the inside out. The
        # qualifier that applies to a given level is the one before it.
        if ty.is_pointer():
            quals = ty.decor.ref_quals[:]
            quals.append(ty.decor.qual)
            for i in range(len(ty.decor.ref_quals)):
                const = 'const' in (quals[i] or '')
                inner = '*%s %s' % ('const' if const else 'mut', inner)
            return inner

        return inner

    def dimension(self, dim):
        """A static array's extent, as a Rust expression."""
        if dim.isdigit():
            return dim
        return str(self.constants[dim])

    # --- zero values, for the generated Default ---

    def zero(self, var):
        return self._zero_of(var.ty)

    def _zero_of(self, ty):
        if ty.is_static_array():
            inner = ty.static_array_size().split('][')
            val = self._zero_base(ty)
            for dim in reversed(inner):
                val = '[%s; %s]' % (val, self.dimension(dim))
            return val
        if ty.is_pointer():
            return 'core::ptr::null_mut()' if 'const' not in (ty.decor.ref_quals[0] or '') \
                else 'core::ptr::null()'
        return self._zero_base(ty)

    def _zero_base(self, ty):
        base = ty.base
        if base.category == VkType.DEFAULT:
            rs = PRIMITIVES[base.name]
            return PRIMITIVE_ZERO.get(rs, '0')
        if base.category == VkType.FUNCPOINTER:
            return 'None'
        if base.category in (VkType.STRUCT, VkType.UNION):
            return '%s::default()' % base.name
        if base.category == VkType.BITMASK:
            # A bitmask is an alias for VkFlags or VkFlags64, so its zero is the alias target's.
            return '%s(0)' % (base.typedef.name if base.typedef else 'VkFlags')
        # Newtypes: enums, handles, base types and bitmasks all wrap a scalar.
        return '%s(0)' % base.name

    # --- declarations ---

    @staticmethod
    def field_name(name):
        return 'r#' + name if name in KEYWORDS else name

    def struct_fields(self, ty):
        return [(self.field_name(v.name), self.field_type(v)) for v in ty.variables]

    def command_params(self, ty):
        """A command's arguments as the fields of its `vn_command_*` struct, reply included."""
        fields = [(self.field_name(v.name), self.field_type(v)) for v in ty.variables]
        if ty.ret:
            fields.append((self.field_name(ty.ret.name), self.field_type(ty.ret)))
        return fields

    def funcpointer(self, ty):
        params = ', '.join(self.field_type(v) for v in ty.variables)
        ret = ''
        if ty.ret and not (self.base_name(ty.ret.ty) == 'core::ffi::c_void'
                           and not ty.ret.ty.is_pointer()):
            ret = ' -> %s' % self.field_type(ty.ret)
        return 'Option<unsafe extern "C" fn(%s)%s>' % (params, ret)

    def enum_repr(self, ty):
        return 'u64' if ty.enums.bitwidth == 64 else 'i32'

    def enum_values(self, ty):
        """Enumerant name and value. vk.xml's negative error codes stay negative."""
        return list(ty.enums.values.items())
