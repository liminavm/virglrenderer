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

# Gen.VariableInfo's validity, restated so this module does not reach into a nested class: the
# member is initialized (an input), uninitialized (an output), or partially so (an output that is
# still a handle or a struct, and so carries an id, an sType or a pNext the guest chose).
Gen_VALID = 0
Gen_INVALID = 1
Gen_PARTIAL = 2

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

    # ------------------------------------------------------------------
    # Statement emission.
    #
    # The C backend fills structures *through* pointers, which is why its VariableInfo carries a
    # deref/const-cast algebra. This one allocates, fills through a `&mut`, and stores the pointer
    # last, so none of that machinery is needed and the generated decode is safe Rust. What may
    # never change is the order in which bytes leave the stream, or which array-size decodes are
    # checked: those decide what the renderer accepts, and a wire round trip cannot see them.
    # ------------------------------------------------------------------

    SCALAR_CATEGORIES = (VkType.DEFAULT, VkType.BASETYPE, VkType.ENUM, VkType.BITMASK)

    class Unsupported(Exception):
        """A member shape the emitter has not been taught.

        Raised at generation time on purpose. A gap that stops the build names itself; a gap that
        emits plausible-looking wrong code costs a day of round-trip bisection.
        """

    def member_expr(self, var):
        return 'val.%s' % self.field_name(var.name)

    def scalar_of(self, ty):
        """The Rust type a scalar member decodes as, or None if it is not a scalar."""
        base = ty.base
        if base.category not in self.SCALAR_CATEGORIES:
            return None
        if base.category == VkType.BITMASK:
            return base.typedef.name if base.typedef else 'VkFlags'
        return self.base_name(base)

    def _substitute_constants(self, expr):
        """vk.xml writes length expressions in C, so they can name an API constant."""
        for key, val in self.constants.items():
            if key in expr:
                expr = expr.replace(key, str(val))
        return expr

    def _len_expr(self, ty, var, level=0):
        """The element count of a dynamic array, as a Rust `u64` expression."""
        exprs = var.attrs.get('len_exprs')
        names = var.attrs.get('len_names')
        if not exprs or len(exprs) != 1 or exprs[0] == 'null-terminated':
            raise self.Unsupported('%s.%s: len %r' % (ty.name, var.name, exprs))
        expr, name = exprs[0], names[0]
        expr = self._substitute_constants(expr)
        if not name:
            return '(%s) as u64' % expr
        if '->' in name or '[' in name:
            raise self.Unsupported('%s.%s: indirect len %r' % (ty.name, var.name, name))
        holders = ty.find_variables(name)
        holder = holders[-1]
        access = 'val.%s' % self.field_name(name)
        if holder.ty.base.category in (VkType.BASETYPE, VkType.ENUM, VkType.BITMASK):
            access += '.0'
        return '(%s) as u64' % expr.replace(name, access)

    def _shape(self, ty, var):
        """How a member is laid out: ('static', n) | ('dynamic', len) | ('pointer',) | ('plain',)."""
        if var.is_blob():
            raise self.Unsupported('%s.%s: blob' % (ty.name, var.name))
        if 'stride' in var.attrs or 'selector' in var.attrs:
            raise self.Unsupported('%s.%s: %s' % (ty.name, var.name, sorted(var.attrs)))
        if var.ty.is_static_array():
            dim = var.ty.static_array_size()
            if '][' in dim:
                raise self.Unsupported('%s.%s: 2-D array' % (ty.name, var.name))
            return ('static', self.dimension(dim))
        if var.is_dynamic_array():
            if var.ty.is_c_string() or var.has_c_string():
                raise self.Unsupported('%s.%s: c string' % (ty.name, var.name))
            if var.ty.indirection_depth() != 1:
                raise self.Unsupported('%s.%s: pointer depth' % (ty.name, var.name))
            return ('dynamic', self._len_expr(ty, var))
        if var.ty.is_pointer():
            if var.ty.indirection_depth() != 1:
                raise self.Unsupported('%s.%s: pointer depth' % (ty.name, var.name))
            return ('pointer',)
        return ('plain',)

    def _elem_call(self, kind, ty, var, validity, alloc):
        """The per-element serializer to call, and whether it takes a scalar type parameter."""
        base = var.ty.base
        scalar = self.scalar_of(var.ty)
        if scalar:
            return ('scalar', scalar)
        if base.category == VkType.FUNCPOINTER:
            raise self.Unsupported('%s.%s: function pointer' % (ty.name, var.name))

        name = 'vn_%s_%s' % (kind, base.name)
        if base.category in (VkType.STRUCT, VkType.UNION):
            if validity == Gen_PARTIAL and base.category == VkType.STRUCT:
                name += '_partial'
            if kind == 'decode' and alloc:
                name += '_temp'
        elif base.category == VkType.HANDLE:
            if kind == 'decode':
                if validity == Gen_VALID:
                    name += '_lookup'
                elif alloc and base.dispatchable:
                    name += '_temp'
        else:
            raise self.Unsupported('%s.%s: category %d' % (ty.name, var.name, base.category))
        return ('call', name)

    def decode_member(self, ty, var, partial, alloc):
        """Rust statements decoding one struct member or command argument."""
        validity = self.gen._get_variable_validity(ty, var, not partial)
        if not self.gen.is_serializable(var):
            return ['dec.set_fatal();']

        shape = self._shape(ty, var)
        m = self.member_expr(var)
        elem_kind, elem = self._elem_call('decode', ty, var, validity, alloc)

        if shape[0] == 'plain':
            if validity == Gen_INVALID:
                return ['/* skip %s */' % m]
            if elem_kind == 'scalar':
                return ['%s = dec.decode_scalar::<%s>();' % (m, elem)]
            return ['%s(dec, &mut %s);' % (elem, m)]

        if shape[0] == 'static':
            n = shape[1]
            if validity == Gen_INVALID:
                return ['/* skip %s */' % m]
            lines = ['let array_size = dec.decode_array_size(%s) as usize;' % n]
            if elem_kind == 'scalar':
                lines.append('dec.decode_scalar_array(&mut %s[..array_size.min(%s)]);' % (m, n))
            else:
                lines.append('for e in %s[..array_size.min(%s)].iter_mut() {' % (m, n))
                lines.append('    %s(dec, e);' % elem)
                lines.append('}')
            return ['{'] + ['    ' + l for l in lines] + ['}']

        ptr = '*const' if var.ty.is_const_pointer() else '*mut'
        null = 'core::ptr::null()' if var.ty.is_const_pointer() else 'core::ptr::null_mut()'

        if shape[0] == 'pointer':
            miss = ['%s = %s;' % (m, null)]
            if not var.is_optional() and var.can_validate():
                miss.append('dec.set_fatal();')
            if not alloc:
                raise self.Unsupported('%s.%s: pointer without temp storage' % (ty.name, var.name))
            hit = ['let Some(p) = dec.alloc_temp::<%s>() else { return };' % self.base_name(var.ty)]
            if validity != Gen_INVALID:
                hit.append('%s(dec, p);' % elem if elem_kind == 'call'
                           else '*p = dec.decode_scalar::<%s>();' % elem)
            hit.append('%s = p as %s _;' % (m, ptr))
            return (['if dec.decode_simple_pointer() {'] + ['    ' + l for l in hit]
                    + ['} else {'] + ['    ' + l for l in miss] + ['}'])

        # dynamic array
        if not alloc:
            raise self.Unsupported('%s.%s: array without temp storage' % (ty.name, var.name))
        count = shape[1]
        hit = ['let n = dec.decode_array_size(%s) as usize;' % count,
               'let Some(a) = dec.alloc_temp_array::<%s>(n) else { return };'
               % self.base_name(var.ty)]
        if validity != Gen_INVALID:
            if elem_kind == 'scalar':
                hit.append('dec.decode_scalar_array(a);')
            else:
                hit.append('for e in a.iter_mut() {')
                hit.append('    %s(dec, e);' % elem)
                hit.append('}')
        hit.append('%s = a.as_%s;' % (m, 'ptr()' if var.ty.is_const_pointer() else 'mut_ptr()'))

        if not var.is_optional() and var.can_validate():
            miss = ['dec.decode_array_size(%s);' % count]
        else:
            miss = ['dec.decode_array_size_unchecked();']
        miss.append('%s = %s;' % (m, null))

        return (['if dec.peek_array_size() != 0 {'] + ['    ' + l for l in hit]
                + ['} else {'] + ['    ' + l for l in miss] + ['}'])

    def encode_member(self, ty, var, partial):
        """Rust statements encoding one struct member.

        Reads array members back through the raw pointer decode stored, which is the one place the
        generated code is unsafe. The invariant is the decoder's: it allocated exactly that many
        elements from its arena, and the arena outlives the encode.
        """
        return self._out_member('encode', ty, var, partial)

    def sizeof_member(self, ty, var, partial):
        """Rust statements accumulating one struct member's wire size into `size`."""
        return self._out_member('sizeof', ty, var, partial)

    def _out_member(self, kind, ty, var, partial):
        validity = self.gen._get_variable_validity(ty, var, not partial)
        if not self.gen.is_serializable(var):
            return ['unreachable!("not serializable");']

        shape = self._shape(ty, var)
        m = self.member_expr(var)
        elem_kind, elem = self._elem_call(kind, ty, var, validity, False)

        def one(expr):
            if kind == 'encode':
                if elem_kind == 'scalar':
                    return 'enc.encode_scalar::<%s>(%s);' % (elem, expr)
                return '%s(enc, &%s);' % (elem, expr)
            if elem_kind == 'scalar':
                return 'size += cs::sizeof_scalar::<%s>();' % elem
            return 'size += %s(proto, &%s);' % (elem, expr)

        def many(expr, count):
            """A whole array: scalars are packed and padded as one, others are per-element."""
            if elem_kind == 'scalar':
                if kind == 'encode':
                    return ['enc.encode_scalar_array::<%s>(%s);' % (elem, expr)]
                return ['size += cs::sizeof_scalar_array::<%s>(%s as usize);' % (elem, count)]
            body = one('e') if kind == 'encode' else 'size += %s(proto, e);' % elem
            return ['for e in %s {' % expr, '    ' + body, '}']

        def array_size(count):
            if kind == 'encode':
                return 'enc.encode_array_size(%s);' % count
            return 'size += cs::sizeof_scalar::<u64>();'

        if shape[0] == 'plain':
            if validity == Gen_INVALID:
                return ['/* skip %s */' % m]
            return [one(m)]

        if shape[0] == 'static':
            n = shape[1]
            lines = [array_size('%s as u64' % n)]
            if validity != Gen_INVALID:
                lines += many('&%s' % m if elem_kind != 'scalar' else '&%s' % m, n)
            return lines

        if shape[0] == 'pointer':
            if validity == Gen_INVALID:
                if kind == 'encode':
                    return ['enc.encode_simple_pointer(!%s.is_null()); /* out */' % m]
                return ['size += cs::sizeof_scalar::<u32>(); /* out */']
            inner = one('*%s' % m)
            if kind == 'encode':
                return ['if enc.encode_simple_pointer(!%s.is_null()) {' % m,
                        '    // SAFETY: non-null here, and it points at the decoder arena entry',
                        '    // this member was decoded into.',
                        '    unsafe { %s }' % inner,
                        '}']
            return ['size += cs::sizeof_scalar::<u32>();',
                    'if !%s.is_null() {' % m,
                    '    // SAFETY: as above.',
                    '    unsafe { %s }' % inner,
                    '}']

        # dynamic array
        count = shape[1]
        slice_expr = ('core::slice::from_raw_parts(%s, (%s) as usize)' % (m, count))
        if validity == Gen_INVALID:
            return [array_size('if %s.is_null() { 0 } else { %s }' % (m, count))]
        lines = ['if !%s.is_null() {' % m, '    ' + array_size(count)]
        lines.append('    // SAFETY: non-null, and the decoder allocated exactly this many')
        lines.append('    // elements from its arena for this member.')
        lines.append('    unsafe {')
        lines += ['        ' + l for l in many(slice_expr, count)]
        lines.append('    }')
        lines.append('} else {')
        lines.append('    ' + array_size('0'))
        lines.append('}')
        return lines

    # ------------------------------------------------------------------
    # Whole-file emission.
    # ------------------------------------------------------------------

    def _fn(self, sig, body, gaps, what):
        """One generated function, or a poisoning stub if the emitter hit a shape it lacks.

        A stub cannot reproduce the guest's bytes, so the wire round trip fails on the first
        command that needs one. That is the loud failure this trades a build error for -- and it
        keeps the other five hundred types buildable while the gap is filled.
        """
        try:
            lines = body()
        except self.Unsupported as e:
            gaps.append(str(e))
            if 'dec:' in sig:
                lines = ['dec.set_fatal();']
            elif 'enc:' in sig:
                lines = ['let _ = val;', 'enc.write(0, &[]);']
            else:
                lines = ['let _ = (proto, val);', '0']
            lines.insert(0, '/* gap: %s */' % e)
        return ['pub fn %s {' % sig] + ['    ' + l for l in lines] + ['}', '']

    def _struct_body(self, kind, ty, variant, gaps):
        skip = 2 if '_self' in variant and ty.s_type else 0
        partial = '_partial' in variant
        alloc = '_temp' in variant
        out = []
        if kind == 'sizeof':
            out.append('let mut size = 0usize;')
        if skip:
            out.append('/* skip val.{sType,pNext} */')
        for var in ty.variables[skip:]:
            if kind == 'decode':
                out += self.decode_member(ty, var, partial, alloc)
            elif kind == 'encode':
                out += self.encode_member(ty, var, partial)
            else:
                out += self.sizeof_member(ty, var, partial)
        if kind == 'sizeof':
            out.append('size')
        return out

    def _chain_condition(self, next_ty):
        """The C generator's protocol gate, as a Rust expression that is true when the struct must
        be skipped."""
        stmt = self.gen.get_type_condition(next_ty)
        if not stmt:
            return None
        out = stmt.replace('vn_cs_renderer_protocol_has_extension(', 'PROTO.has_extension(')
        return out.replace('vn_cs_renderer_protocol_has_api_version(', 'PROTO.has_api_version(')

    def render_serialize(self, gaps):
        """The whole serializer: handles, structs, chains."""
        out = []
        for ty in self.gen.supported_types[VkType.HANDLE]:
            out += self._handle_fns(ty)
        for ty in self.gen.supported_types[VkType.UNION]:
            out += self._union_fns(ty, gaps)
        for ty in self.gen.supported_types[VkType.STRUCT]:
            if ty.s_type:
                out += self._chain_fns(ty, gaps)
            else:
                out += self._plain_struct_fns(ty, gaps)
        return '\n'.join(out)

    def _handle_fns(self, ty):
        objtype = 'VkObjectType::%s' % ty.attrs['c_objtype']
        n = ty.name
        return [
            '/// Handles are pointer-sized on every target this renderer supports, so the guest id',
            '/// lives in the handle slot itself -- what the C calls a direct, never indirect, id.',
            'pub fn vn_sizeof_%s(_proto: &dyn cs::Protocol, _val: &%s) -> usize {' % (n, n),
            '    cs::sizeof_scalar::<u64>()',
            '}',
            '',
            'pub fn vn_encode_%s(enc: &mut Encoder<\'_>, val: &%s) {' % (n, n),
            '    enc.encode_scalar::<u64>(val.0);',
            '}',
            '',
            'pub fn vn_decode_%s(dec: &mut Decoder<\'_>, val: &mut %s) {' % (n, n),
            '    val.0 = dec.decode_scalar::<u64>();',
            '}',
            '',
            'pub fn vn_decode_%s_temp(dec: &mut Decoder<\'_>, val: &mut %s) {' % (n, n),
            '    vn_decode_%s(dec, val);' % n,
            '}',
            '',
            '/// Resolve the guest id to the host object, poisoning this command if it names one',
            '/// the host never created.',
            'pub fn vn_decode_%s_lookup(dec: &mut Decoder<\'_>, val: &mut %s) {' % (n, n),
            '    let id = dec.decode_scalar::<u64>();',
            '    val.0 = dec.lookup_object(ObjectId(id), %s.0);' % objtype,
            '}',
            '',
        ]

    def _union_fns(self, ty, gaps):
        """Unions are selected by a tag the owning struct carries, which the member emitter does
        not yet thread through; the bodies are gaps, but the names have to exist."""
        n = ty.name
        tag = ', tag: %s' % ty.sty.name if ty.is_valid_union() else ''
        why = '%s: union' % n

        def gap(sig, kind):
            gaps.append(why)
            if kind == 'decode':
                body = ['dec.set_fatal();']
            elif kind == 'encode':
                body = ['let _ = val;', 'enc.write(0, &[]);']
            else:
                body = ['let _ = (proto, val);', '0']
            return (['/* gap: %s */' % why, 'pub fn %s {' % sig]
                    + ['    ' + l for l in body] + ['}', ''])

        out = []
        out += gap('vn_sizeof_%s(proto: &dyn cs::Protocol, val: &%s%s) -> usize'
                   % (n, n, tag), 'sizeof')
        out += gap('vn_encode_%s(enc: &mut Encoder<\'_>, val: &%s%s)' % (n, n, tag), 'encode')
        out += gap('vn_decode_%s_temp(dec: &mut Decoder<\'_>, val: &mut %s%s)'
                   % (n, n, tag), 'decode')
        return out

    def _plain_struct_fns(self, ty, gaps):
        n = ty.name
        out = []
        out += self._fn(
            'vn_sizeof_%s(proto: &dyn cs::Protocol, val: &%s) -> usize' % (n, n),
            lambda: self._struct_body('sizeof', ty, '', gaps), gaps, n)
        out += self._fn('vn_encode_%s(enc: &mut Encoder<\'_>, val: &%s)' % (n, n),
                        lambda: self._struct_body('encode', ty, '', gaps), gaps, n)
        out += self._fn('vn_decode_%s_temp(dec: &mut Decoder<\'_>, val: &mut %s)' % (n, n),
                        lambda: self._struct_body('decode', ty, '_temp', gaps), gaps, n)
        return out

    def _chain_fns(self, ty, gaps):
        n = ty.name
        next_types, skipped = self.gen.get_chain(ty)
        out = []

        out += self._fn(
            'vn_sizeof_%s_self(proto: &dyn cs::Protocol, val: &%s) -> usize' % (n, n),
            lambda: self._struct_body('sizeof', ty, '_self', gaps), gaps, n)
        out += self._fn('vn_encode_%s_self(enc: &mut Encoder<\'_>, val: &%s)' % (n, n),
                        lambda: self._struct_body('encode', ty, '_self', gaps), gaps, n)
        out += self._fn('vn_decode_%s_self_temp(dec: &mut Decoder<\'_>, val: &mut %s)' % (n, n),
                        lambda: self._struct_body('decode', ty, '_self_temp', gaps), gaps, n)

        out += self._chain_pnext_sizeof(ty, next_types)
        out += self._chain_pnext_encode(ty, next_types)
        out += self._chain_pnext_decode(ty, next_types)

        out += [
            'pub fn vn_sizeof_%s(proto: &dyn cs::Protocol, val: &%s) -> usize {' % (n, n),
            '    let mut size = cs::sizeof_scalar::<VkStructureType>();',
            '    // SAFETY: pNext points at the chain this struct was decoded with.',
            '    size += unsafe { vn_sizeof_%s_pnext(proto, val.pNext as *const c_void) };' % n,
            '    size += vn_sizeof_%s_self(proto, val);' % n,
            '    size',
            '}',
            '',
            'pub fn vn_encode_%s(enc: &mut Encoder<\'_>, val: &%s) {' % (n, n),
            '    enc.encode_scalar::<VkStructureType>(VkStructureType::%s);' % ty.s_type,
            '    // SAFETY: as above.',
            '    unsafe { vn_encode_%s_pnext(enc, val.pNext as *const c_void) };' % n,
            '    vn_encode_%s_self(enc, val);' % n,
            '}',
            '',
            'pub fn vn_decode_%s_temp<\'a>(dec: &mut Decoder<\'a>, val: &mut %s) {' % (n, n),
            '    let stype = dec.decode_scalar::<VkStructureType>();',
            '    if stype != VkStructureType::%s {' % ty.s_type,
            '        dec.set_fatal();',
            '    }',
            '    val.sType = stype;',
            '    val.pNext = vn_decode_%s_pnext_temp(dec) as _;' % n,
            '    vn_decode_%s_self_temp(dec, val);' % n,
            '}',
            '',
        ]
        return out

    def _chain_pnext_sizeof(self, ty, next_types):
        proto = 'proto'
        n = ty.name
        out = ['/// # Safety',
               '/// `val` is a `pNext` chain of structs this decoder allocated.',
               'pub unsafe fn vn_sizeof_%s_pnext(proto: &dyn cs::Protocol, val: *const c_void) -> usize {' % n]
        if not next_types:
            out += ['    let _ = (proto, val);',
                    '    return cs::sizeof_scalar::<u32>(); /* no known struct */',
                    '}', '']
            return out
        out += ['    let mut pnext = val as *const VkBaseInStructure;',
                '    let mut size = 0usize;',
                '    while !pnext.is_null() {',
                '        let node = unsafe { &*pnext };',
                '        match node.sType {']
        for nt in next_types:
            cond = self._chain_condition(nt)
            out.append('            VkStructureType::%s => {' % nt.s_type)
            if cond:
                out.append('                if %s {' % cond.replace('PROTO', proto))
                out.append('                    pnext = node.pNext;')
                out.append('                    continue;')
                out.append('                }')
            out += [
                '                size += cs::sizeof_scalar::<u32>();',
                '                size += cs::sizeof_scalar::<VkStructureType>();',
                '                let here = unsafe { &*(pnext as *const %s) };' % nt.name,
                '                size += unsafe {',
                '                    vn_sizeof_%s_pnext(proto, here.pNext as *const c_void)' % n,
                '                };',
                '                size += vn_sizeof_%s_self(proto, here);' % nt.name,
                '                return size;',
                '            }',
            ]
        out += ['            _ => {}',
                '        }',
                '        pnext = node.pNext;',
                '    }',
                '    size + cs::sizeof_scalar::<u32>()',
                '}', '']
        return out

    def _chain_pnext_encode(self, ty, next_types):
        proto = 'enc.protocol()'
        n = ty.name
        out = ['/// # Safety',
               '/// `val` is a `pNext` chain of structs this decoder allocated.',
               'pub unsafe fn vn_encode_%s_pnext(enc: &mut Encoder<\'_>, val: *const c_void) {' % n]
        if not next_types:
            out += ['    let _ = val;',
                    '    enc.encode_simple_pointer(false); /* no known struct */',
                    '}', '']
            return out
        out += ['    let mut pnext = val as *const VkBaseInStructure;',
                '    while !pnext.is_null() {',
                '        let node = unsafe { &*pnext };',
                '        match node.sType {']
        for nt in next_types:
            cond = self._chain_condition(nt)
            out.append('            VkStructureType::%s => {' % nt.s_type)
            if cond:
                out.append('                if %s {' % cond.replace('PROTO', proto))
                out.append('                    pnext = node.pNext;')
                out.append('                    continue;')
                out.append('                }')
            out += [
                '                enc.encode_simple_pointer(true);',
                '                enc.encode_scalar::<VkStructureType>(node.sType);',
                '                let here = unsafe { &*(pnext as *const %s) };' % nt.name,
                '                unsafe { vn_encode_%s_pnext(enc, here.pNext as *const c_void) };' % n,
                '                vn_encode_%s_self(enc, here);' % nt.name,
                '                return;',
                '            }',
            ]
        out += ['            _ => {}',
                '        }',
                '        pnext = node.pNext;',
                '    }',
                '    enc.encode_simple_pointer(false);',
                '}', '']
        return out

    def _chain_pnext_decode(self, ty, next_types):
        n = ty.name
        out = ['pub fn vn_decode_%s_pnext_temp<\'a>(dec: &mut Decoder<\'a>) -> *mut c_void {' % n]
        if not next_types:
            out += ['    if dec.decode_simple_pointer() {',
                    '        dec.set_fatal();',
                    '    }',
                    '    core::ptr::null_mut() /* no known struct */',
                    '}', '']
            return out
        out += ['    if !dec.decode_simple_pointer() {',
                '        return core::ptr::null_mut();',
                '    }',
                '    let stype = dec.decode_scalar::<VkStructureType>();',
                '    match stype {']
        for nt in next_types:
            out += [
                '        VkStructureType::%s => {' % nt.s_type,
                '            let Some(p) = dec.alloc_temp::<%s>() else {' % nt.name,
                '                return core::ptr::null_mut();',
                '            };',
                '            p.sType = stype;',
                '            p.pNext = vn_decode_%s_pnext_temp(dec) as _;' % n,
                '            vn_decode_%s_self_temp(dec, p);' % nt.name,
                '            p as *mut %s as *mut c_void' % nt.name,
                '        }',
            ]
        out += ['        _ => {',
                '            /* unexpected struct */',
                '            dec.set_fatal();',
                '            core::ptr::null_mut()',
                '        }',
                '    }',
                '}', '']
        return out
