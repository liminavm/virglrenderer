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

import re

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

    def __init__(self, gen, constants, bitfields=None, member_order=None, handle_parents=None):
        self.gen = gen
        # vk.xml's "API Constants" block, which the C generator never needs (it includes
        # vulkan.h) and which this one does: they are the static array dimensions.
        self.constants = constants
        # Structs vk.xml declares with C bit-fields. Only the layout oracle cares -- see
        # `gen.bitfield_types` for what the two sides do and do not owe each other there.
        self.bitfields = bitfields or {}
        # Each struct's members as the registry declares them. The model holds the order the wire
        # wants instead; `gen.member_order` is where the two part company.
        self.member_order = member_order or {}
        # Each handle's owning handle. The model drops vk.xml's `parent`; `gen.handle_parents`
        # says why, and `pool_children` is the only thing that reads it.
        self.handle_parents = handle_parents or {}

    # --- names ---

    def base_name(self, ty):
        """The Rust name of a type, ignoring any pointer or array decoration."""
        base = ty.base
        if base.category == VkType.DEFAULT:
            return PRIMITIVES[base.name]
        return base.name

    def bitmask_repr(self, ty):
        """The scalar a bitmask newtype wraps.

        vk.xml declares every bitmask as a typedef of `VkFlags` or `VkFlags64`, and those two are
        the whole vocabulary -- a third would be a new width on the wire and is worth the
        assertion rather than a silent 32.
        """
        if ty.typedef is None:
            return 'u32'
        assert ty.typedef.name in ('VkFlags', 'VkFlags64'), \
            '%s is a bitmask over %s, which is a width the encoder does not know' \
            % (ty.name, ty.typedef.name)
        return 'u64' if ty.typedef.name == 'VkFlags64' else 'u32'

    def bits_of(self, ty):
        """The `FlagBits` enum whose values belong in this bitmask, if this build emits one.

        vk.xml names it in `requires` or `bitvalues`. Some bitmasks have none -- reserved-for-
        future-use words with no bits defined yet -- and some name an enum this build does not
        emit, so the answer is optional and the `From` impl is emitted only when it exists.
        """
        bits = getattr(ty, 'requires', None)
        if bits is None:
            return None
        emitted = {e.name for e in self.gen.supported_types[VkType.ENUM]}
        return bits if bits.name in emitted else None

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
        # Newtypes: enums, handles, base types and bitmasks all wrap a scalar.
        return '%s(0)' % base.name

    # --- declarations ---

    @staticmethod
    def field_name(name):
        return 'r#' + name if name in KEYWORDS else name

    def struct_fields(self, ty):
        """A struct's members as they are laid out, which is not the order they serialize in.

        See `gen.member_order`. A type the registry does not declare -- venus\'s own, from the
        private xmls -- keeps the model\'s order, which for those is the declared one.
        """
        return [(self.field_name(v.name), self.field_type(v)) for v in self.laid_out(ty)]

    def laid_out(self, ty):
        """`ty`\'s members in declaration order."""
        order = self.member_order.get(ty.name)
        if not order:
            return ty.variables
        rank = {name: i for i, name in enumerate(order)}
        # A member the registry does not name cannot be ranked, and a partial sort would be worse
        # than none: it would move some members and leave others, which is neither order.
        assert all(v.name in rank for v in ty.variables), ty.name
        return sorted(ty.variables, key=lambda v: rank[v.name])

    def command_params(self, ty):
        """A command's arguments as the fields of its `vn_command_*` struct, reply included.

        The wire's members first, then the host-side shadows -- see `shadows`, which is where the
        reason they are separate members and the reason they come last are both written down.

        Each carries its own visibility, because an array's pointer is not the handlers\' to read.
        See `restricted`.
        """
        shut = self.restricted(ty)

        def vis(f):
            return 'pub(in crate::venus::proto) ' if f in shut else 'pub '

        return [(vis(f), f, rs) for f, rs in self._members(ty)]

    def _members(self, ty):
        """A command's fields as `(name, rust type)`, in wire order, without visibility.

        Split out from `command_params` because `restricted` needs the types to decide what to
        shut in, and `command_params` asks `restricted` -- reading the types through it would be
        a cycle.
        """
        fields = [(self.field_name(v.name), self.param_type(ty, v)) for v in ty.variables]
        if ty.ret:
            fields.append((self.field_name(ty.ret.name), self.field_type(ty.ret)))
        fields += [(f, rs) for f, rs, _ in self.shadows(ty)]
        return fields

    def scalar_rows(self, ty):
        """The members that point at one value, as `(field, element type, mutable)`.

        The single-value half of the array wall. A `*mut` member is where a query writes its
        answer; a `*const` one is an id the guest named. Both are pointers into the batch arena,
        and a handler that dereferences one is a handler doing the decoder's job with none of the
        decoder's knowledge -- which is how `venus/context.rs`, a file that is supposed to hold no
        unsafe at all, came to hold five blocks that were all the same mistake.

        Three shapes are left out on purpose. Arrays already have their own door and their count to
        be reconciled with. `c_char` is a string, and a reference to one byte is a worse answer
        than the pointer -- it looks complete and is not; that wants a `CStr` accessor of its own.
        `c_void` is untyped by construction, so there is no reference to hand out.
        """
        arrays = {f for f, _, _, _ in self._array_rows(ty)}
        shadowed = {f for f, _, _ in self.shadows(ty)}
        rows = []
        for f, rs in self._members(ty):
            if f in arrays:
                continue
            m = re.fullmatch(r'\*(mut|const) (.+)', rs)
            if not m:
                continue
            elem = m.group(2)
            if elem.startswith('*') or 'c_char' in elem or 'c_void' in elem:
                continue
            # A `*mut` member is not automatically one a handler writes. Where a shadow was
            # emitted beside it, the wire member holds the *guest's* id -- venus carries ids
            # directly, so the reply gives them back unchanged -- and the host handle goes in the
            # shadow. Writing the wire member there would hand the guest a host pointer. So the
            # shadow's existence is what says which of the pair is writable, exactly as it does
            # for arrays in `_array_rows`.
            field_mut = m.group(1) == 'mut'
            mutable = field_mut and ('handle_%s' % f) not in shadowed
            rows.append((f, elem, mutable, field_mut))
        return rows

    def string_rows(self, ty):
        """The members that point at a NUL-terminated string, as field names.

        The third door, and the one that needs no count. A counted array is a pair the wall exists
        to reconcile; a C string is not a pair at all -- `Decoder::decode_c_string` copies the
        guest's bytes into the arena and writes the terminator itself, so the length is inside the
        bytes and a second copy of it beside the pointer would be exactly the redundancy the rest
        of this file removes. What the pointer needed was not reconciling but typing.
        """
        return [self.field_name(var.name) for var in ty.variables
                if self._shape_or_none(ty, var) == ('string',)]

    def _shape_or_none(self, ty, var):
        """`_shape`, with an unsupported member reading as no shape rather than an exception."""
        try:
            return self._shape(ty, var)
        except self.Unsupported:
            return None

    def restricted(self, ty):
        """The members of `ty` no handler may reach, as field names.

        An array is a count and a pointer, and the pointer alone means nothing: read with the
        wrong count it is an out-of-bounds slice, and the counts are not even all members --
        eighteen of them come from an out-parameter, from inside another struct, or from
        arithmetic. So the pointer is shut in here with the code that knows its count, and
        `_command_accessors` emits the one door out.

        Only the pointers, not the counts. A count on its own is an integer a handler is free to
        read and cannot make unsound; shutting them in as well would also shut in the several
        that are not array lengths at all.

        The single-value members are shut in beside them, for the same reason and behind the same
        kind of door -- see `scalar_rows`.
        """
        return ({f for f, _, _, _ in self._array_rows(ty)}
                | {f for f, _, _, _ in self.scalar_rows(ty)}
                | set(self.string_rows(ty)))

    def destroy_target(self, ty):
        """The object a `vkDestroy*`/`vkFree*` names, as `(var, shape)`, or None.

        Its *last* handle argument. Every destroy in Vulkan is shaped `(parent, ..., target)`, so
        the earlier handles are the device or pool it came from and must survive it.
        """
        if not (ty.name.startswith('vkDestroy') or ty.name.startswith('vkFree')):
            return None
        handles = [v for v in ty.variables
                   if self.gen.is_serializable(v) and v.ty.base.category == VkType.HANDLE
                   and self.gen._get_variable_validity(ty, v, 'var_in' in v.attrs) == Gen_VALID]
        if not handles:
            return None
        var = handles[-1]
        try:
            return (var, self._shape(ty, var))
        except self.Unsupported:
            return None

    def create_owner(self, ty):
        """The object a create hangs off, as `(var, shape)`, or None.

        Vulkan spells creation `vkCreateX(parent, ..., out)`: the *first* handle argument is what
        owns the result -- the device for most objects, the physical device for a device, and
        nothing at all for an instance. Destroying that parent destroys everything under it and
        names none of them, so the object table only knows to follow if the parentage was recorded
        as the object was made.

        Recorded as the guest's id, never the host handle. A handle the driver is free to reuse
        would let a freshly created parent adopt a dead one's children -- which is the same
        staleness the table exists to prevent, moved one level up.
        """
        if not self.out_handles(ty):
            return None
        for var in ty.variables:
            if not self.gen.is_serializable(var) or var.ty.base.category != VkType.HANDLE:
                continue
            if self.gen._get_variable_validity(ty, var, 'var_in' in var.attrs) != Gen_VALID:
                continue
            try:
                shape = self._shape(ty, var)
            except self.Unsupported:
                return None
            # An owner named by an array is not a parent, and no Vulkan create is shaped that way.
            return (var, shape) if shape[0] == 'plain' else None
        return None

    def out_handles(self, ty):
        """The objects a command creates, as `(var, shape)`.

        An out-handle is a handle member the model calls PARTIAL, which is the same attribute pass
        the serializer makes -- so the create list is derived, not a list of command names someone
        has to keep current.
        """
        out = []
        for var in ty.variables:
            if not self.gen.is_serializable(var) or var.ty.base.category != VkType.HANDLE:
                continue
            if self.gen._get_variable_validity(ty, var, 'var_in' in var.attrs) != Gen_PARTIAL:
                continue
            try:
                out.append((var, self._shape(ty, var)))
            except self.Unsupported:
                continue
        return out

    def out_handle_fields(self, ty):
        """The visible members a create writes its *guest ids* into, by field name.

        The one place in a command struct where a handle slot holds the guest's word rather than
        the host's -- everywhere else the decoder has already replaced the id with the host handle
        (see `vn_decode_*_lookup`). The reading is a fact about the member, not about the call, so
        the accessor over one hands back `cs::Guest<T>` and the rest hand back the bare newtype.
        """
        return {self.field_name(var.name) for var, _ in self.out_handles(ty)}

    def pool_children(self):
        """The pools, as `(pool, child)` pairs -- a handle allocated from another handle.

        Derived from vk.xml, not listed. `parent` says which handle owns each, and the model says
        whether that owner is dispatchable. **A pool is a non-dispatchable owner**, and that is the
        whole rule: everything else a create hands back is parented on `VkDevice` or `VkInstance`,
        `VkDeviceMemory` included -- so without the dispatchable test `vkAllocateMemory` would make
        the device itself a pool, which compiles and is wrong. `VkQueryPool` is named a pool and is
        not one: nothing is parented on it, so it never appears here.

        The parentage is cross-checked against the command rather than trusted: the allocating
        command has to *name* the pool, in its own parameters or one level into the struct it is
        given, or the assert fires. vk.xml's ownership and the call the renderer actually serves
        have to agree before either is emitted.
        """
        handles = {t.name: t for t in self.gen.supported_types[VkType.HANDLE]}
        pools = {}
        for ty in self.gen.supported_types[VkType.COMMAND]:
            if not self.gen.is_serializable(ty):
                continue
            for var, _ in self.out_handles(ty):
                child = var.ty.base
                owner = handles.get(self.handle_parents.get(child.name))
                if owner is None or owner.dispatchable:
                    continue
                assert self._names_handle(ty, owner.name), (
                    '%s allocates %s from %s, which it never names'
                    % (ty.name, child.name, owner.name))
                seen = pools.setdefault(owner.name, child.name)
                assert seen == child.name, (
                    '%s allocates both %s and %s' % (owner.name, seen, child.name))
        return sorted(pools.items())

    def _names_handle(self, ty, name):
        """Whether a command names a handle type, in its parameters or one struct deep.

        One level is as deep as Vulkan puts it: an allocate takes its pool in the `*AllocateInfo`
        it is handed, never further in.
        """
        for var in ty.variables:
            base = var.ty.base
            if base.name == name:
                return True
            if base.category == VkType.STRUCT and any(
                    m.ty.base.name == name for m in base.variables):
                return True
        return False

    def shadows(self, ty):
        """The host-side members of a command's argument struct, as `(field, rust_type, shape)`.

        **Visible members are the wire's side; shadow members are the host's.** The two cannot be
        one member, because each direction needs both halves of the (guest id, host handle) pair at
        the moment the object table is written:

        - A destroy's target member is *replaced* by the host handle at lookup, which is what keeps
          the argument struct callable by the driver with no conversion. `id_<name>` is the guest
          id the decoder read before it was overwritten -- without it a destroy can only say which
          host handle died, and the table is keyed by id.
        - A create's out member must keep the guest id, because the reply re-encodes it and a host
          handle there would leak a host pointer into the guest. `handle_<name>` is where the
          driver writes instead: the decoder arena-allocates it, the handler passes it straight to
          Vulkan, and the pairing is registered from the two together.

        They are appended after every wire member, never interleaved. C struct layout is
        prefix-stable, so venus-protocol's own encoder -- which reads `vn_command_*` through a
        pointer and never sizeofs or allocates one -- sees the same offsets for every member it
        knows about. The layout-parity oracle is what holds that to an `offsetof` rather than to
        this paragraph.
        """
        out = []
        target = self.destroy_target(ty)
        if target:
            var, shape = target
            f = 'id_%s' % self.field_name(var.name)
            out.append((f, 'ObjectId' if shape[0] == 'plain' else '*const ObjectId', shape))
        for var, shape in self.out_handles(ty):
            f = 'handle_%s' % self.field_name(var.name)
            out.append((f, '*mut %s' % self.base_name(var.ty), shape))
        owner = self.create_owner(ty)
        if owner:
            var, shape = owner
            out.append(('id_%s' % self.field_name(var.name), 'ObjectId', shape))
        return out

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

    @staticmethod
    def null_of(var):
        return 'core::ptr::null()' if var.ty.is_const_pointer() else 'core::ptr::null_mut()'

    def member_expr(self, var):
        return 'val.%s' % self.field_name(var.name)

    def is_ref_member(self, ty, var):
        """Whether this member is a reference rather than a pointer.

        A command's arguments are what a handler is handed, and a handler is safe code (CLAUDE.md),
        so a member that can be a reference is one. What cannot:

        - An **array**. A slice is two words where C has one, and `vn_command_*` has to keep C's
          layout -- `render_layout_oracle` is where that is written down and checked. The count and
          the pointer stay, and are reconciled into a slice at the boundary that knows the truth.
        - An **out member**. The driver writes through it, and the reply encoder reads it back;
          a shared reference cannot express either.
        - A **string** or a **blob**. Neither is one pointee, and neither is sized by anything in
          the type.

        Struct members are never references: a `Vk*` is handed to the Vulkan driver, which owns
        the meaning of every pointer in it.
        """
        if ty.category != VkType.COMMAND:
            return False
        t = var.ty
        return (t.is_pointer() and not t.is_static_array() and len(t.decor.ref_quals) == 1
                and t.is_const_pointer() and 'len_names' not in var.attrs
                and not (t.base.category == VkType.DEFAULT and t.base.name in ('void', 'char')))

    def param_type(self, ty, var, life="'a"):
        """The Rust type of a command argument. See `is_ref_member`.

        `life` is the struct\'s lifetime everywhere but the layout table, which names types in a
        static and so has no borrow to name.
        """
        if self.is_ref_member(ty, var):
            return "Option<&%s %s>" % (life, self.base_name(var.ty))
        return self.field_type(var)

    def scalar_of(self, ty):
        """The Rust type a scalar member decodes as, or None if it is not a scalar."""
        base = ty.base
        if base.category not in self.SCALAR_CATEGORIES:
            return None
        return self.base_name(base)

    def _substitute_constants(self, expr):
        """vk.xml writes length expressions in C, so they can name an API constant."""
        for key, val in self.constants.items():
            if key in expr:
                expr = expr.replace(key, str(val))
        return expr

    def _len_expr(self, ty, var, levels=1):
        """The element count of a dynamic array, as a Rust `u64` expression.

        `levels` is how many len expressions the shape accounts for -- two for an array of strings,
        whose inner length is the terminator. A member with more than the shape expects is a gap,
        never a silently ignored dimension.
        """
        exprs = var.attrs.get('len_exprs')
        names = var.attrs.get('len_names')
        if not exprs or len(exprs) != levels or exprs[0] == 'null-terminated':
            raise self.Unsupported('%s.%s: len %r' % (ty.name, var.name, exprs))
        expr, name = exprs[0], names[0]
        expr = self._substitute_constants(expr)
        if not name:
            return '(%s) as u64' % expr
        if '[' in name:
            raise self.Unsupported('%s.%s: indirect len %r' % (ty.name, var.name, name))

        # A count can live behind a pointer -- `pPropertyCount` -- or inside a struct behind one
        # -- `pAllocateInfo->commandBufferCount`. Either way the C guards the read with the
        # pointer's own null check and calls a missing count zero, and so does this.
        parts = name.split('->')
        holders = ty.find_variables(name)
        if not holders or len(holders) != len(parts) or len(parts) > 2:
            raise self.Unsupported('%s.%s: len %r' % (ty.name, var.name, name))
        access = 'val.%s' % self.field_name(parts[0])
        guard = access if len(parts) > 1 or holders[0].ty.is_pointer() else None
        newtype = holders[-1].ty.base.category in (VkType.BASETYPE, VkType.ENUM, VkType.BITMASK)

        # When the holder is a reference member, the null check the C guards this read with is the
        # match itself, and there is no pointer left to dereference. `h` rather than a name from
        # vk.xml, so no member can shadow it.
        if guard and self.is_ref_member(ty, holders[0]):
            access = 'h.%s' % self.field_name(parts[1]) if len(parts) > 1 \
                else ('(*h)' if newtype else '*h')
            if newtype:
                access = '%s.0' % access
            inner = expr.replace(name, access)
            return 'match %s { Some(h) => (%s) as u64, None => 0 }' % (guard, inner)

        if len(parts) > 1:
            access = '(*%s).%s' % (access, self.field_name(parts[1]))
        elif guard:
            access = '(*%s)' % access if newtype else '*%s' % access
        if newtype:
            access = '%s.0' % access
        inner = expr.replace(name, access)
        if guard:
            # SAFETY, at every use: the pointer came out of the decoder's arena, and the decode of
            # the member that holds it ran before the one this length belongs to.
            return '(if %s.is_null() { 0 } else { unsafe { %s } }) as u64' % (guard, inner)
        return '(%s) as u64' % inner

    def _shape(self, ty, var):
        """How a member is laid out: ('static', n) | ('dynamic', len) | ('pointer',) | ('plain',)."""
        if 'stride' in var.attrs:
            raise self.Unsupported('%s.%s: stride' % (ty.name, var.name))
        if var.is_blob():
            return ('blob', self._len_expr(ty, var))
        if var.ty.is_static_array():
            dims = [self.dimension(d) for d in var.ty.static_array_size().split('][')]
            if len(dims) > 2:
                raise self.Unsupported('%s.%s: array depth %d' % (ty.name, var.name, len(dims)))
            total = ' * '.join(dims)
            return ('static', total, len(dims) > 1)
        if var.has_c_string():
            if var.ty.is_c_string():
                return ('string',)
            if var.ty.indirection_depth() != 2:
                raise self.Unsupported('%s.%s: string pointer depth' % (ty.name, var.name))
            return ('string_array', self._len_expr(ty, var, levels=2))
        if var.is_dynamic_array():
            if var.ty.indirection_depth() != 1:
                raise self.Unsupported('%s.%s: pointer depth' % (ty.name, var.name))
            return ('dynamic', self._len_expr(ty, var))
        if var.ty.is_pointer():
            if var.ty.indirection_depth() != 1:
                raise self.Unsupported('%s.%s: pointer depth' % (ty.name, var.name))
            return ('pointer',)
        return ('plain',)

    def _tag_arg(self, ty, var):
        """A tagged union's discriminant, which the owning struct carries in another member."""
        sel = var.attrs.get('selector')
        if not sel:
            return ''
        if '->' in sel or '[' in sel:
            raise self.Unsupported('%s.%s: indirect selector %r' % (ty.name, var.name, sel))
        return ', val.%s' % self.field_name(sel)

    def _elem_call(self, kind, ty, var, validity, alloc):
        """The per-element serializer to call, and whether it takes a scalar type parameter."""
        base = var.ty.base
        scalar = self.scalar_of(var.ty)
        if scalar:
            return ('scalar', scalar, '')
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
        if base.category == VkType.UNION and base.is_valid_union() and kind != 'decode':
            return ('call', name, self._tag_arg(ty, var))
        return ('call', name, '')

    def decode_member(self, ty, var, validity, alloc, capture=None):
        """Rust statements decoding one struct member or command argument.

        `capture` names a shadow member to keep the guest ids in. A handle lookup replaces the id
        with the host handle, and the wire position is gone afterwards, so an id a later step needs
        has to be kept as it goes past. Only a destroy target asks for this; see `shadows`.
        """
        if not self.gen.is_serializable(var):
            return self._dead_member('decode', ty, var)

        shape = self._shape(ty, var)
        m = self.member_expr(var)
        elem_kind, elem, tag = self._elem_call('decode', ty, var, validity, alloc)

        if shape[0] == 'plain':
            if validity == Gen_INVALID:
                return ['/* skip %s */' % m]
            if elem_kind == 'scalar':
                return ['%s = dec.decode_scalar::<%s>();' % (m, elem)]
            if capture:
                return ['val.%s = %s(dec, &mut %s);' % (capture, elem, m)]
            return ['%s(dec, &mut %s%s);' % (elem, m, tag)]

        if shape[0] == 'static':
            n = shape[1]
            # A 2-D array is one flat run of elements on the wire, in the order C would write it.
            flat = '%s.as_flattened_mut()' % m if shape[2] else '%s' % m
            if validity == Gen_INVALID:
                return ['/* skip %s */' % m]
            lines = ['let array_size = dec.decode_array_size(%s) as usize;' % n]
            if elem_kind == 'scalar':
                lines.append('dec.decode_scalar_array(&mut %s[..array_size.min(%s)]);' % (flat, n))
            else:
                lines.append('for e in %s[..array_size.min(%s)].iter_mut() {' % (flat, n))
                lines.append('    %s(dec, e%s);' % (elem, tag))
                lines.append('}')
            return ['{'] + ['    ' + l for l in lines] + ['}']

        ptr = '*const' if var.ty.is_const_pointer() else '*mut'
        ref = self.is_ref_member(ty, var)
        null = 'None' if ref else self.null_of(var)

        if shape[0] == 'blob':
            if validity == Gen_INVALID:
                return ['dec.decode_array_size(%s);' % shape[1], '%s = %s;' % (m, null)]
            # Borrowed from the stream, not copied: the arena is for what the guest does not
            # already hold in a contiguous, correctly sized run of wire bytes.
            hit = ['let n = dec.decode_array_size(%s) as usize;' % shape[1],
                   'let Some(b) = dec.decode_blob(n) else { return };',
                   '%s = b.as_ptr() as %s _;' % (m, ptr)]
            return self._present(shape[1], var, m, null, hit)

        if shape[0] == 'string':
            hit = ['let n = dec.decode_array_size_unchecked() as usize;',
                   'let Some(t) = dec.decode_c_string(n) else { return };',
                   '%s = t.as_ptr() as %s _;' % (m, ptr)]
            return self._present(None, var, m, null, hit)

        if shape[0] == 'string_array':
            if not alloc:
                raise self.Unsupported('%s.%s: strings without temp storage' % (ty.name, var.name))
            # An array of strings is an array of *pointers*, so the arena element is one pointer
            # wide -- not one character, which is what the base type would say.
            hit = ['let n = dec.decode_array_size(%s) as usize;' % shape[1],
                   'let Some(a) = dec.alloc_temp_array::<cs::Ptr>(n) else { return };',
                   'for e in a.iter_mut() {',
                   '    let n = dec.decode_array_size_unchecked() as usize;',
                   '    let Some(t) = dec.decode_c_string(n) else { return };',
                   '    *e = cs::Ptr(t.as_ptr() as *const c_void);',
                   '}',
                   '%s = a.as_ptr() as %s _;' % (m, ptr)]
            return self._present(shape[1], var, m, null, hit)

        if shape[0] == 'pointer':
            miss = ['%s = %s;' % (m, null)]
            if not var.is_optional() and var.can_validate():
                miss.append('dec.set_fatal();')
            if not alloc:
                raise self.Unsupported('%s.%s: pointer without temp storage' % (ty.name, var.name))
            hit = ['let Some(p) = dec.alloc_temp::<%s>() else { return };' % self.base_name(var.ty)]
            if validity != Gen_INVALID:
                hit.append('%s(dec, p%s);' % (elem, tag) if elem_kind == 'call'
                           else '*p = dec.decode_scalar::<%s>();' % elem)
            # The arena hands back `&'a mut T` and the member wants `&'a T`, which is where the
            # decoder's lifetime enters the struct: a reborrow, not a cast.
            hit.append('%s = Some(&*p);' % m if ref else '%s = p as %s _;' % (m, ptr))
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
            elif capture:
                # The ids and the handles are two arrays of the same length, filled in one pass:
                # the lookup hands back the id it just overwrote.
                hit.append('let Some(ids) = dec.alloc_temp_array::<ObjectId>(n) else { return };')
                hit.append('for (e, id) in a.iter_mut().zip(ids.iter_mut()) {')
                hit.append('    *id = %s(dec, e);' % elem)
                hit.append('}')
                hit.append('val.%s = ids.as_ptr();' % capture)
            else:
                hit.append('for e in a.iter_mut() {')
                hit.append('    %s(dec, e%s);' % (elem, tag))
                hit.append('}')
        hit.append('%s = a.as_%s;' % (m, 'ptr()' if var.ty.is_const_pointer() else 'mut_ptr()'))

        return self._present(count, var, m, null, hit)

    def _dead_member(self, kind, ty, var):
        """A member venus refuses to serialize -- `pAllocator`, above all.

        It still costs a word: the guest sends the pointer, and the renderer accepts it only when
        it is absent. Emitting nothing here would leave that word on the stream and desynchronise
        everything after it, which is how the first corpus run found this.
        """
        m = self.member_expr(var)
        if not var.ty.is_pointer() or not var.maybe_null():
            raise self.Unsupported('%s.%s: not serializable' % (ty.name, var.name))
        ref = self.is_ref_member(ty, var)
        null = 'None' if ref else self.null_of(var)
        absent = '%s.is_none()' % m if ref else '%s.is_null()' % m
        present = '%s.is_some()' % m if ref else '!%s.is_null()' % m
        if kind == 'decode':
            return ['if dec.decode_simple_pointer() {',
                    '    dec.set_fatal();',
                    '} else {',
                    '    %s = %s;' % (m, null),
                    '}']
        if kind == 'encode':
            return ['if enc.encode_simple_pointer(%s) {' % present,
                    '    debug_assert!(false, "%s is not serializable");' % var.name,
                    '}']
        return ['size += cs::sizeof_scalar::<u64>();',
                'debug_assert!(%s, "%s is not serializable");' % (absent, var.name)]

    def _present(self, count, var, m, null, hit):
        """The present/absent frame a wire array shares: peek the count, and on zero consume it
        anyway -- the guest sent it either way -- and leave the member null."""
        if count is not None and not var.is_optional() and var.can_validate():
            miss = ['dec.decode_array_size(%s);' % count]
        else:
            miss = ['dec.decode_array_size_unchecked();']
        miss.append('%s = %s;' % (m, null))
        return (['if dec.peek_array_size() != 0 {'] + ['    ' + l for l in hit]
                + ['} else {'] + ['    ' + l for l in miss] + ['}'])

    def encode_member(self, ty, var, validity):
        """Rust statements encoding one struct member.

        Reads array members back through the raw pointer decode stored, which is the one place the
        generated code is unsafe. The invariant is the decoder's: it allocated exactly that many
        elements from its arena, and the arena outlives the encode.
        """
        return self._out_member('encode', ty, var, validity)

    def sizeof_member(self, ty, var, validity):
        """Rust statements accumulating one struct member's wire size into `size`."""
        return self._out_member('sizeof', ty, var, validity)

    def _out_member(self, kind, ty, var, validity):
        if not self.gen.is_serializable(var):
            return self._dead_member(kind, ty, var)

        shape = self._shape(ty, var)
        m = self.member_expr(var)
        elem_kind, elem, tag = self._elem_call(kind, ty, var, validity, False)

        def one(expr):
            if kind == 'encode':
                if elem_kind == 'scalar':
                    return 'enc.encode_scalar::<%s>(%s);' % (elem, expr)
                return '%s(enc, &%s%s);' % (elem, expr, tag)
            if elem_kind == 'scalar':
                return 'size += cs::sizeof_scalar::<%s>();' % elem
            return 'size += %s(proto, &%s%s);' % (elem, expr, tag)

        def many(expr, count):
            """A whole array: scalars are packed and padded as one, others are per-element."""
            if elem_kind == 'scalar':
                if kind == 'encode':
                    return ['enc.encode_scalar_array::<%s>(%s);' % (elem, expr)]
                return ['size += cs::sizeof_scalar_array::<%s>(%s as usize);' % (elem, count)]
            body = one('e') if kind == 'encode' else 'size += %s(proto, e%s);' % (elem, tag)
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
            # A static array is skipped whole when it is an output: its extent is the struct's, so
            # the count on the wire would tell the far side nothing it does not already know.
            if validity == Gen_INVALID:
                return ['/* skip %s */' % m]
            n = shape[1]
            flat = '%s.as_flattened()' % m if shape[2] else '&%s' % m
            return [array_size('%s as u64' % n)] + many(flat, n)

        if shape[0] == 'blob':
            n = shape[1]
            if validity == Gen_INVALID:
                return ['enc.encode_array_size(%s); /* out */' % n] if kind == 'encode' \
                    else ['size += cs::sizeof_scalar::<u64>(); /* out */']
            if kind == 'encode':
                return ['if !%s.is_null() {' % m,
                        '    enc.encode_array_size(%s);' % n,
                        '    // SAFETY: the member points at the run of wire bytes the decoder',
                        '    // borrowed for it, which is exactly this long.',
                        '    unsafe {',
                        '        enc.encode_blob(core::slice::from_raw_parts(',
                        '            %s as *const u8, (%s) as usize));' % (m, n),
                        '    }',
                        '} else {',
                        '    enc.encode_array_size(0);',
                        '}']
            return ['size += cs::sizeof_scalar::<u64>();',
                    'if !%s.is_null() {' % m,
                    '    size += cs::sizeof_blob((%s) as usize);' % n,
                    '}']

        if shape[0] == 'string':
            # A string carries no count of its own on the wire, so its length has to come back out
            # of the bytes: the decoder guaranteed the terminator when it copied them in.
            if kind == 'encode':
                return ['if !%s.is_null() {' % m,
                        '    // SAFETY: NUL-terminated by the decoder, and the arena outlives this.',
                        '    unsafe {',
                        '        let n = cs::c_string_len(%s);' % m,
                        '        enc.encode_array_size(n as u64);',
                        '        enc.encode_blob(core::slice::from_raw_parts(%s as *const u8, n));'
                        % m,
                        '    }',
                        '} else {',
                        '    enc.encode_array_size(0);',
                        '}']
            return ['size += cs::sizeof_scalar::<u64>();',
                    'if !%s.is_null() {' % m,
                    '    // SAFETY: as above.',
                    '    size += cs::sizeof_blob(unsafe { cs::c_string_len(%s) });' % m,
                    '}']

        if shape[0] == 'string_array':
            n = shape[1]
            body = ['let n = cs::c_string_len(*e);']
            if kind == 'encode':
                body += ['enc.encode_array_size(n as u64);',
                         'enc.encode_blob(core::slice::from_raw_parts(*e as *const u8, n));']
            else:
                body += ['size += cs::sizeof_scalar::<u64>() + cs::sizeof_blob(n);']
            lines = ['if !%s.is_null() {' % m]
            lines.append('    ' + ('enc.encode_array_size(%s);' % n if kind == 'encode'
                                   else 'size += cs::sizeof_scalar::<u64>();'))
            lines.append('    // SAFETY: the decoder allocated this many pointers and made each')
            lines.append('    // one NUL-terminated in its arena.')
            lines.append('    unsafe {')
            lines.append('        for e in core::slice::from_raw_parts(%s, (%s) as usize) {'
                         % (m, n))
            lines += ['            ' + l for l in body]
            lines.append('        }')
            lines.append('    }')
            lines.append('} else {')
            lines.append('    ' + ('enc.encode_array_size(0);' if kind == 'encode'
                                   else 'size += cs::sizeof_scalar::<u64>();'))
            lines.append('}')
            return lines

        if shape[0] == 'pointer':
            if validity == Gen_INVALID:
                if kind == 'encode':
                    return ['enc.encode_simple_pointer(!%s.is_null()); /* out */' % m]
                return ['size += cs::sizeof_scalar::<u64>(); /* out */']
            if self.is_ref_member(ty, var):
                # A reference member needs no null check and no unsafe: the absence the wire can
                # express is the one the type can, and the borrow is the decoder's. The binding
                # is what keeps it that way -- `is_some` and then `unwrap` would be two reads of
                # one answer, which is the shape this whole change exists to stop emitting.
                if elem_kind == 'scalar':
                    body = 'enc.encode_scalar::<%s>(*p);' % elem if kind == 'encode' \
                        else 'size += cs::sizeof_scalar::<%s>();' % elem
                else:
                    body = '%s(enc, p%s);' % (elem, tag) if kind == 'encode' \
                        else 'size += %s(proto, p%s);' % (elem, tag)
                head = 'enc.encode_simple_pointer(%s.is_some());' % m if kind == 'encode' \
                    else 'size += cs::sizeof_scalar::<u64>();'
                # A body that never names the pointee needs no binding for it.
                guard = 'if let Some(p) = %s {' % m if 'p' in body.split('(', 1)[-1] \
                    else 'if %s.is_some() {' % m
                return [head, guard, '    ' + body, '}']
            inner = one('*%s' % m)
            if kind == 'encode':
                return ['if enc.encode_simple_pointer(!%s.is_null()) {' % m,
                        '    // SAFETY: non-null here, and it points at the decoder arena entry',
                        '    // this member was decoded into.',
                        '    unsafe { %s }' % inner,
                        '}']
            return ['size += cs::sizeof_scalar::<u64>();',
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
            validity = self.gen._get_variable_validity(ty, var, not partial)
            out += self._member(kind, ty, var, validity, alloc)
        if kind == 'sizeof':
            out.append('size')
        return out

    def _member(self, kind, ty, var, validity, alloc, capture=None):
        if kind == 'decode':
            return self.decode_member(ty, var, validity, alloc, capture)
        if kind == 'encode':
            return self.encode_member(ty, var, validity)
        return self.sizeof_member(ty, var, validity)

    def _chain_condition(self, next_ty):
        """The C generator's protocol gate, as a Rust expression that is true when the struct must
        be skipped."""
        stmt = self.gen.get_type_condition(next_ty)
        if not stmt:
            return None
        out = stmt.replace('vn_cs_renderer_protocol_has_extension(', 'PROTO.has_extension(')
        return out.replace('vn_cs_renderer_protocol_has_api_version(', 'PROTO.has_api_version(')

    @staticmethod
    def _vk_version(macro):
        """`VK_MAKE_API_VERSION(0, 1, 4, 357)` as the number Vulkan packs it into.

        The model hands this out as the C macro text, because its only consumer until now emitted
        C and let the preprocessor do the arithmetic. Rust has no preprocessor, so the generator
        does it -- with Vulkan's own field widths, not a guess at them.
        """
        name, _, rest = macro.partition('(')
        parts = [int(p.strip()) for p in rest.rstrip(')').split(',')]
        if name == 'VK_MAKE_API_VERSION':
            variant, major, minor, patch = parts
        else:
            assert name == 'VK_MAKE_VERSION', macro
            variant = 0
            major, minor, patch = parts
        return (variant << 29) | (major << 22) | (minor << 12) | patch

    def render_info(self, protocol_module):
        """What the guest is told the renderer speaks, and what the renderer checks it against.

        The venus capset carries a bitmask over extension *numbers*, so the guest and renderer agree
        on a protocol before a single command is decoded. The same table answers
        `Protocol::has_extension`, which is why it is generated rather than written: an extension
        this build cannot serialize must not appear in the mask.
        """
        exts = sorted((e for e in self.gen.reg.extensions
                       if e.name in protocol_module.VK_XML_EXTENSION_LIST),
                      key=lambda e: e.name)
        max_number = max(e.number for e in exts)
        assert max_number > 0

        out = ['/// The wire format venus-protocol pins. A guest speaking another one cannot be',
               '/// served at all -- the bytes would parse into different structs.',
               'pub const WIRE_FORMAT_VERSION: u32 = %d;' % protocol_module.VN_WIRE_FORMAT_VERSION,
               '',
               '/// The vk.xml this renderer was generated from. Reported to the guest so it can',
               '/// refuse a renderer older than the structs it means to send.',
               'pub const VK_XML_VERSION: u32 = %d;' % self._vk_version(
                   self.gen.reg.vk_xml_version),
               '',
               '/// The highest extension number in the table, which is what sizes the capset mask.',
               'pub const MAX_EXTENSION_NUMBER: u32 = %d;' % max_number,
               '',
               '/// Every extension this build can serialize: name, number, spec version.',
               '/// Sorted by name, because `extension` binary-searches it.',
               'pub static EXTENSIONS: &[(&str, u32, u32)] = &[']
        for e in exts:
            out.append('    ("%s", %d, %d),' % (e.name, e.number, e.version))
        out += ['];', '']

        out += ['/// The extension\'s entry, or `None` if this build does not serialize it.',
                'pub fn extension(name: &str) -> Option<&\'static (&\'static str, u32, u32)> {',
                '    EXTENSIONS.binary_search_by_key(&name, |e| e.0).ok().map(|i| &EXTENSIONS[i])',
                '}',
                '',
                '/// The spec version the guest is told, or 0 for an extension we do not serialize.',
                '/// Zero is the honest answer: the guest reads it as "not supported".',
                'pub fn spec_version(name: &str) -> u32 {',
                '    extension(name).map_or(0, |e| e.2)',
                '}',
                '',
                '/// The capset\'s extension bitmask, indexed by extension number.',
                'pub fn extension_mask(out: &mut [u32]) {',
                '    for (_, number, _) in EXTENSIONS {',
                '        out[(number / 32) as usize] |= 1 << (number % 32);',
                '    }',
                '}',
                '']
        return '\n'.join(out)

    # --- the reply oracle's fill ---

    # `pNext` stays null: chaining the outputs takes the reachable type set from 42 structs to
    # 235, and the null branch is the one a recorded command already exercises. `sType` is not
    # skipped but pinned -- the C encoder asserts on it, and a fresh output struct has no guest
    # request behind it to have set it.
    FILL_SKIP = frozenset(['pNext'])

    def _out_types(self, commands):
        """Every type a reply encoder can reach, walked from the out members inward.

        Far smaller than the serializer's world -- outputs are handles, results and property
        structs -- which is what makes a typed fill an afternoon rather than a second generator.
        """
        by_name = {t.name: t for cats in self.gen.supported_types.values() for t in cats}
        seen, work = {}, []
        for ty in commands:
            work += [v.ty.base.name for v in self._out_vars(ty)]
        while work:
            name = work.pop()
            if name in seen:
                continue
            ty = by_name.get(name)
            seen[name] = ty
            if ty is not None and ty.category in (VkType.STRUCT, VkType.UNION):
                work += [m.ty.base.name for m in ty.variables]
        return [t for t in seen.values()
                if t is not None and t.category in (VkType.STRUCT, VkType.UNION)]

    @staticmethod
    def _out_vars(ty):
        """A command's reply members: what it returns, and what it was asked to write."""
        return ([ty.ret] if ty.ret else []) + [v for v in ty.variables if 'var_out' in v.attrs]

    @staticmethod
    def _length_members(ty):
        """Members some other member's length names, which must be planted small rather than
        counted out of the same sequence as everything else."""
        names = set()
        for var in ty.variables:
            for name in var.attrs.get('len_names') or []:
                if name:
                    names.add(name.split('->')[0])
        return names

    def _fill_leaf(self, ty, value):
        """Fill one scalar, as an expression of `ty`'s Rust type.

        `as _` rather than a named cast: every one of these is a newtype over a scalar or a
        primitive, and the field it is assigned to already says which.
        """
        base = ty.base
        if base.category == VkType.DEFAULT:
            return '%s as _' % value
        if base.category == VkType.FUNCPOINTER:
            return 'None'
        return '%s(%s as _)' % (base.name, value)

    def _fill_one(self, ty, var, target, value):
        """Fill a single element of `var`'s base type, reached through `target`."""
        base = var.ty.base
        if base.category in (VkType.STRUCT, VkType.UNION):
            return ['vn_fill_%s(a, f, %s);' % (base.name, target)]
        return ['*%s = %s;' % (target, self._fill_leaf(var.ty, value))]

    def _fill_member(self, ty, var, small, gaps):
        """Plant a value in one member, whatever shape it is."""
        m = 'val.%s' % self.field_name(var.name)
        base = var.ty.base
        # The one member whose value is dictated rather than invented. venus-protocol's encoder
        # asserts a chained struct carries its own tag, and an output struct is allocated here
        # rather than decoded, so nothing else would set it.
        if var.name == 'sType' and ty.s_type:
            return ['%s = VkStructureType::%s;' % (m, ty.s_type)]
        # A length is read back by the member it sizes, so it decides how much gets allocated.
        value = 'FILL_COUNT' if var.name in small else 'f.take()'

        try:
            shape = self._shape(ty, var)
        except self.Unsupported as e:
            gaps.append(str(e))
            return ['/* gap: %s */' % e]

        if not self.gen.is_serializable(var):
            return ['/* not serializable: %s */' % m]

        if shape[0] == 'plain':
            if base.category in (VkType.STRUCT, VkType.UNION):
                return ['vn_fill_%s(a, f, &mut %s);' % (base.name, m)]
            return ['%s = %s;' % (m, self._fill_leaf(var.ty, value))]

        if shape[0] == 'static':
            flat = '%s.as_flattened_mut()' % m if shape[2] else '%s' % m
            return (['for e in %s.iter_mut() {' % flat]
                    + ['    ' + l for l in self._fill_one(ty, var, 'e', value)]
                    + ['}'])

        if shape[0] == 'pointer':
            return (['{',
                     '    let p = a.alloc(%s);' % self._zero_base(var.ty)]
                    + ['    ' + l for l in self._fill_one(ty, var, 'p', value)]
                    + ['    %s = p;' % m,
                       '}'])

        if shape[0] == 'dynamic':
            # The length expression carries its own `unsafe` where it derefs a count pointer.
            # SAFETY, there: the member holding that count is declared before this one and has
            # already been planted, so the pointer it reads is ours.
            return (['{',
                     '    let n = (%s) as usize;' % shape[1],
                     '    let s = a.alloc_slice_fill_with(n, |_| %s);' % self._zero_base(var.ty),
                     '    for e in s.iter_mut() {']
                    + ['        ' + l for l in self._fill_one(ty, var, 'e', value)]
                    + ['    }',
                       '    %s = s.as_mut_ptr();' % m,
                       '}'])

        # Blobs and strings carry their own length rules and no recorded reply exercises one, so
        # they are named gaps rather than a guess. They stay zeroed, which the oracle can still
        # compare -- it just cannot tell those bytes apart.
        gaps.append('%s.%s: fill %s' % (ty.name, var.name, shape[0]))
        return ['/* gap: fill %s */' % shape[0]]

    def render_fill(self, gaps):
        """Deterministic contents for a reply's output members.

        Without this the oracle is sensitive to a reply's *shape* -- framing, branch selection,
        skip order -- and blind to its content: a recorded command's outputs are null or zeroed, so
        an encoder reading the wrong member of two same-typed outputs writes identical bytes. This
        is the fourth walk over the model, and the one that makes a swap visible.
        """
        commands = [c for c in self.gen.supported_types[VkType.COMMAND]
                    if self.gen.is_serializable(c)]
        out = []

        for ty in self._out_types(commands):
            small = self._length_members(ty)
            body = []
            for var in ty.variables:
                if var.name in self.FILL_SKIP:
                    continue
                body += self._fill_member(ty, var, small, gaps)
            out += ['#[allow(unused_variables)]',
                    'pub fn vn_fill_%s(a: &Bump, f: &mut Fill, val: &mut %s) {' % (ty.name, ty.name)]
            out += ['    ' + l for l in body]
            out += ['}', '']

        for ty in commands:
            small = self._length_members(ty)
            body = []
            for var in self._out_vars(ty):
                body += self._fill_member(ty, var, small, gaps)
            out += ['#[allow(unused_variables)]',
                    "pub fn vn_fill_%s_outs(a: &Bump, f: &mut Fill, "
                    "val: &mut vn_command_%s<'_>) {" % (ty.name, ty.name)]
            out += ['    ' + l for l in body]
            out += ['}', '']

        out += ['/// Decode one command\'s arguments, plant contents in its outputs, encode the',
                '/// reply, and hand the *same* struct to `also` -- the C encoder it is diffed',
                '/// against.',
                '///',
                '/// One struct, two encoders. `vn_command_*` is `#[repr(C)]`, so the C reads the',
                '/// memory this filled rather than a second construction of it, and there is',
                '/// nothing for the two sides to disagree about before the encoding starts.',
                '///',
                '/// Returns what the sizeof said, so the caller can hold the encoder to it.',
                'pub fn vn_reply_oracle_args(',
                '    dec: &mut Decoder<\'_>,',
                '    enc: &mut Encoder<\'_>,',
                '    a: &Bump,',
                '    f: &mut Fill,',
                '    cmd: VkCommandTypeEXT,',
                '    also: &mut dyn FnMut(*const core::ffi::c_void),',
                ') -> Option<usize> {',
                '    match cmd {']
        for ty in commands:
            n = ty.name
            out += [
                '        VkCommandTypeEXT::%s => {' % ty.attrs['c_type'],
                '            let mut args = vn_command_%s::default();' % n,
                '            vn_decode_%s_args_temp(dec, &mut args);' % n,
                '            if dec.fatal() {',
                '                return Some(0);',
                '            }',
                '            vn_fill_%s_outs(a, f, &mut args);' % n,
                '            let size = vn_sizeof_%s_reply(enc.protocol(), &args);' % n,
                '            vn_encode_%s_reply(enc, &args);' % n,
                '            also(&raw const args as *const core::ffi::c_void);',
                '            Some(size)',
                '        }',
            ]
        out += ['        _ => None,', '    }', '}', '']

        return '\n'.join(out)

    def unlaid_out(self):
        """The types the layout oracle cannot ask a C compiler about, and everything holding one.

        A bit-field has no `offsetof`, and a struct containing one of these types inherits its
        size, so the exclusion has to travel up every containment edge to a fixpoint rather than
        stopping at the seven structs vk.xml declares. `gen.bitfield_types` carries the reason
        the divergence is allowed to stand at all.
        """
        out = set(self.bitfields)
        composites = [ty for kind in (VkType.STRUCT, VkType.UNION)
                      for ty in self.gen.supported_types[kind]]
        grew = True
        while grew:
            grew = False
            for ty in composites:
                if ty.name in out:
                    continue
                if any(v.ty.base.name in out and not v.ty.is_pointer() for v in ty.variables):
                    out.add(ty.name)
                    grew = True
        return out

    def layout_rows(self):
        """Every generated type a C compiler also defines, as `(rust, c, [(rust_f, c_f), ...])`.

        A member is `(rust_field, c_field, rust_type)`. Shadow members are left out: they exist
        on this side alone -- which is the whole reason `shadows` appends them after every wire
        member rather than interleaving them -- so asking a C compiler for their offsets would
        ask it about members it has never heard of. `unlaid_out` takes the rest away.
        """
        skip = self.unlaid_out()

        def member(owner, v):
            return (self.field_name(v.name), v.name, self.param_type(owner, v, "'static"))

        for kind in (VkType.STRUCT, VkType.UNION):
            for ty in self.gen.supported_types[kind]:
                if ty.name in skip:
                    continue
                yield ty.name, ty.name, [member(ty, v) for v in self.laid_out(ty)]
        for ty in self.gen.supported_types[VkType.COMMAND]:
            if not self.gen.is_serializable(ty):
                continue
            # A command reaches an excluded struct only through a pointer, which is a word
            # whatever it points at -- asserted rather than commented, because a command that
            # took one by value would silently compare two different sizes.
            assert not [v for v in ty.variables
                        if v.ty.base.name in skip and not v.ty.is_pointer()], ty.name
            members = [member(ty, v) for v in ty.variables]
            if ty.ret:
                members.append(member(ty, ty.ret))
            yield ('vn_command_%s' % ty.name, 'struct vn_command_%s' % ty.name, members)

    def render_layout_oracle(self):
        """The C half of the layout parity check: what a C compiler makes of the same structs.

        Two generated struct definitions being members-in-order is not the same claim as two
        struct definitions having the same layout, and only the second one is what the reply
        oracle relies on when it casts a pointer to our `vn_command_*` into venus-protocol's own
        encoder, or what the driver relies on when it is handed a `Vk*` we filled. Padding,
        alignment and the pointer-shape of a member are all places where the two can agree in
        source and disagree in memory.

        So both sides are asked the same question -- `offsetof` here, `offset_of!` there -- and
        the answers are compared. The two tables are index-matched rather than name-matched:
        one generator run emits both, so an index is exactly as trustworthy as a name and costs
        no strings in the binary. The names live on the Rust side, where a mismatch is reported.
        """
        types, members = [], []
        for rust, c, fields in self.layout_rows():
            types.append('    { (uint32_t)sizeof(%s), (uint32_t)_Alignof(%s) }, /* %s */'
                         % (c, c, rust))
            for _, cf, _ in fields:
                members.append('    { (uint32_t)offsetof(%s, %s), '
                               '(uint32_t)sizeof(((%s *)0)->%s) },' % (c, cf, c, cf))
        return '\n'.join([
            '/* The layout oracle: offsets and sizes as a C compiler computes them, for the',
            ' * generated Rust to be held to. Index-matched with the tables in `layout.rs`. */',
            '',
            '#include <stddef.h>',
            '#include <stdint.h>',
            '',
            '#include "vn_protocol_renderer.h"',
            '',
            'struct vn_layout_type { uint32_t size; uint32_t align; };',
            'struct vn_layout_member { uint32_t offset; uint32_t size; };',
            '',
            'const struct vn_layout_type vn_layout_types[] = {',
        ] + types + [
            '};',
            '',
            'const struct vn_layout_member vn_layout_members[] = {',
        ] + members + [
            '};',
            '',
            'const size_t vn_layout_type_count =',
            '    sizeof(vn_layout_types) / sizeof(vn_layout_types[0]);',
            'const size_t vn_layout_member_count =',
            '    sizeof(vn_layout_members) / sizeof(vn_layout_members[0]);',
            '',
        ])

    def render_layout_table(self):
        """The Rust half of the layout parity check. See `render_layout_oracle`."""
        types, members = [], []
        for rust, c, fields in self.layout_rows():
            shadowed = c.startswith('struct ')
            # A static holds no borrow, so the command structs are named at `'static` here. The
            # lifetime is not part of a layout, and it cannot be elided in this position.
            named = "%s<'static>" % rust if shadowed else rust
            types.append('    TypeLayout { name: "%s", size: size_of::<%s>(), '
                         'align: align_of::<%s>(), shadowed: %s },'
                         % (rust, named, named, 'true' if shadowed else 'false'))
            for rf, _, rt in fields:
                members.append('    MemberLayout { ty: "%s", name: "%s", '
                               'offset: offset_of!(%s, %s), size: size_of::<%s>() },'
                               % (rust, rf, named, rf, rt))
        return '\n'.join(['pub static TYPES: &[TypeLayout] = &['] + types
                          + ['];', '', 'pub static MEMBERS: &[MemberLayout] = &['] + members
                          + ['];', ''])

    def render_reply_oracle(self):
        """The C half of the reply differential, dispatched by command type.

        The wire round trip proves the request path against bytes a real guest encoder wrote, but
        no recording carries a reply: both replay entry points strip the reply flag, so the 326
        per-command reply wrappers have no witness. The oracle is byte-identity against
        venus-protocol's own renderer encoder, which is what every venus guest in existence
        decodes.

        Both sides encode the *same* `vn_command_*`, not two constructions of one: the generated
        Rust structs are `#[repr(C)]` mirrors, so Rust passes a pointer and C reads that memory.
        Nothing has to agree about filling -- including float bit patterns and padding, which two
        constructions would be free to differ on.
        """
        commands = [c for c in self.gen.supported_types[VkType.COMMAND]
                    if self.gen.is_serializable(c)]
        out = [
            '/* The reply oracle: venus-protocol\'s renderer encoder, reachable from Rust.',
            ' * The generated C is `static inline`, so a wrapper per command is the only way to',
            ' * take its address; a switch keeps the FFI surface at one symbol. */',
            '',
            '#include <stddef.h>',
            '#include <stdint.h>',
            '',
            '#include "vn_protocol_renderer.h"',
            '',
            '/* Distinct because they mean opposite things: a reply that did not fit is retried at',
            ' * a larger size, and a command this encoder has no arm for means the two generators',
            ' * disagree about the command list. One sentinel for both would retry the second until',
            ' * it hit the size ceiling, and name a generator mismatch as an oversized reply. */',
            '#define VN_ORACLE_OVERRAN ((size_t)-1)',
            '#define VN_ORACLE_NO_ARM  ((size_t)-2)',
            '',
            'size_t vn_oracle_reply(int32_t cmd, void *buf, size_t cap, const void *args);',
            '',
            'size_t vn_oracle_reply(int32_t cmd, void *buf, size_t cap, const void *args)',
            '{',
            '    struct vkr_cs_encoder enc;',
            '    vn_oracle_encoder_init(&enc, buf, cap);',
            '',
            '    switch ((VkCommandTypeEXT)cmd) {',
        ]
        for ty in commands:
            out += [
                '    case %s:' % ty.attrs['c_type'],
                '        vn_encode_%s_reply((struct vn_cs_encoder *)&enc,' % ty.name,
                '            (const struct vn_command_%s *)args);' % ty.name,
                '        break;',
            ]
        out += [
            '    default:',
            '        return VN_ORACLE_NO_ARM;',
            '    }',
            '',
            '    return enc.fatal ? VN_ORACLE_OVERRAN : vn_oracle_encoder_len(&enc);',
            '}',
            '',
        ]
        return '\n'.join(out)

    # Not an entry point in a table: it is how every table is loaded, so it is the one symbol
    # linked by name rather than resolved. MESA's commands are excluded for a different reason --
    # they are venus's own protocol, served by a renderer and exported by no driver.
    # `vkGetDeviceProcAddr` stays: it is device-level and reachable only through the instance.
    PROC_SKIP = frozenset(['vkGetInstanceProcAddr'])

    #: First-parameter types that make a command device-level. Anything reached through one of
    #: these dispatches through the device, so it comes from `vkGetDeviceProcAddr` and skips the
    #: loader's trampoline; everything else is instance-level or global.
    PROC_DEVICE_FIRST = frozenset(['VkDevice', 'VkQueue', 'VkCommandBuffer'])
    PROC_INSTANCE_FIRST = frozenset(['VkInstance', 'VkPhysicalDevice'])

    def proc_level(self, ty):
        """Which of the three tables a command belongs in.

        Vulkan does not say this anywhere machine-readable, but it follows from the first
        parameter: dispatch is on the handle, so a command taking a device-dispatchable handle is
        device-level by construction. `vkGetDeviceProcAddr` is the exception the loader forces --
        it is device-level and can only be found through the instance.
        """
        if ty.name == 'vkGetDeviceProcAddr':
            return 'instance'
        first = ty.variables[0].ty.name if ty.variables else ''
        if first in self.PROC_DEVICE_FIRST:
            return 'device'
        if first in self.PROC_INSTANCE_FIRST:
            return 'instance'
        return 'global'

    def proc_signature(self, ty):
        """A command's real C signature -- the one the driver exports, not the wire's.

        It comes from the same model the serializer is generated from, which is the point: a
        proc table transcribed by hand can disagree with the decoder about a parameter, and the
        disagreement is a stack smash rather than a compile error.
        """
        params = ', '.join(self.field_type(v) for v in ty.variables)
        ret = ' -> %s' % self.field_type(ty.ret) if ty.ret else ''
        return 'unsafe extern "C" fn(%s)%s' % (params, ret)

    def render_proc_table(self):
        """The driver's entry points, in three tables loaded the way Vulkan says to load them.

        This is what replaces `ash`. The types are the ones the decoder already speaks, so a
        command's arguments go from the wire to the driver with no conversion and no second
        definition to keep in step -- and MESA's venus-private commands, which no bindings crate
        has ever seen, are in the same model as the rest.
        """
        commands = [c for c in self.gen.supported_types[VkType.COMMAND]
                    if 'MESA' not in c.name and c.name not in self.PROC_SKIP]
        levels = {'global': [], 'instance': [], 'device': []}
        for ty in commands:
            levels[self.proc_level(ty)].append(ty)

        out = []
        for level, tys in [('global', levels['global']), ('instance', levels['instance']),
                           ('device', levels['device'])]:
            name = level.capitalize()
            loader = 'vkGetInstanceProcAddr' if level != 'device' else 'vkGetDeviceProcAddr'
            out += [
                '/// The %s-level entry points, as `%s` hands them back.' % (level, loader),
                '///',
                '/// Every field is optional because the loader answers null for a command the',
                '/// driver does not implement. Reading one through its accessor is what turns',
                '/// that into a panic naming the command, at the call rather than at the crash.',
                '#[derive(Default)]',
                'pub struct %s {' % name,
            ]
            for ty in tys:
                out.append('    fp_%s: Option<%s>,' % (ty.name, self.proc_signature(ty)))
            out += ['}', '',
                    'impl %s {' % name,
                    '    /// Resolve every entry point through `get`.',
                    '    ///',
                    '    /// # Safety',
                    '    ///',
                    '    /// `get` must answer each name with null or with the address of the',
                    '    /// Vulkan command of exactly that name. That is the loader\'s contract',
                    '    /// for `%s`, and nothing else may be passed here: the returned' % loader,
                    '    /// pointer is transmuted to the signature generated from vk.xml.',
                    '    pub unsafe fn load(get: &mut dyn FnMut(&CStr) -> Option<ProcAddr>) -> %s {' % name,
                    '        %s {' % name]
            for ty in tys:
                out.append('            fp_%s: get(c"%s").map(|p| unsafe { transmute(p) }),'
                           % (ty.name, ty.name))
            out += ['        }', '    }', '']
            for ty in tys:
                out += [
                    '    /// `%s`, or a panic naming it if the driver has none.' % ty.name,
                    '    #[inline]',
                    '    pub fn %s(&self) -> %s {' % (ty.name, self.proc_signature(ty)),
                    '        self.fp_%s.expect(' % ty.name,
                    '            "%s: this build advertises it and the driver does not export it")'
                    % ty.name,
                    '    }',
                    '',
                    '    /// Stand `%s` up on a table with no driver behind it.' % ty.name,
                    '    ///',
                    '    /// Test scaffolding. A handler cannot be watched at the boundary that',
                    '    /// matters -- what it hands the driver -- without a driver to hand it',
                    '    /// to, and the real one needs a device, an instance and a loader. This',
                    '    /// takes a plain function of the right shape instead, which vk.xml is',
                    '    /// what pins.',
                    '    #[cfg(test)]',
                    '    pub fn plant_%s(&mut self, f: %s) {' % (ty.name, self.proc_signature(ty)),
                    '        self.fp_%s = Some(f);' % ty.name,
                    '    }',
                    '',
                    '    /// Whether the driver exports `%s` at all.' % ty.name,
                    '    #[inline]',
                    '    pub fn has_%s(&self) -> bool {' % ty.name,
                    '        self.fp_%s.is_some()' % ty.name,
                    '    }',
                    '',
                    '    /// `%s` if the driver exports it, and no opinion about it if not.'
                    % ty.name,
                    '    ///',
                    '    /// The accessor above panics because absence there is a host bug: this',
                    '    /// build asked for a command it advertises. That reasoning does not',
                    '    /// reach a path the guest can steer, where absence means the guest',
                    '    /// named something this driver does not have -- an answer to give back,',
                    '    /// not a reason to take the process down with it.',
                    '    #[inline]',
                    '    pub fn try_%s(&self) -> Option<%s> {' % (ty.name, self.proc_signature(ty)),
                    '        self.fp_%s' % ty.name,
                    '    }',
                    '',
                ]
            out += ['}', '']

            # The census a test can hold the loader to without naming three hundred commands.
            out += ['impl %s {' % name,
                    '    /// How many of this table\'s entry points the driver actually answered.',
                    '    pub fn loaded(&self) -> (usize, usize) {',
                    '        let got = [']
            for ty in tys:
                out.append('            self.fp_%s.is_some(),' % ty.name)
            out += ['        ];',
                    '        (got.iter().filter(|b| **b).count(), got.len())',
                    '    }',
                    '}', '']

        return '\n'.join(out)

    def render_serialize(self, gaps):
        """The whole serializer: handles, structs, chains."""
        out = []
        for ty in self.gen.supported_types[VkType.HANDLE]:
            out += self._handle_fns(ty)
        for ty in self.gen.supported_types[VkType.UNION]:
            out += self._union_fns(ty, gaps)
        # Both variants for every struct, rather than the C's `need_partial` bookkeeping: the
        # partial one differs only in what an output member costs on the wire, and emitting it
        # unconditionally trades generated lines -- which cost nothing -- for an attribute pass.
        for v in ('', '_partial'):
            for ty in self.gen.supported_types[VkType.STRUCT]:
                if ty.s_type:
                    out += self._chain_fns(ty, gaps, v)
                else:
                    out += self._plain_struct_fns(ty, gaps, v)
        commands = [c for c in self.gen.supported_types[VkType.COMMAND]
                    if self.gen.is_serializable(c)]
        for ty in commands:
            out += self._command_fns(ty, gaps)
            out += self._command_accessors(ty, gaps)
        out += self._dispatch_fns(commands, gaps)
        out += self._pool_impls()
        return '\n'.join(out)

    def _pool_impls(self):
        """Which pool holds which kind, for the driver's bookkeeping.

        Emitted beside `impl cs::Handle` because it is the same kind of fact about the same types,
        and derived by `pool_children` so a pool the renderer starts serving arrives here on its
        own. The pairing is what stops a command buffer being filed under a descriptor pool: both
        sides of that transposition are handles, and the bookkeeping cannot otherwise see it.
        """
        out = []
        for pool, child in self.pool_children():
            out += [
                'impl cs::PoolOf for %s {' % pool,
                '    type Child = %s;' % child,
                '}',
                '']
        return out


    def _array_rows(self, ty):
        """Every array `ty` carries, as `(field, element type, count expression, mutable)`.

        The one place that decides what an array is, so the accessor that hands one out and the
        visibility that shuts the pointer away cannot come to different answers.
        """
        rows = []
        shadowed = {f for f, _, _ in self.shadows(ty)}
        for var in ty.variables:
            try:
                shape = self._shape(ty, var)
            except self.Unsupported:
                # The member has no emitted decode either, so there is no array to hand out.
                continue
            if shape[0] == 'dynamic':
                # Same rule as `scalar_rows`, and for the same reason: a `*mut` member is not
                # automatically one a handler writes. Where a shadow was emitted beside it the
                # wire array holds the *guest's* ids and the host handles go in the shadow, so
                # the wire member is read-only. Where no shadow exists there are no ids to keep
                # apart, and a `*mut` out-array is exactly what it looks like -- an array the
                # driver fills in place.
                f = self.field_name(var.name)
                mutable = (not var.ty.is_const_pointer()
                           and ('handle_%s' % f) not in shadowed)
                rows.append((f, self.base_name(var.ty), shape[1], mutable))
            elif shape[0] == 'blob':
                # A blob is an array whose element type vk.xml declines to name: `void`, with a
                # length counted in bytes. So the slice is of `u8` -- `base_name` would say
                # `c_void`, which nothing can be a slice of, and that is the only reason these
                # were not already behind this door.
                #
                # Mutability follows the member, with no shadow to consult: a blob carries no
                # guest ids, so there is nothing to keep out of the reply, and an out-blob is
                # written where it lies.
                rows.append((self.field_name(var.name), 'u8', shape[1],
                             not var.ty.is_const_pointer()))
        for f, rs, shape in self.shadows(ty):
            if shape[0] == 'dynamic':
                mutable = rs.startswith('*mut ')
                rows.append((f, rs.split(' ', 1)[1], shape[1], mutable))
        return rows

    def _command_accessors(self, ty, gaps):
        """The arrays a command carries, as slices, on the struct that carries them.

        An array is a count and a pointer, and a `vn_command_*` has to keep both: a slice is two
        words where C has one, and venus-protocol's own encoder reads these structs through a
        pointer (see `render_layout_oracle`). So the reconciliation happens here instead -- once
        per array, in generated code, which is the only place that can do it soundly.

        The lifetime is what makes that true. The slice is `&'a [T]` where `'a` is the struct's,
        which the decode tied to the arena the elements came from; a free function taking a bare
        count and a bare pointer can only invent a lifetime, and its caller is then free to invent
        a longer one than the arena has. The shadow arrays go the other way -- `&mut` borrowed
        from the struct -- so no two callers can hold one at once.

        What `None` means is deliberately not decided here. See `cs::wire_array`.
        """
        rows = self._array_rows(ty)
        scalars = self.scalar_rows(ty)
        strings = self.string_rows(ty)
        if not rows and not scalars and not strings:
            return []
        out = ["impl<'a> vn_command_%s<'a> {" % ty.name]
        out_handles = self.out_handle_fields(ty)
        for f, elem, count, mutable in rows:
            # As in `_scalar_accessor`: a create's out-array carries the guest's ids.
            read = 'cs::Guest<%s>' % elem if f in out_handles else elem
            sig = ('pub fn %s_mut(&mut self) -> Option<&mut [%s]>' % (f, elem) if mutable
                   else "pub fn %s(&self) -> Option<&'a [%s]>" % (f, read))
            call = 'wire_array_mut' if mutable else 'wire_array'
            # The count expression is the decode's, which counts in `u64` because that is what the
            # wire holds. A slice is indexed in `usize`, and one cast says so once.
            n = self._count_expr(count)
            out += ['    /// Whether the guest sent `%s` at all.' % f,
                    '    ///',
                    '    /// Not the same question as whether it is empty: Vulkan gives a null',
                    '    /// array its own meaning in the enumerations, where it is the guest',
                    '    /// asking how many there are rather than asking for them.',
                    '    pub fn has_%s(&self) -> bool {' % f,
                    '        !self.%s.is_null()' % f,
                    '    }',
                    '',
                    '    /// `%s`, reconciled with the count the guest sent beside it.' % f,
                    '    ' + sig + ' {',
                    "        // The count is the decode's own expression, so the slice can only be",
                    '        // as long as the array the decoder allocated. `val` is what that',
                    '        // expression names.',
                    '        let val = self;',
                    '        // SAFETY: the decoder allocated this member from the batch arena,',
                    "        // sized to that count, and the arena outlives the struct's `'a`.",
                    '        unsafe { cs::%s((%s) as usize, val.%s as *%s _) }'
                    % (call, n, f, 'mut' if mutable else 'const'),
                    '    }',
                    '']
            out += self._planter(ty, f, elem, count, mutable)
        for f, elem, mutable, field_mut in scalars:
            out += self._scalar_accessor(ty, f, elem, mutable, field_mut)
        for f in strings:
            out += self._string_accessor(f)
        return out[:-1] + ['}', '']

    @staticmethod
    def _string_accessor(f):
        """The door onto a member that points at a NUL-terminated string.

        Read-only, and there is no writable counterpart to be had: every string on the wire is
        something the guest named, and a reply that had to hand one back would need storage the
        command struct does not carry.
        """
        return [
            '    /// Whether the guest sent `%s` at all.' % f,
            '    ///',
            '    /// A meaning of its own, not an empty string: an absent name is the guest',
            '    /// declining to narrow the request, which is not the same as naming nothing --',
            "    /// an absent `pLayerName` asks for the implementation's own extensions.",
            '    pub fn has_%s(&self) -> bool {' % f,
            '        !self.%s.is_null()' % f,
            '    }',
            '',
            '    /// `%s`, as the string it points at.' % f,
            '    ///',
            '    /// No length beside it, because there is none to keep in step: the terminator is',
            '    /// the length. See `cs::wire_c_string`.',
            "    pub fn %s(&self) -> Option<&'a core::ffi::CStr> {" % f,
            '        // SAFETY: the decoder copied this string into the batch arena and forced its',
            "        // last byte to NUL, and the arena outlives the struct's `'a`.",
            '        unsafe { cs::wire_c_string(self.%s) }' % f,
            '    }',
            '',
            '    /// Plant `%s` as the decoder would have, terminator and all.' % f,
            '    #[cfg(test)]',
            "    pub fn plant_%s(&mut self, v: &'a core::ffi::CStr) {" % f,
            '        self.%s = v.as_ptr();' % f,
            '    }',
            '',
        ]

    def _scalar_accessor(self, ty, f, elem, mutable, field_mut):
        """The door onto a member that points at one value.

        Readers hand back the arena's lifetime and writers borrow the struct, which is the array
        wall's rule and not an accident of this one: a handler routinely reads the id the guest
        named and writes the answer beside it in the same breath, and those two must be able to be
        held at once. Two writers must not, and are not.
        """
        # A create's out-member is the guest's id, not a handle the driver may be called with,
        # and `Guest` is transparent so the reference is the same reference.
        read = 'cs::Guest<%s>' % elem if f in self.out_handle_fields(ty) else elem
        if mutable:
            sig = 'pub fn %s_mut(&mut self) -> Option<&mut %s>' % (f, elem)
            call, star = 'wire_out', 'mut'
        else:
            sig = "pub fn %s(&self) -> Option<&'a %s>" % (f, read)
            call, star = 'wire_ref', 'const'
        # The door a test plants through follows the *field*, not the accessor. A wire out-handle
        # is a `*mut` the handler may only read, so the two disagree there, and it is the field
        # the assignment has to typecheck against.
        plant_star = 'mut' if field_mut else 'const'
        plant_life = "'a mut" if field_mut else "'a"
        return [
            '    /// Whether the guest sent `%s` at all.' % f,
            '    ///',
            '    /// Not a question a handler may skip: Vulkan gives a null out-parameter its own',
            '    /// meaning, and the accessor answers `None` rather than deciding what it meant.',
            '    pub fn has_%s(&self) -> bool {' % f,
            '        !self.%s.is_null()' % f,
            '    }',
            '',
            '    /// `%s`, as the one value it points at.' % f,
            '    ' + sig + ' {',
            '        // SAFETY: the decoder allocated this member from the batch arena as a',
            "        // single element, and the arena outlives the struct's `'a`.",
            '        unsafe { cs::%s(self.%s as *%s _) }' % (call, f, star),
            '    }',
            '',
            '    /// Plant `%s` as the decoder would have.' % f,
            '    #[cfg(test)]',
            '    pub fn plant_%s(&mut self, v: &%s %s) {' % (f, plant_life, elem),
            '        self.%s = v as *%s _;' % (f, plant_star),
            '    }',
            '',
        ]

    def _member_is_mut(self, ty, f):
        """Whether the emitted member `f` of command `ty` is a `*mut` pointer."""
        for _vis, name, rs in self.command_params(ty):
            if name == f:
                return rs.startswith('*mut ')
        return False

    @staticmethod
    def _count_expr(count):
        """An array's count as the decode spells it, with the wire's `u64` cast taken back off.

        The wire counts in `u64` because that is what it holds; a slice is indexed in `usize`,
        and the accessor casts once rather than carrying two spellings around.
        """
        n = count[:-len(' as u64')] if count.endswith(' as u64') else count
        return n[1:-1] if n.startswith('(') and n.endswith(')') else n

    def _planter(self, ty, f, elem, count, mutable):
        """The way a test builds a command that carries an array.

        The decoder is the only thing that writes these members in a real run, and it writes the
        count and the pointer together. A test that sets them as two fields can set them
        inconsistently, which is the bug the accessor exists to make impossible -- so the test
        door plants a slice and derives both from it.

        Where the count is a member of this struct, it is set from the slice's own length. Where
        it is not -- inside another struct, behind an out-pointer, or arithmetic -- only the
        pointer is set, because the count is somewhere the test has already had to build.
        """
        # Whether the *member* is `*mut` is not whether the *accessor* hands out `&mut`: an
        # enumeration's wire array is `*mut` because C writes host handles into it, while the
        # accessor over it is read-only because what a handler reads there is the guest's ids.
        wr = self._member_is_mut(ty, f)
        life = "'a mut" if wr else "'a"
        # A blob's slice is of `u8` while its member is `c_void`, so the planter has to
        # say so; for every other array the cast is the identity.
        body = ['        self.%s = a.%s as %s _;'
                % (f, 'as_mut_ptr()' if wr else 'as_ptr()', '*mut' if wr else '*const')]
        m = re.fullmatch(r'val\.(\w+)', self._count_expr(count))
        if m:
            ct = next((self.field_type(v) for v in ty.variables if self.field_name(v.name) == m.group(1)), None)
            if ct in ('u32', 'u64', 'usize', 'i32'):
                body.append('        self.%s = a.len() as %s;' % (m.group(1), ct))
        return ['    /// Plant `%s` as the decoder would have, count and pointer together.' % f,
                '    #[cfg(test)]',
                '    pub fn plant_%s(&mut self, a: &%s [%s]) {' % (f, life, elem)] + body + [
                '    }',
                '']

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
            'impl cs::Handle for %s {' % n,
            '    const OBJECT_TYPE: i32 = %s.0;' % objtype,
            '    fn host(self) -> cs::HostHandle {',
            '        cs::HostHandle(self.0)',
            '    }',
            '    fn guest_id(self) -> ObjectId {',
            '        ObjectId(self.0)',
            '    }',
            '    fn from_host(host: cs::HostHandle) -> Self {',
            '        %s(host.0)' % n,
            '    }',
            '    fn null() -> Self {',
            '        %s(0)' % n,
            '    }',
            '}',
            '',
            '/// Resolve the guest id to the host object, poisoning this command if it names one',
            '/// the host never created.',
            '///',
            '/// The id is *replaced* by the host handle, so the argument struct is what the driver',
            '/// is called with and nothing converts it. That is also why the id is returned rather',
            '/// than dropped: after this the wire position is gone, and a destroy still has to say',
            '/// which guest object died. See the shadow members on `vn_command_*`.',
            'pub fn vn_decode_%s_lookup(dec: &mut Decoder<\'_>, val: &mut %s) -> ObjectId {' % (n, n),
            '    let id = ObjectId(dec.decode_scalar::<u64>());',
            '    *val = <%s as cs::Handle>::from_host(dec.lookup_object(id, <%s as cs::Handle>::OBJECT_TYPE));' % (n, n),
            '    id',
            '}',
            '',
        ]

    # A union with no selector still carries a tag on the wire, and venus pins the one it writes
    # so both sides agree without a discriminant Vulkan never gave them. Decode still accepts every
    # case, as the C does.
    UNION_DEFAULT_TAGS = {
        'VkClearColorValue': 2,
        'VkClearValue': 0,
        'VkDeviceOrHostAddressKHR': 0,
        'VkDeviceOrHostAddressConstKHR': 0,
        'VkPipelineExecutableStatisticValueKHR': 2,
    }

    def _union_fns(self, ty, gaps):
        """A union is a tag followed by whichever member the tag names.

        Reading a Rust union member is unsafe by definition -- nothing but the tag says which one
        is live -- so every case body is wrapped. Where the tag comes from is the whole design:
        a selected union takes it from the owning struct, and the rest write a pinned default. Only
        decode can read it off the wire, which is why decode alone takes no tag argument.
        """
        n = ty.name
        valid = ty.is_valid_union()
        pinned = self.UNION_DEFAULT_TAGS.get(n)
        tag = ', tag: %s' % ty.sty.name if valid else ''

        def case(kind, var):
            validity = self.gen._get_variable_validity(ty, var, True)
            body = (self.decode_member(ty, var, validity, True) if kind == 'decode'
                    else self._out_member(kind, ty, var, validity))
            return (['// SAFETY: the tag names this member.', 'unsafe {']
                    + ['    ' + l for l in body] + ['}'])

        def body(kind):
            def go():
                if not valid and pinned is None:
                    raise self.Unsupported('%s: union without a tag' % n)
                out = ['let mut size = 0usize;'] if kind == 'sizeof' else []

                # The pinned-tag encode has no branch at all: one member, always.
                if not valid and kind != 'decode':
                    var = dict(ty.get_union_cases())[pinned]
                    out.append('enc.encode_scalar::<u32>(%du32);' % pinned if kind == 'encode'
                               else 'size += cs::sizeof_scalar::<u32>();')
                    out += case(kind, var)
                    return out + (['size'] if kind == 'sizeof' else [])

                if valid:
                    if kind == 'decode':
                        out.append('let tag = dec.decode_scalar::<%s>();' % ty.sty.name)
                    elif kind == 'encode':
                        out.append('enc.encode_scalar::<%s>(tag);' % ty.sty.name)
                    else:
                        out.append('size += cs::sizeof_scalar::<%s>();' % ty.sty.name)
                    tests = [('tag == %s::%s' % (ty.sty.name, label), var)
                             for label, var in ty.get_union_cases()]
                else:
                    out.append('let tag = dec.decode_scalar::<u32>();')
                    tests = [('tag == %du32' % i, var) for i, var in ty.get_union_cases()]

                for i, (test, var) in enumerate(tests):
                    out.append('%sif %s {' % ('} else ' if i else '', test))
                    out += ['    ' + l for l in case(kind, var)]
                out += ['} else {',
                        '    ' + ('dec.set_fatal();' if kind == 'decode'
                                  else 'debug_assert!(false, "no case for this union tag");'),
                        '}']
                return out + (['size'] if kind == 'sizeof' else [])
            return go

        out = []
        out += self._fn('vn_sizeof_%s(proto: &dyn cs::Protocol, val: &%s%s) -> usize'
                        % (n, n, tag), body('sizeof'), gaps, n)
        out += self._fn('vn_encode_%s(enc: &mut Encoder<\'_>, val: &%s%s)' % (n, n, tag),
                        body('encode'), gaps, n)
        out += self._fn('vn_decode_%s_temp(dec: &mut Decoder<\'_>, val: &mut %s)' % (n, n),
                        body('decode'), gaps, n)
        return out

    def _plain_struct_fns(self, ty, gaps, v=''):
        n = ty.name
        out = []
        out += self._fn(
            'vn_sizeof_%s%s(proto: &dyn cs::Protocol, val: &%s) -> usize' % (n, v, n),
            lambda: self._struct_body('sizeof', ty, v, gaps), gaps, n)
        out += self._fn('vn_encode_%s%s(enc: &mut Encoder<\'_>, val: &%s)' % (n, v, n),
                        lambda: self._struct_body('encode', ty, v, gaps), gaps, n)
        out += self._fn('vn_decode_%s%s_temp(dec: &mut Decoder<\'_>, val: &mut %s)' % (n, v, n),
                        lambda: self._struct_body('decode', ty, v + '_temp', gaps), gaps, n)
        return out

    def _command_fns(self, ty, gaps):
        """One command's request decode, request encode and reply encode.

        The request encode is the driver's side of the wire, which the C renderer never generates.
        It is what makes the differential test a round trip: the recorded bytes came out of mesa's
        encoder, so re-encoding a decoded command and comparing is a diff against the C.
        """
        n = ty.name
        cmd = 'VkCommandTypeEXT::%s' % ty.attrs['c_type']
        args = "vn_command_%s<'_>" % n

        target = self.destroy_target(ty)
        target_var = target[0] if target else None
        owner = self.create_owner(ty)
        owner_var = owner[0] if owner else None

        def members(kind, reply):
            out = []
            for var in ([ty.ret] if reply and ty.ret else []) + list(ty.variables):
                if reply:
                    if 'var_out' not in var.attrs:
                        out.append('/* skip val.%s */' % self.field_name(var.name))
                        continue
                    validity = Gen_VALID
                else:
                    validity = self.gen._get_variable_validity(ty, var, 'var_in' in var.attrs)
                capture = None
                if kind == 'decode' and (var is target_var or var is owner_var):
                    capture = 'id_%s' % self.field_name(var.name)
                out += self._member(kind, ty, var, validity, kind == 'decode', capture)
            return out

        def out_handle_storage():
            """Arena room for the host handles a create will produce.

            Allocated here rather than by a handler because the length is a decoded value, and
            because every handler then looks the same: pass `handle_<name>` where Vulkan wants the
            out pointer, and the driver writes host handles somewhere the guest's ids are not.
            """
            out = []
            for var, shape in self.out_handles(ty):
                m = self.member_expr(var)
                f = 'handle_%s' % self.field_name(var.name)
                base = self.base_name(var.ty)
                n = '1' if shape[0] == 'pointer' else '(%s) as usize' % shape[1]
                out += ['if !%s.is_null() {' % m,
                        '    let Some(a) = dec.alloc_temp_array::<%s>(%s) else { return };' % (base, n),
                        '    val.%s = a.as_mut_ptr();' % f,
                        '}']
            return out

        def request(kind):
            def go():
                if 'need_blob_encode' in ty.attrs:
                    # The reply carries the blob, so its storage is an offset into the encoder the
                    # renderer has not built yet. Nothing here needs it; the round trip does not
                    # reach it, and vkr will want the offset plumbing rather than this shape.
                    raise self.Unsupported('%s: blob storage rides the reply' % n)
                out = ['let mut size = 0usize;'] if kind == 'sizeof' else []
                if kind == 'decode':
                    out += ['/* the header is the caller\'s: it chose this arm with it */']
                elif kind == 'encode':
                    out += ['enc.encode_scalar::<VkCommandTypeEXT>(%s);' % cmd,
                            'enc.encode_scalar::<VkFlags>(cmd_flags);']
                else:
                    out += ['size += cs::sizeof_scalar::<VkCommandTypeEXT>();',
                            'size += cs::sizeof_scalar::<VkFlags>();']
                out += members(kind, False)
                if kind == 'decode':
                    out += out_handle_storage()
                return out + (['size'] if kind == 'sizeof' else [])
            return go

        def reply(kind):
            def go():
                out = ['let mut size = 0usize;'] if kind == 'sizeof' else []
                out += ['enc.encode_scalar::<VkCommandTypeEXT>(%s);' % cmd] if kind == 'encode' \
                    else ['size += cs::sizeof_scalar::<VkCommandTypeEXT>();']
                out += members(kind, True)
                return out + (['size'] if kind == 'sizeof' else [])
            return go

        out = []
        out += self._fn(
            'vn_decode_%s_args_temp<\'a>(dec: &mut Decoder<\'a>, val: &mut vn_command_%s<\'a>)'
            % (n, n), request('decode'), gaps, n)
        out += self._fn('vn_sizeof_%s_args(proto: &dyn cs::Protocol, val: &%s) -> usize'
                        % (n, args), request('sizeof'), gaps, n)
        out += self._fn('vn_encode_%s_args(enc: &mut Encoder<\'_>, cmd_flags: VkFlags, val: &%s)'
                        % (n, args), request('encode'), gaps, n)
        out += self._fn('vn_sizeof_%s_reply(proto: &dyn cs::Protocol, val: &%s) -> usize'
                        % (n, args), reply('sizeof'), gaps, n)
        out += self._fn('vn_encode_%s_reply(enc: &mut Encoder<\'_>, val: &%s)' % (n, args),
                        reply('encode'), gaps, n)
        return out

    def _dispatch_fns(self, commands, gaps):
        """The command table, as a match the compiler turns into a jump table.

        The round trip lives here rather than in the harness because only the generator knows the
        arm list, and an arm that is merely missing has to be a named failure rather than a silent
        pass.
        """
        out = ['/// The name of the command, for a message a human reads. `None` is a type',
               '/// no version of this protocol defines.',
               'pub fn vn_command_name(cmd: VkCommandTypeEXT) -> Option<&\'static str> {',
               '    match cmd {']
        for ty in commands:
            out.append('        VkCommandTypeEXT::%s => Some("%s"),' % (ty.attrs['c_type'], ty.name))
        out += ['        _ => None,', '    }', '}', '']

        out += ['/// Decode one command\'s arguments and encode them straight back, returning the',
                '/// size the encoder should have written. `None` is a command this protocol does',
                '/// not define -- a stream that names one is a stream we cannot follow.',
                'pub fn vn_round_trip_args(',
                '    dec: &mut Decoder<\'_>,',
                '    enc: &mut Encoder<\'_>,',
                '    cmd: VkCommandTypeEXT,',
                '    cmd_flags: VkFlags,',
                ') -> Option<usize> {',
                '    match cmd {']
        for ty in commands:
            n = ty.name
            out += [
                '        VkCommandTypeEXT::%s => {' % ty.attrs['c_type'],
                '            let mut args = vn_command_%s::default();' % n,
                '            vn_decode_%s_args_temp(dec, &mut args);' % n,
                '            if dec.fatal() {',
                '                return Some(0);',
                '            }',
                '            let size = vn_sizeof_%s_args(enc.protocol(), &args);' % n,
                '            vn_encode_%s_args(enc, cmd_flags, &args);' % n,
                '            Some(size)',
                '        }',
            ]
        out += ['        _ => None,', '    }', '}', '']

        out += self._handler_trait(commands, gaps)
        return out

    def _lifecycle(self, ty, gaps):
        """The objects a command creates or destroys, read out of its decoded arguments.

        The emitter already knows which members these are -- an out-handle is a member the model
        calls PARTIAL, a destroy's target is its last input handle -- so saying it here rather than
        in thirty hand-written handlers is the same trade the C makes with `vkr_device_object.py`,
        minus the C.

        Each hook is handed *both* halves of the pairing, because neither member holds both: the
        visible one is the wire's and the shadow is the host's. See `shadows`.
        """
        n = ty.name
        out = []

        # `vkCreateInstance` has no parent handle: an instance is the root of the context's tree.
        own = self.create_owner(ty)
        owner = ('Some(val.id_%s)' % self.field_name(own[0].name)) if own else 'None'

        for var, shape in self.out_handles(ty):
            objtype = 'VkObjectType::%s' % var.ty.base.attrs['c_objtype']
            m = 'val.%s' % self.field_name(var.name)
            s = 'val.handle_%s' % self.field_name(var.name)
            if shape[0] == 'pointer':
                out += [
                    '// A null out-pointer is the guest asking how many there would be, not',
                    '// creating one. A null shadow means no handler ran.',
                    'if !%s.is_null() && !%s.is_null() {' % (m, s),
                    '    // SAFETY: both are non-null and the decoder allocated them in the arena,',
                    '    // one element each.',
                    '    unsafe { h.object_created(%s, ObjectId((*%s).0), cs::HostHandle((*%s).0), %s) };'
                    % (objtype, m, s, owner),
                    '}']
            elif shape[0] == 'dynamic':
                out += [
                    'if !%s.is_null() && !%s.is_null() {' % (m, s),
                    '    for i in 0..(%s) as usize {' % shape[1],
                    '        // SAFETY: the decoder allocated both arrays with that many elements,',
                    '        // from the same count.',
                    '        unsafe { h.object_created(%s, ObjectId((*%s.add(i)).0), cs::HostHandle((*%s.add(i)).0), %s) };'
                    % (objtype, m, s, owner),
                    '    }',
                    '}']
            else:
                gaps.append('%s.%s: %s out handle' % (n, var.name, shape[0]))
                out.append('/* gap: %s.%s: %s out handle */' % (n, var.name, shape[0]))

        target = self.destroy_target(ty)
        if target:
            var, shape = target
            objtype = 'VkObjectType::%s' % var.ty.base.attrs['c_objtype']
            f = 'val.id_%s' % self.field_name(var.name)
            if shape[0] == 'plain':
                out.append('h.object_destroyed(%s, %s);' % (objtype, f))
            elif shape[0] == 'dynamic':
                m = 'val.%s' % self.field_name(var.name)
                out += [
                    'if !%s.is_null() {' % f,
                    '    for i in 0..(%s) as usize {' % shape[1],
                    '        // SAFETY: the decoder filled this array alongside the handles, from',
                    '        // the same count.',
                    '        h.object_destroyed(%s, unsafe { *%s.add(i) });' % (objtype, f),
                    '    }',
                    '}']
            else:
                gaps.append('%s.%s: %s destroy target' % (n, var.name, shape[0]))
                out.append('/* gap: %s.%s: %s destroy target */' % (n, var.name, shape[0]))

        return out

    def _handler_trait(self, commands, gaps):
        """The renderer's side of the wire: one method per command, and the match that reaches it.

        Every method defaults to `unsupported`, so a renderer implements the commands it serves and
        inherits a loud, uniform answer for the ~600 it does not. That default is what lets vkr grow
        one command at a time without the generator being touched again.

        The handler takes its arguments by `&mut` because a command's outputs are members of the
        same struct its inputs came from -- the reply encoder reads back what the handler wrote,
        exactly as the C does.
        """
        out = ['/// Every command venus defines, as a method a renderer overrides.',
               'pub trait Commands {',
               '    /// A command this renderer does not serve. The generated default calls it, so',
               '    /// an unimplemented command is answered the same way everywhere -- and the',
               '    /// renderer decides whether that is a poisoned context or a logged no-op.',
               '    fn unsupported(&mut self, cmd: VkCommandTypeEXT);',
               '',
               '    /// A command created this object: the id the guest chose, and the handle',
               '    /// the driver returned into the shadow member beside it.',
               '    ///',
               '    /// Called after the handler, which is what makes `host` meaningful -- a',
               '    /// handler that ran and failed leaves it zero, and one that never ran leaves',
               '    /// the shadow null and this uncalled. Registering the pairing is the',
               '    /// renderer\'s to do: the generator does not get to decide what a zero handle',
               '    /// means.',
               '    /// `owner` is the guest id of the object the create hung off -- the device',
               '    /// for most things, the physical device for a device, `None` for an instance,',
               '    /// which owns itself. Destroying it destroys everything under it without a',
               '    /// command naming any of them, so this is the only moment the parentage is',
               '    /// there to record.',
               '    fn object_created(',
               '        &mut self,',
               '        ty: VkObjectType,',
               '        id: ObjectId,',
               '        host: cs::HostHandle,',
               '        owner: Option<ObjectId>,',
               '    ) {',
               '        let _ = (ty, id, host, owner);',
               '    }',
               '',
               '    /// A command destroyed this object, named by the guest id the decoder kept',
               '    /// when the lookup overwrote it with the host handle.',
               '    ///',
               '    /// Called after the handler, which still needed that host handle to destroy.',
               '    fn object_destroyed(&mut self, ty: VkObjectType, id: ObjectId) {',
               '        let _ = (ty, id);',
               '    }',
               '']
        for ty in commands:
            n = ty.name
            out += ["    fn %s(&mut self, args: &mut vn_command_%s<'_>) {" % (n, n),
                    '        let _ = args;',
                    '        self.unsupported(VkCommandTypeEXT::%s);' % ty.attrs['c_type'],
                    '    }']
        out += ['}', '']

        out += ['/// Decode one command, run it, and encode its reply when the guest asked for one.',
                '///',
                '/// `enc` is `Some` exactly when the command header carried the reply flag. Replay',
                '/// strips that flag, which is why a replayed stream needs no reply buffer at all.',
                '///',
                '/// `None` is a command type this protocol does not define -- a stream naming one',
                '/// is a stream we cannot follow, and the caller poisons the ring.',
                'pub fn vn_dispatch_command(',
                '    dec: &mut Decoder<\'_>,',
                '    enc: Option<&mut Encoder<\'_>>,',
                '    cmd: VkCommandTypeEXT,',
                '    h: &mut dyn Commands,',
                ') -> Option<()> {',
                '    match cmd {']
        for ty in commands:
            n = ty.name
            life = self._lifecycle(ty, gaps)
            out += [
                '        VkCommandTypeEXT::%s => {' % ty.attrs['c_type'],
                '            let mut args = vn_command_%s::default();' % n,
                '            vn_decode_%s_args_temp(dec, &mut args);' % n,
                '            if dec.fatal() {',
                '                return Some(());',
                '            }',
                '            h.%s(&mut args);' % n,
            ] + (['            let val = &args;']
                 + ['            ' + l for l in life] if life else []) + [
                '            if let Some(enc) = enc {',
                '                vn_encode_%s_reply(enc, &args);' % n,
                '            }',
                '            Some(())',
                '        }',
            ]
        out += ['        _ => None,', '    }', '}', '']
        return out

    def _chain_fns(self, ty, gaps, v=''):
        n = ty.name
        next_types, skipped = self.gen.get_chain(ty)
        out = []

        out += self._fn(
            'vn_sizeof_%s_self%s(proto: &dyn cs::Protocol, val: &%s) -> usize' % (n, v, n),
            lambda: self._struct_body('sizeof', ty, '_self' + v, gaps), gaps, n)
        out += self._fn('vn_encode_%s_self%s(enc: &mut Encoder<\'_>, val: &%s)' % (n, v, n),
                        lambda: self._struct_body('encode', ty, '_self' + v, gaps), gaps, n)
        out += self._fn('vn_decode_%s_self%s_temp(dec: &mut Decoder<\'_>, val: &mut %s)'
                        % (n, v, n),
                        lambda: self._struct_body('decode', ty, '_self' + v + '_temp', gaps),
                        gaps, n)

        out += self._chain_pnext_sizeof(ty, next_types, v)
        out += self._chain_pnext_encode(ty, next_types, v)
        out += self._chain_pnext_decode(ty, next_types, v)

        out += [
            'pub fn vn_sizeof_%s%s(proto: &dyn cs::Protocol, val: &%s) -> usize {' % (n, v, n),
            '    let mut size = cs::sizeof_scalar::<VkStructureType>();',
            '    // SAFETY: pNext points at the chain this struct was decoded with.',
            '    size += unsafe { vn_sizeof_%s_pnext%s(proto, val.pNext as *const c_void) };'
            % (n, v),
            '    size += vn_sizeof_%s_self%s(proto, val);' % (n, v),
            '    size',
            '}',
            '',
            'pub fn vn_encode_%s%s(enc: &mut Encoder<\'_>, val: &%s) {' % (n, v, n),
            '    enc.encode_scalar::<VkStructureType>(VkStructureType::%s);' % ty.s_type,
            '    // SAFETY: as above.',
            '    unsafe { vn_encode_%s_pnext%s(enc, val.pNext as *const c_void) };' % (n, v),
            '    vn_encode_%s_self%s(enc, val);' % (n, v),
            '}',
            '',
            'pub fn vn_decode_%s%s_temp<\'a>(dec: &mut Decoder<\'a>, val: &mut %s) {' % (n, v, n),
            '    let stype = dec.decode_scalar::<VkStructureType>();',
            '    if stype != VkStructureType::%s {' % ty.s_type,
            '        dec.set_fatal();',
            '    }',
            '    val.sType = stype;',
            '    val.pNext = vn_decode_%s_pnext%s_temp(dec) as _;' % (n, v),
            '    vn_decode_%s_self%s_temp(dec, val);' % (n, v),
            '}',
            '',
        ]
        return out

    def _chain_pnext_sizeof(self, ty, next_types, v=''):
        proto = 'proto'
        n = ty.name
        out = ['/// # Safety',
               '/// `val` is a `pNext` chain of structs this decoder allocated.',
               'pub unsafe fn vn_sizeof_%s_pnext%s(proto: &dyn cs::Protocol, val: *const c_void) -> usize {' % (n, v)]
        if not next_types:
            out += ['    let _ = (proto, val);',
                    '    return cs::sizeof_scalar::<u64>(); /* no known struct */',
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
                '                size += cs::sizeof_scalar::<u64>(); /* simple pointer */',
                '                size += cs::sizeof_scalar::<VkStructureType>();',
                '                let here = unsafe { &*(pnext as *const %s) };' % nt.name,
                '                size += unsafe {',
                '                    vn_sizeof_%s_pnext%s(proto, here.pNext as *const c_void)' % (n, v),
                '                };',
                '                size += vn_sizeof_%s_self%s(proto, here);' % (nt.name, v),
                '                return size;',
                '            }',
            ]
        out += ['            _ => {}',
                '        }',
                '        pnext = node.pNext;',
                '    }',
                '    size + cs::sizeof_scalar::<u64>()',
                '}', '']
        return out

    def _chain_pnext_encode(self, ty, next_types, v=''):
        proto = 'enc.protocol()'
        n = ty.name
        out = ['/// # Safety',
               '/// `val` is a `pNext` chain of structs this decoder allocated.',
               'pub unsafe fn vn_encode_%s_pnext%s(enc: &mut Encoder<\'_>, val: *const c_void) {' % (n, v)]
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
                '                unsafe { vn_encode_%s_pnext%s(enc, here.pNext as *const c_void) };' % (n, v),
                '                vn_encode_%s_self%s(enc, here);' % (nt.name, v),
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

    def _chain_pnext_decode(self, ty, next_types, v=''):
        n = ty.name
        out = ['pub fn vn_decode_%s_pnext%s_temp<\'a>(dec: &mut Decoder<\'a>) -> *mut c_void {' % (n, v)]
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
                '            p.pNext = vn_decode_%s_pnext%s_temp(dec) as _;' % (n, v),
                '            vn_decode_%s_self%s_temp(dec, p);' % (nt.name, v),
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
