#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright © 2026 the limina authors
"""Generate the GLES and EGL bindings from the Khronos registries.

    gen.py --outdir DIR [--registry DIR]

Emits `types.rs`, `gles.rs` and `egl.rs` into DIR: the C types the two APIs are declared in, every
constant the chosen features and extensions require, and one proc table per API whose signatures
are the registry's own. A binding transcribed by hand can disagree with the driver about a
parameter, and the disagreement is a stack smash rather than a compile error -- which is the
reason `vulkan.rs` gives for generating its tables from vk.xml, and it holds here unchanged.

Which entry points exist is a decision this file makes, once, in the lists below. GLES is the
target (`docs/rust-rewrite.md`, P3): every core command through 3.2 is in the table, so that
"3.1 host, and these of 3.2's extensions" is a runtime census rather than a build-time guess, plus
the extensions vrend reaches for by name. EGL is 1.0 through 1.5 plus what the surfaceless,
image-importing, fence-exporting winsys needs.
"""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent

GLES_FEATURES = [
    'GL_ES_VERSION_2_0', 'GL_ES_VERSION_3_0', 'GL_ES_VERSION_3_1', 'GL_ES_VERSION_3_2',
]

# What vrend asks a GLES host for beyond the core, by the names it uses. An extension here costs
# nothing when the driver lacks it -- its entry points load as `None` and its census says so.
GLES_EXTENSIONS = [
    'GL_OES_EGL_image',
    'GL_OES_EGL_image_external',
    'GL_EXT_EGL_image_storage',
    'GL_OES_texture_3D',
    'GL_OES_texture_buffer',
    'GL_EXT_texture_buffer',
    'GL_OES_texture_view',
    'GL_EXT_texture_view',
    'GL_OES_texture_storage_multisample_2d_array',
    'GL_OES_viewport_array',
    'GL_OES_sample_shading',
    'GL_OES_draw_buffers_indexed',
    'GL_EXT_draw_buffers_indexed',
    'GL_OES_draw_elements_base_vertex',
    'GL_EXT_draw_elements_base_vertex',
    'GL_EXT_base_instance',
    'GL_EXT_multi_draw_indirect',
    'GL_EXT_draw_transform_feedback',
    'GL_OES_geometry_shader',
    'GL_EXT_geometry_shader',
    'GL_OES_tessellation_shader',
    'GL_EXT_tessellation_shader',
    'GL_OES_gpu_shader5',
    'GL_EXT_gpu_shader5',
    'GL_OES_shader_io_blocks',
    'GL_OES_shader_multisample_interpolation',
    'GL_OES_texture_cube_map_array',
    'GL_EXT_texture_cube_map_array',
    'GL_OES_primitive_bounding_box',
    'GL_OES_copy_image',
    'GL_EXT_copy_image',
    'GL_EXT_clear_texture',
    'GL_EXT_buffer_storage',
    'GL_EXT_memory_object',
    'GL_EXT_memory_object_fd',
    'GL_EXT_polygon_offset_clamp',
    'GL_EXT_texture_border_clamp',
    'GL_OES_texture_border_clamp',
    'GL_EXT_texture_mirror_clamp_to_edge',
    'GL_EXT_texture_format_BGRA8888',
    'GL_EXT_read_format_bgra',
    'GL_EXT_texture_norm16',
    'GL_EXT_texture_sRGB_R8',
    'GL_EXT_texture_sRGB_RG8',
    'GL_EXT_texture_sRGB_decode',
    'GL_EXT_sRGB_write_control',
    'GL_EXT_color_buffer_float',
    'GL_EXT_color_buffer_half_float',
    'GL_OES_texture_float',
    'GL_OES_texture_half_float',
    'GL_OES_texture_float_linear',
    'GL_OES_texture_half_float_linear',
    'GL_OES_depth_texture',
    'GL_OES_depth24',
    'GL_OES_depth32',
    'GL_OES_packed_depth_stencil',
    'GL_OES_rgb8_rgba8',
    'GL_EXT_texture_type_2_10_10_10_REV',
    'GL_OES_vertex_half_float',
    'GL_OES_vertex_type_10_10_10_2',
    'GL_EXT_texture_compression_s3tc',
    'GL_EXT_texture_compression_s3tc_srgb',
    'GL_EXT_texture_compression_rgtc',
    'GL_EXT_texture_compression_bptc',
    'GL_ANGLE_texture_compression_dxt3',
    'GL_ANGLE_texture_compression_dxt5',
    'GL_KHR_texture_compression_astc_ldr',
    'GL_EXT_texture_compression_astc_decode_mode',
    'GL_EXT_texture_filter_anisotropic',
    'GL_EXT_texture_query_lod',
    'GL_EXT_texture_shadow_lod',
    'GL_EXT_shader_framebuffer_fetch',
    'GL_EXT_shader_io_blocks',
    'GL_EXT_shader_integer_mix',
    'GL_EXT_shader_implicit_conversions',
    'GL_EXT_shader_non_constant_global_initializers',
    'GL_EXT_separate_shader_objects',
    'GL_EXT_blend_func_extended',
    'GL_EXT_blend_minmax',
    'GL_KHR_blend_equation_advanced',
    'GL_KHR_debug',
    'GL_KHR_robustness',
    'GL_EXT_robustness',
    'GL_KHR_robust_buffer_access_behavior',
    'GL_EXT_disjoint_timer_query',
    'GL_EXT_occlusion_query_boolean',
    'GL_NV_conditional_render',
    'GL_EXT_multisampled_render_to_texture',
    'GL_EXT_framebuffer_blit_layers',
    'GL_MESA_framebuffer_flip_y',
    'GL_MESA_shader_integer_functions',
    'GL_EXT_clip_cull_distance',
    'GL_EXT_clip_control',
    'GL_EXT_depth_clamp',
    'GL_EXT_primitive_bounding_box',
    'GL_EXT_shader_pixel_local_storage',
    'GL_EXT_shadow_samplers',
    'GL_OES_standard_derivatives',
    'GL_OES_element_index_uint',
    'GL_OES_vertex_array_object',
    'GL_OES_mapbuffer',
    'GL_EXT_map_buffer_range',
    'GL_OES_get_program_binary',
    'GL_OES_required_internalformat',
    'GL_OES_surfaceless_context',
    'GL_EXT_unpack_subimage',
    'GL_NV_pack_subimage',
    'GL_NV_image_formats',
    'GL_NV_read_depth',
    'GL_NV_read_stencil',
    'GL_NV_read_depth_stencil',
    'GL_EXT_window_rectangles',
]

EGL_FEATURES = ['EGL_VERSION_1_0', 'EGL_VERSION_1_1', 'EGL_VERSION_1_2', 'EGL_VERSION_1_3',
                'EGL_VERSION_1_4', 'EGL_VERSION_1_5']

EGL_EXTENSIONS = [
    'EGL_EXT_platform_base',
    'EGL_EXT_platform_device',
    'EGL_EXT_device_base',
    'EGL_EXT_device_enumeration',
    'EGL_EXT_device_query',
    'EGL_MESA_platform_surfaceless',
    'EGL_KHR_surfaceless_context',
    'EGL_KHR_create_context',
    'EGL_KHR_no_config_context',
    'EGL_KHR_image',
    'EGL_KHR_image_base',
    'EGL_KHR_image_pixmap',
    'EGL_KHR_gl_texture_2D_image',
    'EGL_KHR_gl_renderbuffer_image',
    'EGL_KHR_gl_colorspace',
    'EGL_KHR_fence_sync',
    'EGL_KHR_wait_sync',
    'EGL_KHR_reusable_sync',
    'EGL_ANDROID_native_fence_sync',
    'EGL_MESA_image_dma_buf_export',
    'EGL_EXT_image_dma_buf_import',
    'EGL_EXT_image_dma_buf_import_modifiers',
    'EGL_KHR_debug',
    'EGL_KHR_get_all_proc_addresses',
    'EGL_KHR_client_get_all_proc_addresses',
]

# The C types the registries are written in, as Rust. Both registries share the Khronos base
# types; the GL ones are `<ptype>`s and the EGL ones are declared in eglplatform.h, which is why
# neither can be read out of its XML. Each becomes a type alias of the same name, so a generated
# signature reads as the registry's.
TYPES = {
    'GLenum': 'u32', 'GLboolean': 'u8', 'GLbitfield': 'u32', 'GLbyte': 'i8', 'GLshort': 'i16',
    'GLint': 'i32', 'GLsizei': 'i32', 'GLubyte': 'u8', 'GLushort': 'u16', 'GLuint': 'u32',
    'GLfloat': 'f32', 'GLclampf': 'f32', 'GLdouble': 'f64', 'GLclampd': 'f64', 'GLchar': 'c_char',
    'GLvoid': 'c_void', 'GLintptr': 'isize', 'GLsizeiptr': 'isize', 'GLint64': 'i64',
    'GLuint64': 'u64', 'GLint64EXT': 'i64', 'GLuint64EXT': 'u64', 'GLfixed': 'i32',
    'GLhalf': 'u16', 'GLhalfNV': 'u16', 'GLclampx': 'i32',
    'GLsync': '*mut c_void', 'GLeglImageOES': '*mut c_void', 'GLeglClientBufferEXT': '*mut c_void',
    'GLDEBUGPROCKHR': 'GLDEBUGPROC',
    'EGLint': 'i32', 'EGLBoolean': 'u32', 'EGLenum': 'u32', 'EGLAttrib': 'isize',
    'EGLAttribKHR': 'isize', 'EGLTime': 'u64', 'EGLTimeKHR': 'u64', 'EGLuint64KHR': 'u64',
    'EGLnsecsANDROID': 'i64', 'EGLNativeFileDescriptorKHR': 'i32',
    'EGLDisplay': '*mut c_void', 'EGLContext': '*mut c_void', 'EGLConfig': '*mut c_void',
    'EGLSurface': '*mut c_void', 'EGLImage': '*mut c_void', 'EGLImageKHR': '*mut c_void',
    'EGLSync': '*mut c_void', 'EGLSyncKHR': '*mut c_void', 'EGLClientBuffer': '*mut c_void',
    'EGLDeviceEXT': '*mut c_void', 'EGLLabelKHR': '*mut c_void', 'EGLObjectKHR': '*mut c_void',
    'EGLNativeDisplayType': '*mut c_void', 'EGLNativeWindowType': '*mut c_void',
    'EGLNativePixmapType': '*mut c_void',
    '__eglMustCastToProperFunctionPointerType': 'Option<ProcAddr>',
}

# Type names that are not aliases of a primitive but declared in full in `types.rs`.
DECLARED = {'GLDEBUGPROC', 'EGLDEBUGPROCKHR'}

TYPES_RS = '''
// The C types GLES and EGL are declared in.

pub use core::ffi::{c_char, c_int, c_void};

/// An entry point as `eglGetProcAddress` returns it, before it is given a signature. Never called
/// through this type: the tables transmute it to the registry's signature for the name it was
/// asked for, which is the only thing that makes the transmute sound.
pub type ProcAddr = unsafe extern "C" fn();

pub type GLDEBUGPROC = Option<
    unsafe extern "C" fn(
        source: GLenum,
        ty: GLenum,
        id: GLuint,
        severity: GLenum,
        length: GLsizei,
        message: *const GLchar,
        user: *const c_void,
    ),
>;

pub type EGLDEBUGPROCKHR = Option<
    unsafe extern "C" fn(
        error: EGLenum,
        command: *const c_char,
        message_type: EGLint,
        thread_label: EGLLabelKHR,
        object_label: EGLLabelKHR,
        message: *const c_char,
    ),
>;
'''


def render_types():
    out = [TYPES_RS]
    for name, rust in TYPES.items():
        if name != rust:
            out.append('pub type %s = %s;' % (name, rust))
    return '\n'.join(out) + '\n'


def rust_type(ctype):
    """A registry parameter's C type, as Rust.

    The registry writes them as C: `const GLvoid *`, `GLuint *`, `const GLchar *const*`. Pointers
    nest from the outside in, and `const` binds to what precedes it, so the text is read right to
    left: each `*` is one pointer, mutable unless a `const` sits before it.
    """
    text = ctype.replace('struct ', '').strip()
    if text == 'void':
        return None
    tokens = re.findall(r'const|\*|[A-Za-z_][A-Za-z0-9_]*', text)
    base = next(t for t in tokens if t not in ('const', '*'))
    if base == 'void':
        rust = 'c_void'
    elif base in ('char', 'int'):
        rust = {'char': 'c_char', 'int': 'c_int'}[base]
    elif base in DECLARED:
        rust = base
    else:
        rust = TYPES[base] if TYPES[base] in ('c_void', 'c_char', 'c_int') else base
    if rust == 'c_void' and '*' not in tokens:
        raise ValueError('bare void as a type: %s' % ctype)
    # Walk pointers outermost first: the rightmost `*` is the outermost.
    stars = [i for i, t in enumerate(tokens) if t == '*']
    out = rust
    for i in reversed(stars):
        inner_const = tokens[i - 1] == 'const' if i > 0 else False
        # The innermost pointee's constness is the `const` before the base type.
        if i == stars[0]:
            inner_const = 'const' in tokens[:i]
        out = ('*const %s' if inner_const else '*mut %s') % out
    return out


def parse_proto(elem):
    """(name, C return type) from a <proto>."""
    name = elem.find('name').text
    text = ''.join(elem.itertext()).replace(name, '', 1)
    return name, text.strip()


def parse_param(elem):
    name = elem.find('name').text
    text = ''.join(elem.itertext())
    return name, text[: text.rfind(name)].strip()


class Registry:
    def __init__(self, path):
        self.root = ET.parse(path).getroot()
        self.commands = {}
        for c in self.root.find('commands'):
            name, ret = parse_proto(c.find('proto'))
            params = [parse_param(p) for p in c.findall('param')]
            self.commands[name] = (ret, params)
        self.enums = {}
        for group in self.root.findall('enums'):
            for e in group.findall('enum'):
                if 'api' in e.attrib and e.attrib['api'] != 'gles2':
                    continue
                if e.attrib['name'] in self.enums and 'api' not in e.attrib:
                    continue
                self.enums[e.attrib['name']] = (e.attrib['value'], e.attrib.get('type', ''))

    def requirements(self, api, features, extensions):
        """The (commands, enums) the named features and extensions require, in registry order."""
        commands, enums = [], []
        seen_c, seen_e = set(), set()

        def take(block):
            for req in block.findall('require'):
                if 'api' in req.attrib and req.attrib['api'] != api:
                    continue
                for c in req.findall('command'):
                    if c.attrib['name'] not in seen_c:
                        seen_c.add(c.attrib['name'])
                        commands.append(c.attrib['name'])
                for e in req.findall('enum'):
                    if e.attrib['name'] not in seen_e:
                        seen_e.add(e.attrib['name'])
                        enums.append(e.attrib['name'])

        found = set()
        for f in self.root.findall('feature'):
            if f.attrib['api'] == api and f.attrib['name'] in features:
                found.add(f.attrib['name'])
                take(f)
        for x in self.root.find('extensions').findall('extension'):
            if x.attrib['name'] in extensions:
                assert api in x.attrib['supported'].split('|'), \
                    '%s is not an %s extension' % (x.attrib['name'], api)
                found.add(x.attrib['name'])
                take(x)
        missing = set(features) | set(extensions)
        missing -= found
        assert not missing, 'not in the registry: %s' % sorted(missing)
        return commands, enums


def signature(reg, name):
    ret, params = reg.commands[name]
    args = ', '.join(rust_type(t) for _, t in params)
    r = rust_type(ret)
    return 'unsafe extern "C" fn(%s)%s' % (args, ' -> %s' % r if r else '')


def render_consts(reg, names, prefix):
    out = []
    for n in names:
        value, ty = reg.enums[n]
        cast = re.match(r'EGL_CAST\((\w+),(.+)\)', value)
        if cast:
            cty, cv = cast.group(1), cast.group(2).strip()
            if TYPES[cty].startswith('*'):
                assert cv == '0', value
                out.append('pub const %s: %s = core::ptr::null_mut();' % (n, cty))
            else:
                out.append('pub const %s: %s = %s;' % (n, cty, cv))
            continue
        v = int(value, 0)
        if ty == 'ull' or v > 0xffffffff:
            out.append('pub const %s: u64 = %s;' % (n, value))
        elif v < 0:
            out.append('pub const %s: i32 = %s;' % (n, value))
        else:
            out.append('pub const %s: %senum = %s;' % (n, prefix, value))
    return out


def render_table(reg, struct, commands, loader):
    out = [
        '/// The entry points, as `%s` hands them back.' % loader,
        '///',
        '/// Every field is optional because the driver answers null for a command it does not',
        '/// export. Reading one through its accessor is what turns that into a panic naming the',
        '/// command, at the call rather than at the crash.',
        '#[derive(Default)]',
        'pub struct %s {' % struct,
    ]
    for c in commands:
        out.append('    fp_%s: Option<%s>,' % (c, signature(reg, c)))
    out += ['}', '',
            'impl %s {' % struct,
            '    /// Resolve every entry point through `get`.',
            '    ///',
            '    /// # Safety',
            '    ///',
            '    /// `get` must answer each name with null or with the address of the entry point',
            '    /// of exactly that name: the returned pointer is transmuted to the signature the',
            '    /// registry gives that name.',
            '    pub unsafe fn load(get: &mut dyn FnMut(&CStr) -> Option<ProcAddr>) -> %s {' % struct,
            '        %s {' % struct]
    for c in commands:
        out.append('            fp_%s: get(c"%s").map(|p| unsafe { transmute(p) }),' % (c, c))
    out += ['        }', '    }', '']
    for c in commands:
        sig = signature(reg, c)
        out += [
            '    /// `%s`, or a panic naming it if the driver has none.' % c,
            '    #[inline]',
            '    pub fn %s(&self) -> %s {' % (c, sig),
            '        self.fp_%s.expect("%s: this build relies on it and the driver does not export it")'
            % (c, c),
            '    }',
            '',
            '    /// Stand `%s` up on a table with no driver behind it. Test scaffolding.' % c,
            '    #[cfg(test)]',
            '    pub fn plant_%s(&mut self, f: %s) {' % (c, sig),
            '        self.fp_%s = Some(f);' % c,
            '    }',
            '',
            '    /// Whether the driver exports `%s` at all.' % c,
            '    #[inline]',
            '    pub fn has_%s(&self) -> bool {' % c,
            '        self.fp_%s.is_some()' % c,
            '    }',
            '',
            '    /// `%s` if the driver exports it, and no opinion about it if not.' % c,
            '    #[inline]',
            '    pub fn try_%s(&self) -> Option<%s> {' % (c, sig),
            '        self.fp_%s' % c,
            '    }',
            '',
        ]
    out += ['    /// The entry points the driver did not answer for, by name.',
            '    pub fn missing(&self) -> Vec<&\'static str> {',
            '        let mut out = Vec::new();']
    for c in commands:
        out.append('        if self.fp_%s.is_none() { out.push("%s"); }' % (c, c))
    out += ['        out', '    }', '}', '']
    return out


def render_census(reg, struct, blocks):
    """Per feature and extension, the names of its commands, so a caller can ask a table
    whether it has all of an extension rather than probing one entry point and hoping."""
    out = ['impl %s {' % struct,
           '    /// The commands a feature or extension requires, by its registry name.',
           '    pub fn requires(name: &str) -> Option<&\'static [&\'static str]> {',
           '        Some(match name {']
    for name, cmds in blocks:
        out.append('            "%s" => &[%s],' % (name, ', '.join('"%s"' % c for c in cmds)))
    out += ['            _ => return None,', '        })', '    }', '',
            '    /// Whether every command `name` requires is exported.',
            '    pub fn has_all_of(&self, name: &str) -> bool {',
            '        let missing = self.missing();',
            '        Self::requires(name).is_some_and(|cmds| cmds.iter().all(|c| !missing.contains(c)))',
            '    }', '}', '']
    return out


def per_block_commands(reg, api, names):
    blocks = []
    for n in names:
        cmds, _ = reg.requirements(api, [n], [n])
        blocks.append((n, cmds))
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--registry', default=str(HERE / 'registry'))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    banner = '// GENERATED by virglrs/gl-gen/gen.py from the Khronos registries -- do not edit.\n'

    (out / 'types.rs').write_text(banner + render_types())

    gl = Registry(Path(args.registry) / 'gl.xml')
    commands, _ = gl.requirements('gles2', GLES_FEATURES, GLES_EXTENSIONS)
    body = [banner, 'use super::types::*;', 'use core::ffi::CStr;', 'use core::mem::transmute;', '']
    # Every constant the registry has, not only the chosen features': the format tables name
    # desktop enums that a GLES driver refuses at the probe, and the probe needs their values.
    body += render_consts(gl, list(gl.enums), 'GL')
    body.append('')
    body += render_table(gl, 'Gles', commands, 'eglGetProcAddress')
    body += render_census(gl, 'Gles', per_block_commands(gl, 'gles2', GLES_FEATURES + GLES_EXTENSIONS))
    (out / 'gles.rs').write_text('\n'.join(body))

    egl = Registry(Path(args.registry) / 'egl.xml')
    commands, enums = egl.requirements('egl', EGL_FEATURES, EGL_EXTENSIONS)
    commands = [c for c in commands if c != 'eglGetProcAddress']
    body = [banner, 'use super::types::*;', 'use core::ffi::CStr;', 'use core::mem::transmute;', '']
    body += render_consts(egl, enums, 'EGL')
    body.append('')
    body += render_table(egl, 'Egl', commands, 'eglGetProcAddress')
    body += render_census(egl, 'Egl', per_block_commands(egl, 'egl', EGL_FEATURES + EGL_EXTENSIONS))
    (out / 'egl.rs').write_text('\n'.join(body))


if __name__ == '__main__':
    main()
