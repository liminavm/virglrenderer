// SPDX-License-Identifier: MIT
// Copyright © 2026 the limina authors

//! What the host's GL can do, decided once at init.
//!
//! The C's `feature_list`, GLES column: a feature is present when the context's version reaches
//! the version that made it core, or when the driver advertises one of the extensions that
//! provide it. Some features are extension-only (`Unavail` as the core version); some are core
//! at a version this host never has, and are here so the table stays the C's, one row per row.

use std::collections::BTreeSet;

/// The GLES version a feature became core at, or never.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Core {
    Unavail,
    Gles(u32),
}

macro_rules! features {
    ($($name:ident = ($core:expr, [$($ext:literal),*]),)+) => {
        /// One capability of the host GL.
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        #[allow(non_camel_case_types)]
        pub enum Feature { $($name,)+ }

        impl Feature {
            const ALL: &'static [Feature] = &[$(Feature::$name,)+];

            fn core(self) -> Core {
                match self { $(Feature::$name => $core,)+ }
            }

            fn extensions(self) -> &'static [&'static str] {
                match self { $(Feature::$name => &[$($ext),*],)+ }
            }

            pub fn name(self) -> &'static str {
                match self { $(Feature::$name => stringify!($name),)+ }
            }
        }
    };
}

use Core::{Gles, Unavail};

features! {
    amd_pinned_memory = (Unavail, ["GL_AMD_pinned_memory"]),
    arb_or_gles_ext_texture_buffer = (Unavail, ["GL_EXT_texture_buffer"]),
    arb_robustness = (Unavail, ["GL_ARB_robustness"]),
    arb_buffer_storage = (Unavail, ["GL_EXT_buffer_storage"]),
    arrays_of_arrays = (Gles(31), ["GL_ARB_arrays_of_arrays"]),
    ati_meminfo = (Unavail, ["GL_ATI_meminfo"]),
    atomic_counters = (Gles(31), ["GL_ARB_shader_atomic_counters"]),
    base_instance = (Unavail, ["GL_ARB_base_instance", "GL_EXT_base_instance"]),
    barrier = (Gles(31), ["GL_ARB_shader_image_load_store"]),
    bind_vertex_buffers = (Unavail, []),
    bit_encoding = (Unavail, ["GL_ARB_shader_bit_encoding"]),
    blend_equation_advanced = (Gles(32), ["GL_KHR_blend_equation_advanced"]),
    clear_texture = (Unavail, ["GL_ARB_clear_texture", "GL_EXT_clear_texture"]),
    clip_control = (Unavail, ["GL_ARB_clip_control", "GL_EXT_clip_control"]),
    compute_shader = (Gles(31), ["GL_ARB_compute_shader"]),
    copy_image = (Gles(32), ["GL_ARB_copy_image", "GL_EXT_copy_image", "GL_OES_copy_image"]),
    conditional_render_inverted = (Unavail, ["GL_ARB_conditional_render_inverted"]),
    conservative_depth = (Unavail, ["GL_ARB_conservative_depth", "GL_EXT_conservative_depth"]),
    cube_map_array = (Gles(32), ["GL_ARB_texture_cube_map_array", "GL_EXT_texture_cube_map_array", "GL_OES_texture_cube_map_array"]),
    cull_distance = (Unavail, ["GL_ARB_cull_distance", "GL_EXT_clip_cull_distance"]),
    draw_instance = (Gles(30), ["GL_ARB_draw_instanced"]),
    draw_parameters = (Unavail, ["ARB_shader_draw_parameters"]),
    dual_src_blend = (Unavail, ["GL_ARB_blend_func_extended", "GL_EXT_blend_func_extended"]),
    depth_clamp = (Unavail, ["GL_ARB_depth_clamp", "GL_EXT_depth_clamp", "GL_NV_depth_clamp"]),
    enhanced_layouts = (Unavail, ["GL_ARB_enhanced_layouts"]),
    egl_image = (Unavail, ["GL_OES_EGL_image"]),
    egl_image_storage = (Unavail, ["GL_EXT_EGL_image_storage"]),
    fb_no_attach = (Gles(31), ["GL_ARB_framebuffer_no_attachments"]),
    framebuffer_fetch = (Unavail, ["GL_EXT_shader_framebuffer_fetch"]),
    framebuffer_fetch_non_coherent = (Unavail, ["GL_EXT_shader_framebuffer_fetch_non_coherent"]),
    geometry_shader = (Gles(32), ["GL_EXT_geometry_shader", "GL_OES_geometry_shader"]),
    gl_conditional_render = (Unavail, []),
    gl_prim_restart = (Gles(30), []),
    gles_khr_robustness = (Unavail, ["GL_KHR_robustness"]),
    gles31_compatibility = (Gles(31), ["ARB_ES3_1_compatibility"]),
    gles31_vertex_attrib_binding = (Gles(31), ["GL_ARB_vertex_attrib_binding"]),
    gpu_shader5 = (Gles(32), ["GL_ARB_gpu_shader5", "GL_EXT_gpu_shader5", "GL_OES_gpu_shader5"]),
    group_vote = (Unavail, ["GL_ARB_shader_group_vote"]),
    images = (Gles(31), ["GL_ARB_shader_image_load_store"]),
    indep_blend = (Gles(32), ["GL_EXT_draw_buffers2", "GL_OES_draw_buffers_indexed"]),
    indep_blend_func = (Gles(32), ["GL_ARB_draw_buffers_blend", "GL_OES_draw_buffers_indexed"]),
    indirect_draw = (Gles(31), ["GL_ARB_draw_indirect"]),
    indirect_params = (Unavail, ["GL_ARB_indirect_parameters"]),
    khr_debug = (Gles(32), ["GL_KHR_debug"]),
    memory_object = (Unavail, ["GL_EXT_memory_object"]),
    memory_object_fd = (Unavail, ["GL_EXT_memory_object_fd"]),
    mesa_invert = (Unavail, ["GL_MESA_pack_invert"]),
    ms_scaled_blit = (Unavail, ["GL_EXT_framebuffer_multisample_blit_scaled"]),
    multisample = (Gles(30), ["GL_ARB_texture_multisample"]),
    multi_draw_indirect = (Unavail, ["GL_ARB_multi_draw_indirect", "GL_EXT_multi_draw_indirect"]),
    nv_conditional_render = (Unavail, ["GL_NV_conditional_render"]),
    nv_prim_restart = (Unavail, ["GL_NV_primitive_restart"]),
    shader_noperspective_interpolation = (Unavail, ["GL_NV_shader_noperspective_interpolation", "GL_EXT_gpu_shader4"]),
    nvx_gpu_memory_info = (Unavail, ["GL_NVX_gpu_memory_info"]),
    pipeline_statistics_query = (Unavail, ["GL_ARB_pipeline_statistics_query"]),
    polygon_offset_clamp = (Unavail, ["GL_ARB_polygon_offset_clamp", "GL_EXT_polygon_offset_clamp"]),
    occlusion_query = (Unavail, ["GL_ARB_occlusion_query"]),
    occlusion_query_boolean = (Gles(30), ["GL_EXT_occlusion_query_boolean", "GL_ARB_occlusion_query2"]),
    qbo = (Unavail, ["GL_ARB_query_buffer_object"]),
    robust_buffer_access = (Unavail, ["GL_ARB_robust_buffer_access_behavior", "GL_KHR_robust_buffer_access_behavior"]),
    sample_mask = (Gles(31), ["GL_ARB_texture_multisample"]),
    sample_shading = (Gles(32), ["GL_ARB_sample_shading", "GL_OES_sample_shading"]),
    samplers = (Gles(30), ["GL_ARB_sampler_objects"]),
    sampler_border_colors = (Gles(32), ["GL_ARB_sampler_objects", "GL_EXT_texture_border_clamp", "GL_OES_texture_border_clamp"]),
    separate_shader_objects = (Gles(31), ["GL_ARB_seperate_shader_objects"]),
    shader_clock = (Unavail, ["GL_ARB_shader_clock"]),
    ssbo = (Gles(31), ["GL_ARB_shader_storage_buffer_object"]),
    ssbo_barrier = (Gles(31), ["GL_ARB_shader_storage_buffer_object"]),
    srgb_write_control = (Unavail, ["GL_EXT_sRGB_write_control"]),
    stencil_texturing = (Gles(31), ["GL_ARB_stencil_texturing"]),
    storage_multisample = (Gles(31), ["GL_ARB_texture_storage_multisample"]),
    tessellation = (Gles(32), ["GL_ARB_tessellation_shader", "GL_OES_tessellation_shader", "GL_EXT_tessellation_shader"]),
    texture_array = (Gles(30), ["GL_EXT_texture_array"]),
    texture_barrier = (Unavail, ["GL_ARB_texture_barrier"]),
    texture_buffer_range = (Gles(32), ["GL_ARB_texture_buffer_range"]),
    texture_gather = (Gles(31), ["GL_ARB_texture_gather"]),
    texture_mirror_clamp_to_edge = (Unavail, ["GL_ATI_texture_mirror_once", "GL_EXT_texture_mirror_clamp", "GL_ARB_texture_mirror_clamp_to_edge", "GL_EXT_texture_mirror_clamp_to_edge"]),
    texture_mirror_clamp = (Unavail, ["GL_ATI_texture_mirror_once", "GL_EXT_texture_mirror_clamp"]),
    texture_mirror_clamp_to_border = (Unavail, ["GL_EXT_texture_mirror_clamp"]),
    texture_multisample = (Gles(31), ["GL_ARB_texture_multisample"]),
    texture_query_lod = (Unavail, ["GL_ARB_texture_query_lod", "GL_EXT_texture_query_lod"]),
    texture_shadow_lod = (Unavail, ["GL_EXT_texture_shadow_lod"]),
    texture_srgb_decode = (Unavail, ["GL_EXT_texture_sRGB_decode"]),
    texture_storage = (Gles(30), ["GL_ARB_texture_storage"]),
    texture_view = (Unavail, ["GL_ARB_texture_view", "GL_OES_texture_view", "GL_EXT_texture_view"]),
    timer_query = (Unavail, ["GL_ARB_timer_query", "GL_EXT_disjoint_timer_query"]),
    transform_feedback = (Gles(30), ["GL_EXT_transform_feedback"]),
    transform_feedback2 = (Gles(30), ["GL_ARB_transform_feedback2"]),
    transform_feedback3 = (Unavail, ["GL_ARB_transform_feedback3"]),
    transform_feedback_overflow_query = (Unavail, ["GL_ARB_transform_feedback_overflow_query"]),
    txqs = (Unavail, ["GL_ARB_shader_texture_image_samples"]),
    ubo = (Gles(30), ["GL_ARB_uniform_buffer_object"]),
    viewport_array = (Unavail, ["GL_ARB_viewport_array", "GL_OES_viewport_array"]),
    implicit_msaa = (Unavail, ["GL_EXT_multisampled_render_to_texture"]),
    anisotropic_filter = (Unavail, ["GL_EXT_texture_filter_anisotropic", "GL_ARB_texture_filter_anisotropic"]),
    seamless_cubemap_per_texture = (Unavail, ["GL_AMD_seamless_cubemap_per_texture"]),
    vs_layer_viewport = (Unavail, ["GL_AMD_vertex_shader_layer"]),
    vs_viewport_index = (Unavail, ["GL_AMD_vertex_shader_viewport_index"]),
    // Not in the C's list, which tests these by name where it needs them.
    s3tc = (Unavail, ["GL_EXT_texture_compression_s3tc"]),
    rgtc = (Unavail, ["GL_EXT_texture_compression_rgtc"]),
    bptc = (Unavail, ["GL_EXT_texture_compression_bptc"]),
    astc = (Unavail, ["GL_KHR_texture_compression_astc_ldr"]),
    etc2 = (Gles(30), []),
    color_buffer_float = (Gles(32), ["GL_EXT_color_buffer_float"]),
    nv_read_depth = (Unavail, ["GL_NV_read_depth"]),
    nv_read_depth_stencil = (Unavail, ["GL_NV_read_depth_stencil"]),
    nv_read_stencil = (Unavail, ["GL_NV_read_stencil"]),
    // Two the C calls without a feature: epoxy would abort on the missing symbol.
    texture_3d_attach = (Unavail, ["GL_OES_texture_3D"]),
    storage_multisample_2d_array = (Gles(32), ["GL_OES_texture_storage_multisample_2d_array"]),
}

/// The features this host has.
pub struct Features {
    have: BTreeSet<Feature>,
    /// The context's GLES version as `major * 10 + minor`, the C's spelling.
    pub gles_version: u32,
    extensions: BTreeSet<String>,
}

impl Features {
    /// Decide every feature from a context's version and the extensions it advertises.
    pub fn probe(gles_version: u32, extensions: impl IntoIterator<Item = String>) -> Features {
        let extensions: BTreeSet<String> = extensions.into_iter().collect();
        let have = Feature::ALL
            .iter()
            .copied()
            .filter(|f| {
                let core = match f.core() {
                    Gles(v) => gles_version >= v,
                    Unavail => false,
                };
                core || f.extensions().iter().any(|e| extensions.contains(*e))
            })
            .collect();
        Features { have, gles_version, extensions }
    }

    pub fn has(&self, f: Feature) -> bool {
        self.have.contains(&f)
    }

    /// Withdraw a feature the driver advertises but the winsys cannot honour -- sRGB write
    /// control without `EGL_KHR_gl_colorspace`, as `vrend_renderer_init` withdraws it.
    pub fn clear(&mut self, f: Feature) {
        self.have.remove(&f);
    }

    /// Withdraw every feature whose entry points the driver did not actually hand over: what a
    /// driver advertises and what it exports are two answers to one question, and this is where
    /// they are made one, so a `Gl` wrapper behind a feature can take its proc for granted.
    pub fn reconcile(&mut self, gl: &super::gl::Gl) {
        for (feature, proc_name) in gl.missing_procs() {
            if self.have.remove(&feature) {
                eprintln!(
                    "[virglrs] vrend: {} advertised without {proc_name}: withdrawn",
                    feature.name()
                );
            }
        }
    }

    pub fn has_extension(&self, name: &str) -> bool {
        self.extensions.contains(name)
    }

    pub fn present(&self) -> impl Iterator<Item = Feature> + '_ {
        self.have.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_feature_is_core_by_version_or_provided_by_an_extension() {
        let f = Features::probe(31, ["GL_KHR_robustness".to_string()]);
        assert!(f.has(Feature::texture_storage));
        assert!(f.has(Feature::compute_shader));
        assert!(!f.has(Feature::geometry_shader));
        assert!(f.has(Feature::gles_khr_robustness));
        assert!(!f.has(Feature::qbo));
        let g = Features::probe(32, ["GL_EXT_buffer_storage".to_string()]);
        assert!(g.has(Feature::geometry_shader));
        assert!(g.has(Feature::arb_buffer_storage));
        assert!(!g.has(Feature::gles_khr_robustness));
    }
}
