// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-limina-exception
// Copyright © 2026 the limina authors
//
// Print the layout of every struct that crosses the C ABI, one field per line, as a file that can
// be diffed. A layout mismatch is the failure mode nothing else in the harness catches: the Rust
// side declares its own #[repr(C)] structs by hand, a wrong offset compiles clean on both sides,
// and the corruption shows up somewhere else entirely at runtime.
//
// Built and run by abi-fixture.sh. Sizes and offsets come from the compiler, so this is the
// authority for the platform it is built on -- not a transcription anyone has to keep in step.
#include "virgl_hw.h"
#include "virglrenderer.h"

#include <stddef.h>
#include <stdio.h>

#define S(type) printf("struct %s size=%zu align=%zu\n", #type, sizeof(struct type), _Alignof(struct type))
#define F(type, field) \
   printf("  %s.%s offset=%zu size=%zu\n", #type, #field, \
          offsetof(struct type, field), sizeof(((struct type *)0)->field))
#define D(name) printf("define %s = %d\n", #name, (int)(name))

int main(void)
{
   D(VIRGL_RENDERER_CALLBACKS_VERSION);

   S(virgl_renderer_gl_ctx_param);
   F(virgl_renderer_gl_ctx_param, major_ver);
   F(virgl_renderer_gl_ctx_param, minor_ver);
   F(virgl_renderer_gl_ctx_param, shared);
   F(virgl_renderer_gl_ctx_param, compat_ctx);

   S(virgl_renderer_callbacks);
   F(virgl_renderer_callbacks, version);
   F(virgl_renderer_callbacks, write_fence);
   F(virgl_renderer_callbacks, create_gl_context);
   F(virgl_renderer_callbacks, destroy_gl_context);
   F(virgl_renderer_callbacks, make_current);
   F(virgl_renderer_callbacks, get_drm_fd);
   F(virgl_renderer_callbacks, write_context_fence);
   F(virgl_renderer_callbacks, get_server_fd);
   F(virgl_renderer_callbacks, get_egl_display);

   S(virgl_renderer_resource_create_args);
   F(virgl_renderer_resource_create_args, handle);
   F(virgl_renderer_resource_create_args, target);
   F(virgl_renderer_resource_create_args, format);
   F(virgl_renderer_resource_create_args, bind);
   F(virgl_renderer_resource_create_args, width);
   F(virgl_renderer_resource_create_args, height);
   F(virgl_renderer_resource_create_args, depth);
   F(virgl_renderer_resource_create_args, array_size);
   F(virgl_renderer_resource_create_args, last_level);
   F(virgl_renderer_resource_create_args, nr_samples);
   F(virgl_renderer_resource_create_args, flags);

   S(virgl_renderer_resource_info);
   F(virgl_renderer_resource_info, handle);
   F(virgl_renderer_resource_info, virgl_format);
   F(virgl_renderer_resource_info, width);
   F(virgl_renderer_resource_info, height);
   F(virgl_renderer_resource_info, depth);
   F(virgl_renderer_resource_info, flags);
   F(virgl_renderer_resource_info, tex_id);
   F(virgl_renderer_resource_info, stride);
   F(virgl_renderer_resource_info, drm_fourcc);
   F(virgl_renderer_resource_info, fd);

   S(virgl_renderer_resource_create_blob_args);
   F(virgl_renderer_resource_create_blob_args, res_handle);
   F(virgl_renderer_resource_create_blob_args, ctx_id);
   F(virgl_renderer_resource_create_blob_args, blob_mem);
   F(virgl_renderer_resource_create_blob_args, blob_flags);
   F(virgl_renderer_resource_create_blob_args, blob_id);
   F(virgl_renderer_resource_create_blob_args, size);
   F(virgl_renderer_resource_create_blob_args, iovecs);
   F(virgl_renderer_resource_create_blob_args, num_iovs);

   S(virgl_renderer_resource_import_blob_args);
   F(virgl_renderer_resource_import_blob_args, res_handle);
   F(virgl_renderer_resource_import_blob_args, blob_mem);
   F(virgl_renderer_resource_import_blob_args, fd_type);
   F(virgl_renderer_resource_import_blob_args, fd);
   F(virgl_renderer_resource_import_blob_args, size);

   S(virgl_renderer_hdr);
   F(virgl_renderer_hdr, stype);
   F(virgl_renderer_hdr, stype_version);
   F(virgl_renderer_hdr, size);

   S(virgl_renderer_export_query);
   F(virgl_renderer_export_query, hdr);
   F(virgl_renderer_export_query, in_resource_id);
   F(virgl_renderer_export_query, out_num_fds);
   F(virgl_renderer_export_query, in_export_fds);
   F(virgl_renderer_export_query, out_fourcc);
   F(virgl_renderer_export_query, out_fds);
   F(virgl_renderer_export_query, out_strides);
   F(virgl_renderer_export_query, out_offsets);
   F(virgl_renderer_export_query, out_modifier);

   S(virgl_renderer_supported_structures);
   F(virgl_renderer_supported_structures, hdr);
   F(virgl_renderer_supported_structures, in_stype_version);
   F(virgl_renderer_supported_structures, out_supported_structures_mask);

   S(virgl_renderer_resource_info_ext);
   F(virgl_renderer_resource_info_ext, version);
   F(virgl_renderer_resource_info_ext, base);
   F(virgl_renderer_resource_info_ext, has_dmabuf_export);
   F(virgl_renderer_resource_info_ext, planes);
   F(virgl_renderer_resource_info_ext, modifiers);
   F(virgl_renderer_resource_info_ext, d3d_tex2d);

   /* Not in virglrenderer.h, but it crosses the ABI all the same: every transfer call takes one. */
   S(virgl_box);
   F(virgl_box, x);
   F(virgl_box, y);
   F(virgl_box, z);
   F(virgl_box, w);
   F(virgl_box, h);
   F(virgl_box, d);

   return 0;
}
