/**************************************************************************
 *
 * Copyright (C) 2014 Red Hat Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#include <stdio.h>
#include <time.h>

#include <epoxy/gl.h>

#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#ifndef _WIN32
#include <sys/mman.h>
#else
#include "mman_win32.h"
#endif

#include "pipe/p_state.h"
#include "util/u_format.h"
#include "util/u_math.h"
#include "vkr_allocator.h"
#include "drm_renderer.h"
#include "proxy/proxy_renderer.h"
#include "vrend/vrend_renderer.h"
#include "vrend/vrend_winsys.h"

#ifndef WIN32
#include "util/libsync.h"
#endif

#include "virglrenderer.h"
#include "virtgpu_drm.h"

#include "virgl_context.h"
#include "virgl_fence.h"
#include "virgl_resource.h"
#include "virgl_util.h"

struct global_state {
   bool client_initialized;
   void *cookie;
   int flags;
   const struct virgl_renderer_callbacks *cbs;

   bool resource_initialized;
   bool context_initialized;
   bool winsys_initialized;
   bool vrend_initialized;
   bool proxy_initialized;
   bool external_winsys_initialized;
   bool drm_initialized;
   bool fence_initialized;
};

static struct global_state state;

/* limina: VKR_FD_TRACE-gated teardown tracing (matches vkr_fd_trace()); attributes
 * which teardown commands actually reach the renderer for the carrier-fd ratchet. */
static bool virgl_fd_trace(void)
{
   static int on = -1;
   if (on < 0) {
      const char *e = getenv("VKR_FD_TRACE");
      on = e && e[0] && strcmp(e, "0") != 0;
   }
   return on;
}

/* new API - just wrap internal API for now */

static int virgl_renderer_resource_create_internal(struct virgl_renderer_resource_create_args *args,
                                                   UNUSED struct iovec *iov, UNUSED uint32_t num_iovs,
                                                   void *image)
{
   struct virgl_resource *res;
   struct pipe_resource *pipe_res;
   struct vrend_renderer_resource_create_args vrend_args =  { 0 };
   uint32_t map_info;

   if (!state.vrend_initialized && !state.drm_initialized)
      return EINVAL;

   /* do not accept handle 0 */
   if (args->handle == 0)
      return EINVAL;

   if (virgl_resource_lookup(args->handle))
      return -EINVAL;

   vrend_args.target = args->target;
   vrend_args.format = args->format;
   vrend_args.bind = args->bind;
   vrend_args.width = args->width;
   vrend_args.height = args->height;
   vrend_args.depth = args->depth;
   vrend_args.array_size = args->array_size;
   vrend_args.nr_samples = args->nr_samples;
   vrend_args.last_level = args->last_level;
   vrend_args.flags = args->flags;

   pipe_res = vrend_renderer_resource_create(&vrend_args, image);
   if (!pipe_res)
      return EINVAL;

   map_info = vrend_renderer_resource_get_map_info(pipe_res);
   res = virgl_resource_create_from_pipe(args->handle, pipe_res, iov, num_iovs);
   if (!res)
      return -ENOMEM;

   res->map_info = map_info;

   return 0;
}

int virgl_renderer_resource_create(struct virgl_renderer_resource_create_args *args,
                                   struct iovec *iov, uint32_t num_iovs)
{
   TRACE_FUNC();
   return virgl_renderer_resource_create_internal(args, iov, num_iovs, NULL);
}

int virgl_renderer_resource_import_eglimage(struct virgl_renderer_resource_create_args *args, void *image)
{
   TRACE_FUNC();
   return virgl_renderer_resource_create_internal(args, NULL, 0, image);
}

void virgl_renderer_resource_set_priv(uint32_t res_handle, void *priv)
{
   struct virgl_resource *res = virgl_resource_lookup(res_handle);
   if (!res)
      return;

   res->private_data = priv;
}

void *virgl_renderer_resource_get_priv(uint32_t res_handle)
{
   struct virgl_resource *res = virgl_resource_lookup(res_handle);
   if (!res)
      return NULL;

   return res->private_data;
}

static bool detach_resource(struct virgl_context *ctx, void *data)
{
   struct virgl_resource *res = data;
   ctx->detach_resource(ctx, res);
   return true;
}

void virgl_renderer_resource_unref(uint32_t res_handle)
{
   struct virgl_resource *res = virgl_resource_lookup(res_handle);
   struct virgl_context_foreach_args args;

   if (virgl_fd_trace())
      virgl_error("[FDTRACE] resource_unref res=%u %s", res_handle,
                  res ? "found" : "MISSING");
   if (!res)
      return;

   /* A resource destroyed while its VMM mapping is live leaked the mapping:
    * virgl_resource_destroy_func() closes the fd and frees the struct but never
    * munmaps res->mapped. Two such leaks per venus context (the instance ring +
    * reply shmem blobs) exhaust the host address space after sustained use
    * (mmap/CTX_CREATE start failing with ENOMEM). Unmap here so unref is safe
    * regardless of whether the VMM balanced its map calls. */
   if (res->mapped)
      virgl_renderer_resource_unmap(res_handle);

   args.callback = detach_resource;
   args.data = res;
   virgl_context_foreach(&args);

   virgl_resource_remove(res->res_id);
}

void virgl_renderer_fill_caps(uint32_t set, uint32_t version,
                              void *caps)
{
   switch (set) {
   case VIRTGPU_DRM_CAPSET_VIRGL:
   case VIRTGPU_DRM_CAPSET_VIRGL2:
      if (state.vrend_initialized)
         vrend_renderer_fill_caps(set, version, (union virgl_caps *)caps);
      break;
   case VIRTGPU_DRM_CAPSET_VENUS:
      if (state.proxy_initialized)
         proxy_get_capset(set, caps);
      break;
   case VIRTGPU_DRM_CAPSET_DRM:
      if (state.drm_initialized)
         drm_renderer_capset(caps);
      break;
   default:
      break;
   }
}

static void per_context_fence_retire(struct virgl_context *ctx,
                                     uint32_t ring_idx,
                                     uint64_t fence_id)
{
   state.cbs->write_context_fence(state.cookie,
                                  ctx->ctx_id,
                                  ring_idx,
                                  fence_id);
}

int virgl_renderer_context_create_with_flags(uint32_t ctx_id,
                                             uint32_t ctx_flags,
                                             uint32_t nlen,
                                             const char *name)
{
   uint32_t capset_id = ctx_flags & VIRGL_RENDERER_CONTEXT_FLAG_CAPSET_ID_MASK;
   struct virgl_context *ctx;
   int ret;

   TRACE_FUNC();

   /* user context id must be greater than 0 */
   if (ctx_id == 0)
      return EINVAL;

   /* unsupported flags */
   if (ctx_flags & ~VIRGL_RENDERER_CONTEXT_FLAG_CAPSET_ID_MASK)
      return EINVAL;

   ctx = virgl_context_lookup(ctx_id);
   if (ctx) {
      return ctx->capset_id == capset_id ? 0 : EINVAL;
   }

   switch (capset_id) {
   case VIRTGPU_DRM_CAPSET_VIRGL:
   case VIRTGPU_DRM_CAPSET_VIRGL2:
      if (!state.vrend_initialized)
         return EINVAL;
      ctx = vrend_renderer_context_create(ctx_id, nlen, name);
      break;
   case VIRTGPU_DRM_CAPSET_VENUS:
      if (!state.proxy_initialized)
         return EINVAL;
      ctx = proxy_context_create(ctx_id, ctx_flags, nlen, name);
      break;
   case VIRTGPU_DRM_CAPSET_DRM:
      if (!state.drm_initialized)
         return EINVAL;
      if (state.cbs->version >= 2 && state.cbs->get_drm_fd)
         ctx = drm_renderer_create(nlen, name, state.cbs->get_drm_fd(state.cookie));
      else
         ctx = drm_renderer_create(nlen, name, -1);
      break;
   default:
      return EINVAL;
      break;
   }
   if (!ctx)
      return ENOMEM;

   ctx->ctx_id = ctx_id;
   ctx->in_fence_fd = -1;
   ctx->capset_id = capset_id;
   ctx->fence_retire = per_context_fence_retire;

   ret = virgl_context_add(ctx);
   if (ret) {
      ctx->destroy(ctx);
      return ret;
   }

   return 0;
}

int virgl_renderer_context_create(uint32_t handle, uint32_t nlen, const char *name)
{
   return virgl_renderer_context_create_with_flags(handle,
                                                   VIRTGPU_DRM_CAPSET_VIRGL2,
                                                   nlen,
                                                   name);
}

void virgl_renderer_context_destroy(uint32_t handle)
{
   TRACE_FUNC();
   if (virgl_fd_trace())
      virgl_error("[FDTRACE] context_destroy ctx=%u", handle);
   virgl_context_remove(handle);
}

int virgl_renderer_submit_cmd(void *buffer,
                              int ctx_id,
                              int ndw)
{
   TRACE_FUNC();
   struct virgl_context *ctx = virgl_context_lookup(ctx_id);
   if (!ctx)
      return EINVAL;

   if (ndw < 0 || (unsigned)ndw > UINT32_MAX / sizeof(uint32_t))
      return EINVAL;

   if (((uintptr_t)buffer & 3) != 0)
      return EFAULT;

   return ctx->submit_cmd(ctx, buffer, (uint32_t)ndw * sizeof(uint32_t));
}

int virgl_renderer_transfer_write_iov(uint32_t handle,
                                      uint32_t ctx_id,
                                      int level,
                                      uint32_t stride,
                                      uint32_t layer_stride,
                                      struct virgl_box *box,
                                      uint64_t offset,
                                      struct iovec *iovec,
                                      unsigned int iovec_cnt)
{
   TRACE_FUNC();

   struct virgl_resource *res = virgl_resource_lookup(handle);
   struct vrend_transfer_info transfer_info;

   if (!res)
      return EINVAL;

   transfer_info.level = level;
   transfer_info.stride = stride;
   transfer_info.layer_stride = layer_stride;
   transfer_info.box = (struct pipe_box *)box;
   transfer_info.offset = offset;
   transfer_info.iovec = iovec;
   transfer_info.iovec_cnt = iovec_cnt;
   transfer_info.synchronized = false;

   if (ctx_id) {
      struct virgl_context *ctx = virgl_context_lookup(ctx_id);
      if (!ctx)
         return EINVAL;

      return ctx->transfer_3d(ctx, res, &transfer_info,
                              VIRGL_TRANSFER_TO_HOST);
   } else {
      if (!res->pipe_resource)
         return EINVAL;

      return vrend_renderer_transfer_pipe(res->pipe_resource, &transfer_info,
                                          VIRGL_TRANSFER_TO_HOST);
   }
}

int virgl_renderer_transfer_read_iov(uint32_t handle, uint32_t ctx_id,
                                     uint32_t level, uint32_t stride,
                                     uint32_t layer_stride,
                                     struct virgl_box *box,
                                     uint64_t offset, struct iovec *iovec,
                                     int iovec_cnt)
{
   TRACE_FUNC();
   struct virgl_resource *res = virgl_resource_lookup(handle);
   struct vrend_transfer_info transfer_info;

   if (!res)
      return EINVAL;

   transfer_info.level = level;
   transfer_info.stride = stride;
   transfer_info.layer_stride = layer_stride;
   transfer_info.box = (struct pipe_box *)box;
   transfer_info.offset = offset;
   transfer_info.iovec = iovec;
   transfer_info.iovec_cnt = iovec_cnt;
   transfer_info.synchronized = false;

   if (ctx_id) {
      struct virgl_context *ctx = virgl_context_lookup(ctx_id);
      if (!ctx)
         return EINVAL;

      return ctx->transfer_3d(ctx, res, &transfer_info,
                              VIRGL_TRANSFER_FROM_HOST);
   } else {
      if (!res->pipe_resource)
         return EINVAL;

      return vrend_renderer_transfer_pipe(res->pipe_resource, &transfer_info,
                                          VIRGL_TRANSFER_FROM_HOST);
   }
}

int virgl_renderer_resource_attach_iov(int res_handle, struct iovec *iov,
                                       int num_iovs)
{
   TRACE_FUNC();
   struct virgl_resource *res = virgl_resource_lookup(res_handle);
   if (!res)
      return EINVAL;

   return virgl_resource_attach_iov(res, iov, num_iovs);
}

void virgl_renderer_resource_detach_iov(int res_handle, struct iovec **iov_p, int *num_iovs_p)
{
   TRACE_FUNC();
   struct virgl_resource *res = virgl_resource_lookup(res_handle);
   if (!res)
      return;

   if (iov_p)
      *iov_p = (struct iovec *)res->iov;
   if (num_iovs_p)
      *num_iovs_p = res->iov_count;

   virgl_resource_detach_iov(res);
}

int virgl_renderer_create_fence(int client_fence_id, UNUSED uint32_t ctx_id)
{
   TRACE_FUNC();
   const uint32_t fence_id = (uint32_t)client_fence_id;
   if (state.vrend_initialized)
      return vrend_renderer_create_ctx0_fence(fence_id);
   return EINVAL;
}

int virgl_renderer_context_create_fence(uint32_t ctx_id,
                                        uint32_t flags,
                                        uint32_t ring_idx,
                                        uint64_t fence_id)
{
   TRACE_FUNC();
   struct virgl_context *ctx = virgl_context_lookup(ctx_id);
   if (!ctx)
      return -EINVAL;

   assert(state.cbs->version >= 3 && state.cbs->write_context_fence);
   return ctx->submit_fence(ctx, flags, ring_idx, fence_id);
}

void virgl_renderer_context_poll(uint32_t ctx_id)
{
   TRACE_FUNC();
   struct virgl_context *ctx = virgl_context_lookup(ctx_id);
   if (!ctx)
      return;

   ctx->retire_fences(ctx);
}

int virgl_renderer_context_get_poll_fd(uint32_t ctx_id)
{
   struct virgl_context *ctx = virgl_context_lookup(ctx_id);
   if (!ctx)
      return -1;

   return ctx->get_fencing_fd(ctx);
}

void virgl_renderer_force_ctx_0(void)
{
   if (state.vrend_initialized)
      vrend_renderer_force_ctx_0();
}

void virgl_renderer_ctx_attach_resource(int ctx_id, int res_handle)
{
   TRACE_FUNC();
   struct virgl_context *ctx = virgl_context_lookup(ctx_id);
   struct virgl_resource *res = virgl_resource_lookup(res_handle);
   if (!ctx || !res)
      return;
   ctx->attach_resource(ctx, res);
}

void virgl_renderer_ctx_detach_resource(int ctx_id, int res_handle)
{
   TRACE_FUNC();
   struct virgl_context *ctx = virgl_context_lookup(ctx_id);
   struct virgl_resource *res = virgl_resource_lookup(res_handle);
   if (virgl_fd_trace())
      virgl_error("[FDTRACE] ctx_detach_resource ctx=%d res=%d%s%s", ctx_id,
                  res_handle, ctx ? "" : " NO-CTX", res ? "" : " NO-RES");
   if (!ctx || !res)
      return;
   ctx->detach_resource(ctx, res);
}

static int virgl_renderer_resource_get_info_common(int res_handle,
                                                   struct virgl_renderer_resource_info *info,
                                                   UNUSED void **d3d_tex2d)
{
   int ret = 0;

   TRACE_FUNC();
   struct virgl_resource *res = virgl_resource_lookup(res_handle);

   if (!res)
      return EINVAL;
   if (!info)
      return EINVAL;

   info->handle = res_handle;
   info->fd = res->fd;

   if (!res->pipe_resource)
      return 0;

   vrend_renderer_resource_get_info(res->pipe_resource,
                                    (struct vrend_renderer_resource_info *)info);

#ifdef WIN32
   if (d3d_tex2d)
      ret = vrend_renderer_resource_d3d11_texture2d(res->pipe_resource, d3d_tex2d);
#endif

   return ret;
}

int virgl_renderer_resource_get_info(int res_handle,
                                     struct virgl_renderer_resource_info *info)
{
   TRACE_FUNC();
   int ret;

   if ((ret = virgl_renderer_resource_get_info_common(res_handle, info, NULL)) != 0)
       return ret;

   if (state.winsys_initialized) {
      return vrend_winsys_get_attrs_for_texture(info->tex_id,
                                                info->virgl_format,
                                                &info->drm_fourcc,
                                                NULL,
                                                NULL,
                                                NULL);
   }

   return 0;
}

int virgl_renderer_resource_get_info_ext(int res_handle,
                                         struct virgl_renderer_resource_info_ext *info_ext)
{
   TRACE_FUNC();
   int ret;

   if ((ret = virgl_renderer_resource_get_info_common(res_handle,
                                                      &info_ext->base,
                                                      &info_ext->d3d_tex2d)) != 0)
      return ret;

   info_ext->version = VIRGL_RENDERER_RESOURCE_INFO_EXT_VERSION;

   if (state.winsys_initialized) {
      return vrend_winsys_get_attrs_for_texture(info_ext->base.tex_id,
                                                info_ext->base.virgl_format,
                                                &info_ext->base.drm_fourcc,
                                                &info_ext->has_dmabuf_export,
                                                &info_ext->planes,
                                                &info_ext->modifiers);
   }

   return 0;
}

void virgl_renderer_get_cap_set(uint32_t cap_set, uint32_t *max_ver,
                                uint32_t *max_size)
{
   TRACE_FUNC();

   /* this may be called before virgl_renderer_init */
   switch (cap_set) {
   case VIRTGPU_DRM_CAPSET_VIRGL:
   case VIRTGPU_DRM_CAPSET_VIRGL2:
      vrend_renderer_get_cap_set(cap_set, max_ver, max_size);
      break;
   case VIRTGPU_DRM_CAPSET_VENUS:
      *max_ver = 0;
      *max_size = proxy_get_capset(cap_set, NULL);
      break;
   case VIRTGPU_DRM_CAPSET_DRM:
      *max_ver = 0;
      *max_size = drm_renderer_capset(NULL);
      break;
   default:
      *max_ver = 0;
      *max_size = 0;
      break;
   }
}

void virgl_renderer_get_rect(int resource_id, struct iovec *iov, unsigned int num_iovs,
                             uint32_t offset, int x, int y, int width, int height)
{
   TRACE_FUNC();
   struct virgl_resource *res = virgl_resource_lookup(resource_id);
   if (!res || !res->pipe_resource)
      return;

   vrend_renderer_get_rect(res->pipe_resource, iov, num_iovs, offset, x, y,
                           width, height);
}


static void ctx0_fence_retire(uint64_t fence_id, UNUSED void *retire_data)
{
   // ctx0 fence_id is created from uint32_t but stored internally as uint64_t,
   // so casting back to uint32_t doesn't result in data loss.
   assert((fence_id >> 32) == 0);
   state.cbs->write_fence(state.cookie, (uint32_t)fence_id);
}

static virgl_renderer_gl_context create_gl_context(int scanout_idx, struct virgl_gl_ctx_param *param)
{
   struct virgl_renderer_gl_ctx_param vparam;

   if (state.winsys_initialized)
      return vrend_winsys_create_context(param);

   vparam.version = 2;
   vparam.shared = param->shared;
   vparam.compat_ctx = param->compat_ctx;
   vparam.major_ver = param->major_ver;
   vparam.minor_ver = param->minor_ver;
   return state.cbs->create_gl_context(state.cookie, scanout_idx, &vparam);
}

static void destroy_gl_context(virgl_renderer_gl_context ctx)
{
   if (state.winsys_initialized) {
      vrend_winsys_destroy_context(ctx);
      return;
   }

   state.cbs->destroy_gl_context(state.cookie, ctx);
}

static int make_current(virgl_renderer_gl_context ctx)
{
   int ret;

   if (state.winsys_initialized)
      return vrend_winsys_make_context_current(ctx);

   ret = state.cbs->make_current(state.cookie, 0, ctx);
   if (ret && state.cbs->version >= 4) {
      virgl_error("%s: Error switching context: %d\n", __func__, ret);
      assert(!ret && "Failed to switch GL context");
      return -1;
   }

   return 0;
}

static virgl_renderer_gl_context create_gl_context_surfaceless(int scanout_idx, struct virgl_gl_ctx_param *param)
{
   struct virgl_renderer_gl_ctx_param vparam;

   if (state.winsys_initialized || state.external_winsys_initialized)
      return vrend_winsys_create_context(param);

   vparam.version = 2;
   vparam.shared = param->shared;
   vparam.major_ver = param->major_ver;
   vparam.minor_ver = param->minor_ver;
   vparam.compat_ctx = param->compat_ctx;
   return state.cbs->create_gl_context(state.cookie, scanout_idx, &vparam);
}

static void destroy_gl_context_surfaceless(virgl_renderer_gl_context ctx)
{
   if (state.winsys_initialized || state.external_winsys_initialized) {
      vrend_winsys_destroy_context(ctx);
      return;
   }

   state.cbs->destroy_gl_context(state.cookie, ctx);
}

static int make_current_surfaceless(virgl_renderer_gl_context ctx)
{
   int ret;

   if (state.winsys_initialized || state.external_winsys_initialized)
      return vrend_winsys_make_context_current(ctx);

   ret = state.cbs->make_current(state.cookie, 0, ctx);
   if (ret && state.cbs->version >= 4) {
      virgl_error("%s: Error switching surfaceless context: %d\n",
                  __func__, ret);
      assert(!ret && "Failed to switch GL context");
      return -1;
   }

   return 0;
}

static int get_drm_fd(void)
{
   if (state.cbs->get_drm_fd)
      return state.cbs->get_drm_fd(state.cookie);

   return -1;
}

static const struct vrend_if_cbs vrend_cbs = {
   ctx0_fence_retire,
   create_gl_context,
   destroy_gl_context,
   make_current,
   get_drm_fd,
   create_gl_context_surfaceless,
   destroy_gl_context_surfaceless,
   make_current_surfaceless,
};

static int
proxy_renderer_cb_get_server_fd(uint32_t version)
{
   if (state.cbs && state.cbs->version >= 3 && state.cbs->get_server_fd)
      return state.cbs->get_server_fd(state.cookie, version);
   else
      return -1;
}

static const struct proxy_renderer_cbs proxy_cbs = {
   proxy_renderer_cb_get_server_fd,
};

void *virgl_renderer_get_cursor_data(uint32_t resource_id, uint32_t *width, uint32_t *height)
{
   struct virgl_resource *res = virgl_resource_lookup(resource_id);
   if (!res || !res->pipe_resource)
      return NULL;

   vrend_renderer_force_ctx_0();
   return vrend_renderer_get_cursor_contents(res->pipe_resource,
                                             width,
                                             height);
}

static bool
virgl_context_foreach_retire_fences(struct virgl_context *ctx,
                                    UNUSED void* data)
{
   /* vrend contexts are polled explicitly by the caller */
   if (ctx->capset_id != VIRTGPU_DRM_CAPSET_VIRGL &&
       ctx->capset_id != VIRTGPU_DRM_CAPSET_VIRGL2 &&
       !(state.flags & VIRGL_RENDERER_ASYNC_FENCE_CB))
   {
      assert(ctx->retire_fences);
      ctx->retire_fences(ctx);
   }
   return true;
}

void virgl_renderer_poll(void)
{
   TRACE_FUNC();
   if (state.vrend_initialized)
      vrend_renderer_poll();

   struct virgl_context_foreach_args args;
   args.callback = virgl_context_foreach_retire_fences;
   virgl_context_foreach(&args);
}

void virgl_renderer_cleanup(UNUSED void *cookie)
{
   TRACE_FUNC();
   if (state.vrend_initialized)
      vrend_renderer_prepare_reset();

   if (state.context_initialized)
      virgl_context_table_cleanup();

   if (state.resource_initialized)
      virgl_resource_table_cleanup();

   if (state.proxy_initialized)
      proxy_renderer_fini();

   if (state.vrend_initialized)
      vrend_renderer_fini();

   if (state.fence_initialized)
      virgl_fence_table_cleanup();

   if (state.winsys_initialized || state.external_winsys_initialized)
      vrend_winsys_cleanup();

   if (state.drm_initialized)
      drm_renderer_fini();

   /* vkr_allocator_init is called on-demand upon the first map */
   vkr_allocator_fini();

   memset(&state, 0, sizeof(state));
}

int virgl_renderer_init(void *cookie, int flags, struct virgl_renderer_callbacks *cbs)
{
   TRACE_INIT();
   TRACE_FUNC();

   int ret;

   /* VIRGL_RENDERER_THREAD_SYNC is a hint and can be silently ignored */
   if (!has_eventfd() || getenv("VIRGL_DISABLE_MT"))
      flags &= ~VIRGL_RENDERER_THREAD_SYNC;

   if (state.client_initialized && (state.cookie != cookie ||
                                    state.flags != flags ||
                                    state.cbs != cbs)) {
      virgl_error("renderer already initialized\n");
      return -EBUSY;
   }

   if (!state.client_initialized) {
      if (!cbs ||
          cbs->version < 1 ||
          cbs->version > VIRGL_RENDERER_CALLBACKS_VERSION) {
         virgl_error("invalid renderer callbacks\n");
         return -1;
      }

      state.cookie = cookie;
      state.flags = flags;
      state.cbs = cbs;
      state.client_initialized = true;
   }

   if (!state.resource_initialized) {
      const struct virgl_resource_pipe_callbacks *pipe_cbs =
         (flags & VIRGL_RENDERER_NO_VIRGL) ? NULL :
         vrend_renderer_get_pipe_callbacks();

      ret = virgl_resource_table_init(pipe_cbs);
      if (ret) {
         virgl_error("failed to initialize virgl resources\n");
         goto fail;
      }
      state.resource_initialized = true;
   }

   if (!state.context_initialized) {
      ret = virgl_context_table_init();
      if (ret) {
         virgl_error("failed to initialize virgl context\n");
         goto fail;
      }
      state.context_initialized = true;
   }

   if (!state.winsys_initialized && !(flags & VIRGL_RENDERER_NO_VIRGL) &&
       (flags & (VIRGL_RENDERER_USE_EGL | VIRGL_RENDERER_USE_GLX))) {
      int drm_fd = -1;

      if (flags & VIRGL_RENDERER_USE_EGL) {
         if (cbs->version >= 2 && cbs->get_drm_fd)
            drm_fd = cbs->get_drm_fd(cookie);
      }

      ret = vrend_winsys_init(flags, drm_fd);
      if (ret) {
         if (drm_fd >= 0)
            close(drm_fd);
         virgl_error("failed to initialize vrend winsys\n");
         goto fail;
      }
      state.winsys_initialized = true;
   }

   if (!state.winsys_initialized && !state.external_winsys_initialized &&
       state.cbs && state.cbs->version >= 4 && state.cbs->get_egl_display) {
      void *egl_display = NULL;

      if (!cbs->create_gl_context || !cbs->destroy_gl_context ||
          !cbs->make_current) {
         virgl_error("invalid renderer gl callbacks\n");
         ret = EINVAL;
         goto fail;
      }

      egl_display = state.cbs->get_egl_display(cookie);

      if (!egl_display) {
         virgl_error("failed to get egl display\n");
         ret = -1;
         goto fail;
      }
      ret = vrend_winsys_init_external(egl_display);

      if (ret) {
         virgl_error("failed to initialize vrend winsys\n");
         ret = -1;
         goto fail;
      }

      state.external_winsys_initialized = true;
   }

   if (!state.vrend_initialized && !(flags & VIRGL_RENDERER_NO_VIRGL)) {
      uint32_t renderer_flags = 0;

      if (!cookie || !cbs) {
         virgl_error("invalid renderer vrend callbacks\n");
         ret = -1;
         goto fail;
      }

      if (flags & VIRGL_RENDERER_THREAD_SYNC)
         renderer_flags |= VREND_USE_THREAD_SYNC;
      if (flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
         renderer_flags |= VREND_USE_ASYNC_FENCE_CB;
      if (flags & VIRGL_RENDERER_USE_EXTERNAL_BLOB)
         renderer_flags |= VREND_USE_EXTERNAL_BLOB;
      if (flags & VIRGL_RENDERER_USE_VIDEO)
         renderer_flags |= VREND_USE_VIDEO;
      if (flags & VIRGL_RENDERER_D3D11_SHARE_TEXTURE)
         renderer_flags |= VREND_D3D11_SHARE_TEXTURE;
      if (flags & VIRGL_RENDERER_COMPAT_PROFILE)
         renderer_flags |= VREND_USE_COMPAT_CONTEXT;
      if (flags & VIRGL_RENDERER_USE_GLES)
         renderer_flags |= VREND_USE_GLES;
      if (flags & VIRGL_RENDERER_VENUS)
         renderer_flags |= VREND_USE_GBM_LAYOUT;

      ret = vrend_renderer_init(&vrend_cbs, renderer_flags);
      if (ret) {
         virgl_error("failed to initialize vrend renderer\n");
         goto fail;
      }
      state.vrend_initialized = true;
   }

   if (!state.proxy_initialized && (flags & VIRGL_RENDERER_RENDER_SERVER)) {
      ret = proxy_renderer_init(&proxy_cbs, flags | VIRGL_RENDERER_NO_VIRGL);
      if (ret) {
         virgl_error("failed to initialize venus renderer\n");
         goto fail;
      }
      state.proxy_initialized = true;
   }

   if ((flags & VIRGL_RENDERER_ASYNC_FENCE_CB) &&
       (flags & VIRGL_RENDERER_DRM)) {
      int drm_fd = -1;

      if (cbs->version >= 2 && cbs->get_drm_fd)
         drm_fd = cbs->get_drm_fd(cookie);

      ret = drm_renderer_init(drm_fd);
      if (ret) {
         virgl_error("failed to initialize drm renderer\n");
         goto fail;
      }
      state.drm_initialized = true;
   }

   if (!state.fence_initialized) {
      ret = virgl_fence_table_init();
      if (ret) {
         virgl_error("failed to initialize fence table\n");
         goto fail;
      }
      state.fence_initialized = true;
   }

   return 0;

fail:
   virgl_renderer_cleanup(NULL);
   return ret;
}

int virgl_renderer_get_fd_for_texture(uint32_t tex_id, int *fd)
{
   TRACE_FUNC();
   if (state.winsys_initialized)
      return vrend_winsys_get_fd_for_texture(tex_id, fd);
   return -1;
}

int virgl_renderer_get_fd_for_texture2(uint32_t tex_id, int *fd, int *stride, int *offset)
{
   TRACE_FUNC();
   if (state.winsys_initialized)
      return vrend_winsys_get_fd_for_texture2(tex_id, fd, stride, offset);
   return -1;
}

void virgl_renderer_reset(void)
{
   TRACE_FUNC();
   if (state.vrend_initialized)
      vrend_renderer_prepare_reset();

   if (state.context_initialized)
      virgl_context_table_reset();

   if (state.resource_initialized)
      virgl_resource_table_reset();

   if (state.proxy_initialized)
      proxy_renderer_reset();

   if (state.vrend_initialized)
      vrend_renderer_reset();

   if (state.drm_initialized)
      drm_renderer_reset();
}

int virgl_renderer_get_poll_fd(void)
{
   TRACE_FUNC();
   if (state.vrend_initialized)
      return vrend_renderer_get_poll_fd();

   return -1;
}

static
void virgl_null_logger(UNUSED const char *fmt, UNUSED va_list va)
{
}

/* Compatibility layer for the virgl_set_debug_callback function */
static inline void virgl_legacy_logger_wrapper(virgl_debug_callback_type cb,
                                               const char *fmt,
                                               ...)
{
   va_list va;
   va_start(va, fmt);
   cb(fmt, va);
   va_end(va);
}

/* The logger need to be wrapped into a structure to be given as void* */
struct virgl_legacy_logger_holder {
   virgl_debug_callback_type logger;
};

static void virgl_legacy_logger(UNUSED enum virgl_log_level_flags log_level,
                                const char *message,
                                void* user_data)
{
   struct virgl_legacy_logger_holder *log_cb = user_data;
   virgl_legacy_logger_wrapper(log_cb->logger, "%s", message);
}

static struct virgl_legacy_logger_holder legacy_logger = { virgl_null_logger };

virgl_debug_callback_type virgl_set_debug_callback(virgl_debug_callback_type cb)
{
   virgl_debug_callback_type previous_cb = legacy_logger.logger;
   legacy_logger.logger = cb;
   virgl_log_set_handler(virgl_legacy_logger, &legacy_logger, NULL);
   return previous_cb;
}

void virgl_set_log_callback(virgl_log_callback_type cb,
                            void* user_data,
                            virgl_free_data_callback_type free_user_data_cb)
{
   virgl_log_set_handler(cb, user_data, free_user_data_cb);
}

static int virgl_renderer_export_query(void *execute_args, uint32_t execute_size)
{
   struct virgl_resource *res;
   struct virgl_renderer_export_query *export_query = execute_args;
   if (execute_size != sizeof(struct virgl_renderer_export_query))
      return -EINVAL;

   if (export_query->hdr.size != sizeof(struct virgl_renderer_export_query))
      return -EINVAL;

   res = virgl_resource_lookup(export_query->in_resource_id);
   if (!res)
      return -EINVAL;


   if (res->pipe_resource) {
      return vrend_renderer_export_query(res->pipe_resource, export_query);
   } else if (!export_query->in_export_fds) {
      /* Untyped resources are expected to be exported with
       * virgl_renderer_resource_export_blob instead and have no type
       * information.  But when this is called to query (in_export_fds is
       * false) an untyped resource, we should return sane values.
       */
      export_query->out_num_fds = 1;
      export_query->out_fourcc = 0;
      export_query->out_fds[0] = -1;
      export_query->out_strides[0] = 0;
      export_query->out_offsets[0] = 0;
      export_query->out_modifier = DRM_FORMAT_MOD_INVALID;
      return 0;
   } else {
      return -EINVAL;
   }
}

static int virgl_renderer_supported_structures(void *execute_args, uint32_t execute_size)
{
   struct virgl_renderer_supported_structures *supported_structures = execute_args;
   if (execute_size != sizeof(struct virgl_renderer_supported_structures))
      return -EINVAL;

   if (supported_structures->hdr.size != sizeof(struct virgl_renderer_supported_structures))
      return -EINVAL;

   if (supported_structures->in_stype_version == 0) {
      supported_structures->out_supported_structures_mask =
         VIRGL_RENDERER_STRUCTURE_TYPE_EXPORT_QUERY |
         VIRGL_RENDERER_STRUCTURE_TYPE_SUPPORTED_STRUCTURES;
   } else {
      supported_structures->out_supported_structures_mask = 0;
   }

   return 0;
}

int virgl_renderer_execute(void *execute_args, uint32_t execute_size)
{
   TRACE_FUNC();
   struct virgl_renderer_hdr *hdr = execute_args;
   if (hdr->stype_version != 0)
      return -EINVAL;

   switch (hdr->stype) {
      case VIRGL_RENDERER_STRUCTURE_TYPE_SUPPORTED_STRUCTURES:
         return virgl_renderer_supported_structures(execute_args, execute_size);
      case VIRGL_RENDERER_STRUCTURE_TYPE_EXPORT_QUERY:
         return virgl_renderer_export_query(execute_args, execute_size);
      default:
         return -EINVAL;
   }
}

int virgl_renderer_resource_create_blob(const struct virgl_renderer_resource_create_blob_args *args)
{
   TRACE_FUNC();
   struct virgl_resource *res;
   struct virgl_context *ctx;
   /* Zero-init: get_blob() only sets iosurface_id on the IOSurface-backed (macOS) path,
    * so an uninitialized field would leave non-IOSurface resources with stack garbage that
    * SET_SCANOUT_BLOB then misreads as a valid IOSurface id (limina tier-2 #30). */
   struct virgl_context_blob blob = {0};
   bool has_host_storage;
   bool has_guest_storage;
   int ret;

   switch (args->blob_mem) {
   case VIRGL_RENDERER_BLOB_MEM_GUEST:
      has_host_storage = false;
      has_guest_storage = true;
      break;
   case VIRGL_RENDERER_BLOB_MEM_HOST3D:
      has_host_storage = true;
      has_guest_storage = false;
      break;
   case VIRGL_RENDERER_BLOB_MEM_HOST3D_GUEST:
      has_host_storage = true;
      has_guest_storage = true;
      break;
   default:
      return -EINVAL;
   }

   /* user resource id must be greater than 0 */
   if (args->res_handle == 0)
      return -EINVAL;

   /* user resource id must be unique */
   if (virgl_resource_lookup(args->res_handle))
      return -EINVAL;

   if (args->size == 0)
      return -EINVAL;
   if (has_guest_storage) {
      const size_t iov_size = vrend_get_iovec_size(args->iovecs, args->num_iovs);
      if (iov_size < args->size)
         return -EINVAL;
   } else {
      if (args->num_iovs)
         return -EINVAL;
   }

   if (!has_host_storage) {
      res = virgl_resource_create_from_iov(args->res_handle,
                                           args->iovecs,
                                           args->num_iovs);
      if (!res)
         return -ENOMEM;

      res->map_info = VIRGL_RENDERER_MAP_CACHE_CACHED;
      return 0;
   }

   ctx = virgl_context_lookup(args->ctx_id);
   if (!ctx)
      return -EINVAL;

   ret = ctx->get_blob(ctx, args->res_handle, args->blob_id, args->size, args->blob_flags, &blob);
   if (ret)
      return ret;

   if (blob.type == VIRGL_RESOURCE_OPAQUE_HANDLE) {
      assert(!(args->blob_flags & VIRGL_RENDERER_BLOB_FLAG_USE_SHAREABLE));
      res = virgl_resource_create_from_opaque_handle(ctx, args->res_handle, blob.u.opaque_handle);
      if (!res)
         return -ENOMEM;
   } else if (blob.type != VIRGL_RESOURCE_FD_INVALID) {
      res = virgl_resource_create_from_fd(args->res_handle,
                                          blob.type,
                                          blob.u.fd,
                                          args->iovecs,
                                          args->num_iovs,
                                          &blob.vulkan_info);
      if (!res)
         return -ENOMEM;
   } else {
      res = virgl_resource_create_from_pipe(args->res_handle,
                                            blob.u.pipe_resource,
                                            args->iovecs,
                                            args->num_iovs);
      if (!res)
         return -ENOMEM;
   }

   res->map_info = blob.map_info;
   res->map_size = args->size;
   res->iosurface_id = blob.iosurface_id;
   /* limina tier-2 (macOS) #28: borrow MoltenVK's own mapping for a HOST_VISIBLE blob so the VMM
    * hv_vm_maps the exact memory the GPU binds (one mapping, guest+GPU coherent). 0 = fd path. */
   res->map_ptr = blob.map_ptr;

   return 0;
}

/* limina tier-2 (macOS): return the global IOSurface id backing a scanout resource, or 0 if
 * the resource is not IOSurface-backed. libkrun's SET_SCANOUT_BLOB uses this to present the
 * IOSurface zero-copy (worker and vkr are in the same process, so the id is a valid handle)
 * instead of reading the resource's SHM carrier. Mirrors virgl_renderer_resource_get_map_ptr. */
int virgl_renderer_resource_get_iosurface_id(uint32_t res_handle, uint32_t *iosurface_id)
{
   TRACE_FUNC();
   struct virgl_resource *res = virgl_resource_lookup(res_handle);
   if (!res)
      return -EINVAL;

   *iosurface_id = res->iosurface_id;
   return 0;
}

#ifdef __APPLE__
/* Defined in venus/vkr_metal_helpers.m (forward-declared to avoid pulling the Vulkan-typed
 * vkr_metal_helpers.h into this TU). */
int vkr_mtl_iosurface_read(uint32_t id, void *dst, uint32_t dst_stride, uint32_t height);
#endif

/* limina: copy a scanout resource's presented IOSurface into a CPU buffer (top-down BGRA,
 * dst_stride bytes/row, height rows). The headless capture display sink uses this because venus
 * scanout blobs have no CPU transfer_read — the frame only exists in the IOSurface's shared
 * storage. Worker and vkr share a process, so res->iosurface_id resolves via vkr's registry.
 * Returns 0 on success, -EINVAL if the resource is not IOSurface-backed or the read failed. */
int virgl_renderer_resource_read_iosurface(uint32_t res_handle, void *dst, uint32_t dst_stride,
                                           uint32_t height)
{
   TRACE_FUNC();
   struct virgl_resource *res = virgl_resource_lookup(res_handle);
   if (!res || !res->iosurface_id)
      return -EINVAL;
#ifdef __APPLE__
   return vkr_mtl_iosurface_read(res->iosurface_id, dst, dst_stride, height) ? 0 : -EINVAL;
#else
   (void)dst;
   (void)dst_stride;
   (void)height;
   return -EINVAL;
#endif
}

#ifdef ENABLE_SAME_PROCESS_RENDER_SERVER
/* Defined in server/render_state.c (compiled into the lib in thread render-server
 * mode; server/ is not on the include path for this TU, hence the local decls). */
void render_state_limina_dump_state(void);
bool render_state_limina_journal_export(uint32_t ctx_id, void **out_buf, size_t *out_size);
void render_state_limina_journal_unpin(uint32_t ctx_id, uint64_t key);
uint64_t render_state_limina_journal_seq(uint32_t ctx_id);
bool render_state_limina_replay_begin(uint32_t ctx_id);
bool render_state_limina_replay_submit(uint32_t ctx_id, void *cmd, uint32_t size);
bool render_state_limina_replay_ring_cmd(uint32_t ctx_id,
                                         uint64_t ring_id,
                                         void *cmd,
                                         uint32_t size);
bool render_state_limina_replay_end(uint32_t ctx_id);
int render_state_limina_memory_census(uint32_t ctx_id, uint64_t **out_pairs,
                                      uint32_t *out_count);
int render_state_limina_sync_export(uint32_t ctx_id, void **out_buf, size_t *out_size);
int render_state_limina_sync_restore(uint32_t ctx_id, const void *data, size_t size);
bool render_state_limina_memory_read(uint32_t ctx_id, uint64_t mem_id, void *buf,
                                     uint64_t size);
bool render_state_limina_memory_write(uint32_t ctx_id, uint64_t mem_id, const void *buf,
                                      uint64_t size);
#endif

/* limina M9.3 diagnostics: log the live venus (vkr) context table — per context:
 * rings, object counts by VkObjectType, resources, sync queues. Thread-safe (takes
 * the render-server renderer lock). The libkrun gpu worker calls this on the first
 * stale-context submission after a snapshot restore, and periodically under
 * LIMINA_GPU_TRACE_VKR=1; the per-type object tally on a healthy session is the
 * retain-and-replay bill of materials. */
void virgl_renderer_limina_dump_state(void)
{
   TRACE_FUNC();
#ifdef ENABLE_SAME_PROCESS_RENDER_SERVER
   if (state.proxy_initialized) {
      render_state_limina_dump_state();
      return;
   }
   virgl_info("[GPUTRACE] vkr state: venus proxy not initialized\n");
#else
   virgl_info("[GPUTRACE] vkr state: unavailable (out-of-process render server)\n");
#endif
}

/* limina snapshot-replay (limina M9.3 P1): venus journal export + replay. Same
 * availability rules as the state dump above — same-process render server only.
 * See vkr_renderer.h for the replay contract (replay buffers must be mutable;
 * one journal entry per submit/ring_cmd call). */

int virgl_renderer_limina_journal_export(uint32_t ctx_id, void **out_buf, uint64_t *out_size)
{
   TRACE_FUNC();
#ifdef ENABLE_SAME_PROCESS_RENDER_SERVER
   if (state.proxy_initialized) {
      size_t size = 0;
      if (!render_state_limina_journal_export(ctx_id, out_buf, &size))
         return -EINVAL;
      *out_size = size;
      return 0;
   }
#endif
   (void)ctx_id;
   (void)out_buf;
   (void)out_size;
   return -ENOTSUP;
}

uint64_t virgl_renderer_limina_journal_seq(uint32_t ctx_id)
{
#ifdef ENABLE_SAME_PROCESS_RENDER_SERVER
   if (state.proxy_initialized)
      return render_state_limina_journal_seq(ctx_id);
#endif
   (void)ctx_id;
   return 0;
}

void virgl_renderer_limina_journal_unpin(uint32_t ctx_id, uint64_t key)
{
#ifdef ENABLE_SAME_PROCESS_RENDER_SERVER
   if (state.proxy_initialized) {
      render_state_limina_journal_unpin(ctx_id, key);
      return;
   }
#endif
   (void)ctx_id;
   (void)key;
}

int virgl_renderer_limina_replay_begin(uint32_t ctx_id)
{
   TRACE_FUNC();
#ifdef ENABLE_SAME_PROCESS_RENDER_SERVER
   if (state.proxy_initialized)
      return render_state_limina_replay_begin(ctx_id) ? 0 : -EINVAL;
#endif
   (void)ctx_id;
   return -ENOTSUP;
}

int virgl_renderer_limina_replay_submit(uint32_t ctx_id, void *cmd, uint32_t size)
{
#ifdef ENABLE_SAME_PROCESS_RENDER_SERVER
   if (state.proxy_initialized)
      return render_state_limina_replay_submit(ctx_id, cmd, size) ? 0 : -EINVAL;
#endif
   (void)ctx_id;
   (void)cmd;
   (void)size;
   return -ENOTSUP;
}

int virgl_renderer_limina_replay_ring_cmd(uint32_t ctx_id,
                                          uint64_t ring_id,
                                          void *cmd,
                                          uint32_t size)
{
#ifdef ENABLE_SAME_PROCESS_RENDER_SERVER
   if (state.proxy_initialized)
      return render_state_limina_replay_ring_cmd(ctx_id, ring_id, cmd, size) ? 0 : -EINVAL;
#endif
   (void)ctx_id;
   (void)ring_id;
   (void)cmd;
   (void)size;
   return -ENOTSUP;
}

int virgl_renderer_limina_replay_end(uint32_t ctx_id)
{
   TRACE_FUNC();
#ifdef ENABLE_SAME_PROCESS_RENDER_SERVER
   if (state.proxy_initialized)
      return render_state_limina_replay_end(ctx_id) ? 0 : -EINVAL;
#endif
   (void)ctx_id;
   return -ENOTSUP;
}

int virgl_renderer_limina_sync_export(uint32_t ctx_id, void **out_buf, uint64_t *out_size)
{
   TRACE_FUNC();
#ifdef ENABLE_SAME_PROCESS_RENDER_SERVER
   if (state.proxy_initialized) {
      size_t size = 0;
      if (render_state_limina_sync_export(ctx_id, out_buf, &size))
         return -EINVAL;
      *out_size = size;
      return 0;
   }
#endif
   (void)ctx_id;
   (void)out_buf;
   (void)out_size;
   return -ENOTSUP;
}

int virgl_renderer_limina_sync_restore(uint32_t ctx_id, const void *data, uint64_t size)
{
   TRACE_FUNC();
#ifdef ENABLE_SAME_PROCESS_RENDER_SERVER
   if (state.proxy_initialized)
      return render_state_limina_sync_restore(ctx_id, data, size);
#endif
   (void)ctx_id;
   (void)data;
   (void)size;
   return -ENOTSUP;
}

int virgl_renderer_limina_memory_census(uint32_t ctx_id, uint64_t **out_pairs,
                                        uint32_t *out_count)
{
   TRACE_FUNC();
#ifdef ENABLE_SAME_PROCESS_RENDER_SERVER
   if (state.proxy_initialized)
      return render_state_limina_memory_census(ctx_id, out_pairs, out_count);
#endif
   (void)ctx_id;
   (void)out_pairs;
   (void)out_count;
   return -ENOTSUP;
}

int virgl_renderer_limina_memory_read(uint32_t ctx_id, uint64_t mem_id, void *buf,
                                      uint64_t size)
{
#ifdef ENABLE_SAME_PROCESS_RENDER_SERVER
   if (state.proxy_initialized)
      return render_state_limina_memory_read(ctx_id, mem_id, buf, size) ? 0 : -EINVAL;
#endif
   (void)ctx_id;
   (void)mem_id;
   (void)buf;
   (void)size;
   return -ENOTSUP;
}

int virgl_renderer_limina_memory_write(uint32_t ctx_id, uint64_t mem_id, const void *buf,
                                       uint64_t size)
{
#ifdef ENABLE_SAME_PROCESS_RENDER_SERVER
   if (state.proxy_initialized)
      return render_state_limina_memory_write(ctx_id, mem_id, buf, size) ? 0 : -EINVAL;
#endif
   (void)ctx_id;
   (void)mem_id;
   (void)buf;
   (void)size;
   return -ENOTSUP;
}

int virgl_renderer_resource_map(uint32_t res_handle, void **out_map, uint64_t *out_size)
{
   TRACE_FUNC();
   int ret = 0;
   void *map = NULL;
   uint64_t map_size = 0;
   struct virgl_context *ctx = NULL;
   struct virgl_resource *res = virgl_resource_lookup(res_handle);
   if (!res || res->mapped)
      return -EINVAL;

   /* limina #28: a HOST_VISIBLE venus blob borrows MoltenVK's own vkMapMemory pointer; hand it
    * back directly (no mmap). Lifetime is owned by the VkDeviceMemory (unmapped in vkFreeMemory),
    * so we do NOT cache it in res->mapped (which the fd paths munmap on unmap). */
   if (res->map_ptr) {
      *out_map = (void *)(uintptr_t)res->map_ptr;
      *out_size = res->map_size;
      return 0;
   }

   if (res->pipe_resource) {
      ret = vrend_renderer_resource_map(res->pipe_resource, &map, &map_size);
      if (!ret) {
         res->map_size = map_size;
         res->mapped_from_pipe_resource = true;
      }
   } else {
      enum virgl_resource_fd_type fd_type = res->fd_type;
      enum virgl_resource_fd_type export_fd_type = res->fd_type;
      int fd = res->fd;

      if (fd_type == VIRGL_RESOURCE_OPAQUE_HANDLE) {
         ctx = virgl_context_lookup(res->opaque_handle_context_id);
         if (!ctx)
            return -EINVAL;

         if (!ctx->resource_map) {
            /* Create a transient dmabuf. */
            export_fd_type = virgl_resource_export_fd(res, &fd);
         }
      }

      switch (export_fd_type) {
      case VIRGL_RESOURCE_FD_DMABUF:
      case VIRGL_RESOURCE_FD_SHM:
         map = mmap(NULL, res->map_size, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
         map_size = res->map_size;
         break;
      case VIRGL_RESOURCE_FD_OPAQUE:
         ret = vkr_allocator_resource_map(res, &map, &map_size);
         break;
      case VIRGL_RESOURCE_OPAQUE_HANDLE:
         map = ctx->resource_map(ctx, res, NULL, PROT_WRITE | PROT_READ, MAP_SHARED);
         map_size = res->map_size;
         break;
      case VIRGL_RESOURCE_FD_INVALID:
         /* Avoid a default case so that -Wswitch will tell us at compile time
          * if a new virgl resource type is added without being handled here.
          */
         break;
      }

      if (export_fd_type != fd_type)
         close(fd);
   }

   if (!map || map == MAP_FAILED)
      return -EINVAL;

   res->mapped = map;
   *out_map = map;
   *out_size = map_size;
   return ret;
}

int virgl_renderer_resource_map_fixed(uint32_t res_handle, void *addr)
{
   void *map = NULL;
   struct virgl_context *ctx = NULL;
   struct virgl_resource *res = virgl_resource_lookup(res_handle);

   if (!res)
      return -EINVAL;

   enum virgl_resource_fd_type fd_type = res->fd_type;
   enum virgl_resource_fd_type export_fd_type = res->fd_type;
   int fd = res->fd;

   if (fd_type == VIRGL_RESOURCE_OPAQUE_HANDLE) {
      ctx = virgl_context_lookup(res->opaque_handle_context_id);
      if (!ctx)
         return -EINVAL;

      if (!ctx->resource_map) {
         /* Create a transient dmabuf. */
         export_fd_type = virgl_resource_export_fd(res, &fd);
      }
   }

   switch (export_fd_type) {
      case VIRGL_RESOURCE_FD_DMABUF:
      case VIRGL_RESOURCE_FD_SHM:
         map = mmap(addr, res->map_size, PROT_WRITE | PROT_READ,
                    MAP_FIXED | MAP_SHARED, fd, 0);
         break;
      case VIRGL_RESOURCE_OPAQUE_HANDLE:
         map = ctx->resource_map(ctx, res, addr, PROT_WRITE | PROT_READ,
                                 MAP_FIXED | MAP_SHARED);
         break;
      case VIRGL_RESOURCE_FD_OPAQUE:
      case VIRGL_RESOURCE_FD_INVALID:
         /* Avoid a default case so that -Wswitch will tell us at compile time
          * if a new virgl resource type is added without being handled here.
          */
      break;
   }

   if (export_fd_type != fd_type)
      close(fd);

   if (!map)
      return -EOPNOTSUPP;

   if (map == MAP_FAILED)
      return -EINVAL;

   return 0;
}

/* limina: thin shim over virgl_renderer_resource_map() that hands back the host map address as an
 * integer. The limina worker's macOS blob path (rutabaga map_ptr -> libkrun resource_map_blob)
 * needs the host pointer to hv_vm_map a venus blob into the guest; the krunkit fork exposed a
 * cached res->map_ptr, which upstream replaced with the void**-returning resource_map(). We map
 * on first call (resource_map caches it in res->mapped) and return res->mapped thereafter, so
 * repeated calls are idempotent like the old accessor. */
int virgl_renderer_resource_get_map_ptr(uint32_t res_handle, uint64_t *map_ptr)
{
   TRACE_FUNC();
   struct virgl_resource *res = virgl_resource_lookup(res_handle);
   if (!res)
      return -EINVAL;

   /* limina #28: HOST_VISIBLE venus blob — borrowed MoltenVK pointer, return it as-is. */
   if (res->map_ptr) {
      *map_ptr = res->map_ptr;
      return 0;
   }

   if (!res->mapped) {
      void *map = NULL;
      uint64_t map_size = 0;
      int ret = virgl_renderer_resource_map(res_handle, &map, &map_size);
      if (ret)
         return ret;
   }

   *map_ptr = (uint64_t)(uintptr_t)res->mapped;
   return 0;
}

int virgl_renderer_resource_unmap(uint32_t res_handle)
{
   TRACE_FUNC();
   int ret = 0;
   struct virgl_resource *res = virgl_resource_lookup(res_handle);

   /* limina #28: borrowed MoltenVK pointer (HOST_VISIBLE venus blob) — never cached in res->mapped
    * and never munmap'd here; the VkDeviceMemory owns it and vkUnmapMemory's it in vkFreeMemory. */
   if (res && res->map_ptr)
      return 0;

   if (!res || !res->mapped)
      return -EINVAL;

   if (res->mapped_from_pipe_resource) {
      assert(res->pipe_resource);
      ret = vrend_renderer_resource_unmap(res->pipe_resource);
   } else {
      switch (res->fd_type) {
      case VIRGL_RESOURCE_FD_DMABUF:
      case VIRGL_RESOURCE_FD_SHM:
      case VIRGL_RESOURCE_OPAQUE_HANDLE:
         ret = munmap(res->mapped, res->map_size);
         break;
      case VIRGL_RESOURCE_FD_OPAQUE:
         ret = vkr_allocator_resource_unmap(res);
         break;
      case VIRGL_RESOURCE_FD_INVALID:
         /* Avoid a default case so that -Wswitch will tell us at compile time
          * if a new virgl resource type is added without being handled here.
          */
         ret = -EINVAL;
         break;
      }
   }

   assert(!ret);
   res->mapped = NULL;
   return ret;
}

int virgl_renderer_resource_get_map_info(uint32_t res_handle, uint32_t *map_info)
{
   TRACE_FUNC();
   struct virgl_resource *res = virgl_resource_lookup(res_handle);
   if (!res)
      return -EINVAL;

   if ((res->map_info & VIRGL_RENDERER_MAP_CACHE_MASK) ==
       VIRGL_RENDERER_MAP_CACHE_NONE)
      return -EINVAL;

   *map_info = res->map_info;
   return 0;
}

int
virgl_renderer_resource_export_blob(uint32_t res_id, uint32_t *fd_type, int *fd)
{
   TRACE_FUNC();
   struct virgl_resource *res = virgl_resource_lookup(res_id);
   if (!res)
      return -EINVAL;

   /* limina #28: a HOST_VISIBLE venus blob is shared by pointer (map_ptr), has no fd, and must
    * never be fd-exported. Bail before virgl_resource_export_fd, which for our OPAQUE_HANDLE
    * vehicle would call the (absent) proxy-context export_fd callback and crash. rutabaga's
    * create_blob calls export_blob(res).ok() unconditionally; returning -EINVAL => handle=None. */
   if (res->map_ptr)
      return -EINVAL;

   switch (virgl_resource_export_fd(res, fd)) {
   case VIRGL_RESOURCE_FD_DMABUF:
      *fd_type = VIRGL_RENDERER_BLOB_FD_TYPE_DMABUF;
      break;
   case VIRGL_RESOURCE_FD_OPAQUE:
      *fd_type = VIRGL_RENDERER_BLOB_FD_TYPE_OPAQUE;
      break;
   case VIRGL_RESOURCE_FD_SHM:
      *fd_type = VIRGL_RENDERER_BLOB_FD_TYPE_SHM;
      break;
   case VIRGL_RESOURCE_OPAQUE_HANDLE:
   case VIRGL_RESOURCE_FD_INVALID:
      /* Avoid a default case so that -Wswitch will tell us at compile time if a
       * new virgl resource type is added without being handled here.
       */
      return -EINVAL;
   }

   return 0;
}

int
virgl_renderer_resource_import_blob(const struct virgl_renderer_resource_import_blob_args *args)
{
   TRACE_FUNC();
   struct virgl_resource *res;

   /* user resource id must be greater than 0 */
   if (args->res_handle == 0)
      return -EINVAL;

   /* user resource id must be unique */
   if (virgl_resource_lookup(args->res_handle))
      return -EINVAL;

   switch (args->blob_mem) {
   case VIRGL_RENDERER_BLOB_MEM_HOST3D:
   case VIRGL_RENDERER_BLOB_MEM_GUEST_VRAM:
      break;
   default:
      return -EINVAL;
   }

   enum virgl_resource_fd_type fd_type = VIRGL_RESOURCE_FD_INVALID;
   switch (args->fd_type) {
   case VIRGL_RENDERER_BLOB_FD_TYPE_DMABUF:
      fd_type = VIRGL_RESOURCE_FD_DMABUF;
      break;
   case VIRGL_RENDERER_BLOB_FD_TYPE_OPAQUE:
      fd_type = VIRGL_RESOURCE_FD_OPAQUE;
      break;
   case VIRGL_RENDERER_BLOB_FD_TYPE_SHM:
      fd_type = VIRGL_RESOURCE_FD_SHM;
      break;
   default:
      return -EINVAL;
   }

   if (args->fd < 0)
      return -EINVAL;
   if (args->size == 0)
      return -EINVAL;

   res = virgl_resource_create_from_fd(args->res_handle,
                                       fd_type,
                                       args->fd,
                                       NULL,
                                       0,
                                       NULL);
   if (!res)
      return -ENOMEM;

   res->map_info = 0;
   res->map_size = args->size;

   return 0;
}

int
virgl_renderer_export_fence(uint64_t client_fence_id, int *fd)
{
   TRACE_FUNC();

   /* transfers FD ownership to caller */
   *fd = virgl_fence_get_fd(client_fence_id);
   if (*fd >= 0)
      return 0;

   return -EINVAL;
}

int virgl_renderer_export_signalled_fence(void)
{
   TRACE_FUNC();

   /* transfers FD ownership to caller, returns -1 on failure */
   return virgl_fence_get_last_signalled_fence_fd();
}

static int attach_in_fence_fd(struct virgl_context *ctx, int fence_fd)
{
   int ret = -EINVAL;

#ifndef WIN32
   ret = sync_accumulate("virglrenderer", &ctx->in_fence_fd, fence_fd);
#endif
   close(fence_fd);

   return ret;
}

/* Special entrypoint for vtest, which has received a real fence fd,
 * not a fence-id
 */
int virgl_renderer_attach_fence(int ctx_id, int fence_fd)
{
   TRACE_FUNC();
   struct virgl_context *ctx = virgl_context_lookup(ctx_id);
   if (!ctx)
      return EINVAL;

   return attach_in_fence_fd(ctx, fence_fd);
}

int virgl_renderer_get_fence_fd(uint64_t fence_id)
{
   return virgl_fence_get_fd(fence_id);
}

static int virgl_renderer_context_attach_in_fence(struct virgl_context *ctx,
                                                  uint64_t fence_id)
{
   int ret;

   /*
    * FD will be -1 in two cases:
    *
    *    1. Fence was signalled and retired.
    *    2. Fence ID is invalid. Virglrenderer doesn't take responsibility
    *       for handling invalid fences and assumes that all supplied fence
    *       IDs are always valid. It's caller's responsibility to validate
    *       fence IDs.
    */
   int fd = virgl_fence_get_fd(fence_id);
   if (fd < 0)
      return 0;

   ret = attach_in_fence_fd(ctx, fd);
   if (ret)
      virgl_error("%s: sync_accumulate failed for fence_id=%" PRIu64 " err=%d\n",
                  __func__, fence_id, ret);

   return ret;
}

static int virgl_renderer_context_attach_in_fences(struct virgl_context *ctx,
                                                   uint64_t *fence_ids,
                                                   uint32_t num_fences)
{
   TRACE_FUNC();

   if (!ctx->supports_fence_sharing)
      return -EINVAL;

   for (uint32_t i = 0; i < num_fences; i++) {
      int ret = virgl_renderer_context_attach_in_fence(ctx, fence_ids[i]);
      if (ret)
         return ret;
   }

   return 0;
}

int virgl_renderer_submit_cmd2(void *buffer,
                               int ctx_id,
                               int ndw,
                               uint64_t *in_fence_ids,
                               uint32_t num_in_fences)
{
   TRACE_FUNC();
   struct virgl_context *ctx = virgl_context_lookup(ctx_id);
   if (!ctx)
      return EINVAL;

   if (((uintptr_t)buffer & 3) != 0)
      return EFAULT;

   if (ndw < 0 || (unsigned)ndw > UINT32_MAX / sizeof(uint32_t))
      return EINVAL;

   if (num_in_fences) {
      int err = virgl_renderer_context_attach_in_fences(ctx, in_fence_ids, num_in_fences);
      if (err)
         return err;
   }

   return ctx->submit_cmd(ctx, buffer, (uint32_t)ndw * sizeof(uint32_t));
}

int virgl_renderer_get_dev_fd(int ctx_id)
{
   struct virgl_context *ctx = virgl_context_lookup(ctx_id);
   if (!ctx)
      return -EINVAL;

   if (!ctx->get_device_fd)
      return -ENODEV;

   return ctx->get_device_fd(ctx);
}
