/*
 * Copyright 2021 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef RENDER_STATE_H
#define RENDER_STATE_H

#include "render_common.h"

bool
render_state_init(uint32_t init_flags);

void
render_state_fini(void);

/* limina M9.3 diagnostics: thread-safe vkr context-table dump (takes the
 * renderer lock; no-op log line when the renderer isn't initialized). */
void
render_state_limina_dump_state(void);

/* gkvm snapshot-replay (limina M9.3 P1): journal export + replay, all under
 * the renderer lock. See vkr_renderer.c for the replay contract. */
bool
render_state_limina_journal_export(uint32_t ctx_id, void **out_buf, size_t *out_size);

uint64_t
render_state_limina_journal_seq(uint32_t ctx_id);

void
render_state_limina_journal_unpin(uint32_t ctx_id, uint64_t key);

bool
render_state_limina_replay_begin(uint32_t ctx_id);

bool
render_state_limina_replay_submit(uint32_t ctx_id, void *cmd, uint32_t size);

bool
render_state_limina_replay_ring_cmd(uint32_t ctx_id,
                                    uint64_t ring_id,
                                    void *cmd,
                                    uint32_t size);

bool
render_state_limina_replay_end(uint32_t ctx_id);

bool
render_state_create_context(struct render_context *ctx,
                            uint32_t flags,
                            uint32_t name_len,
                            const char *name);

void
render_state_destroy_context(uint32_t ctx_id);

bool
render_state_submit_cmd(uint32_t ctx_id, void *cmd, uint32_t size);

bool
render_state_submit_fence(uint32_t ctx_id,
                          uint32_t flags,
                          uint64_t ring_idx,
                          uint64_t fence_id);

bool
render_state_create_resource(uint32_t ctx_id,
                             uint32_t res_id,
                             uint64_t blob_id,
                             uint64_t blob_size,
                             uint32_t blob_flags,
                             enum virgl_resource_fd_type *out_fd_type,
                             int *out_res_fd,
                             uint32_t *out_map_info,
                             struct virgl_resource_vulkan_info *out_vulkan_info,
                             uint32_t *out_iosurface_id,
                             uint64_t *out_map_ptr);

bool
render_state_import_resource(uint32_t ctx_id,
                             uint32_t res_id,
                             enum virgl_resource_fd_type fd_type,
                             int fd,
                             uint64_t size,
                             uint32_t iosurface_id,
                             uint64_t map_ptr);

void
render_state_destroy_resource(uint32_t ctx_id, uint32_t res_id);

#endif /* RENDER_STATE_H */
