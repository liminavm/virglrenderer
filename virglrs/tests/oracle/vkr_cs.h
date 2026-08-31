/*
 * SPDX-License-Identifier: MIT
 * Copyright © 2026 the limina authors
 */

/*
 * The command stream venus-protocol's generated renderer encoder writes through, cut down to the
 * reply path.
 *
 * The subproject ships this interface as no-op stubs (`tests/vkr_cs.h`) for a compile-only test.
 * The oracle needs it to actually encode, so this is the real thing -- but only the encode half:
 * the decode and temp-allocation entry points abort rather than return a plausible value, because
 * nothing on the reply path may reach them and a silent zero would look like agreement.
 */

#ifndef VKR_CS_H
#define VKR_CS_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <vulkan/vulkan.h>

typedef uint64_t vkr_object_id;

struct vkr_object {
    union {
        uint64_t u64;
    } handle;
};

struct vkr_cs_encoder {
    uint8_t *cur;
    uint8_t *end;
    uint8_t *base;
    bool fatal;
};

struct vkr_cs_decoder;

static inline void
vn_oracle_encoder_init(struct vkr_cs_encoder *enc, void *buf, size_t cap)
{
    enc->base = (uint8_t *)buf;
    enc->cur = enc->base;
    enc->end = enc->base + cap;
    enc->fatal = false;
}

/* `(size_t)-1` rather than a short count: a reply that overran its buffer wrote nothing
 * trustworthy, and the caller must not diff it against ours as though it had. */
static inline size_t
vn_oracle_encoder_len(const struct vkr_cs_encoder *enc)
{
    return enc->fatal ? (size_t)-1 : (size_t)(enc->cur - enc->base);
}

static inline bool
vkr_cs_encoder_acquire(struct vkr_cs_encoder *enc)
{
    return true;
}

static inline void
vkr_cs_encoder_release(struct vkr_cs_encoder *enc)
{
}

/* `size` is the wire span and `val_size` the host width; the difference is the padding a
 * sub-word scalar sits in. Advancing by `size` while copying `val_size` is what leaves that
 * padding at whatever the buffer was pre-filled with -- which is why the caller pre-fills both
 * sides identically before diffing. */
static inline void
vkr_cs_encoder_write(struct vkr_cs_encoder *enc, size_t size, const void *val, size_t val_size)
{
    assert(val_size <= size);

    if (size > (size_t)(enc->end - enc->cur)) {
        enc->fatal = true;
        return;
    }

    memcpy(enc->cur, val, val_size);
    enc->cur += size;
}

static inline void *
vkr_cs_encoder_get_blob_storage(struct vkr_cs_encoder *enc, size_t offset, size_t size)
{
    if (size > (size_t)(enc->end - enc->cur)) {
        enc->fatal = true;
        return NULL;
    }

    void *storage = enc->cur;
    enc->cur += size;
    return storage;
}

/*
 * Handles.
 *
 * `vkr_cs_handle_indirect_id` is false for every object type on a 64-bit host, so production
 * reads the id straight out of the handle slot -- which is the convention the Rust side already
 * stores. The static assert is what keeps that agreement from going quiet on a host where it
 * stops holding.
 */
static_assert(sizeof(VkInstance) == sizeof(vkr_object_id),
              "the oracle assumes handles are id-wide; see vkr_cs_handle_indirect_id");

static inline bool
vkr_cs_handle_indirect_id(VkObjectType type)
{
    return false;
}

static inline vkr_object_id
vkr_cs_handle_load_id(const void **handle, VkObjectType type)
{
    return *(const vkr_object_id *)handle;
}

static inline void
vkr_cs_handle_store_id(void **handle, vkr_object_id id, VkObjectType type)
{
    *(vkr_object_id *)handle = id;
}

/* The decode half. Reaching any of these from a reply encode is a generator bug, and returning
 * something harmless would hide it behind a passing diff. */
static inline void
vkr_cs_decoder_set_fatal(const struct vkr_cs_decoder *dec)
{
    abort();
}

static inline bool
vkr_cs_decoder_get_fatal(const struct vkr_cs_decoder *dec)
{
    abort();
}

static inline void
vkr_cs_decoder_read(struct vkr_cs_decoder *dec, size_t size, void *val, size_t val_size)
{
    abort();
}

static inline void
vkr_cs_decoder_peek(const struct vkr_cs_decoder *dec, size_t size, void *val, size_t val_size)
{
    abort();
}

static inline struct vkr_object *
vkr_cs_decoder_lookup_object(const struct vkr_cs_decoder *dec, vkr_object_id id, VkObjectType type)
{
    abort();
}

static inline void
vkr_cs_decoder_reset_temp_pool(struct vkr_cs_decoder *dec)
{
    abort();
}

static inline void *
vkr_cs_decoder_alloc_temp(struct vkr_cs_decoder *dec, size_t size)
{
    abort();
}

static inline void *
vkr_cs_decoder_alloc_temp_array(struct vkr_cs_decoder *dec, size_t size, size_t count)
{
    abort();
}

static inline void *
vkr_cs_decoder_get_blob_storage(struct vkr_cs_decoder *dec, size_t size)
{
    abort();
}

#endif /* VKR_CS_H */
