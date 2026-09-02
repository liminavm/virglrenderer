/**************************************************************************
 *
 * Copyright (C) 2026 Gustavo Noronha Silva
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

/* The classic re-creation journal, driven with raw command dwords: no renderer, no GL. */

#include <check.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "virgl_protocol.h"
#include "vrend/vrend_journal.h"

#define KLASS_CREATE 6u

/* One exported entry: 8-byte seq, cmd_type, klass + 3 pad, 8-byte ring key, size, payload. */
struct exported_entry {
   uint32_t cmd_type;
   uint8_t klass;
   uint32_t size;
   const uint32_t *payload;
};

static uint32_t export_entries(const struct vrend_journal *j, struct exported_entry *out,
                               uint32_t max, void **buf_out)
{
   void *buf;
   size_t size;
   ck_assert(vrend_journal_export(j, &buf, &size));
   ck_assert_uint_ge(size, 16);
   const uint8_t *p = buf;
   uint32_t magic, version, count;
   memcpy(&magic, p, 4);
   memcpy(&version, p + 4, 4);
   memcpy(&count, p + 8, 4);
   ck_assert_uint_eq(magic, 0x524a4b56u);
   ck_assert_uint_eq(version, 1);
   p += 16;
   for (uint32_t i = 0; i < count && i < max; i++) {
      p += 8; /* seq */
      memcpy(&out[i].cmd_type, p, 4);
      p += 4;
      out[i].klass = *p;
      p += 4; /* klass + pad */
      p += 8; /* ring key */
      memcpy(&out[i].size, p, 4);
      p += 4;
      out[i].payload = (const uint32_t *)p;
      p += out[i].size;
   }
   ck_assert_uint_le((size_t)(p - (const uint8_t *)buf), size);
   *buf_out = buf;
   return count;
}

static void record_create_codec(struct vrend_journal *j, uint32_t handle)
{
   uint32_t cmd[9] = { VIRGL_CMD0(VIRGL_CCMD_CREATE_VIDEO_CODEC, 0, 8),
                       handle, 12 /* profile */, 1 /* entrypoint */, 1 /* chroma */,
                       0, 1920, 1080, 8 };
   vrend_journal_record(j, cmd, 9);
}

static void record_create_buffer(struct vrend_journal *j, uint32_t handle)
{
   uint32_t cmd[7] = { VIRGL_CMD0(VIRGL_CCMD_CREATE_VIDEO_BUFFER, 0, 6),
                       handle, 20 /* format */, 1920, 1080, 101, 102 };
   vrend_journal_record(j, cmd, 7);
}

static void record_destroy(struct vrend_journal *j, uint32_t cmd_id, uint32_t handle)
{
   uint32_t cmd[2] = { VIRGL_CMD0(cmd_id, 0, 1), handle };
   vrend_journal_record(j, cmd, 2);
}

/* A video codec and its target buffers are created once and used every frame: they
 * must be retained for replay, or the guest decodes into a codec the host no longer
 * has after a restore. */
START_TEST(video_creates_are_retained)
{
   struct vrend_journal *j = vrend_journal_create(1);
   struct vrend_journal_census c;
   struct exported_entry e[4];
   void *buf;

   record_create_codec(j, 7);
   record_create_buffer(j, 9);
   vrend_journal_census(j, &c);
   ck_assert_uint_eq(c.skipped, 0);
   ck_assert_uint_eq(c.live, 2);

   uint32_t count = export_entries(j, e, 4, &buf);
   ck_assert_uint_eq(count, 2);
   ck_assert_uint_eq(e[0].cmd_type, VIRGL_CCMD_CREATE_VIDEO_CODEC);
   ck_assert_uint_eq(e[0].klass, KLASS_CREATE);
   ck_assert_uint_eq(e[0].size, 9 * 4);
   ck_assert_uint_eq(e[0].payload[1], 7);
   ck_assert_uint_eq(e[1].cmd_type, VIRGL_CCMD_CREATE_VIDEO_BUFFER);
   ck_assert_uint_eq(e[1].klass, KLASS_CREATE);
   ck_assert_uint_eq(e[1].size, 7 * 4);
   ck_assert_uint_eq(e[1].payload[1], 9);
   free(buf);

   vrend_journal_destroy(j);
}
END_TEST

/* Destroys tombstone their create by handle, and only that handle. */
START_TEST(video_destroys_prune_by_handle)
{
   struct vrend_journal *j = vrend_journal_create(1);
   struct vrend_journal_census c;

   record_create_codec(j, 7);
   record_create_codec(j, 8);
   record_create_buffer(j, 9);
   record_create_buffer(j, 10);

   record_destroy(j, VIRGL_CCMD_DESTROY_VIDEO_BUFFER, 9);
   vrend_journal_census(j, &c);
   ck_assert_uint_eq(c.live, 3);
   ck_assert_uint_eq(c.pruned, 1);

   /* a buffer handle does not name a codec */
   record_destroy(j, VIRGL_CCMD_DESTROY_VIDEO_CODEC, 10);
   vrend_journal_census(j, &c);
   ck_assert_uint_eq(c.live, 3);

   record_destroy(j, VIRGL_CCMD_DESTROY_VIDEO_CODEC, 7);
   record_destroy(j, VIRGL_CCMD_DESTROY_VIDEO_CODEC, 8);
   record_destroy(j, VIRGL_CCMD_DESTROY_VIDEO_BUFFER, 10);
   vrend_journal_census(j, &c);
   ck_assert_uint_eq(c.live, 0);
   ck_assert_uint_eq(c.pruned, 4);

   vrend_journal_destroy(j);
}
END_TEST

/* The video context hangs off the vrend context, not a sub-context: its creates must
 * not be bracketed by a SET_SUB_CTX at export, whatever sub was current. */
START_TEST(video_creates_are_not_sub_scoped)
{
   struct vrend_journal *j = vrend_journal_create(1);
   struct exported_entry e[4];
   void *buf;

   uint32_t set_sub[2] = { VIRGL_CMD0(VIRGL_CCMD_SET_SUB_CTX, 0, 1), 3 };
   vrend_journal_record(j, set_sub, 2);
   record_create_codec(j, 7);

   /* SET_SUB_CTX itself is retained (latest-wins); the codec create follows it
    * without a second bracket. */
   uint32_t count = export_entries(j, e, 4, &buf);
   ck_assert_uint_eq(count, 2);
   ck_assert_uint_eq(e[0].cmd_type, VIRGL_CCMD_SET_SUB_CTX);
   ck_assert_uint_eq(e[1].cmd_type, VIRGL_CCMD_CREATE_VIDEO_CODEC);
   free(buf);

   vrend_journal_destroy(j);
}
END_TEST

/* Per-frame video commands are transient: re-issued every frame, never retained, and
 * not "unknown" either — the skipped count must keep meaning "a command the journal
 * has no rule for". */
START_TEST(video_frame_commands_are_transient)
{
   struct vrend_journal *j = vrend_journal_create(1);
   struct vrend_journal_census c;

   uint32_t begin[3] = { VIRGL_CMD0(VIRGL_CCMD_BEGIN_FRAME, 0, 2), 7, 9 };
   uint32_t decode[6] = { VIRGL_CMD0(VIRGL_CCMD_DECODE_BITSTREAM, 0, 5), 7, 9, 11, 13, 4096 };
   uint32_t end[3] = { VIRGL_CMD0(VIRGL_CCMD_END_FRAME, 0, 2), 7, 9 };
   vrend_journal_record(j, begin, 3);
   vrend_journal_record(j, decode, 6);
   vrend_journal_record(j, end, 3);

   vrend_journal_census(j, &c);
   ck_assert_uint_eq(c.live, 0);
   ck_assert_uint_eq(c.skipped, 0);

   vrend_journal_destroy(j);
}
END_TEST

static Suite *journal_suite(void)
{
   Suite *s = suite_create("vrend_journal");
   TCase *tc = tcase_create("video");
   tcase_add_test(tc, video_creates_are_retained);
   tcase_add_test(tc, video_destroys_prune_by_handle);
   tcase_add_test(tc, video_creates_are_not_sub_scoped);
   tcase_add_test(tc, video_frame_commands_are_transient);
   suite_add_tcase(s, tc);
   return s;
}

int main(void)
{
   Suite *s = journal_suite();
   SRunner *sr = srunner_create(s);
   srunner_run_all(sr, CK_NORMAL);
   int failed = srunner_ntests_failed(sr);
   srunner_free(sr);
   return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
