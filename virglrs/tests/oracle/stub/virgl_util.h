/* SPDX-License-Identifier: MIT
 * Copyright © 2026 the limina authors
 *
 * A stand-in for src/virgl_util.h, on the oracle build's include path ahead of the real one.
 *
 * The video serializers use exactly one thing from it -- virgl_error, to say why they refused a
 * stream. The real header reaches virglrenderer.h and mesa's hash table for everything else, and
 * virglrenderer.h wants a virgl-version.h that only meson writes. Building the oracle against a
 * meson build directory would tie a unit test to a tree someone has to have configured first,
 * for a logging function. So the serializers get the declaration and nothing else.
 */
#ifndef VIRGL_UTIL_H
#define VIRGL_UTIL_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>

void virgl_error(const char *fmt, ...);

#endif /* VIRGL_UTIL_H */
