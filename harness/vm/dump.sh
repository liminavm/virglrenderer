#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright © 2026 the limina authors
#
# Ask a running capture for its dump, then check what came back.
#
#   dump.sh venus | dump.sh vrend
#
# The check is not a formality: a corpus is only worth pinning as a fixture if it is
# structurally sound, and the failures worth catching (a lost head record, an orphan context, a
# miscounted header) all read as a working capture until something tries to replay it.
set -euo pipefail
cd "$(dirname "$0")"
RIG="$(pwd)"

case "${1:-}" in
  venus) FIFO="$RIG/captures/venus.fifo"; OUT="$RIG/captures/venus.vkrc" ;;
  vrend) FIFO="$RIG/captures/vrend.fifo"; OUT="$RIG/captures/vrend.bin" ;;
  *) echo "usage: dump.sh {venus|vrend}" >&2; exit 2 ;;
esac

[ -p "$FIFO" ] || { echo "no FIFO at $FIFO — is a capture running?" >&2; exit 1; }
echo x > "$FIFO"

for _ in $(seq 1 50); do
  [ -s "$OUT" ] && break
  sleep 0.2
done
[ -s "$OUT" ] || { echo "nothing written to $OUT" >&2; exit 1; }

ls -lh "$OUT"
if [ "$1" = venus ]; then
  python3 "$RIG/../replay/vkr-record-decode.py" "$OUT" --check
  python3 "$RIG/../replay/vkr-record-decode.py" "$OUT"
else
  python3 "$RIG/../replay/vrend-trace-decode.py" "$OUT"
fi
