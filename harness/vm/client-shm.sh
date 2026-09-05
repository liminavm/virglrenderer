#!/bin/bash
# shm-only workload: a GTK4 terminal with every GPU path switched off, so its surface reaches
# the compositor as a wl_shm buffer. The client issues no GL at all; everything the corpus
# records is the shell uploading and sampling that buffer.
export GSK_RENDERER=cairo
export GDK_DEBUG=gl-disable
export LIBGL_ALWAYS_SOFTWARE=1
setsid ptyxis -x "bash -c 'i=0; while :; do i=\$((i+1)); printf \"%s  shm corpus line %06d  %s\n\" \"\$(date +%H:%M:%S.%N)\" \$i \"\$(head -c 40 /dev/urandom | base64 | tr -d =)\"; sleep 0.08; done'" &
sleep 3
echo "ptyxis pids: $(pgrep -c ptyxis)"
