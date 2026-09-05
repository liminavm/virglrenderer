#!/bin/bash
# Vulkan client on the stock guest. Its Vulkan goes out through venus; the classic tracer on
# this guest records the OTHER half -- the GL shell importing and compositing the buffer the
# Vulkan client produced. That cross-path import is the point of this corpus.
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/virtio_icd.aarch64.json
setsid vkcube --wsi wayland &
sleep 3
echo "vkcube pids: $(pgrep -c vkcube)"
