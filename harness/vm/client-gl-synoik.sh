#!/bin/bash
# GL client on the Vulkan-compositor guest: the mirror of vkclient.sh. The client draws through
# classic virgl and the compositor imports the result into its Vulkan world, so the venus
# recorder captures the import side of a cross-path buffer.
#
# Two things an SSH shell gets wrong on its own, both silently:
#   * The zink/virgl environment is the session's, not the shell's. Without it the GL stack falls
#     back to llvmpipe -- the client draws, the window appears, the capture completes, and none of
#     it touched the path being measured. Take it from the compositor process itself.
#   * WAYLAND_DISPLAY is absent from the compositor's own environ, because it is the compositor.
#     Find the socket instead of assuming wayland-0; synoik's is wayland-1.
# glxgears is not a route here: it is GLX, and there is no X server on this guest.
set -u
COMP=$(pgrep -u "$(id -u)" -x synoik || pgrep -u "$(id -u)" -x gnome-shell || true)
COMP=${COMP%%$'\n'*}
if [ -n "$COMP" ]; then
  echo "compositor pid=$COMP ($(tr -d '\0' < /proc/$COMP/comm))"
  while IFS= read -r -d '' kv; do
    case "$kv" in
      LD_LIBRARY_PATH=*|MESA_*|GALLIUM_*|ZINK_*|LIBGL_*|EGL_*|VK_*|XDG_RUNTIME_DIR=*)
        export "$kv"; echo "  inherited: $kv" ;;
    esac
  done < /proc/$COMP/environ
else
  echo "WARNING: no compositor found; the client would fall back to llvmpipe" >&2
fi
: "${XDG_RUNTIME_DIR:=/run/user/$(id -u)}"; export XDG_RUNTIME_DIR
for s in "$XDG_RUNTIME_DIR"/wayland-*; do
  case "$s" in *.lock) continue ;; esac
  [ -S "$s" ] && { export WAYLAND_DISPLAY="$(basename "$s")"; break; }
done
echo "WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-UNSET}"
setsid glmark2-wayland --run-forever -b shading -b texture -b build > /tmp/gl.log 2>&1 &
sleep 6
echo "glmark2 pids: $(pgrep -c glmark2)"
echo "--- renderer the client actually got:"
grep -iE 'GL_RENDERER|GL_VENDOR|GL_VERSION' /tmp/gl.log || head -c 400 /tmp/gl.log
