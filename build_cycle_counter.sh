#!/usr/bin/env bash
# build_cycle_counter.sh — Compile cycle_counter.c into cycle_counter.so
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Locate qemu-plugin.h inside ./qemu
HEADER="$(find ./qemu -name "qemu-plugin.h" -path "*/plugins/*" 2>/dev/null | head -n 1)"
if [ -z "$HEADER" ]; then
    echo "ERROR: qemu-plugin.h not found under ./qemu" >&2
    echo "       Expected location: ./qemu/include/plugins/qemu-plugin.h" >&2
    exit 1
fi

# We need two -I entries:
#   1) The 'plugins/' directory so that #include <qemu-plugin.h> is found here
#      *before* the system /usr/include/qemu-plugin.h (which may be an older API).
#   2) The parent include directory for any other QEMU internal headers.
PLUGIN_DIR="$(dirname "$HEADER")"         # .../include/plugins
INCLUDE_DIR="$(dirname "$PLUGIN_DIR")"    # .../include

echo "Using header : $HEADER"
echo "Plugin dir   : $PLUGIN_DIR"
echo "Include path : $INCLUDE_DIR"

# Resolve glib-2.0 flags
if ! pkg-config --exists glib-2.0; then
    echo "ERROR: glib-2.0 not found via pkg-config" >&2
    exit 1
fi
GLIB_CFLAGS="$(pkg-config --cflags glib-2.0)"
GLIB_LIBS="$(pkg-config --libs glib-2.0)"

CC="${CC:-gcc}"
CFLAGS="-O2 -fPIC -shared -std=gnu11 -Wall -Wextra -Wno-unused-parameter"
CFLAGS="$CFLAGS -fvisibility=hidden"
# Suppress the unused-parameter warning — many QEMU plugin callbacks receive
# vcpu_index or userdata that we intentionally ignore.
CFLAGS="$CFLAGS -Wno-unused-parameter"

echo "Compiling cycle_counter.c ..."
$CC $CFLAGS \
    -I"$PLUGIN_DIR" \
    -I"$INCLUDE_DIR" \
    $GLIB_CFLAGS \
    cycle_counter.c \
    -o cycle_counter.so \
    $GLIB_LIBS

echo "Build succeeded: ./cycle_counter.so"