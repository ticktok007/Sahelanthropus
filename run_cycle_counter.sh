#!/usr/bin/env bash
# run_cycle_counter.sh — Run a RISC-V binary under QEMU with the cost plugin
#
# Usage:
#   ./run_cycle_counter.sh <binary>                    # prints JSON to stdout
#   ./run_cycle_counter.sh <binary> <output.json>      # writes JSON to file
#   cache_size=8192 line_size=128 ./run_cycle_counter.sh <binary>
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Validate arguments ----
if [ $# -lt 1 ]; then
    echo "Usage: $0 <riscv-binary> [output.json]" >&2
    exit 1
fi

BINARY="$1"
OUT_JSON="${2:-}"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: binary not found: $BINARY" >&2
    exit 1
fi
if [ ! -x "$BINARY" ]; then
    echo "ERROR: binary is not executable: $BINARY" >&2
    exit 1
fi

# ---- Locate QEMU ----
QEMU_BIN="$(find ./qemu/build -name "qemu-riscv64" -type f -executable 2>/dev/null | head -n 1)"
if [ -z "$QEMU_BIN" ]; then
    echo "ERROR: qemu-riscv64 not found under ./qemu/build" >&2
    echo "       Run ./build_qemu.sh first." >&2
    exit 1
fi

# ---- Build plugin if stale or missing ----
if [ ! -f "./cycle_counter.so" ] || \
   [ "./cycle_counter.c" -nt "./cycle_counter.so" ] || \
   [ "./build_cycle_counter.sh" -nt "./cycle_counter.so" ]; then
    echo "cycle_counter.so is missing or out of date — rebuilding ..."
    ./build_cycle_counter.sh
fi

# ---- Build plugin argument string ----
CACHE_SIZE="${cache_size:-4096}"
LINE_SIZE="${line_size:-64}"

PLUGIN_ARGS="cache_size=${CACHE_SIZE},line_size=${LINE_SIZE}"

if [ -n "$OUT_JSON" ]; then
    mkdir -p "$(dirname "$OUT_JSON")"
    PLUGIN_ARGS="${PLUGIN_ARGS},output=${OUT_JSON}"
else
    PLUGIN_ARGS="${PLUGIN_ARGS},stdout=1"
fi

# ---- Run ----
echo "Running: $QEMU_BIN -plugin ./cycle_counter.so,${PLUGIN_ARGS} $BINARY" >&2
exec "$QEMU_BIN" \
    -plugin "./cycle_counter.so,${PLUGIN_ARGS}" \
    "$BINARY"
# exec preserves the guest program's exit status