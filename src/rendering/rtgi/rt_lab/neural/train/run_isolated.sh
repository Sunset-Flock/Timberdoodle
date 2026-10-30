#!/usr/bin/env bash
# Runs the Debug exe on a test plan in an ISOLATED working directory, for when the user's own app is running.
# Their exe (RelWithDebInfo) and ours share the repo root's caches (spv/, spv_raw/, imgui.ini) and hot reload shaders from
# ./src, so we run from a private copy of src + settings with junctions to the big read-only folders. Results (the plan's
# "out" folder, e.g. perf_tests/...) are copied back into the repo afterwards.
#
#   run_isolated.sh <plan.json> <log file> [exe]
#
# Junctions are removed with `cmd /c rmdir` ONLY (rm -rf would follow them into the real folders).
set -u
PLAN="$1"
LOG="$2"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
EXE="${3:-$REPO/build/cl-x86_64-windows-msvc/bin/Debug/TimberDoodle.exe}"
ISO="$REPO/../tido_isolated_$$"

mkdir -p "$ISO"
cp -r "$REPO/src" "$ISO/src"
cp -r "$REPO/settings" "$ISO/settings"
for d in assets deps media; do
    cmd //c mklink //J "$(cygpath -w "$ISO/$d")" "$(cygpath -w "$REPO/$d")" > /dev/null
done

(cd "$ISO" && "$EXE" --test-plan "$(cygpath -w "$PLAN")" > "$LOG" 2>&1)
STATUS=$?

# copy results back (everything the run created outside the copied / linked folders)
for d in "$ISO"/*/; do
    name="$(basename "$d")"
    case "$name" in src|settings|assets|deps|media|spv|spv_raw|tg_dbg_cache) continue ;; esac
    mkdir -p "$REPO/$name"
    cp -r "$d"/. "$REPO/$name"/
done

for d in assets deps media; do
    cmd //c rmdir "$(cygpath -w "$ISO/$d")" > /dev/null
done
rm -rf "$ISO"
echo "isolated run exit $STATUS (log: $LOG)"
exit $STATUS
