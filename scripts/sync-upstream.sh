#!/usr/bin/env bash
# Moves the pin to current upstream master, re-applies patches/ (3-way), regenerates them, rebuilds, runs the NVFP4 tests.
# Does NOT commit or push; the owner reviews docs/PINNED_COMMIT and patches/ and commits.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/llama.cpp"
"$ROOT/scripts/fetch-llama.sh"
cd "$SRC"
git fetch origin master
NEW="$(git rev-parse origin/master)"
git stash push -u -q -m "fp4-unlock patches before sync"   # current patched state, in case the re-apply fails
git checkout -q --detach "$NEW"
git apply --3way "$ROOT"/patches/*.patch || { echo "patches need manual rebase on $NEW (stash holds the old state)"; exit 1; }
git stash drop -q
echo "$NEW" > "$ROOT/docs/PINNED_COMMIT"
"$ROOT/scripts/regen-patches.sh"
echo "pinned $NEW"
"$ROOT/scripts/build.sh" p3
"$ROOT/build-p3/bin/test-backend-ops" test -b CUDA0 -o MUL_MAT -p "nvfp4" | tail -3
