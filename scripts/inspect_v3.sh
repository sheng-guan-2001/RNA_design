#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ -n "${V3_ORIGINAL:-}" && -d "$V3_ORIGINAL" ]]; then
 for f in 03_generated_CAI085.manifest.json generate_CAI085.log local.log; do
  if [[ -f "$V3_ORIGINAL/$f" ]]; then echo "=== $f ==="; head -n 40 "$V3_ORIGINAL/$f"; fi
 done
fi
cd "$REPO/CodonTransformer-main"
for m in m4_q95_benchmark.generate_q95_m4 m4_extreme_cai08.generate_q95_m4; do
 echo "=== $m ==="
 python -m "$m" --help
 done
