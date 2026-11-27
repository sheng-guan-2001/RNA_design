#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${V3_MODULE:?Set V3_MODULE in local.env after checking the original generation entry point.}"
: "${CHECKPOINT:?Set CHECKPOINT to the trained M4 bundle.}"
GENERATED_CSV="${GENERATED_CSV:-$RUN/v3_generated.csv}"
need "$CHECKPOINT" "$FINAL_TARGETS" "$RUN/v3_local/02_q95_local_prefixes.csv"
args=(--module "$V3_MODULE" --checkpoint "$CHECKPOINT" --prefixes "$RUN/v3_local/02_q95_local_prefixes.csv" --targets "$FINAL_TARGETS" --reference "$PREP/14_reserved_test_four_objectives.csv" --output "$GENERATED_CSV")
if [[ -n "${V3_ORIGINAL:-}" && -f "$V3_ORIGINAL/03_generated_CAI085.manifest.json" ]]; then
 args+=(--manifest "$V3_ORIGINAL/03_generated_CAI085.manifest.json")
fi
# Default is a dry run; pass --execute only after inspecting the command.
python scripts/generate_v3.py "${args[@]}" "$@"
