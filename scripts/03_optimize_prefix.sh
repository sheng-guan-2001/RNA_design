#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need "$PREP/14_reserved_test_four_objectives.csv" "$LOCAL_TARGETS" "$HEG_FASTA"
cd "$REPO/CodonTransformer-main"
python -u -m m4_cai_aware_local.prepare_cai_aware_local \
 --test-csv "$PREP/14_reserved_test_four_objectives.csv" --targets-json "$LOCAL_TARGETS" \
 --heg-fasta "$HEG_FASTA" --output "$RUN/v3_local" --workers 8 "$@"
