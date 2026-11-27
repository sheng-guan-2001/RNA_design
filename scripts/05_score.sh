#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
GENERATED_CSV="${GENERATED_CSV:-$RUN/v3_generated.csv}"
need "$GENERATED_CSV" "$HEG_FASTA"
[[ ! -e "$RUN/v3_scored.csv" ]] || { echo 'Score file already exists; choose another RUN.'; exit 1; }
cd "$REPO/CodonTransformer-main"
python -u -m conditional_model.score_m4 --generated "$GENERATED_CSV" --heg-fasta "$HEG_FASTA" --output "$RUN/v3_scored.csv" --workers 12 "$@"
