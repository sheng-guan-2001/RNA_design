#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
python -u scripts/train_from_config.py --model-path "$BASE_MODEL" --tokenizer-path "$TOKENIZER" --prepared "$PREP" --output "$RUN/train_m4" "$@"
