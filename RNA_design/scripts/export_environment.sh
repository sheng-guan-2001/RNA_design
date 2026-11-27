#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p local_configs
for env in protein_RNA CodonTransformer; do
 conda run -n "$env" python -m pip freeze > "local_configs/${env}_pip_freeze.txt"
 conda list -n "$env" --explicit > "local_configs/${env}_conda_explicit.txt"
 conda run -n "$env" python --version > "local_configs/${env}_python.txt"
 done
conda run -n protein_RNA RNAfold --version > local_configs/RNAfold_version.txt
conda run -n protein_RNA RNAplfold --version > local_configs/RNAplfold_version.txt
