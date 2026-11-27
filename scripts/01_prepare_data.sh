#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
COLI="${COLI_DIR:-$DATA_ROOT/dataset/ColiFormer-Data}"
CT_CSV="${CT_CSV:-$DATA_ROOT/dataset/CodonTransformer/dataset.csv}"
CANDIDATE_CSV="${CANDIDATE_CSV:-$COLI/audit_results/05_candidate_training_CDS.csv}"
TAXONOMY_CSV="${TAXONOMY_CSV:-$COLI/audit_results/03_CAI_accessions_NEED_TAXONOMY_CHECK.csv}"
TEST_CSV="${TEST_CSV:-$COLI/audit_results/06_reserved_test_CDS.csv}"
need "$CT_CSV" "$CANDIDATE_CSV" "$TAXONOMY_CSV" "$TEST_CSV" "$COLI/CAI.csv" "$HEG_FASTA"
command -v RNAfold >/dev/null
command -v RNAplfold >/dev/null
[[ ! -d "$PREP" || -z "$(ls -A "$PREP")" ]] || { echo "Prepared output already exists: $PREP; choose a new PREP or skip this step."; exit 1; }
python -u create_train_dataset/prepare_ecoli_codontransformer_fixed.py \
 --ct-csv "$CT_CSV" --coli-dir "$COLI" --candidate-csv "$CANDIDATE_CSV" \
 --cai-csv "$COLI/CAI.csv" --taxonomy-csv "$TAXONOMY_CSV" --test-csv "$TEST_CSV" \
 --heg-fasta "$HEG_FASTA" --organism-map "$REPO/CodonTransformer-main/src/organism2id.pkl" \
 --outdir "$PREP" --temperature 37 --rnaplfold-window 150 --rnaplfold-span 100 --organism-id 51
