# RNA_design

Region-aware multi-objective synonymous mRNA sequence design using local structure optimization and dual-condition CodonTransformer.

## 1. Installation

Python 3.10 (Linux, CUDA 12.1 for the pinned PyTorch build).

```bash
pip install -r requirements.txt
```

Install [ViennaRNA](https://www.tbi.univie.ac.at/RNA/) command-line tools (`RNAfold`, `RNAplfold`) if not already available:

```bash
conda install -c conda-forge -c bioconda viennarna
```

## 2. Data and pretrained model

### 2.1 Expression datasets (Stage 1)

The following three experimental expression datasets are used to evaluate local RNA folding and accessibility in E. coli. They are not used to train the conditional sequence generator.

| Dataset | Expression measurement | Download |
|---|---|---|
| Nieuwkoop 2023 | Red fluorescent protein (mRFP) expression | [mRFP_Expression.csv](https://github.com/Sanofi-Public/CodonBERT/blob/master/benchmarks/CodonBERT/data/fine-tune/mRFP_Expression.csv) |
| Schmitz 2021 | Green fluorescent protein (GFP) expression | [Figshare](https://figshare.com/s/51c4820b2bee85c94007) |
| Boël 2016 | Protein expression score (0–5) | [PMC supplementary data](https://pmc.ncbi.nlm.nih.gov/articles/PMC5054687/) — `Boel2015NESG6348ProteinDataset.csv` |

For Nieuwkoop, the CSV is a redistributed copy; consult the original publication for provenance. For Schmitz, the Figshare link is cited in the original article's Data Availability section; direct access could not be verified. For Boël, locate the named CSV in the original article's supplementary materials.

The raw datasets require preprocessing and sequence/label alignment before validation; they are not the exact processed input files used in our experiments.

If included in this repository, the Stage 1 validation scripts are located in `part_Verification/`.

### 2.2 Training datasets and pretrained model (Stage 2)

| Resource | Download |
|---|---|
| CodonTransformer training data (`dataset.csv`) | [Hugging Face](https://huggingface.co/datasets/adibvafa/CodonTransformer) / [Zenodo](https://zenodo.org/records/12509224) |
| ColiFormer data (`CAI.csv`, `finetune_set.json`, test data) | [Hugging Face](https://huggingface.co/datasets/saketh11/ColiFormer-Data) |
| CodonTransformer pretrained model and tokenizer | [Hugging Face](https://huggingface.co/adibvafa/CodonTransformer) |
| E. coli CAI reference (`ecol.heg.fasta`) | [CodonAdaptationIndex](https://github.com/Benjamin-Lee/CodonAdaptationIndex) (example sequences; verify against the original reference) |

Copy `local.env.example` to `local.env` and set local paths.

Use the original audited ColiFormer candidate, taxonomy and reserved-test CSV files, or rebuild them using `create_train_dataset/audit_coliformer_training.py` (see `--help`). These audit files are not upstream downloads.

To reuse prepared data, set `PREP` in `local.env` and skip step 1. Use the original reserved-test split for exact reproduction.

## 3. Run

Run from the repository root in the same Python environment.

| Step | Command | Main source |
|---|---|---|
| 1. Prepare data | `bash scripts/01_prepare_data.sh` | `create_train_dataset/prepare_ecoli_codontransformer_fixed.py` |
| 2. Train M4 | `bash scripts/02_train.sh` | `conditional_model/train_m4.py` |
| 3. Optimize local prefixes | `bash scripts/03_optimize_prefix.sh` | `m4_cai_aware_local/prepare_cai_aware_local.py` |
| 4. Generate synonymous CDS | `bash scripts/04_generate.sh --execute` | confirmed Q95/CAI-aware generator |
| 5. Score sequences | `bash scripts/05_score.sh` | `conditional_model/score_m4.py` |

After training, set `CHECKPOINT` to the selected M4 checkpoint bundle.

Before step 4, use `bash scripts/inspect_v3.sh` to check the original generator and set `V3_MODULE` in `local.env`. Running `bash scripts/04_generate.sh` previews its command.

If the generator writes a different CSV name, set `GENERATED_CSV` to that file before scoring.

Original training and target configs are in `configs/` when recovered from the run. Set `LOCAL_TARGETS` and `FINAL_TARGETS` accordingly. The exact final V3 generator and search parameters still need confirmation from the original run logs.

Results are saved under `runs/reproduction_01/` by default.

This repository contains core design code, not data or trained weights. The original CodonTransformer license and attribution are retained.

Computational scores are not experimental expression measurements.
