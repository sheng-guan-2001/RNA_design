#!/usr/bin/env bash
set -euo pipefail
python --version
python -m pip check
RNAfold --version
RNAplfold --version
python -c 'import RNA, torch, transformers, Bio, numpy, pandas; print("ViennaRNA", RNA.__version__, "PyTorch", torch.__version__, "CUDA", torch.cuda.is_available())'
