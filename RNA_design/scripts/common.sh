#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[[ -f local.env ]] || { echo 'Copy local.env.example to local.env and edit the paths.' >&2; exit 1; }
source local.env
RUN="${RUN:-$REPO/runs/reproduction_01}"
PREP="${PREP:-$RUN/prepared}"
# Keep any ViennaRNA scratch files inside this project, not the system /tmp.
mkdir -p "$RUN/work"
export TMPDIR="$RUN/work"
export REPO RUN PREP
need() { for f in "$@"; do [[ -e "$f" ]] || { echo "Missing input: $f" >&2; exit 1; }; done; }
