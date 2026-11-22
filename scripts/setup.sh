#!/usr/bin/env bash
# One-time setup: submodule, python env, A-MDM checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."

git submodule update --init --recursive
uv sync

AMDM=third_party/AMDM
if [ ! -f "$AMDM/output/base/amdm_lafan1/model_param.pth" ]; then
  mkdir -p "$AMDM/output/dl"
  # checkpoints from the A-MDM README (Google Drive)
  uv run gdown 1a3emD8C5hN4wbTuPYhI0WytSMCuGoLNs -O "$AMDM/output/dl/LAFAN1_15step.zip"
  uv run gdown 1-ju_XW9JsHrBuLORl8-n7jJiry5C_3Fn -O "$AMDM/output/dl/100STYLE_25step.zip"
  for z in "$AMDM"/output/dl/*.zip; do unzip -o -q "$z" -d "$AMDM"; done   # zips contain output/...
fi
echo "ready: uv run python exp2_viewer.py"
