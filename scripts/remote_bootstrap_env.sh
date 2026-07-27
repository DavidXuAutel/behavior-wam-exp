#!/bin/bash
# Create isolated training env for MoT-WAM (2x H100).
# Do not overwrite the kairos conda env.
set -euo pipefail

export PATH=/home/a25689/bin:${PATH}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-/home/a25689/micromamba}
export MAMBA_EXE=${MAMBA_EXE:-/home/a25689/bin/micromamba}

ENV=mot-wam
LOG_DIR=/home/a25689/behavior-wam-exp/reports
mkdir -p "${LOG_DIR}"

if ! micromamba env list | grep -E "(^|[[:space:]])${ENV}([[:space:]]|$)" >/dev/null; then
  micromamba create -y -n "${ENV}" python=3.10
else
  echo "env ${ENV} already exists"
fi

# Prefer micromamba run to avoid activate + set -u interactions.
micromamba run -n "${ENV}" python -m pip install -U pip
micromamba run -n "${ENV}" python -m pip install \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu126
micromamba run -n "${ENV}" python -m pip install \
  pyyaml numpy pandas pyarrow tqdm rich einops safetensors

micromamba run -n "${ENV}" python -c \
  'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count()); assert torch.cuda.is_available(); assert torch.cuda.device_count() >= 2; print("OK")'
