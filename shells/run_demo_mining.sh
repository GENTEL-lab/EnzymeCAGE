#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${DATA_DIR:-dataset/demo}"
P2RANK_HOME="${P2RANK_HOME:-tools/p2rank_2.5.1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/full/seed_42}"
MODEL_NAME="${MODEL_NAME:-best_model.pth}"
THREADS="${THREADS:-4}"
ENV_NAME="${ENV_NAME:-enzymecage}"

PYTHON_CMD=()

resolve_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$ROOT_DIR/$path"
  fi
}

ensure_python_cmd() {
  local candidate

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_CMD=("${PYTHON_BIN}")
    return
  fi

  candidate="${HOME}/miniconda3/envs/${ENV_NAME}/bin/python"
  if [[ -x "$candidate" ]]; then
    PYTHON_CMD=("$candidate")
    return
  fi

  candidate="${HOME}/anaconda3/envs/${ENV_NAME}/bin/python"
  if [[ -x "$candidate" ]]; then
    PYTHON_CMD=("$candidate")
    return
  fi

  if command -v conda >/dev/null 2>&1; then
    PYTHON_CMD=("conda" "run" "-n" "${ENV_NAME}" "python")
    return
  fi

  echo "Unable to locate Python for env '${ENV_NAME}'." >&2
  echo "Set PYTHON_BIN=/path/to/python or add conda to PATH." >&2
  exit 1
}

require_java17() {
  local version_line
  local major

  if ! command -v java >/dev/null 2>&1; then
    echo "java is not on PATH. Install OpenJDK 17+ first." >&2
    exit 1
  fi

  version_line="$(java -version 2>&1 | head -n 1)"
  major="$(printf '%s\n' "$version_line" | sed -E 's/.*version "([0-9]+)(\.[0-9]+)?(\.[0-9]+)?".*/\1/')"

  if [[ -z "$major" || ! "$major" =~ ^[0-9]+$ || "$major" -lt 17 ]]; then
    echo "Java 17+ is required by P2Rank. Current version: $version_line" >&2
    exit 1
  fi
}

ensure_python_cmd

DATA_DIR_ABS="$(resolve_path "$DATA_DIR")"
P2RANK_HOME_ABS="$(resolve_path "$P2RANK_HOME")"
CHECKPOINT_DIR_ABS="$(resolve_path "$CHECKPOINT_DIR")"
RESULT_PATH="${DATA_DIR_ABS}/predictions/mining_${MODEL_NAME%.pth}_ranked.csv"

if [[ ! -d "$DATA_DIR_ABS" ]]; then
  echo "Dataset directory not found: $DATA_DIR_ABS" >&2
  exit 1
fi

if [[ ! -d "$P2RANK_HOME_ABS" ]]; then
  echo "P2Rank directory not found: $P2RANK_HOME_ABS" >&2
  echo "Download it first or override P2RANK_HOME=/path/to/p2rank_2.5.1." >&2
  exit 1
fi

require_java17

cd "$ROOT_DIR"

echo "Running EnzymeCAGE mining demo"
echo "data_dir: $DATA_DIR_ABS"
echo "p2rank_home: $P2RANK_HOME_ABS"
echo "checkpoint_dir: $CHECKPOINT_DIR_ABS"
echo "model_name: $MODEL_NAME"

"${PYTHON_CMD[@]}" scripts/run_mining_pipeline.py \
  --data_dir "$DATA_DIR_ABS" \
  --p2rank_home "$P2RANK_HOME_ABS" \
  --checkpoint_dir "$CHECKPOINT_DIR_ABS" \
  --model_name "$MODEL_NAME" \
  --threads "$THREADS"

echo "Ranked results: $RESULT_PATH"
