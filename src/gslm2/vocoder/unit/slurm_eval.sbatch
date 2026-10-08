#!/bin/bash

# Synthesize a unit manifest, then evaluate Whisper/Qwen3-Omni/Wav2Vec2 WER and UTMOS.
#
# Usage:
#   sbatch src/metasr_misc/unit_vocoder/slurm_eval.sbatch \
#     WORKDIR VOCODER_NAME EVAL_SET UNIT_MANIFEST
#
# EVAL_SET: ljspeech_test, librispeech_dev-clean_1k_min6s, or
#           librispeech_test-clean_1k_min6s.
# Install runtime packages into the environment used below with:
#   conda activate metasr && uv pip install -e ".[unit-vocoder]"

#SBATCH --job-name=unit-vocoder-eval
#SBATCH --output=slurm/%x_%j/output.log
#SBATCH --error=slurm/%x_%j/error.log
#SBATCH --account=devai
#SBATCH --qos=h100_devai_high
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=720
#SBATCH --mem=400G

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: sbatch $0 WORKDIR VOCODER_NAME EVAL_SET UNIT_MANIFEST" >&2
    exit 2
fi

WORKDIR=$1
VOCODER_NAME=$2
EVAL_SET=$3
UNIT_MANIFEST=$4
ASR_MANIFEST_ROOT=/datasets/metaspeechssl/yosukeh/assets/manifests

case "${EVAL_SET}" in
    ljspeech_test)
        ASR_MANIFEST=${ASR_MANIFEST_ROOT}/ljspeech/ljspeech_test_asr.jsonl
        ;;
    librispeech_dev-clean_1k_min6s)
        ASR_MANIFEST=${ASR_MANIFEST_ROOT}/librispeech/librispeech_dev-clean_1k_min6s_asr.jsonl
        ;;
    librispeech_test-clean_1k_min6s)
        ASR_MANIFEST=${ASR_MANIFEST_ROOT}/librispeech/librispeech_test-clean_1k_min6s_asr.jsonl
        ;;
    *)
        echo "Unknown EVAL_SET: ${EVAL_SET}" >&2
        exit 2
        ;;
esac

EVAL_DIR=${WORKDIR}/eval/${VOCODER_NAME}/${EVAL_SET}
GENERATIONS=${EVAL_DIR}/audio
mkdir -p "${EVAL_DIR}"

echo "---- Start date: $(date)"
echo "Vocoder: ${WORKDIR}/${VOCODER_NAME}"
echo "Unit manifest: ${UNIT_MANIFEST}"
echo "ASR manifest: ${ASR_MANIFEST}"
echo "Evaluation directory: ${EVAL_DIR}"

source /home/yosukeh/miniconda3/etc/profile.d/conda.sh
conda activate metasr_fs2
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
python -c "import accelerate, torchaudio, utmos_pytorch, whisper; from transformers import Qwen3OmniMoeProcessor"

srun python -m metasr_misc.unit_vocoder.evaluation "${UNIT_MANIFEST}" "${ASR_MANIFEST}"
srun python -m metasr_misc.unit_vocoder.inference \
    "${GENERATIONS}" "${WORKDIR}/${VOCODER_NAME}" "${UNIT_MANIFEST}"
# srun python -m metasr_misc.unit_vocoder.wer.torchaudio \
#     "${GENERATIONS}" "${ASR_MANIFEST}" "${EVAL_DIR}/asr_wav2vec2_large_lv60k_960h.jsonl"
# srun python -m metasr_misc.unit_vocoder.wer.whisper \
#     "${GENERATIONS}" "${ASR_MANIFEST}" "${EVAL_DIR}/asr_whisper_large-v3.jsonl"
srun python -m metasr_misc.unit_vocoder.wer.qwen3_omni \
    "${GENERATIONS}" "${ASR_MANIFEST}" "${EVAL_DIR}/asr_qwen3_omni_30b_a3b.jsonl"
srun python -m metasr_misc.unit_vocoder.mos.utmos \
    "${GENERATIONS}" "${ASR_MANIFEST}" "${EVAL_DIR}/utmos_v1.jsonl"

echo "---- End date: $(date)"
