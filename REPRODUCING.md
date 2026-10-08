# Reproducing the GSLM2 paper

This file maps each paper table/figure to the commands that produce it.

## Training

| Artifact | Config | Command |
|---|---|---|
| SpidR 2.0 encoder (LS-960h) | `configs/encoder/spidr2.0-ls960.yaml` | `gslm2-train -A <acct> -N 2 -G 8 -Q <qos> --dump <dir> configs/encoder/spidr2.0-ls960.yaml` |
| SpidR 2.0 encoder (VoxPopuli En/Fr/De) | `configs/encoder/spidr2.0-voxpopuli-{en,fr,de}.yaml` | same, 2 nodes |
| Single-stream SLM | `configs/slm/ll-*.yaml`, `configs/slm/voxpopuli-*.yaml` | `gslm2-train ...` (nodes per config) |
| GSLM2 multi-stream SLM | `configs/gslm2/ll-*.yaml` | `gslm2-train ...` (extract units first, see below) |
| Text LM | `configs/textlm/*.yaml` | `gslm2-train ...` (train BPE first, see `DATA.md`) |
| Scaling (2.5B/6.5B) | `configs/conv/*.yaml` | needs FSDP (not yet in the native trainer) + data access |
| Duration predictor | `configs/duration/{single,multi}-stream.yaml` | `gslm2-duration-predictor --config ...` |
| Unit vocoder | `configs/vocoder/{single,multi}-stream.yaml` | `gslm2-unit-vocoder --config ...` |

## Unit extraction (GSLM2 / vocoder / duration inputs)

```bash
# single-stream deduped (SLM + discrete evals)
gslm2-extract-units --config configs/extract/spidr2-l7-codes-dedup.yaml
# six-stream cross-stream deduped (GSLM2)
gslm2-extract-units --config configs/extract/spidr2-l5to10-codes-multistream.yaml
# k-means baseline units (SpidR L6 / HuBERT L11)
gslm2-extract-units --config configs/extract/spidr-l6-kmeans.yaml
```

## Encoder evaluation (`tab:encoder-eval`, `fig:frontend-eval`)

```bash
CKPT=/path/to/spidr2.0-seed0-final.pt
# ABX triphone (continuous + discrete), dev-clean / dev-other
gslm2-eval-abx $CKPT abx.csv dev-clean.tsv triphone-dev-clean.item \
  --speaker_mode all --context_mode within --layers 10,11,12
gslm2-eval-abx $CKPT abx-discrete.csv dev-clean.tsv triphone-dev-clean.item \
  --speaker_mode all --context_mode within --layers 10,11,12 --codebook_proba
# MAP word discovery
gslm2-eval-map $CKPT map.csv dev-clean-map.tsv dev-clean-words.jsonl spidr --layers 10,11,12
# Codebook quality: purity / PNMI / bitrate + P(phone|code) + durations
gslm2-eval-codebook $CKPT spidr dev-clean-map.tsv dev-clean.jsonl codebook-out/
# DiscoPhon cross-lingual (per language; eng shown)
gslm2-eval-discophon $CKPT spidr manifest-eng-test.csv alignment-eng-test.txt \
  triphone-eng-test.item discophon.csv eng --layers 5,6,7,8,9,10,11,12
# Stream similarity between codebook layers (Fig. stream similarity)
python scripts/analyze_codebook_pairwise_nmi.py --root codebook-out/ --split zr2021-dev-clean-map --out similarity/
# Layerwise plots
python scripts/plot_abx.py abx.csv -o abx.png
python scripts/plot_map.py map.csv -o map.png
python scripts/plot_pnmi_layerwise.py codebook-out/ -o pnmi.png
```

## Spoken-LM evaluation (sWuggy / sBLIMP / StoryCloze, `tab:slm`)

```bash
# single-stream: audio pairs in, on-the-fly encoding with the training encoder
gslm2-eval-slm --config configs/slm/ll-6kh-300m.yaml \
  --checkpoint /path/to/slm-ckpt.pt --pairs-dir /path/to/sLM21-pairs \
  --benchmarks lexical_test_invocab lexical_test_oov syntactic_test sSC tSC \
  --out results/slm-ll-6kh-300m.json
# text LM: same CLI, text pair CSVs (positive/negative columns)
gslm2-eval-slm --config configs/textlm/lh-6kh-300m.yaml \
  --checkpoint /path/to/textlm-ckpt.pt --pairs-dir /path/to/text-pairs \
  --benchmarks syntactic_test --out results/textlm.json
```

Build audio pair manifests from ZeroSpeech downloads once:

```bash
python scripts/create_slm_pair_manifests.py --data-dir /path/to/sLM21-dataset \
  --output-dir /path/to/sLM21-pairs   # see scripts/PAIR_MANIFEST_FORMAT.md
```

Multi-stream (GSLM2) scores pre-extracted `|`-joined unit pairs: extract the
benchmark audio with `manifests_test=[benchmark.csv]` (6-stream config), pair
with `gslm2.eval.slm.create_pairs.create_paired_file`, then run
`gslm2-eval-slm` with the GSLM2 training config over the `pairs/` dir.

## Resynthesis (`tab:speech_resynthesis`)

```bash
# 1. LJ Speech units (non-deduped oracle + deduped), 250 held-out utts
gslm2-extract-units --config configs/extract/spidr2-l7-codes-dup.yaml
# 2. oracle vs predicted durations, WER (Whisper/Qwen3-Omni) + UTMOS
gslm2-eval-resynthesis --oracle-manifest feats/l7-dup/unit_vocoder/ljspeech_valid.jsonl \
  --dedup-manifest feats/l7-dedup/unit_vocoder/ljspeech_valid.jsonl \
  --asr-manifest ljspeech-valid-asr.jsonl --vocoder-checkpoint runs/vocoder-single/vocoder-last \
  --duration-checkpoint runs/duration-single/best.pt --asr whisper --out-dir results/resynth
```

## Generation (ASR-PPL, NRS, LLMJ), LT-Swap, multilingual cascade

Under construction (Phase 5).
