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

## Spoken-LM evaluation (sWuggy / sBLIMP / StoryCloze)

(Coming next: native benchmark CLIs over `eval/slm`; validation during
training via the `slm`/`lm_loss` specs in `configs/slm/*.yaml` already works.)

## Generation, LT-Swap, resynthesis, multilingual cascade

Under construction (Phase 5). Resynthesis pieces exist:
`gslm2.vocoder.unit.evaluation`, `wer/*`, `mos/utmos.py`.
