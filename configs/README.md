# Canonical paper configs

Translated from the paper's training tables (`tab:lm_training_params`,
`tab:transformer_params`, Appendix Training Configurations) into the native
`gslm2-train` schema. All paths are `/path/to/...` placeholders.

## Layout

- `encoder/` — SpidR 2.0 on LibriSpeech-960h and VoxPopuli En/Fr/De 6Kh
- `slm/` — single-stream SLMs (Libri-Light 600h/6Kh/60Kh, VoxPopuli 6Kh)
- `gslm2/` — multi-stream GSLM2 (6 streams, layers 5–10)
- `textlm/` — text-LM toplines (Libriheavy / VoxPopuli ASR transcripts)
- `conv/` — 600Kh/2.5B and 6Mh/6.5B scaling configs (need FSDP + data access)
- `duration/`, `vocoder/` — synthesis configs for `gslm2-duration-predictor`
  and `gslm2-unit-vocoder`

## Translation notes (native vs paper pipeline)

- **Batches.** The paper counts global unit/token batches; the native SLM
  batches on-the-fly audio in samples. Per-GPU `max_batch_length` is set from
  the measured deduped rate (~14 units/s at layer 7): e.g. 81920 units/GPU →
  93.6M samples. GSLM2/text rows follow `global / (world × ctx)` exactly.
- **Losses.** GSLM2 `gen/multi_pred/self_distil = 0.1/1.0/0.1` is the paper's
  α=β=0.1; stream order `[5,0,1,4,3,2]` is layers (10,5,6,9,8,7).
- **Init.** Single-stream/text use `standard` init; GSLM2 uses the native
  `fairseq2`-scheme init (same distributions, no fairseq2 dependency).
- **VoxPopuli encoders** reuse the LS-960 recipe (steps kept at 400k; the paper
  does not report VP encoder steps separately). VP text LMs reuse the 6Kh
  text recipe (20k steps).
- **6.5B decay floor** `final_lr_scale: 0.1` is assumed (not reported).
- **Text tokenizers** are per-scale BPE-4096 (`scripts/train_tokenizer.py
  --type bpe --vocab-size 4096`); set `model.vocab_size`/`pad_idx` from the
  trained tokenizer.
- **Validation.** `slm`/`lm_loss` online validation supports on-the-fly SLM
  sources only; GSLM2/text configs ship without validation blocks until the
  pre-extracted/text runners land.
