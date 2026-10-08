# Model zoo

Released checkpoints (links to be added on release):

- SpidR 2.0 encoder (LibriSpeech-960h, seed 0)
- Single-stream SLMs (600h/90M, 6Kh/300M, 60Kh/1.2B)
- GSLM2 multi-stream SLMs (600h/90M, 6Kh/300M, 60Kh/1.2B)
- Duration predictors and unit vocoders (single- and multi-stream)

External baselines used in the paper:

- DinoSR official checkpoint (loadable via `gslm2.models.load_pretrained("dinosr-original")`)
- SpidR official checkpoint
- HuBERT Base (via torchaudio)
