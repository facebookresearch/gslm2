# ZeroSpeech and StoryCloze evaluation manifests

See [shared setup](../data.md#before-you-start) for environment and path conventions.

Create the lexical dev/test invocab and OOV, syntactic dev/test, tSC, and sSC
manifests for the [LM experiment configs](../../configs/lm/README.md).

## Download datasets

- **ZeroSpeech sLM21:** [ZeroSpeech downloads](https://download.zerospeech.com/).
  Download the lexical and syntactic audio archives with their `gold.csv` labels.
- **tSC and sSC:** [SpokenStoryCloze](https://github.com/slp-rl/SpokenStoryCloze).
  Follow the repository's instructions to download the audio and accompanying
  text files.
- **French and German tSC and sBLIMP:** see the
  [multilingual text release](#french-and-german-tsc-and-sblimp) below.
  SLM evaluation requires synthesizing audio from the released text pairs.

Choose the download directory in step 1, then extract the audio archives into
the layout shown there. For sources with published checksums, verify the
downloads before extraction.

## 1. Set the dataset and output directories

Activate the LM environment and choose the download and manifest directories:

```bash
conda activate gslm2_lm

export BENCHMARK_AUDIO_ROOT=/path/to/benchmarks
export BENCHMARK_MANIFEST_DIR="$PWD/build/eval_manifests/raw"
mkdir -p "$BENCHMARK_MANIFEST_DIR"
```

Extract the datasets into this layout:

```text
<BENCHMARK_AUDIO_ROOT>/
├── sLM21-dataset/
│   ├── lexical/
│   │   ├── dev/       # WAV files and gold.csv
│   │   └── test/      # WAV files and gold.csv
│   └── syntactic/
│       ├── dev/       # WAV files and gold.csv
│       └── test/      # WAV files and gold.csv
├── tSC/               # WAV files, plus accompanying text files
└── sSC/               # WAV files, plus accompanying text files
```

## 2. Create per-audio CSVs

Scan the WAV files and write their paths and sample counts:

```bash
for benchmark in tSC sSC; do
  python -m gslm2.encoder.data.write_manifest \
    "$BENCHMARK_AUDIO_ROOT/$benchmark" \
    "$BENCHMARK_MANIFEST_DIR/$benchmark.csv" --ext .wav --num-workers 8 || exit 1
done

for task in lexical syntactic; do
  for split in dev test; do
    python -m gslm2.encoder.data.write_manifest \
      "$BENCHMARK_AUDIO_ROOT/sLM21-dataset/$task/$split" \
      "$BENCHMARK_MANIFEST_DIR/${task}_${split}.csv" --ext .wav --num-workers 8 || exit 1
  done
done
```

The syntactic, tSC, and sSC manifests are ready to use.
`lexical_dev.csv` and `lexical_test.csv` contain both vocabulary categories;
split them in the next step.

## 3. Split lexical manifests into invocab and OOV

Run the [lexical manifest splitter](../../scripts/split_lexical_manifests.py) on
the intermediate CSVs:

```bash
python scripts/split_lexical_manifests.py \
  --slm21-root "$BENCHMARK_AUDIO_ROOT/sLM21-dataset" \
  --manifests-dir "$BENCHMARK_MANIFEST_DIR"
```

The splitter keeps both recordings of a pair in the same category. See
[the pairing rules](formats.md#speech-benchmark-pairing) for the frequency definition.

Outputs are four CSVs in `BENCHMARK_MANIFEST_DIR`. Use `--output-dir` for another
destination or `--splits dev` for only the development split.

## 4. Set the LM config paths

Set `slm.feature_extraction.manifests_test` to the eight generated CSVs.
Keep their filenames: preparation uses each stem to select the pairing logic.
Set `slm.feature_extraction.slm21_root` to the downloaded `sLM21-dataset`
directory containing the gold labels:

```yaml
slm:
  feature_extraction:
    slm21_root: /path/to/benchmarks/sLM21-dataset
    manifests_test:
    - /path/to/eval_manifests/raw/lexical_dev_invocab.csv
    - /path/to/eval_manifests/raw/lexical_dev_oov.csv
    - /path/to/eval_manifests/raw/lexical_test_invocab.csv
    - /path/to/eval_manifests/raw/lexical_test_oov.csv
    - /path/to/eval_manifests/raw/syntactic_dev.csv
    - /path/to/eval_manifests/raw/syntactic_test.csv
    - /path/to/eval_manifests/raw/tSC.csv
    - /path/to/eval_manifests/raw/sSC.csv
```

When running only these eight benchmarks, use these training validation metrics
in `slm.train.dataset.valid_metrics.names`:

- `lexical_dev_invocab`
- `lexical_dev_oov`
- `syntactic_dev`

For LT-Swap entries in the templates, follow
[LT-Swap evaluation manifests](ltswap.md). For French and German benchmarks,
follow the [multilingual text release](#french-and-german-tsc-and-sblimp) below.

Continue with [LM preparation and training](../../configs/lm/README.md#launch).
The pipeline creates tokenized pairs under `test/pairs/*.jsonl`. For optional
audio pair CSVs, see [SLM Pair Manifest Format](../../scripts/PAIR_MANIFEST_FORMAT.md).

## Expected outputs

For the complete datasets at the paths above:

| Manifest | Audio rows | Evaluation pairs |
|---|---:|---:|
| `lexical_dev_invocab.csv` | 40,000 | 20,000 |
| `lexical_dev_oov.csv` | 40,000 | 20,000 |
| `lexical_test_invocab.csv` | 160,000 | 80,000 |
| `lexical_test_oov.csv` | 160,000 | 80,000 |
| `syntactic_dev.csv` | 50,400 | 25,200 |
| `syntactic_test.csv` | 126,000 | 63,000 |
| `tSC.csv` | 3,742 | 1,871 |
| `sSC.csv` | 3,742 | 1,871 |

## French and German tSC and sBLIMP

We release French (`fr`) and German (`de`) extensions of tSC (StoryCloze) and
sBLIMP (grammaticality evaluation) as **text only**. The translations are
outputs of Llama 3.3. **Audio is not included; synthesize the texts with a TTS
system before evaluating an SLM.** Text LMs can use the inline text pairs
directly with the [LM text preparation pipeline](../../configs/lm/README.md).

### Download text pairs

Download the JSONLs from the
[multilingual tSC and sBLIMP release directory](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/multilingual_tsc_sblimp/).
See the [release README](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/multilingual_tsc_sblimp/README.md)
for the schema, synthesis guidance, and license terms, and
[RELEASE.json](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/multilingual_tsc_sblimp/RELEASE.json)
for pair counts and checksums. The data files are:

- French: [tSC](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/multilingual_tsc_sblimp/fr/tsc_fr.jsonl)
  and [sBLIMP](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/multilingual_tsc_sblimp/fr/blimp_fr.jsonl).
- German: [tSC](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/multilingual_tsc_sblimp/de/tsc_de.jsonl)
  and [sBLIMP](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/multilingual_tsc_sblimp/de/blimp_de.jsonl).

Download the accompanying
[SHA256SUMS](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/multilingual_tsc_sblimp/SHA256SUMS)
and [LICENSE](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/multilingual_tsc_sblimp/LICENSE).
Save the release files under `MULTILINGUAL_BENCHMARK_DIR`, preserving the
`fr/` and `de/` subdirectories:

```bash
export MULTILINGUAL_BENCHMARK_DIR=/path/to/multilingual_tsc_sblimp
export MULTILINGUAL_BENCHMARK_URL=https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/multilingual_tsc_sblimp
mkdir -p "$MULTILINGUAL_BENCHMARK_DIR"/{fr,de}
for filename in README.md LICENSE RELEASE.json SHA256SUMS \
  fr/tsc_fr.jsonl fr/blimp_fr.jsonl de/tsc_de.jsonl de/blimp_de.jsonl; do
  curl -fL "$MULTILINGUAL_BENCHMARK_URL/$filename" \
    -o "$MULTILINGUAL_BENCHMARK_DIR/$filename" || exit 1
done
```

### Verify downloads

Verify the downloaded files against the release checksums before preparation:

```bash
(cd "$MULTILINGUAL_BENCHMARK_DIR" && sha256sum -c SHA256SUMS) || exit 1
```

### Prepare audio for SLM evaluation

Each JSONL record contains `positive` and `negative` objects with inline `text`
and a stable `file_name` identifier. Synthesize both texts using a voice for the
appropriate language, keeping the same voice and synthesis settings within
each pair. Save mono 16 kHz WAVs as `<file_name>.wav` in separate directories
for each language and benchmark, then create per-audio CSVs with the
[manifest writer](../data.md#before-you-start).

Preserve the source JSONL's positive/negative association when preparing
evaluation pairs. tSC filenames have different leading indices for the two
sides; their shared UUID identifies the pair. Removing only `_correct` or
`_incorrect` does not recover the tSC pair ID.

The release includes CC-BY-NC-4.0 and additional Llama 3.3 translation and
third-party terms; consult its README and LICENSE before use.
