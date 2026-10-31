# LJSpeech vocoder manifests

See [shared setup](../data.md#before-you-start) for environment and path conventions.

The framewise extraction configs use LJSpeech for unit vocoder training. Prepare
mono 16 kHz WAVs and the three `ljspeech_{train,valid,test}.csv` manifests with
the fixed [split IDs bundled in this repo](../../manifests/README.md#ljspeech).

## 1. Download and extract audio

Download the LJSpeech-1.1 archive from the
[LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/).
It contains mono WAVs at 22,050 Hz.

Save the archive under `LJSPEECH_DATA_DIR`, then extract it:

```bash
export LJSPEECH_DATA_DIR=/path/to/ljspeech
mkdir -p "$LJSPEECH_DATA_DIR"
curl -fL --continue-at - \
  https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 \
  -o "$LJSPEECH_DATA_DIR/LJSpeech-1.1.tar.bz2" || exit 1
tar -xjf "$LJSPEECH_DATA_DIR/LJSpeech-1.1.tar.bz2" -C "$LJSPEECH_DATA_DIR"
```

## 2. Resample audio and create manifests

Use the native environment, which includes FFmpeg and the audio dependencies:

```bash
conda activate gslm2
export LJSPEECH_PREP_DIR="$PWD/build/ljspeech"
python scripts/prepare_ljspeech_manifests.py \
  --data-dir "$LJSPEECH_DATA_DIR/LJSpeech-1.1" \
  --output-dir "$LJSPEECH_PREP_DIR"
```

Outputs are mono 16 kHz PCM WAVs under `wavs_16k/` and CSVs under `manifests/`.
CSVs contain absolute audio paths and sample counts from the prepared WAVs.
Reruns reuse valid WAVs. Optional flags: `--workers` for conversion concurrency
and `--ffmpeg /path/to/ffmpeg` for another executable.

See the [bundled split IDs and counts](../../manifests/README.md#ljspeech)
for the fixed experiment splits.

## 3. Set the extraction config paths

Set the vocoder target to the generated CSVs:

```yaml
targets:
  unit_vocoder:
    manifests:
    - /path/to/prepared-ljspeech/manifests/ljspeech_train.csv
    - /path/to/prepared-ljspeech/manifests/ljspeech_valid.csv
    - /path/to/prepared-ljspeech/manifests/ljspeech_test.csv
```

Keep the CSV filenames: extraction uses them to name the vocoder JSONL inputs.
Continue with [full-rate extraction](../synthesis.md#prepare-training-data).
