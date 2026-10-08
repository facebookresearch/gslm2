import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from tqdm import tqdm

from gslm2.config import DataConfig, ModelType
from gslm2.data.speech.dataset import Sample, speech_dataset
from gslm2.models import build_model


def count_codebooks_layers(checkpoint: str | Path) -> tuple[int, int]:
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    consume_prefix_in_state_dict_if_present(state_dict, "module.")
    parameters: list[str] = list(state_dict.keys())
    student_layers, codebooks = set(), set()
    for param in parameters:
        if param.startswith("student.layers."):
            student_layers.add(int(param.split(".")[2]))
        elif param.startswith("codebooks."):
            codebooks.add(int(param.split(".")[1]))
    assert sorted(student_layers) == list(range(len(student_layers)))
    assert sorted(codebooks) == list(range(len(codebooks)))
    return len(codebooks), len(student_layers)


def validate_layers(checkpoint: str | Path, layers: list[int] | None, *, codebook_proba: bool) -> list[int]:
    num_codebooks, max_layer = count_codebooks_layers(checkpoint)
    min_layer = max_layer - num_codebooks + 1 if codebook_proba else 1
    if layers is None:
        return list(range(min_layer, max_layer + 1))
    assert all(min_layer <= layer <= max_layer for layer in layers)
    return layers


def temp_dir() -> Path:
    """Scratch dir for extracted features: SLURM-aware, else system temp."""
    if "JOBSCRATCH" in os.environ:
        return Path(os.environ["JOBSCRATCH"])
    if "SLURM_JOB_ID" in os.environ and (tmp := Path("/scratch")).is_dir():
        gpu_tmp = tmp / os.environ["SLURM_JOB_ID"] / os.environ.get("SLURM_PROCID", "0")
        gpu_tmp.mkdir(parents=True, exist_ok=True)
        return gpu_tmp
    import tempfile

    return Path(tempfile.mkdtemp(prefix="gslm2_eval_"))


def extract_representations(  # noqa: PLR0913
    checkpoint: Path,
    model_type: ModelType,
    output_dir: Path,
    manifests: list[Path],
    layers: list[int],
    *,
    codebook_proba: bool,
    onehot: bool,
) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint=checkpoint, model_type=model_type).to(device).eval()
    datasets = {p.stem: speech_dataset(DataConfig(manifest=p, normalize=True)) for p in set(manifests)}
    lengths = {name: len(dataset) for name, dataset in datasets.items()}

    with torch.inference_mode(), tqdm(total=sum(lengths.values())) as pbar:
        for name, dataset in datasets.items():
            pbar.set_description(name)
            for i, sample in enumerate[Sample](dataset):
                outputs = (
                    model.get_codebooks(sample.waveform.to(device).unsqueeze(0), onehot=onehot)
                    if codebook_proba
                    else model.get_intermediate_outputs(sample.waveform.to(device).unsqueeze(0))
                )
                filename = Path(dataset.filenames[i]).with_suffix(".pt").name
                for layer in layers:
                    output_file = output_dir / f"{name}/{layer}/{filename}"
                    output_file.parent.mkdir(exist_ok=True, parents=True)
                    assert outputs[layer - 1] is not None
                    torch.save(outputs[layer - 1].squeeze().float().cpu(), output_file)
                pbar.update()


def extract_codes(  # noqa: PLR0913
    checkpoint: Path,
    model_type: ModelType,
    output_dir: Path,
    manifests: list[Path],
    layers: list[int],
    *,
    deduplicate: bool = False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint=checkpoint, model_type=model_type).to(device).eval()
    datasets = {p.stem: speech_dataset(DataConfig(manifest=p, normalize=True)) for p in set(manifests)}
    lengths = {name: len(dataset) for name, dataset in datasets.items()}

    with torch.inference_mode(), tqdm(total=sum(lengths.values())) as pbar:
        for name, dataset in datasets.items():
            codes = defaultdict(list)
            pbar.set_description(name)
            for i, sample in enumerate[Sample](dataset):
                onehots = model.get_codebooks(sample.waveform.to(device).unsqueeze(0), onehot=True)
                for layer in layers:
                    assert onehots[layer - 1] is not None
                    code = onehots[layer - 1].argmax(dim=1)
                    if deduplicate:
                        code = torch.unique_consecutive(code)
                    codes[layer].append({"audio": dataset.filenames[i], "codes": " ".join(map(str, code.tolist()))})
                pbar.update()
            for layer, data in codes.items():
                (output_dir / f"{layer}").mkdir(exist_ok=True, parents=True)
                with (output_dir / f"{layer}/{name}.jsonl").open("w") as f:
                    f.write("\n".join(map(json.dumps, data)) + "\n")
