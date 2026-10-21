"""Keep track of metrics."""

import torch


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float | torch.Tensor, n: int = 1) -> None:
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def log(self) -> float:
        avg = self.avg
        if isinstance(avg, torch.Tensor):
            avg = avg.item()
        self.reset()
        return avg


class Records:
    """Record average meters for multiple values."""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.meters = {name: AverageMeter() for name in names}

    def __getitem__(self, name: str) -> AverageMeter:
        return self.meters[name]

    def __contains__(self, name: str) -> bool:
        return name in self.meters

    def log(self) -> dict[str, float]:
        return {name: meter.log() for name, meter in self.meters.items()}
