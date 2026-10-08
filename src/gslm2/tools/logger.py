"""Logging utilities."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pathlib import Path


class ColoredFormatter(logging.Formatter):
    COLORS: ClassVar[dict[str, str]] = {
        "NOTSET": "\033[0m",  # No color
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"  # Reset color code

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        return f"{self.COLORS[record.levelname]}{message}{self.RESET}"


def init_logger(file: str | Path | None = None, *, with_colors: bool = False) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    template = "[spidr] %(asctime)s - %(levelname)s - %(name)s - %(message)s"
    formatter = (ColoredFormatter if with_colors else logging.Formatter)(template)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    stdout.setFormatter(formatter)
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.WARNING)
    stderr.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(stdout)
    logger.addHandler(stderr)

    if file is not None:
        handler = logging.FileHandler(file, "a")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def log_info_box(title: str, data: dict[str, object]) -> None:
    """Log key-value pairs inside a formatted box with the given title.

    Values are converted to strings via ``str()``. The box width adapts
    to the longest row so everything stays aligned.
    """
    rows = [(k, str(v)) for k, v in data.items()]
    # minimum content width = title + 4 padding chars
    min_width = max(len(title) + 4, *(len(k) + len(v) + 5 for k, v in rows))
    w = min_width + (min_width % 2)  # keep even for symmetry

    border = "═" * (w + 2)
    title_pad = w - len(title)
    l_pad = title_pad // 2
    r_pad = title_pad - l_pad

    lines = [
        f"╔{border}╗",
        f"║ {' ' * l_pad}{title}{' ' * r_pad} ║",
        f"╠{border}╣",
    ]
    for key, val in rows:
        content = f"  {key}: {val}"
        lines.append(f"║{content:<{w + 2}}║")
    lines.append(f"╚{border}╝")

    logging.getLogger().info("\n%s", "\n".join(lines))


def log_model_params(model: object) -> None:
    """Log total, trainable, and frozen parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    log_info_box(
        "Model Parameters",
        {
            "Total": f"{total:>12,}  ({total / 1e6:7.2f}M)",
            "Trainable": f"{trainable:>12,}  ({trainable / 1e6:7.2f}M)",
            "Frozen": f"{frozen:>12,}  ({frozen / 1e6:7.2f}M)",
        },
    )
