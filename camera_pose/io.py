from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def numpy_to_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: numpy_to_list(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [numpy_to_list(item) for item in value]
    return value


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(numpy_to_list(data), handle, sort_keys=False, allow_unicode=True)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return data


def write_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def append_markdown_log(path: Path, title: str, lines: list[str]) -> None:
    ensure_dir(path.parent)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {timestamp} - {title}\n\n")
        for line in lines:
            handle.write(f"- {line}\n")
