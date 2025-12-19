# app/model.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelBundle:
    model: torch.nn.Module
    tokenizer: object
    device: str
    dtype: torch.dtype
    source: str


_MODEL: Optional[ModelBundle] = None
_LOAD_ERROR: Optional[str] = None


def get_load_error() -> str | None:
    return _LOAD_ERROR


def _pick_device_and_dtype() -> tuple[str, torch.dtype]:
    # Mac: предпочтительно MPS, иначе CPU
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _is_local_model_dir(p: Path) -> bool:
    # Минимальный признак HF-совместимой папки модели
    return p.is_dir() and (p / "config.json").exists()


def _resolve_source() -> tuple[str, bool]:
    """
    Returns: (source, is_local)
    Priority:
      1) MODEL_DIR (local path)
      2) MODEL_SOURCE (HF repo id)
    """
    model_dir = os.environ.get("MODEL_DIR", "").strip()
    if model_dir:
        p = Path(model_dir).expanduser().resolve()
        if _is_local_model_dir(p):
            return str(p), True
        # Если указали MODEL_DIR, но папка невалидна — это явная ошибка конфигурации
        raise RuntimeError(
            f"MODEL_DIR is set but not a valid model folder: {p}. "
            f"Expected to find config.json in that directory."
        )

    model_source = os.environ.get("MODEL_SOURCE", "").strip()
    if not model_source:
        # дефолт — ваш HF репо (можно менять)
        model_source = "NaruMaru/ege-checker-qwen2p5-0p5b-demo"
    return model_source, False


def get_model() -> Optional[ModelBundle]:
    global _MODEL, _LOAD_ERROR
    if _MODEL is not None:
        return _MODEL

    try:
        source, is_local = _resolve_source()
        device, dtype = _pick_device_and_dtype()

        # Важно: для локальной папки source — это путь, и AutoTokenizer/AutoModel умеют это.
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=True,
            local_files_only=is_local,
        )

        model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=dtype,
            device_map=None,  # на Mac лучше без device_map
            trust_remote_code=True,
            local_files_only=is_local,
        )
        model.eval()
        model.to(device)

        _MODEL = ModelBundle(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            source=source if not is_local else f"local:{source}",
        )
        _LOAD_ERROR = None
        return _MODEL

    except Exception as e:
        _MODEL = None
        _LOAD_ERROR = f"{type(e).__name__}: {e}"
        return None
