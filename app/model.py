# app/model.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelBundle:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: str
    dtype: torch.dtype
    source: str  # MODEL_ID or MODEL_DIR


_MODEL: Optional[ModelBundle] = None
_LOAD_ERROR: Optional[str] = None


def _pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _pick_dtype(device: str) -> torch.dtype:
    # На CPU bf16 может работать, но чаще стабильнее float32.
    # Для GPU — bf16 если доступно, иначе fp16.
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_model() -> Optional[ModelBundle]:
    """
    Пытается загрузить модель из:
      1) HF Hub по MODEL_ID
      2) локальной папки по MODEL_DIR (fallback)

    Возвращает ModelBundle или None.
    Ошибка хранится в _LOAD_ERROR.
    """
    global _MODEL, _LOAD_ERROR
    if _MODEL is not None:
        return _MODEL

    device = _pick_device()
    dtype = _pick_dtype(device)

    model_id = os.getenv("MODEL_ID", "").strip()
    model_dir = os.getenv("MODEL_DIR", "").strip()

    source = None
    if model_id:
        source = model_id
    elif model_dir:
        source = model_dir
    else:
        # дефолт: папка merged_model рядом с проектом
        source = os.path.join(os.getcwd(), "merged_model")

    try:
        # Токен нужен только если репозиторий private
        token = os.getenv("HF_TOKEN", None)

        tokenizer = AutoTokenizer.from_pretrained(
            source,
            token=token,
            trust_remote_code=True,
        )

        # device_map="auto" — удобно для GPU, на CPU не нужно
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                source,
                token=token,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                source,
                token=token,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(device)

        model.eval()
        _MODEL = ModelBundle(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            source=source,
        )
        _LOAD_ERROR = None
        return _MODEL

    except Exception as e:
        _LOAD_ERROR = f"{type(e).__name__}: {e}"
        _MODEL = None
        return None


def get_model() -> Optional[ModelBundle]:
    return _MODEL if _MODEL is not None else load_model()


def get_load_error() -> Optional[str]:
    return _LOAD_ERROR
