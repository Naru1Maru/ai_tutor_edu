# app/api.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model import get_model, get_load_error


app = FastAPI(title="MVP AI Service (EGE Solution Checker)")


class PredictRequest(BaseModel):
    condition: str
    student_solution: str
    reference_solution: str | None = None
    use_reference: bool = True
    answer_hint: str | None = None
    max_new_tokens: int = 220


class PredictResponse(BaseModel):
    verdict: int  # 1=верно, 0=неверно
    verdict_text: str
    explanation: str
    raw: str


@app.on_event("startup")
def _startup():
    get_model()


@app.get("/health")
def health():
    m = get_model()
    return {
        "status": "ok",
        "model_loaded": m is not None,
        "model_source": None if m is None else m.source,
        "load_error": get_load_error(),
    }


@app.get("/info")
def info():
    m = get_model()
    return {
        "model_loaded": m is not None,
        "model_source": None if m is None else m.source,
        "device": None if m is None else m.device,
        "dtype": None if m is None else str(m.dtype),
        "tokenizer": None if m is None else type(m.tokenizer).__name__,
        "load_error": get_load_error(),
    }


def _build_prompt(req: PredictRequest) -> str:
    parts = [
        "Ты — эксперт по проверке решений ЕГЭ по математике.",
        "Верни ровно в таком формате:",
        "Вердикт: верно|неверно",
        "Пояснение: ...",
        "",
        f"Условие:\n{req.condition.strip()}",
        "",
        f"Решение ученика:\n{req.student_solution.strip()}",
    ]
    if req.use_reference and req.reference_solution:
        parts += ["", f"Эталонное решение:\n{req.reference_solution.strip()}"]
    if req.answer_hint:
        parts += ["", f"Эталонный ответ (подсказка): {req.answer_hint.strip()}"]
    return "\n".join(parts).strip() + "\n"


def _parse_output(text: str) -> tuple[int, str, str]:
    # Очень простая разборка
    verdict = 0
    verdict_text = "неверно"
    explanation = ""

    lower = text.lower()
    if "вердикт:" in lower:
        # берём первую строку с вердиктом
        for line in text.splitlines():
            if line.lower().startswith("вердикт:"):
                v = line.split(":", 1)[1].strip().lower()
                if "верно" in v and "неверно" not in v:
                    verdict = 1
                    verdict_text = "верно"
                else:
                    verdict = 0
                    verdict_text = "неверно"
                break

    if "пояснение:" in lower:
        for line in text.splitlines():
            if line.lower().startswith("пояснение:"):
                explanation = line.split(":", 1)[1].strip()
                break

    return verdict, verdict_text, explanation


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    m = get_model()
    if m is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not available. load_error={get_load_error()}",
        )

    prompt = _build_prompt(req)

    inputs = m.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(m.model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = m.model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=False,
            eos_token_id=m.tokenizer.eos_token_id,
            pad_token_id=m.tokenizer.pad_token_id,
        )

    text = m.tokenizer.decode(out[0], skip_special_tokens=True)
    raw = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()

    verdict, verdict_text, explanation = _parse_output(raw)

    return PredictResponse(
        verdict=verdict,
        verdict_text=verdict_text,
        explanation=explanation,
        raw=raw,
    )
