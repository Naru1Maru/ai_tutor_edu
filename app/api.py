# app/api.py
from __future__ import annotations

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse


from .model import get_model, get_load_error


app = FastAPI(
    title="MVP AI Service (EGE Solution Checker)",
    version="0.1.0",
    description=(
        "FastAPI-сервис для проверки письменных решений ЕГЭ по математике.\n\n"
        "Эндпоинт **/predict** принимает условие и решение ученика, опционально — эталон и подсказку ответа.\n"
        "Возвращает вердикт и короткое пояснение."
    ),
    swagger_ui_parameters={
        # приятнее пользоваться
        "docExpansion": "list",
        "displayRequestDuration": True,
        "tryItOutEnabled": True,  # чтобы Try it out был доступен сразу
    },
)


class PredictRequest(BaseModel):
    condition: str = Field(
        ...,
        title="Условие",
        description="Текст условия задачи.",
        examples=["Хорда AB делит окружность на две части..."],
        min_length=1,
    )
    student_solution: str = Field(
        ...,
        title="Решение ученика",
        description="Решение в свободной форме (как написал ученик).",
        examples=["Решение... Ответ: -105."],
        min_length=1,
    )
    reference_solution: str | None = Field(
        default=None,
        title="Эталонное решение (опционально)",
        description="Если есть — эталонное решение/разбор.",
        examples=["Решение... Ответ: 105."],
    )
    use_reference: bool = Field(
        default=True,
        title="Использовать эталон",
        description="Если true и reference_solution задан — модель учитывает эталон при проверке.",
        examples=[True],
    )
    answer_hint: str | None = Field(
        default=None,
        title="Подсказка ответа (опционально)",
        description="Короткая подсказка: итоговый ответ (например, '105').",
        examples=["105"],
    )
    max_new_tokens: int = Field(
        default=220,
        title="Лимит генерации",
        description="Ограничение на количество новых токенов при генерации ответа.",
        ge=1,
        le=1024,
        examples=[220],
    )

    # Большой “красивый” пример, который будет показываться в Swagger как выбираемый example
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "condition": "Хорда AB делит окружность на две части...",
                    "student_solution": "Решение... Ответ: -105.",
                    "reference_solution": "Решение... Ответ: 105.",
                    "use_reference": True,
                    "answer_hint": "105",
                    "max_new_tokens": 220,
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    verdict: int = Field(
        ...,
        title="Вердикт (0/1)",
        description="1 — верно, 0 — неверно.",
        examples=[0],
    )
    verdict_text: str = Field(
        ...,
        title="Текст вердикта",
        description="Человекочитаемая форма вердикта.",
        examples=["неверно"],
    )
    explanation: str = Field(
        ...,
        title="Пояснение",
        description="Короткое пояснение, что не так (или почему верно).",
        examples=["В вычислениях получился один результат, но в конце указан другой ответ."],
    )
    raw: str = Field(
        ...,
        title="Raw-ответ модели",
        description="Сырой текст, который вернула модель (после промпта).",
        examples=["Вердикт: неверно\nПояснение: ..."],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "verdict": 0,
                    "verdict_text": "неверно",
                    "explanation": "В вычислениях получился один результат, но в конце указан другой ответ.",
                    "raw": "Вердикт: неверно\nПояснение: В вычислениях ...",
                }
            ]
        }
    }


@app.on_event("startup")
def _startup():
    get_model()


@app.get("/health", tags=["service"], summary="Health-check")
def health():
    m = get_model()
    return {
        "status": "ok",
        "model_loaded": m is not None,
        "model_source": None if m is None else m.source,
        "load_error": get_load_error(),
    }


@app.get("/info", tags=["service"], summary="Информация о модели/окружении")
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
    verdict = 0
    verdict_text = "неверно"
    explanation = ""

    lower = text.lower()
    if "вердикт:" in lower:
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


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    summary="Проверить решение",
    description="Проверяет решение ученика и возвращает вердикт и пояснение.",
)
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
UI_HTML = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>EGE Solution Checker</title>
  <style>
    :root { --bg:#0b1220; --card:#0f1b33; --muted:#9bb0d0; --text:#e7efff; --accent:#4da3ff; --ok:#41d17a; --bad:#ff5a6a; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; background: radial-gradient(1200px 700px at 20% 0%, #14254a 0%, var(--bg) 50%); color:var(--text); }
    .wrap { max-width: 980px; margin: 0 auto; padding: 28px 16px 60px; }
    .top { display:flex; gap:14px; align-items:center; justify-content:space-between; flex-wrap:wrap; }
    .title { font-size: 28px; font-weight: 800; letter-spacing: .2px; }
    .subtitle { color:var(--muted); margin-top:6px; line-height:1.4; }
    .badge { padding:6px 10px; border:1px solid rgba(255,255,255,.12); border-radius:999px; color:var(--muted); font-size:12px; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 18px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    .card { background: rgba(15,27,51,.85); border: 1px solid rgba(255,255,255,.08); border-radius: 16px; padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,.25); }
    label { display:block; color: var(--muted); font-size: 13px; margin: 4px 0 6px; }
    textarea, input[type="text"], input[type="number"] {
      width:100%; box-sizing:border-box;
      background: rgba(7,12,24,.6); color: var(--text);
      border: 1px solid rgba(255,255,255,.10);
      border-radius: 12px; padding: 12px;
      outline: none;
    }
    textarea { min-height: 140px; resize: vertical; }
    .row { display:flex; gap: 10px; align-items:center; flex-wrap:wrap; }
    .row > * { flex: 1; }
    .row .small { flex: 0 0 auto; }
    .btn {
      background: linear-gradient(180deg, rgba(77,163,255,.95), rgba(77,163,255,.75));
      border: 0; color:#061025; font-weight: 800;
      padding: 12px 14px; border-radius: 12px; cursor:pointer;
      transition: transform .08s ease;
    }
    .btn:disabled { opacity:.55; cursor:not-allowed; }
    .btn:active { transform: translateY(1px); }
    .muted { color: var(--muted); font-size: 12px; }
    .out { white-space: pre-wrap; line-height: 1.5; }
    .verdict { display:inline-flex; gap:8px; align-items:center; padding:8px 10px; border-radius:999px; font-weight:800; }
    .verdict.ok { background: rgba(65,209,122,.16); color: var(--ok); border:1px solid rgba(65,209,122,.25); }
    .verdict.bad{ background: rgba(255,90,106,.16); color: var(--bad); border:1px solid rgba(255,90,106,.25); }
    .hr { height:1px; background: rgba(255,255,255,.08); margin: 12px 0; }
    .footer { margin-top: 14px; color: var(--muted); font-size: 12px; }
    .link { color: var(--accent); text-decoration: none; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div>
        <div class="title">Проверка решения ЕГЭ (MVP)</div>
        <div class="subtitle">Вставьте условие и решение. Нажмите «Проверить» — получите вердикт и объяснение.</div>
      </div>
      <div class="badge" id="status">Проверяем статус модели…</div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="row">
          <div>
            <label>Условие</label>
            <textarea id="condition" placeholder="Вставьте условие задачи..."></textarea>
          </div>
        </div>

        <div class="row" style="margin-top:10px;">
          <div>
            <label>Решение ученика</label>
            <textarea id="student_solution" placeholder="Вставьте решение ученика..."></textarea>
          </div>
        </div>

        <div class="row" style="margin-top:10px;">
          <div>
            <label>Эталонное решение (необязательно)</label>
            <textarea id="reference_solution" placeholder="Если есть — вставьте эталон..."></textarea>
          </div>
        </div>

        <div class="row" style="margin-top:10px;">
          <div class="small" style="display:flex; gap:10px; align-items:center;">
            <input id="use_reference" type="checkbox" checked />
            <label for="use_reference" style="margin:0;">Учитывать эталон</label>
          </div>
          <div>
            <label>Подсказка ответа (необязательно)</label>
            <input id="answer_hint" type="text" placeholder="например, 105" />
          </div>
          <div>
            <label>max_new_tokens</label>
            <input id="max_new_tokens" type="number" min="1" max="1024" value="220" />
          </div>
        </div>

        <div class="row" style="margin-top:12px;">
          <button class="btn" id="btn" onclick="runCheck()">Проверить</button>
          <div class="muted" id="hint">Ответ придёт в читаемом виде (не JSON).</div>
        </div>
      </div>

      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap;">
          <div style="font-weight:800; font-size:16px;">Результат</div>
          <div id="verdictBadge" class="verdict" style="display:none;"></div>
        </div>
        <div class="hr"></div>
        <div id="explanation" class="out muted">Пока пусто. Нажмите «Проверить».</div>

        <div class="footer">
          Для разработчиков доступно <a class="link" href="/docs">/docs</a>.
        </div>
      </div>
    </div>
  </div>

  <script>
    async function refreshStatus() {
      try {
        const r = await fetch("/health");
        const j = await r.json();
        const el = document.getElementById("status");
        if (j.model_loaded) {
          el.textContent = "Модель готова: " + (j.model_source || "loaded");
          el.style.borderColor = "rgba(65,209,122,.35)";
          el.style.color = "#baf7d0";
        } else {
          el.textContent = "Модель не загружена: " + (j.load_error || "unknown");
          el.style.borderColor = "rgba(255,90,106,.35)";
          el.style.color = "#ffd0d5";
        }
      } catch (e) {
        document.getElementById("status").textContent = "Статус недоступен";
      }
    }

    function setLoading(isLoading) {
      const b = document.getElementById("btn");
      b.disabled = isLoading;
      b.textContent = isLoading ? "Проверяем…" : "Проверить";
    }

    function showResult(verdict, verdictText, explanation) {
      const badge = document.getElementById("verdictBadge");
      badge.style.display = "inline-flex";
      badge.className = "verdict " + (verdict === 1 ? "ok" : "bad");
      badge.textContent = verdict === 1 ? "Верно" : "Неверно";

      const out = document.getElementById("explanation");
      out.className = "out";
      out.textContent = explanation || "(пояснение пустое)";
    }

    async function runCheck() {
      setLoading(true);
      const payload = {
        condition: document.getElementById("condition").value || "",
        student_solution: document.getElementById("student_solution").value || "",
        reference_solution: document.getElementById("reference_solution").value || null,
        use_reference: document.getElementById("use_reference").checked,
        answer_hint: document.getElementById("answer_hint").value || null,
        max_new_tokens: Number(document.getElementById("max_new_tokens").value || 220)
      };

      try {
        const r = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        if (!r.ok) {
          const t = await r.text();
          showResult(0, "error", "Ошибка сервера: " + t);
          return;
        }

        const j = await r.json();
        showResult(j.verdict, j.verdict_text, j.explanation);
      } catch (e) {
        showResult(0, "error", "Сетевая ошибка: " + (e?.message || e));
      } finally {
        setLoading(false);
      }
    }

    refreshStatus();
  </script>
</body>
</html>
"""

@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def landing():
    return HTMLResponse(UI_HTML)
