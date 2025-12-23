# AI Tutor EDU — MVP сервиса проверки решений ЕГЭ по математике

> **AI Tutor EDU** — это минимально жизнеспособный (MVP) веб‑сервис на базе LLM, который автоматически проверяет письменные решения задач ЕГЭ по математике, выносит вердикт (*верно / неверно*) и даёт краткое объяснение.

Проект демонстрирует **полный production‑цикл**:

* inference LLM через FastAPI
* деплой в Docker
* CI/CD сборки образа через GitHub Actions
* публикацию образа в GitHub Container Registry (GHCR)
* запуск сервиса на любой машине с Docker

---

## Что умеет сервис

* принимает:

  * условие задачи
  * решение ученика
  * (опционально) эталонное решение
  * (опционально) эталонный ответ
* формирует строгий prompt для LLM
* возвращает:

  * числовой вердикт (1 / 0)
  * текстовый вердикт ("верно" / "неверно")
  * краткое пояснение
* имеет health‑endpoint для мониторинга

---

## Используемая модель

По умолчанию сервис использует модель с HuggingFace Hub:

```
NaruMaru/ege-checker-qwen2p5-0p5b-demo
```

Поддерживаются:

* **публичные** модели HF (без токена)
* **приватные** модели HF (через `HF_TOKEN`)

Модель подгружается **при старте сервиса**.

---

## Архитектура

```
FastAPI
  ├── /health      — статус сервиса и модели
  ├── /info        — информация о модели
  └── /predict     — основной inference endpoint

Transformers (HF)
  └── AutoModelForCausalLM

Docker
  └── GitHub Container Registry (ghcr.io)
```

---

## Структура репозитория

```
ai_tutor_edu/
├── app/
│   ├── api.py          # FastAPI endpoints
│   └── model.py        # загрузка и кэш модели
├── requirements.txt
├── Dockerfile
├── .env.example
├── README.md
└── .github/
    └── workflows/
        └── docker-build.yml
```

---

## Локальный запуск (без Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export MODEL_SOURCE="NaruMaru/ege-checker-qwen2p5-0p5b-demo"
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

Открыть:

* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) — Swagger
* [http://127.0.0.1:8000/](http://127.0.0.1:8000/ui)   — пользовательский интерфейс

---

## Docker: локальная сборка

```bash
docker build -t ai-tutor-edu:local .
```

Запуск:

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_SOURCE="NaruMaru/ege-checker-qwen2p5-0p5b-demo" \
  -v hf_cache:/root/.cache/huggingface \
  ai-tutor-edu:local
```

---

## CI/CD: GitHub Actions

При каждом `push` в ветку `main` автоматически:

1. собирается Docker‑образ
2. публикуется в **GitHub Container Registry (GHCR)**

Workflow:

```
.github/workflows/docker-build.yml
```

---

## Публичный Docker‑образ (GHCR)

Актуальный образ:

```
ghcr.io/naru1maru/ai_tutor_edu:latest
```

Стянуть:

```bash
docker pull ghcr.io/naru1maru/ai_tutor_edu:latest
```

Для того чтобы использовать docker на Mac с Apple Silicon:

```bash
docker pull --platform=linux/amd64 ghcr.io/naru1maru/ai_tutor_edu:latest
```
---

## Ручной деплой из GHCR

### 1. (Опционально) логин в GHCR

```bash
echo "$GITHUB_TOKEN" | docker login ghcr.io -u naru1maru --password-stdin
```

### 2. Запуск сервиса

**Публичная модель HF:**

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_SOURCE="NaruMaru/ege-checker-qwen2p5-0p5b-demo" \
  -v hf_cache:/root/.cache/huggingface \
  ghcr.io/naru1maru/ai_tutor_edu:latest
```
Если на Mac, то добавить: --platform=linux/amd64
 
---

## Проверка работы

### Health

```bash
curl http://127.0.0.1:8000/health
```

### Predict

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "condition": "Хорда AB делит окружность на две части, градусные величины которых относятся как 5 : 7. Под каким углом видна эта хорда из точки C, принадлежащей меньшей дуге окружности? Ответ дайте в градусах.",
    "student_solution": "Решение. Из точки C хорда AB видна под углом ACB. Пусть большая часть окружности равна 7x, тогда меньшая равна 5x. 7x + 5x = 360°  ⇒  12x = 360°  ⇒  x = 30°. Значит, меньшая дуга окружности равна 150°, а большая — 210°. Вписанный угол равен половине дуги, на которую он опирается. Так как точка C лежит на меньшей дуге, то угол ACB опирается на меньшую дугу 150°, следовательно, ∠ACB = 150° / 2 = 75°. Ответ: 75.",
    "reference_solution": "Решение. Из точки C хорда AB видна под углом ACB. Пусть большая часть окружности равна 7x, тогда меньшая равна 5x. 7x + 5x = 360°  ⇒  12x = 360°  ⇒  x = 30°. Значит, меньшая дуга окружности равна 150°, а большая — 210°. Вписанный угол равен половине дуги, на которую он опирается, а угол ACB (точка C на меньшей дуге) опирается на большую дугу 210°, поэтому ∠ACB = 210° / 2 = 105°. Ответ: 105.",
    "use_reference": true,
    "answer_hint": "105",
    "max_new_tokens": 220
  }'
```

---

## Веб-интерфейс для проверки решений

При переходе по ссылке на сервис пользователю открывается **удобная веб-страница**, где можно:

* вставить **условие задачи**
* ввести **решение ученика**
* (опционально) указать **эталонное решение или правильный ответ**
* нажать кнопку **«Проверить»**
* получить:

  * вердикт (*верно / неверно*)
  * текстовое объяснение ошибки или подтверждение корректности решения

> API (`/predict`) и Swagger-документация остаются доступными для разработчиков,
> но основной сценарий использования — через веб-страницу.

<img width="1345" height="777" alt="Снимок экрана 2025-12-20 в 03 13 08" src="https://github.com/user-attachments/assets/3e560358-80d7-4dd4-8e1c-deb836318868" />

---

## Участники проекта

* Конохова Екатерина Михайловна
* Дистлер Марина Алексеевна 
* Гадиев Михаил Искандерович 
* Чубарова Дарья Алексеевна 

