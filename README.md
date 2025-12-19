# MVP AI Service (EGE Solution Checker)

FastAPI сервис для проверки решения ученика по условию задачи ЕГЭ (демо-версия).
Веса модели не хранятся в репозитории и подключаются при запуске контейнера.

## Run locally (без Docker)
```bash
pip install -r requirements.txt
set MODEL_DIR=.\merged_model
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
