FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/merged_model

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
