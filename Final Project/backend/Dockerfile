FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY backend ./backend
COPY artifacts ./artifacts
COPY ci ./ci

RUN pip install --upgrade pip \
    && pip install .

EXPOSE 8000
CMD ["uvicorn", "giftcard_sentiment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
