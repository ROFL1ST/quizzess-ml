FROM python:3.11-slim

ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
ENV PORT 8080
WORKDIR $APP_HOME

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); \
    from transformers import AutoTokenizer, AutoModel; \
    m='indobenchmark/indobert-base-p1'; \
    AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]