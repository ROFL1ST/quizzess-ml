FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
ENV PORT=7860
ENV HF_HOME=/app/.cache/huggingface

WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model will be downloaded at runtime from HuggingFace Hub
# rofl1st/llama-3.1-8b-essay-grader

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
