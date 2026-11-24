FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend backend
COPY agent agent
COPY frontend frontend
COPY tests tests
COPY docs docs
# voice worker image only: download Silero + turn-detector weights at build time
ARG WITH_MODELS=0
RUN if [ "$WITH_MODELS" = "1" ]; then python -m agent.worker download-files; fi
EXPOSE 8000
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
