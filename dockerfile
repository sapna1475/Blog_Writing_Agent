FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 

WORKDIR /app

# Non-root user (security best practice, and a good interview talking point)
RUN useradd --create-home --uid 1000 appuser

# Install dependencies first so Docker caches this layer
# and only re-runs it when requirements change
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY backend.py .

# Output dir for generated blogs (ephemeral in the container)
ENV BLOG_OUTPUT_DIR=/app/blog_output
RUN mkdir -p /app/blog_output && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]