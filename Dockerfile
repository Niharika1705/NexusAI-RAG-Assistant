FROM python:3.10-slim

WORKDIR /app

# Install system utilities if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . /app

# Create necessary directories
RUN mkdir -p /app/backend/knowledge_base /app/backend/chroma_db && \
    chmod -R 777 /app/backend/knowledge_base /app/backend/chroma_db

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Start Uvicorn serving FastAPI backend on 0.0.0.0:7860
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "7860"]
