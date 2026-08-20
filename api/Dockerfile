FROM python:3.10-slim

WORKDIR /app

# Force Python standard output and error streams to be unbuffered
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code and models
COPY api/ ./api/
COPY models/ ./models/

# Expose port
EXPOSE 7860

# Run the API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
