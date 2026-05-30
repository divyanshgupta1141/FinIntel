FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required for compile/build of some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source files
COPY . .

# Expose port 8000 for web access
EXPOSE 8000

# Provision schemas automatically on startup, then start the FastAPI application
CMD ["sh", "-c", "python -c 'import asyncio; from database import init_db; asyncio.run(init_db())' && uvicorn main:app --host 0.0.0.0 --port 8000"]
