FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-api.txt .
COPY pyproject.toml .

# CRITICAL: Install numpy first with compatible version
RUN pip install --no-cache-dir "numpy>=1.24,<2.0"

# Install PyTorch (CPU version)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install scikit-image from source to match numpy version
RUN pip install --no-cache-dir --no-binary scikit-image scikit-image

# Install other dependencies
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart python-dotenv matplotlib tqdm python-chess pyfastnoisesimd

# Copy application files
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Pre-download PyTorch models to cache them in image
RUN python preload_models.py

# Expose port
EXPOSE 5000

# Set environment variables
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "app.py"]
