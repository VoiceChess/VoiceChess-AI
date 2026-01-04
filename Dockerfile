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
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart python-dotenv matplotlib tqdm python-chess

# Copy application files
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Set PyTorch cache directory to a location inside the image
ENV TORCH_HOME=/app/.torch
ENV XDG_CACHE_HOME=/app/.cache

# Create cache directories
RUN mkdir -p /app/.torch /app/.cache

# Pre-download PyTorch models to cache them in image
RUN python preload_models.py

# Verify models are cached
RUN ls -lah /app/.torch/hub/checkpoints/ || echo "Warning: Models may not be cached"

# Expose port
EXPOSE 5000

# Set environment variables
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "app.py"]
