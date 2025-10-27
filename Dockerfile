FROM python:3.13-slim

# Dependencies for Tensorflow and OpenCV    
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.3 /uv /usr/local/bin/uv

# Copy uv config files
COPY pyproject.toml uv.lock* ./

# Install dependencies with uv
RUN uv sync --frozen

COPY . .

EXPOSE 8888
