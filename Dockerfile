FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for protoc
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install protoc (Protocol Buffers compiler)
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v25.5/protoc-25.5-linux-x86_64.zip \
    && unzip protoc-25.5-linux-x86_64.zip -d /usr/local \
    && rm protoc-25.5-linux-x86_64.zip

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Set environment variable to ensure protobuf can find protoc
ENV PROTOC=/usr/local/bin/protoc

CMD ["python", "app.py"]