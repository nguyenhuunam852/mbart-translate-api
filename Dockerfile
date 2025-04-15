FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for protoc and huggingface-cli
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install protoc (Protocol Buffers compiler)
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v25.5/protoc-25.5-linux-x86_64.zip \
    && unzip protoc-25.5-linux-x86_64.zip -d /usr/local \
    && rm protoc-25.5-linux-x86_64.zip

RUN pip install --no-cache-dir huggingface_hub[cli]

# Check if huggingface-cli is installed and working
RUN huggingface-cli --version && \
        huggingface-cli whoami || echo "huggingface-cli is installed but may require login or config"

RUN mkdir -p /app/models
# Download the model file from Hugging Face
RUN huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.q5_0.gguf --local-dir /app/models --local-dir-use-symlinks False

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Set environment variable to ensure protobuf can find protoc
ENV PROTOC=/usr/local/bin/protoc

# (Optional) Set environment variable for model file if needed by app.py
ENV MODEL_FILE=/app/models/llama-2-7b-chat.q5_0.gguf

CMD ["python", "app.py"]