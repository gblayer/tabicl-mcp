FROM python:3.11-slim

WORKDIR /app

# System deps (TabICL may need torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY tabicl_mcp/ ./tabicl_mcp/

# Install CPU-only torch first (saves ~2 GB vs full torch)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir .

# Default: HTTP/SSE on port 7860 (HuggingFace Spaces expects 7860)
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "tabicl-mcp-http 0.0.0.0 $PORT"]
