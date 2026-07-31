FROM python:3.11-slim

WORKDIR /app

# CPU-only torch first — much smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY tabicl_mcp/ ./tabicl_mcp/
RUN pip install --no-cache-dir .

# Pre-download TabICL checkpoints into the image so the first user request
# doesn't pay for the download (free Spaces already pay a wake-from-sleep cost).
ENV HF_HOME=/app/.cache/huggingface
RUN python - <<'EOF'
import numpy as np
from tabicl import TabICLClassifier, TabICLRegressor

rng = np.random.default_rng(0)
X = rng.random((24, 3)).astype("float32")
clf = TabICLClassifier(); clf.fit(X, (X[:, 0] > 0.5).astype(int)); clf.predict(X[:4])
reg = TabICLRegressor(); reg.fit(X, X[:, 0]); reg.predict(X[:4])
EOF
# HF Spaces runs the container as a non-root user
RUN chmod -R a+rwX /app/.cache

ENV PORT=7860 \
    TABICL_MCP_REPORTS_DIR=/tmp/tabicl-reports
EXPOSE 7860

CMD ["tabicl-mcp-http"]
