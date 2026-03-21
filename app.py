"""
HuggingFace Spaces entry point.
HF Spaces expects a file called app.py at the root.
This simply starts the HTTP/SSE MCP server on port 7860.
"""
import os
from tabicl_mcp.server import serve_http

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    serve_http(host="0.0.0.0", port=port)
