"""HuggingFace Spaces entry point.

HF Spaces expects app.py at the repo root. Starts the Streamable HTTP MCP
server; HF provides $PORT (7860) and $SPACE_HOST (used for report links).
"""
from tabicl_mcp.server import serve_http

if __name__ == "__main__":
    serve_http()
