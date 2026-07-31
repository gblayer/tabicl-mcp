#!/usr/bin/env bash
# Deploy main to the HuggingFace Space.
#
# The Space needs a YAML config header at the top of README.md, and HF's git
# rejects raw binary files (docs/images/). So we keep a snapshot branch
# `hf-deploy` = content of main, minus docs/, plus the header — and push that.
set -euo pipefail
cd "$(dirname "$0")/.."

git switch -q hf-deploy
git rm -rq --ignore-unmatch . >/dev/null
git checkout -q main -- .
rm -rf docs
printf -- '---\ntitle: TabICL MCP\nemoji: 🤖\ncolorFrom: blue\ncolorTo: green\nsdk: docker\napp_port: 7860\npinned: false\n---\n\n' \
  | cat - README.md > README.md.tmp && mv README.md.tmp README.md
git add -A
if git diff --cached --quiet; then
  echo "Nothing to deploy — hf-deploy already matches main."
else
  git commit -q -m "deploy: sync with main @ $(git rev-parse --short main)"
fi
git push hf hf-deploy:main
git switch -q main
echo "Deployed. Space will rebuild: https://huggingface.co/spaces/gblayer/tabicl-mcp"
