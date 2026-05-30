#!/usr/bin/env bash
# arXiv Digest — one-command setup, in the spirit of Night-Notes.
set -e

echo "▶ arXiv Digest setup"

# 1. Python deps
echo "  installing Python dependencies ..."
pip install -r requirements.txt --quiet

# 2. Detect Ollama + a Qwen model
if ! command -v ollama >/dev/null 2>&1; then
  echo "  ⚠ Ollama not found. Install from https://ollama.com/download, then re-run."
  exit 1
fi

MODEL=$(ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -i qwen | head -n1)
if [ -z "$MODEL" ]; then
  echo "  no Qwen model found. Pulling qwen2.5:14b (this is a few GB) ..."
  ollama pull qwen2.5:14b
  MODEL="qwen2.5:14b"
fi
echo "  using Ollama model: $MODEL"

# 3. Patch the model name into arxiv_digest.py
python3 - "$MODEL" <<'PY'
import re, sys, pathlib
model = sys.argv[1]
p = pathlib.Path("arxiv_digest.py")
txt = p.read_text()
txt = re.sub(r'OLLAMA_MODEL\s*=\s*".*?"', f'OLLAMA_MODEL = "{model}"', txt, count=1)
p.write_text(txt)
print(f"  patched OLLAMA_MODEL = {model}")
PY

# 4. Warm the model so the first run is fast
echo "  warming the model ..."
ollama run "$MODEL" "ready" >/dev/null 2>&1 || true

echo "✓ Done. Run it with:  python3 arxiv_digest.py"
echo "  Reports land in ./reports/ and open in your browser."
