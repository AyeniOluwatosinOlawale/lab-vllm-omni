#!/usr/bin/env bash
# Phase 0 — Environment verification and smoke test
set -euo pipefail

VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-../vllm-omni}"
PORT=8091
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"

echo "========================================"
echo " vLLM-Omni Lab — Phase 0: Setup Verify"
echo "========================================"

# 1. Python version
echo ""
echo "[1/5] Python version"
python3 --version

# 2. GPU check
echo ""
echo "[2/5] GPU check"
python3 -c "
import torch
if not torch.cuda.is_available():
    print('WARNING: No CUDA GPU detected. Most labs require a GPU.')
else:
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name}  {props.total_memory // 1024**3} GiB')
"

# 3. vLLM-Omni import + version check
REQUIRED_VLLM_OMNI="0.30.0"
echo ""
echo "[3/5] vLLM-Omni import (required: $REQUIRED_VLLM_OMNI)"
python3 -c "
import vllm_omni, sys
installed = vllm_omni.__version__
required = '$REQUIRED_VLLM_OMNI'
print(f'  vllm_omni version: {installed}')
if installed != required:
    print(f'  WARNING: version mismatch — expected {required}, got {installed}')
    print(f'  Fix: pip install vllm-omni=={required}')
    sys.exit(1)
else:
    print(f'  OK: version matches requirements.txt')
" 2>/dev/null \
  || echo "  WARNING: vllm_omni not installed. Run: pip install vllm-omni==$REQUIRED_VLLM_OMNI"

# 4. vLLM-Omni collect_env
echo ""
echo "[4/5] Environment info"
if [ -f "$VLLM_OMNI_DIR/collect_env.py" ]; then
  python3 "$VLLM_OMNI_DIR/collect_env.py" 2>/dev/null | head -30
else
  echo "  vllm-omni source not found at $VLLM_OMNI_DIR — set VLLM_OMNI_DIR to your clone path"
fi

# 5. Serve smoke test
echo ""
echo "[5/5] Serve smoke test (text-only, no GPU memory warmup)"
echo "  Starting server on port $PORT ..."
vllm serve "$MODEL" --omni --no-async-chunk --port $PORT \
  --max-model-len 4096 --gpu-memory-utilization 0.5 &
SERVER_PID=$!

echo "  Waiting for server health ..."
for i in $(seq 1 30); do
  sleep 5
  if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "  Server ready."
    break
  fi
  echo "  Attempt $i/30 ..."
done

echo "  Sending smoke request ..."
curl -s http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly: OK\"}],
    \"modalities\": [\"text\"],
    \"max_tokens\": 5
  }" | python3 -m json.tool

kill $SERVER_PID 2>/dev/null || true
echo ""
echo "Phase 0 complete. Proceed to phase1-ar-audio/lab1_baseline.sh"
