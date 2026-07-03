#!/usr/bin/env bash
set -euo pipefail

echo "[LAYER] Using existing MAT A6000 environment by default."
echo "Python:"
python --version

python - <<'PY'
import importlib
for name in ["torch", "diffusers", "transformers", "numpy", "PIL", "matplotlib"]:
    try:
        mod = importlib.import_module(name)
        print(f"{name}: {getattr(mod, '__version__', 'available')}")
    except Exception as exc:
        print(f"{name}: missing ({exc})")
PY

echo
echo "If a dependency is missing, install it into the existing A6000 env, not into this repo."
