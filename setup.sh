#!/usr/bin/env bash
# Установка с чистого листа: окружение .venv, пакеты, проверка на примерах из папки модели.
#   ./setup.sh                 # PYTHON=python3.10 ./setup.sh — другой интерпретатор (нужен 3.9–3.12)
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
TORCH_VERSION=2.8.0

[ -d serving ] || { echo "нет папки serving/ (модель и веса): распакуйте архив целиком" >&2; exit 1; }
"$PYTHON" -c 'import sys; assert (3, 9) <= sys.version_info[:2] <= (3, 12), sys.version' \
  || { echo "нужен Python 3.9–3.12: PYTHON=python3.11 ./setup.sh" >&2; exit 1; }

"$PYTHON" -m venv .venv
.venv/bin/pip install -q --upgrade pip
if [ "$(uname)" = "Linux" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
  .venv/bin/pip install -q "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu
else
  .venv/bin/pip install -q "torch==$TORCH_VERSION"
fi
.venv/bin/pip install -q -r requirements.txt

echo "== тесты кода (синтетика, без модели)"
RECSYS_ENV_FILE="" .venv/bin/python -m pytest -q tests
echo "== проверка модели: 20 запросов test_public, сверка с сабмитом (загрузка ~1 мин)"
.venv/bin/python -m recsys.inference check --bundle serving --queries serving/examples/queries.parquet \
  --users serving/examples/users.parquet --reference serving/examples/reference.parquet -n 20
echo "готово: .venv/bin/python -m recsys.inference serve --bundle serving --port 8000"
