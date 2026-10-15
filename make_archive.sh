#!/usr/bin/env bash
# На машине разработки: примеры для проверки в serving/examples/ и архив dist/recsys-vps-final6.tar.gz
# (код ветки — только закоммиченное, git archive — и папка модели serving/).
#   PYTHON=../recsys/.venv/bin/python ./make_archive.sh <папка датасета> <сабмит final6: query_id, top20>
set -euo pipefail
cd "$(dirname "$0")"
DATA="$1"; REF="$2"; PY="${PYTHON:-python3}"
[ -f serving/config.yaml ] || { echo "нет serving/: сначала python -m recsys.inference export (README)" >&2; exit 1; }

"$PY" - "$DATA" "$REF" <<'PYEOF'
import os, sys
import pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
data, ref = sys.argv[1:3]
q = pd.read_parquet(f"{data}/test_public_queries.parquet", columns=["query_id", "user_id", "split", "query"])
q = q.sample(20, random_state=0).reset_index(drop=True)
kind = pq.read_schema(f"{data}/test_public.parquet").field("user_id").type
users = pq.read_table(f"{data}/test_public.parquet",
                      columns=["user_id", "split", "history", "history_ts", "user_profile", "user_demographics"],
                      filters=pc.is_in(pc.field("user_id"), value_set=pa.array(sorted(set(q.user_id)), type=kind))).to_pandas()
r = pd.read_parquet(ref)
r = r[r.query_id.isin(q.query_id)].reset_index(drop=True)
assert len(r) == len(q), "в сабмите нет части примеров"
os.makedirs("serving/examples", exist_ok=True)
q.to_parquet("serving/examples/queries.parquet", index=False)
users.to_parquet("serving/examples/users.parquet", index=False)
r.to_parquet("serving/examples/reference.parquet", index=False)
print(f"serving/examples: {len(q)} запросов, {len(users)} пользователей, эталон {ref}")
PYEOF

mkdir -p dist
OUT=dist/recsys-vps-final6.tar
git archive --format=tar --prefix=recsys-vps/ -o "$OUT" HEAD
if tar --version 2>/dev/null | grep -q GNU; then
  tar -rf "$OUT" --transform 's,^serving,recsys-vps/serving,' serving
else
  tar -rf "$OUT" -s ',^serving,recsys-vps/serving,' serving
fi
gzip -f "$OUT"
ls -lh "$OUT.gz"
