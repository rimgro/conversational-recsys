"""Generate notebooks/datasphere_embed_top200.ipynb (drag-and-drop for DataSphere)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "datasphere_embed_top200.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


CELLS = [
    md(
        """# EmbeddingGemma-2 → top-200 ранкинги для всех диалогов (DataSphere)

**Что делает ноутбук:** считает эмбеддинги всех запросов (`train_queries` + `test_public_queries`)
моделью `google/embeddinggemma-2` и сохраняет **топ-200 треков** для каждого запроса
(точный поиск по всему каталогу 64 016 треков, GPU).

**Как запустить (drag-and-drop):**
1. Откройте проект DataSphere с **GPU-конфигурацией** (V100 достаточно).
2. Загрузите этот ноутбук и запустите все ячейки (Run All).
3. Результат: `~/project/top200_gemma.parquet` (можно скачать из файлового браузера).

**Зависимости:** `sentence-transformers==6.1.0`, `transformers>=5.19`, `lancedb`, `pyarrow`, `pandas`, `tqdm`.
`torch` уже стоит в GPU-конфигурации (cu126).

**Что настраивается:** пути, сплиты, `TOPK`, размеры батчей, `HF_TOKEN` (модель публичная, токен не обязателен).
"""
    ),
    md("## 1. Установка зависимостей"),
    code(
        """# torch уже есть в GPU-конфигурации; ставим только то, чего нет
import os, subprocess, sys
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), "project", "hf_cache"))  # до импорта transformers
def sh(cmd):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, text=True)

sh(f"{sys.executable} -m pip install -q 'sentence-transformers==6.1.0' 'transformers>=5.19' lancedb pyarrow pandas tqdm huggingface_hub")
import torch, sentence_transformers, transformers, lancedb
print("torch", torch.__version__, "| cuda_build", torch.version.cuda, "| CUDA:", torch.cuda.is_available())
print("sentence-transformers", sentence_transformers.__version__, "| transformers", transformers.__version__, "| lancedb", lancedb.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA недоступна: откройте DataSphere с GPU-конфигурацией (V100) и перезапустите ядро.")
"""
    ),
    md("## 2. Конфигурация"),
    code(
        """from pathlib import Path
import os, glob, json

HOME = Path(os.path.expanduser("~/"))
PROJECT = HOME / "project"

# Куда класть результат
OUT_PARQUET = PROJECT / "top200_gemma.parquet"
WORKDIR = PROJECT / "gemma_top200_work"
WORKDIR.mkdir(parents=True, exist_ok=True)

# Индекс треков (LanceDB). Если нет — скачаем из GitHub-релиза hnsw-v1.
LANCEDB_DIR = PROJECT / "lancedb"
LANCEDB_URL = "https://github.com/rimgro/conversational-recsys/releases/download/hnsw-v1/lancedb.zip"

# Сплиты и параметры
SPLITS = ["test_public"]          # можно: ["train", "test_public"]
TOPK = 200
EMBED_BATCH = 256
SEARCH_BATCH = 512
MAX_QUERIES = None                # для смоука: например, 5000
QUERY_PREFIX = "task: search result | query: "
HF_HOME = PROJECT / "hf_cache"
os.environ.setdefault("HF_HOME", str(HF_HOME))
print("OUT_PARQUET:", OUT_PARQUET)
print("SPLITS:", SPLITS, "| TOPK:", TOPK, "| MAX_QUERIES:", MAX_QUERIES)
"""
    ),
    md("## 3. HF-токен (опционально)"),
    code(
        """# Модель публичная, но если в проекте есть HF_TOKEN — логинимся (иногда нужно для gated-моделей/лимитов).
from pathlib import Path
token = os.environ.get("HF_TOKEN")
env_file = PROJECT / "conversational-recsys" / ".env"
if not token and env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.strip().startswith("HF_TOKEN="):
            token = line.split("=", 1)[1].strip().strip('"').strip("'")
if token:
    from huggingface_hub import login
    login(token=token)
    print("HF login: ok")
else:
    print("HF_TOKEN не найден — продолжаем анонимно")
"""
    ),
    md("## 4. Загрузка запросов (все диалоги)"),
    code(
        """import pandas as pd

def find_queries():
    cands = []
    for base in [PROJECT / "datasets", PROJECT / "conversational-recsys", HOME / "datasets", PROJECT]:
        if base.exists():
            cands += [Path(p) for p in glob.glob(str(base / "**" / "*_queries.parquet"), recursive=True)]
    return sorted(set(cands))

files = find_queries()
print("найдены файлы запросов:")
for f in files:
    print("  ", f)
if not files:
    raise SystemExit("Не найдены *_queries.parquet; положите датасет Music4All-CRS в ~/project")

frames = []
for split in SPLITS:
    match = [f for f in files if f.name == f"{split}_queries.parquet"]
    if not match:
        print(f"WARN: нет {split}_queries.parquet"); continue
    df = pd.read_parquet(match[0])
    print(f"{split}: {len(df)} запросов, колонки {list(df.columns)}")
    frames.append(df)
queries = pd.concat(frames, ignore_index=True).drop_duplicates("query_id").reset_index(drop=True)
if MAX_QUERIES:
    queries = queries.head(MAX_QUERIES)
print("итого запросов:", len(queries))
queries[["query_id", "split", "query"]].head()
"""
    ),
    md("## 5. Индекс треков (LanceDB)"),
    code(
        """import zipfile, urllib.request, numpy as np, lancedb

def ensure_lancedb():
    if (LANCEDB_DIR / "tracks.lance").exists():
        return LANCEDB_DIR
    zip_path = WORKDIR / "lancedb.zip"
    if not zip_path.exists():
        print("скачиваем индекс:", LANCEDB_URL, flush=True)
        urllib.request.urlretrieve(LANCEDB_URL, zip_path)
    print("распаковываем", zip_path, flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(LANCEDB_DIR)
    return LANCEDB_DIR

db_dir = ensure_lancedb()
table = lancedb.connect(str(db_dir)).open_table("tracks")
arrow = table.to_arrow()
track_ids = arrow.column("id").to_pylist()
flat = arrow.column("vector_combined").combine_chunks().flatten().to_numpy(zero_copy_only=False)
track_vecs = np.asarray(flat, dtype="float32").reshape(len(track_ids), -1)
norms = np.linalg.norm(track_vecs, axis=1, keepdims=True); norms[norms == 0] = 1.0
track_vecs = track_vecs / norms
print("треков:", len(track_ids), "| dim:", track_vecs.shape[1])
"""
    ),
    md("## 6. Модель EmbeddingGemma-2"),
    code(
        """from sentence_transformers import SentenceTransformer
model = SentenceTransformer("google/embeddinggemma-2", device="cuda", trust_remote_code=True)
model.max_seq_length = 256
_ = model.encode(["warm up"], normalize_embeddings=True, show_progress_bar=False)
print("модель загружена на", next(model.parameters()).device)
"""
    ),
    md("## 7. Эмбеддинги запросов (с кэшем и резюмированием)"),
    code(
        """emb_path = WORKDIR / f"query_emb_{'_'.join(SPLITS)}.npy"
ids_path = WORKDIR / f"query_ids_{'_'.join(SPLITS)}.json"
texts = [QUERY_PREFIX + str(t) for t in queries["query"].tolist()]

if emb_path.exists() and ids_path.exists() and json.loads(ids_path.read_text()) == queries["query_id"].tolist():
    query_vecs = np.load(emb_path)
    print("эмбеддинги загружены из кэша:", query_vecs.shape)
else:
    import time
    vecs = np.zeros((len(texts), 768), dtype="float32")
    t0 = time.time()
    for s in range(0, len(texts), EMBED_BATCH):
        batch = texts[s : s + EMBED_BATCH]
        v = model.encode(batch, batch_size=EMBED_BATCH, normalize_embeddings=True, show_progress_bar=False)
        vecs[s : s + len(batch)] = np.asarray(v, dtype="float32")
        if s % (EMBED_BATCH * 20) == 0:
            done = s + len(batch)
            print(f"  {done}/{len(texts)} ({100*done/len(texts):.1f}%) {done/max(1e-9, time.time()-t0):.0f} запр/с", flush=True)
    query_vecs = vecs
    np.save(emb_path, query_vecs)
    ids_path.write_text(json.dumps(queries["query_id"].tolist()))
    print("готово за", round(time.time() - t0), "с ->", emb_path)
print("query_vecs:", query_vecs.shape)
"""
    ),
    md("## 8. Топ-200 треков (точный поиск на GPU)"),
    code(
        """import torch, time

# Точный поиск по всему каталогу: косинус = скалярное произведение нормализованных векторов.
T = torch.from_numpy(track_vecs).cuda()
if torch.cuda.is_available():
    T = T.half()  # fp16 для скорости; при необходимости замените на .float()

all_ids, all_scores = [], []
t0 = time.time()
with torch.no_grad():
    for s in range(0, len(query_vecs), SEARCH_BATCH):
        q = torch.from_numpy(query_vecs[s : s + SEARCH_BATCH]).cuda()
        if T.dtype == torch.float16:
            q = q.half()
        sims = q @ T.T
        vals, idx = torch.topk(sims, k=min(TOPK, T.shape[0]), dim=1)
        idx = idx.cpu().numpy(); vals = vals.float().cpu().numpy()
        for row_ids, row_scores in zip(idx, vals):
            all_ids.append([track_ids[i] for i in row_ids])
            all_scores.append([float(x) for x in row_scores])
        if s % (SEARCH_BATCH * 20) == 0:
            done = s + len(idx)
            print(f"  {done}/{len(query_vecs)} ({100*done/len(query_vecs):.1f}%)", flush=True)
print("поиск за", round(time.time() - t0), "с")
print("пример:", all_ids[0][:5], [round(x, 3) for x in all_scores[0][:5]])
"""
    ),
    md("## 9. Сохранение результата"),
    code(
        """import pyarrow as pa, pyarrow.parquet as pq

out = pd.DataFrame({
    "query_id": queries["query_id"].tolist(),
    "split": queries["split"].tolist(),
    "top200": all_ids,
    "scores": all_scores,
})
schema = pa.schema([
    ("query_id", pa.string()),
    ("split", pa.string()),
    ("top200", pa.list_(pa.string())),
    ("scores", pa.list_(pa.float32())),
])
pq.write_table(pa.Table.from_pandas(out, schema=schema, preserve_index=False), OUT_PARQUET, compression="zstd")
print("сохранено:", OUT_PARQUET)
print("строк:", len(out), "| размер, МБ:", round(OUT_PARQUET.stat().st_size / 1e6, 1))
out.head(3)
"""
    ),
    md(
        """## 10. Проверка

Быстрая проверка качества (если рядом есть qrels):
```python
qrels = pd.read_parquet(<путь к *_qrels.parquet>)
# nDCG@20 = 1/log2(rank+2) для target_m4a_id
```
"""
    ),
]


nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT, "cells:", len(nb["cells"]))
