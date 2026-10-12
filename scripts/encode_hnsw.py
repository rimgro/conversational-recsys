"""Векторы для кандгена hnsw (recsys/candgen/dense.py): треки — из индекса hnsw-v1, запросы — EmbeddingGemma-2.

    python scripts/encode_hnsw.py                                  # треки + запросы ranker_train, valid, test
    python scripts/encode_hnsw.py --parts submit                   # ещё весь test_public (для сабмита)
    python scripts/encode_hnsw.py --lancedb ~/project/lancedb      # индекс уже скачан и распакован

В --out (по умолчанию candgen.sources.hnsw.dir = artifacts/hnsw):
  item_vectors.npy, item_ids.npy     векторы треков (vector_combined: метаданные + текст песни) и их m4a_id
  query_vectors_<часть>.npz          query_id, vectors (float16)

Векторы треков берутся из LanceDB релиза hnsw-v1 (тот же индекс, что у сервиса HNSW): скачивается lancedb.zip.
Вектор запроса — тот же рецепт, что у сервиса (recsys/candgen/query_embedder.py). На GPU — минуты,
на CPU — около часа на 45k запросов. Подготовка модели и доступа к ней: python scripts/make_embed.py.
Нужны пакеты lancedb и sentence-transformers. Модель предобучена и наших запросов не видела: части можно считать любые.
"""
import argparse
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # recsys из корня репозитория; запускать из корня: python scripts/<скрипт>.py

import numpy as np

from recsys.candgen.query_embedder import GemmaQueryEmbedder
from recsys.config import load_config
from recsys.data.dataset import Dataset

LANCEDB_URL = "https://github.com/rimgro/conversational-recsys/releases/download/hnsw-v1/lancedb.zip"


def item_vectors(lancedb_dir: str, out: str) -> None:
    """LanceDB (таблица tracks: id, vector_combined) -> item_vectors.npy, item_ids.npy."""
    import lancedb
    table = lancedb.connect(lancedb_dir).open_table("tracks").to_arrow()
    ids = np.array(table.column("id").to_pylist(), dtype=str)
    flat = table.column("vector_combined").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    vectors = np.asarray(flat, dtype=np.float32).reshape(len(ids), -1)
    np.save(os.path.join(out, "item_vectors.npy"), vectors)
    np.save(os.path.join(out, "item_ids.npy"), ids)
    print(f"треки: {vectors.shape} -> {out}")


def download_lancedb(work: str) -> str:
    target = os.path.join(work, "lancedb")
    if os.path.exists(os.path.join(target, "tracks.lance")):
        return target
    zip_path = os.path.join(work, "lancedb.zip")
    if not os.path.exists(zip_path):
        print("скачиваю", LANCEDB_URL, flush=True)
        urllib.request.urlretrieve(LANCEDB_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target)
    return target


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--parts", default="ranker_train,valid,test", help="части запросов (recsys/data/parts.py)")
    p.add_argument("--out", help="папка (по умолчанию candgen.sources.hnsw.dir)")
    p.add_argument("--lancedb", help="распакованный индекс hnsw-v1 (по умолчанию скачать в --out)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", help="cuda / cpu (по умолчанию — сама)")
    p.add_argument("--config", action="append", default=[])
    a = p.parse_args(argv)

    cfg = load_config(["configs/default.yaml"] + a.config)
    out = a.out or cfg["candgen"]["sources"]["hnsw"]["dir"]
    os.makedirs(out, exist_ok=True)
    if not os.path.exists(os.path.join(out, "item_vectors.npy")):
        item_vectors(a.lancedb or download_lancedb(out), out)
    embedder = GemmaQueryEmbedder(device=a.device)
    ds = Dataset(cfg)
    for part in [x.strip() for x in a.parts.split(",") if x.strip()]:
        q = ds.part(part)
        path = os.path.join(out, f"query_vectors_{part}.npz")
        t0, vectors = time.time(), []
        texts = q["query"].astype(str).tolist()
        for start in range(0, len(texts), a.batch_size):
            vectors.append(embedder.encode_batch(texts[start:start + a.batch_size], a.batch_size).astype(np.float16))
            done = start + len(vectors[-1])
            if done % (a.batch_size * 20) < a.batch_size or done == len(texts):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"{part}: {done}/{len(texts)}  {rate:.0f} запр/с  осталось ~{(len(texts) - done) / rate / 60:.0f} мин",
                      flush=True)
        np.savez(path, query_id=q["query_id"].astype(str).to_numpy(), vectors=np.concatenate(vectors))
        print(f"{part}: {len(texts)} запросов -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
