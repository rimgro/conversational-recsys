"""Предподсчёт локального BM25-индекса карточек треков (bm25 и relisten): один раз на машину или проект.

    python make_index.py                  # собрать, если в кэше нет, и проверить запросом
    python make_index.py --rebuild        # пересобрать (например, после изменения рецепта карточки)
    !python make_index.py                 # из ячейки ноутбука DataSphere

Индекс собирается из tracks_meta (data.crs.dir) за ~минуту и кладётся в data.cache_dir (cache/bm25_cards_*.npz);
evaluate.py, evaluate_candgen.py и ноутбуки дальше только загружают его. Без файла индекс собирается при первом
запуске пайплайна сам. Ключ кэша — tracks_meta (размер, время), рецепт карточки (CARD_VERSION) и bm25_index.k1 / b:
другой tracks_meta или k1 / b — другой файл. Веса полей (bm25_index.weights) применяются при запросе.
"""
import argparse
import os
import sys
import time

from recsys.config import load_config
from recsys.data.load import load_catalog
from recsys.retrieval.local_index import card_index_path, load_card_index
from recsys.retrieval.sources import query_terms
from recsys.schemas import DialogSummary

CHECK = DialogSummary(query="mellow jazz with female vocals", include_tags=["jazz", "female vocals"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", help="папка датасета; по умолчанию data.crs.dir")
    p.add_argument("--config", action="append", default=[], help="ещё YAML поверх configs/default.yaml")
    p.add_argument("--rebuild", action="store_true", help="пересобрать, даже если индекс есть в кэше")
    a = p.parse_args(argv)

    over = {"data": {"crs": {"dir": a.data_dir}}} if a.data_dir else None
    cfg = load_config(["configs/default.yaml"] + a.config, over)
    t0 = time.time()
    catalog = load_catalog(cfg)
    print(f"каталог: {len(catalog)} треков за {time.time() - t0:.1f} c")
    path = card_index_path(cfg, catalog)
    if path is None:
        print("кэш выключен (data.cache_dir / data.use_cache или не Music4All-CRS): индекс будет собираться при запуске")
    elif a.rebuild and os.path.exists(path):
        os.remove(path)
    t0 = time.time()
    index = load_card_index(cfg, catalog)
    print(f"индекс: {len(index.fields)} полей ({', '.join(index.fields)}), загрузка {time.time() - t0:.1f} c"
          + (f", файл {path} ({os.path.getsize(path) / 2**20:.0f} МБ)" if path and os.path.exists(path) else ""))

    pos, scores = index.search(query_terms(CHECK), 5, weights=cfg.get("bm25_index", {}).get("weights"))
    print(f"\nпроверка: «{CHECK.query}»")
    for p_, s in zip(pos, scores):
        print(f"  {s:6.2f}  {catalog.df.at[p_, 'artist']} — {catalog.df.at[p_, 'title']}")
    if len(pos) == 0:
        print("индекс ничего не нашёл — что-то не так", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
