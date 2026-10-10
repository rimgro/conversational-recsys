"""Проверка semantic ID перед обучением двухбашенного кандгена (файлы и формат: docs/semantic_ids.md).

    python check_semantic_ids.py                       # пути из configs/default.yaml (semantic_ids, data.crs.dir)
    python check_semantic_ids.py --out outputs/semantic_ids_check   # ещё и группы коллизий / дублей в csv

Что печатает:
  1. коды: сколько значений у каждой позиции и сколько реально используется;
  2. коллизии: сколько треков делят первые 1 / 2 / 3 / все коды с другим треком (одинаковый полный код —
     одинаковый вектор в трековой башне, модель не отличит);
  3. порядок строк: чекпоинт RQ-VAE по MuQ из tracks_meta (по возрастанию m4a_id) воспроизводит коды;
     доля совпадений и ошибка квантования;
  4. дубли песен: один ISRC или одинаковые «артист + название» без версии — для полного softmax это негативы,
     которые на деле та же песня; сколько из них ещё и делят код.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from evaluate import config_path
from recsys.config import load_config
from recsys.data.crs import split_path
from recsys.semantic_ids import RQVAE, load_semantic_ids, semantic_paths


def code_usage(codes: np.ndarray) -> None:
    for k, col in enumerate(codes.T):
        counts = np.bincount(col)
        used = counts[counts > 0]
        print(f"  позиция {k}: значения {col.min()}..{col.max()}, используется {len(used)}; "
              f"треков на значение: медиана {int(np.median(used))}, макс {used.max()}")


def collisions(ids: np.ndarray, codes: np.ndarray) -> pd.DataFrame:
    """Печатает долю треков с общим префиксом кода; -> треки с одинаковыми первыми тремя кодами."""
    df = pd.DataFrame(codes, columns=[f"c{k}" for k in range(codes.shape[1])])
    n = len(df)
    for k in range(1, codes.shape[1] + 1):
        prefix = df.iloc[:, :k]
        shared = int(prefix.duplicated(keep=False).sum())
        print(f"  первые {k}: {len(prefix.drop_duplicates())} разных, делят с другим треком {shared} ({shared / n:.2%})")
    three = df.iloc[:, :3]
    same = three.duplicated(keep=False).to_numpy()
    return df[same].assign(m4a_id=ids[same]).sort_values(list(three.columns))


def reproduce(meta_path: str, ids: np.ndarray, codes: np.ndarray, checkpoint: str) -> None:
    rq = RQVAE.load(checkpoint)
    meta = pq.read_table(meta_path, columns=["m4a_id", "muq_embedding"]).to_pandas().set_index("m4a_id")
    x = np.stack(meta.loc[ids, "muq_embedding"].to_numpy()).astype(np.float32)
    z = rq.encode(x)
    got = rq.quantize(z)
    L = got.shape[1]
    match = got == codes[:, :L]
    resid = z - rq.codebooks[np.arange(L)[None, :], got].sum(1)
    print(f"  чекпоинт: вход {x.shape[1]}, z {z.shape[1]}, кодбуки {rq.codebooks.shape}")
    print(f"  совпадение кодов по уровням: {np.round(match.mean(0), 4).tolist()}, все {L}: {match.all(1).mean():.2%}")
    print(f"  ошибка квантования |z - сумма центроидов| / |z|: "
          f"{np.mean(np.linalg.norm(resid, axis=1) / np.linalg.norm(z, axis=1)):.3f}")
    if not match.all():
        print("  внимание: коды не воспроизводятся — порядок строк или чекпоинт не те", file=sys.stderr)


_VERSION = re.compile(r"\s*[\(\[].*?[\)\]]|\s+-\s+.*$")   # «(Remastered 2011)», «[Live]», «- Radio Edit»


def duplicates(meta_path: str, ids: np.ndarray, codes: np.ndarray) -> pd.DataFrame:
    meta = pq.read_table(meta_path, columns=["m4a_id", "m4a_artist", "m4a_song", "isrc"]).to_pandas()
    meta["song"] = [f"{str(a).lower().strip()} || {_VERSION.sub('', str(t).lower()).strip()}"
                    for a, t in zip(meta["m4a_artist"], meta["m4a_song"])]
    isrc = meta["isrc"].where(meta["isrc"].notna() & (meta["isrc"] != ""))
    by_isrc = isrc.map(isrc.value_counts()).fillna(0) > 1
    by_song = meta["song"].map(meta["song"].value_counts()) > 1
    dup = meta[by_isrc | by_song].copy()
    print(f"  один ISRC: {int(by_isrc.sum())} треков ({by_isrc.mean():.2%}); одинаковые артист + название: "
          f"{int(by_song.sum())} ({by_song.mean():.2%}); всего с дублем: {len(dup)} ({len(dup) / len(meta):.2%})")
    pos = {t: i for i, t in enumerate(ids)}
    dup["code"] = [tuple(codes[pos[t], :3]) for t in dup["m4a_id"]]
    same = dup.groupby(["song", "code"])["m4a_id"].transform("size") > 1
    print(f"  из них делят и первые три кода: {int(same.sum())} треков; остальные — разные векторы, "
          f"в полном softmax это негатив, который на деле та же песня")
    return dup


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=config_path("default.yaml"))
    p.add_argument("--out", help="папка: collisions.csv (общие первые три кода) и duplicates.csv")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    paths = semantic_paths(cfg)
    meta_path = split_path(cfg["data"]["crs"]["dir"], "tracks_meta")
    sid = load_semantic_ids(cfg, checkpoint=False)
    print(f"[semantic_ids] {paths['codes']}: {sid.codes.shape}, треков в tracks_meta: {len(sid)}")

    print("\n1. коды")
    code_usage(sid.codes)
    print("\n2. коллизии")
    coll = collisions(sid.track_ids, sid.codes)
    print("\n3. порядок строк и чекпоинт")
    if os.path.exists(paths["checkpoint"]):
        reproduce(meta_path, sid.track_ids, sid.codes, paths["checkpoint"])
    else:
        print(f"  нет {paths['checkpoint']}: не проверяем")
    print("\n4. дубли песен")
    dup = duplicates(meta_path, sid.track_ids, sid.codes)

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        coll.to_csv(os.path.join(a.out, "collisions.csv"), index=False)
        dup.to_csv(os.path.join(a.out, "duplicates.csv"), index=False)
        print(f"\nгруппы сохранены в {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
