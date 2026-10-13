"""Векторы названий треков («артист — название») для lazy-кандгена (twotower.item.text, docs/lazy.md).

    python scripts/encode_item_text.py                                   # -> artifacts/item_text/qwen3emb.npz

Qwen3-Embedding (twotower.model.name), последний токен (EOS), L2; без инструкции — так в Qwen3-Embedding кодируются
документы. Файл: ids (m4a_id), vectors (n, hidden) float16. 64k коротких строк: пара минут на GPU, ~10 на Mac.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # recsys из корня репозитория; запускать из корня: python scripts/<скрипт>.py

import numpy as np
import pandas as pd

from train_twotower import build_cfg
from recsys.twotower.build import load_codes, tracks_meta_path
from recsys.twotower.data import TokenizerSpec
from recsys.twotower.model import load_base
from recsys.twotower.train import autocast, device_attention, device_precision, pick_device


def main(argv=None) -> int:
    import torch
    import torch.nn.functional as F
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="artifacts/item_text/qwen3emb.npz")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--config", action="append", default=[])
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--device")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    cfg = build_cfg(a)
    mcfg = cfg["twotower"]["model"]
    ids = load_codes(cfg).sid.track_ids
    meta = pd.read_parquet(tracks_meta_path(cfg), columns=["m4a_id", "m4a_artist", "m4a_song"])
    meta = meta.drop_duplicates("m4a_id").set_index("m4a_id").reindex(ids)
    texts = [f"{a_} — {s_}" for a_, s_ in zip(meta["m4a_artist"].fillna(""), meta["m4a_song"].fillna(""))]

    device = pick_device(a.device)
    precision = device_precision(device, mcfg.get("precision", "bf16"))
    base, tok = load_base(mcfg["name"], precision, device_attention(device, mcfg.get("attn_implementation", "sdpa")))
    base.to(device).eval()
    spec = TokenizerSpec.from_hf(tok)
    out, t0 = [], time.time()
    for s in range(0, len(texts), a.batch_size):
        seqs = [t[:62] + [spec.eos_id] for t in spec.tokenize(texts[s:s + a.batch_size])]
        T = max(map(len, seqs))
        x = torch.full((len(seqs), T), spec.pad_id, dtype=torch.long)
        m = torch.zeros_like(x)
        for i, q in enumerate(seqs):                     # паддинг слева: EOS — последняя позиция у всех
            x[i, T - len(q):], m[i, T - len(q):] = torch.tensor(q), 1
        with torch.no_grad(), autocast(device, precision):
            h = base(input_ids=x.to(device), attention_mask=m.to(device)).last_hidden_state[:, -1]
        out.append(F.normalize(h.float(), dim=-1).cpu().numpy().astype(np.float16))
        if (s // a.batch_size) % 50 == 0:
            print(f"[item_text] {min(s + a.batch_size, len(texts))}/{len(texts)}, {time.time() - t0:.0f} c")
    v = np.concatenate(out)
    if not np.isfinite(v.astype(np.float32)).all():
        raise RuntimeError("NaN в векторах названий")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, ids=np.asarray(ids).astype(str), vectors=v)
    print(f"[item_text] {v.shape} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
