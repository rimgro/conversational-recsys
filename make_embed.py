"""Подготовка эмбеддинга запросов для HNSW (EmbeddingGemma-2): один раз на машину или проект.

    python make_embed.py              # проверить пакеты, скачать веса, построить вектор, проверить на сервере HNSW
    python make_embed.py --no-server  # без проверки на сервере
    !python make_embed.py             # из ячейки ноутбука DataSphere

GPU (cuda, float16), если он есть, иначе CPU (float32). Веса: в DataSphere — /home/jupyter/project/hf_cache
(диск проекта), локально — ~/.cache/huggingface; дальше inference.ipynb и evaluate.py берут модель оттуда.
Нужен доступ к модели на Hugging Face: принять лицензию google/embeddinggemma-2 и задать HF_TOKEN
(в DataSphere — секрет проекта; локально — переменная окружения или файл .env). Код выхода 1 при ошибке.
"""
import argparse
import os
import sys
import time

from recsys.config import load_env
from recsys.retrieval.query_embedder import MODEL_NAME, GemmaQueryEmbedder, setup_cache

INSTALL_DATASPHERE = """  # DataSphere: драйвер NVIDIA с CUDA 12.x — torch под CUDA 12.6, затем перезапустить ядро
  !pip install --progress-bar on "torch==2.11.0" "torchaudio==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu126
  !pip install --progress-bar on "sentence-transformers==6.1.0" "transformers>=5.19" huggingface_hub"""
INSTALL_LOCAL = """  # локально (CPU), Python 3.10+:
  pip install torch "sentence-transformers==6.1.0" "transformers>=5.19" huggingface_hub"""
CHECK_QUERY = "play smells like teen spirit by nirvana"


def step(title: str) -> None:
    print(f"\n== {title}", flush=True)


def check_packages() -> bool:
    step("1. пакеты")
    if sys.version_info < (3, 10):
        print(f"Python {sys.version.split()[0]}: transformers 5 и sentence-transformers 6 требуют Python 3.10+")
        print(INSTALL_LOCAL)
        return False
    try:
        import sentence_transformers
        import torch
        import transformers
    except ImportError as e:
        print(f"не хватает пакета: {e.name}\n{INSTALL_DATASPHERE}\n{INSTALL_LOCAL}")
        return False
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) | sentence-transformers "
          f"{sentence_transformers.__version__} | transformers {transformers.__version__}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    elif torch.version.cuda and os.path.exists("/usr/bin/nvidia-smi"):
        print("GPU есть, но torch его не видит: сборка torch не под драйвер (см. nvidia-smi). Считаем на CPU.\n"
              + INSTALL_DATASPHERE)
    else:
        print("GPU нет — вектор считается на CPU (медленнее, но работает)")
    return True


def download() -> bool:
    step("2. веса " + MODEL_NAME)
    print("кэш:", setup_cache())
    from huggingface_hub import snapshot_download
    try:
        path = snapshot_download(MODEL_NAME, token=os.environ.get("HF_TOKEN") or None)
    except Exception as e:  # 401/403 — нет токена или не принята лицензия; сеть
        print(f"не скачалось: {type(e).__name__}: {e}\n"
              "  проверьте: лицензия модели принята на huggingface.co, HF_TOKEN задан (секрет проекта / .env)")
        return False
    print("веса:", path)
    return True


def build(embedder: GemmaQueryEmbedder) -> bool:
    step("3. вектор запроса")
    t = time.time()
    try:
        embedder.encode("warm up")  # первый вызов загружает модель
    except Exception as e:
        print(f"модель не загрузилась: {type(e).__name__}: {e}")
        return False
    print(f"модель загружена на {embedder.device} за {time.time() - t:.1f} c")
    times = []
    for q in ("calm jazz for a late evening", "fast high energy rock in minor key", "a song about rain"):
        t = time.time()
        vec = embedder.encode(q)
        times.append(time.time() - t)
    print(f"{len(vec)} чисел, {sorted(times)[1] * 1000:.0f} мс на запрос")
    return len(vec) == 768


def check_server(embedder: GemmaQueryEmbedder) -> bool:
    step("4. совместимость с индексом на сервере HNSW")
    from recsys.retrieval.remote import CandgenClient, CandgenError
    if not os.environ.get("HNSW_URL"):
        print("HNSW_URL не задан (.env / секрет проекта) — проверка пропущена")
        return True
    client = CandgenClient("${HNSW_URL}", api_key="${HNSW_API_KEY}", timeout=30)
    try:
        resp = client.post("/hnsw/search", {"vector": embedder.encode(CHECK_QUERY), "k": 5})
    except CandgenError as e:
        print("сервер не ответил:", e)
        return False
    for item in resp.get("items", []):
        print(f"  {item['score']:.3f}  {item['artist']} — {item['title']}")
    ok = bool(resp.get("items")) and "nirvana" in resp["items"][0]["artist"].lower()
    print("вектор совместим с индексом" if ok else
          "первым должен быть Nirvana — Smells Like Teen Spirit: рецепт вектора разошёлся с индексом")
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--no-server", action="store_true", help="не проверять на сервере HNSW")
    p.add_argument("--device", help="cuda / cpu (по умолчанию — сама)")
    a = p.parse_args(argv)
    load_env()  # HF_TOKEN, HNSW_URL, HNSW_API_KEY из .env, если он есть
    if not check_packages() or not download():
        return 1
    embedder = GemmaQueryEmbedder(device=a.device)
    if not build(embedder):
        return 1
    if not a.no_server and not check_server(embedder):
        return 1
    print("\nготово: inference.ipynb и evaluate.py --candgen строят вектор для HNSW сами")
    return 0


if __name__ == "__main__":
    sys.exit(main())
