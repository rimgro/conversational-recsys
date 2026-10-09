import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["RECSYS_ENV_FILE"] = ""  # тесты не зависят от локального .env с адресами и ключами

from recsys.config import load_config  # noqa: E402
from recsys.data import load_data  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    # сервис HNSW знает только каталог датасета: на синтетике работают relisten, audio и bm25,
    # HNSW проверяет tests/test_remote.py против поддельного сервера
    return load_config(os.path.join(ROOT, "configs", "default.yaml"),
                       {"data": {"source": "synthetic", "synthetic": {"n_tracks": 1500, "n_users": 60}},
                        "retrieval": {"hnsw": {"enabled": False}}})


@pytest.fixture(scope="session")
def data(cfg):
    return load_data(cfg, verbose=False)
