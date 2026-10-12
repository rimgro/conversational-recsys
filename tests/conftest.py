import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))  # скрипты: import make_embed, ...
os.environ["RECSYS_ENV_FILE"] = ""  # тесты не зависят от локального .env с адресами и ключами

from recsys.config import load_config  # noqa: E402
from recsys.resources import Resources  # noqa: E402

OURS = {"card_bm25": {"enabled": True}, "relisten": {"enabled": True}, "audio_seed": {"enabled": True}}


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    """Синтетика (без файлов датасета), кэш и прогоны — во временной папке; все кандгены, кроме плотных
    (им нужны векторы с GPU); ранкер — порядок отбора (обучение CatBoost — отдельный тест)."""
    tmp = tmp_path_factory.mktemp("recsys")
    return load_config(os.path.join(ROOT, "configs", "default.yaml"), {
        "data": {"source": "synthetic", "synthetic": {"n_tracks": 1500, "n_users": 60},
                 "cache_dir": str(tmp / "cache")},
        "candgen": {"sources": OURS},
        "ranker": {"type": "stub"},
        "run": {"dir": str(tmp / "runs")},
    })


@pytest.fixture(scope="session")
def res(cfg):
    return Resources(cfg, verbose=False)


@pytest.fixture(scope="session")
def catalog(res):
    return res.catalog


@pytest.fixture(scope="session")
def requests(res):
    """Запросы части test (test_public синтетики) с ответами."""
    return res.dataset.requests("test")
