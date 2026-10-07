import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from recsys.config import load_config  # noqa: E402
from recsys.data import load_data  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(os.path.join(ROOT, "configs", "default.yaml"),
                       {"data": {"source": "synthetic", "synthetic": {"n_tracks": 1500, "n_users": 100,
                                                                      "n_dialogs": 60}}})


@pytest.fixture(scope="session")
def data(cfg):
    return load_data(cfg, verbose=False)
