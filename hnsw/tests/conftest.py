import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def items():
    # a, b близко друг к другу; c — в другую сторону; d без вектора
    return pd.DataFrame({
        'm4a_id': ['a', 'b', 'c', 'd'],
        'muq_embedding': [np.array([1.0, 0.0, 0.0]), np.array([0.9, 0.1, 0.0]), np.array([0.0, 0.0, 2.0]), None],
    })
