import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def items():
    return pd.DataFrame({
        'spotify_id': ['a', 'b', 'c', 'd', 'e', 'f'],
        'genres': ['pop,dance pop', "['rock','hard rock']", None, '',
                   'r&b,k-pop', 'alternative rock,indie pop'],
    })
