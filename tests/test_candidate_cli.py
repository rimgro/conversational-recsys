"""End-to-end CLI filtering must overfetch and use the full discovery history."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('transformers')
from recsys.two_tower_qwen import bundle

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/predict_qwen.py'
spec = importlib.util.spec_from_file_location('predict_cli', SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class Index:
    def __init__(self):
        self.calls = []

    def set_ef(self, value):
        self.ef = value

    def knn_query(self, vector, k):
        self.calls.append(k)
        return np.arange(k)[None, :], np.linspace(0, 1, k)[None, :]


class Retriever:
    def __init__(self, *args, **kwargs):
        self.catalog = [{'m4a_id': str(i), 'spotify_id': 'group' + str(i//2),
            'artist': 'artist' + str((i//2) % 5)} for i in range(2100)]
        self.ids = [r['m4a_id'] for r in self.catalog]
        self.index = Index()
        self.manifest = {'bundle_id': 'matched-test-bundle'}

    def encode_query(self, query, history):
        self.received_history = history
        return np.array([1., 0.], dtype=np.float32)


def test_ann_overfetch_preserves_full_history_and_returns_200_canonical(monkeypatch, tmp_path, capsys):
    model = Retriever()
    monkeypatch.setattr(bundle, 'QwenRetriever', lambda *a, **kw: model)
    history = [str(2*i) for i in range(450)]
    path = tmp_path / 'history.json'
    path.write_text(json.dumps(history))
    monkeypatch.setattr(sys, 'argv', ['predict', '--bundle', str(tmp_path), '--query', 'query',
        '--history', str(path), '--device', 'cpu', '--discovery'])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result['returned'] == 200
    assert model.received_history == history
    assert model.index.calls == [1024, 2048]
    keys = [int(c['m4a_id'])//2 for c in result['candidates']]
    assert len(set(keys)) == 200 and min(keys) >= 450


def test_exact_filters_reference_aliases_and_artist(monkeypatch, tmp_path, capsys):
    model = Retriever()
    monkeypatch.setattr(bundle, 'QwenRetriever', lambda *a, **kw: model)
    vectors = np.column_stack((np.linspace(1, 0, 2100), np.zeros(2100))).astype(np.float32)
    np.save(tmp_path / 'track_vectors.npy', vectors)
    monkeypatch.setattr(sys, 'argv', ['predict', '--bundle', str(tmp_path), '--query', 'query',
        '--device', 'cpu', '--exact', '--query-type', 'similar_to',
        '--exclude-m4a-id', '0', '--exclude-artist', 'artist0'])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result['returned'] == 200 and not model.index.calls
    keys = [int(c['m4a_id'])//2 for c in result['candidates']]
    assert len(set(keys)) == 200 and 0 not in keys
    assert all(g % 5 != 0 for g in keys)
    scores = [c['score'] for c in result['candidates']]
    assert all(a >= b for a, b in zip(scores, scores[1:]))
