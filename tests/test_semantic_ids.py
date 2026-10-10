"""recsys/semantic_ids.py на маленьком RQ-VAE: квантование, центроиды, выравнивание по трекам."""
import numpy as np
import pytest

from recsys.semantic_ids import RQVAE, SemanticIDs


def _rqvae(seed=0, d_in=8, d=4, K=5, L=3):
    rng = np.random.default_rng(seed)
    w = {"encoder.0.weight": rng.normal(size=(d_in, d_in)), "encoder.0.bias": rng.normal(size=d_in),
         "encoder.2.weight": rng.normal(size=(d, d_in)), "encoder.2.bias": rng.normal(size=d),
         "encoder.4.weight": rng.normal(size=(d, d))}
    w.update({f"codebooks.{k}": rng.normal(size=(K, d)) for k in range(L)})
    return RQVAE({k: v.astype(np.float32) for k, v in w.items()})


def test_quantize_is_greedy_residual():
    rq = _rqvae()
    z = rq.encode(np.random.default_rng(1).normal(size=(20, 8)))
    codes = rq.quantize(z)
    assert codes.shape == (20, 3) and codes.max() < 5
    r = z.copy()
    for k, cb in enumerate(rq.codebooks):  # тот же жадный выбор, посчитанный в лоб
        assert (np.linalg.norm(r[:, None] - cb[None], axis=2).argmin(1) == codes[:, k]).all()
        r -= cb[codes[:, k]]


def test_semantic_ids_vectors_and_alignment():
    rq = _rqvae()
    codes = np.c_[rq.quantize(rq.encode(np.random.default_rng(2).normal(size=(6, 8)))), np.zeros(6, int)]
    codes[1, :3] = codes[0, :3]
    codes[1, 3] = 1  # коллизия первых трёх — различает четвёртая позиция
    sid = SemanticIDs(["t5", "t1", "t3", "t0", "t4", "t2"], codes, rq.codebooks)
    assert sid.vocab_sizes == [5, 5, 5, 2]
    assert sid.centroids().shape == (6, 3, 4)
    assert np.allclose(sid.quantized()[0], sum(rq.codebooks[k][codes[0, k]] for k in range(3)))
    assert (sid.aligned(["t1", "t5"]) == codes[[1, 0]]).all()
    with pytest.raises(KeyError):
        sid.positions(["нет такого"])
    with pytest.raises(ValueError):
        SemanticIDs(["a"], codes)
