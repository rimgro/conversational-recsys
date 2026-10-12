"""CPU checks for feature semantics, joint gradients and portable item inference."""
import io
import unittest

import numpy as np
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise unittest.SkipTest("Optional PyTorch is required for training-tower tests") from exc

from recsys.track_tower import (
    TrackPreprocessor, TrackTransformerTower, TransformerTwoTower,
    category_tokens, lyric_chunks, pack_features, pool_lyric_chunks, export_item_tower,
)


class TrackTowerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.rng = np.random.default_rng(7)
        self.records = [
            {"m4a_genres_full": "rock, pop", "lang": "en", "key": 0,
             "energy": .8, "tempo": 120, "duration_ms": 180000, "release_year": "2001"},
            {"m4a_tags_full": "rock, calm", "energy": 0, "tempo": 0},
            {},
        ]
        self.prep = TrackPreprocessor().fit(self.records)
        cats, numbers, structured = self.prep.transform(self.records)
        present = np.column_stack([np.ones(3), [1, 1, 0], [1, 0, 0], structured])
        self.features = torch.tensor(pack_features(
            self.rng.normal(size=(3, 128)), self.rng.normal(size=(3, 384)),
            self.rng.normal(size=(3, 384)), cats, numbers, present))
        self.tower = TrackTransformerTower(384, len(self.prep.vocabulary)).eval()

    def test_preprocessing_identity_missing_and_unknown(self):
        tokens = category_tokens({"m4a_genres_full": "rock", "m4a_tags_full": "rock"})
        self.assertEqual(len(set(tokens)), 2)  # The two fields cannot alias.
        self.assertEqual(category_tokens({"key": 0, "mode": 1, "is_instrumental": False}),
                         category_tokens({"key": 0.0, "mode": "1.0", "is_instrumental": "false"}))
        restored = TrackPreprocessor.from_dict(self.prep.to_dict())
        original = self.prep.transform(self.records)
        for a, b in zip(original, restored.transform(self.records)):
            np.testing.assert_array_equal(a, b)
        cats, numbers, present = original
        self.assertEqual(numbers[1, 6 + 1], 1)  # energy=0 is observed.
        self.assertEqual(numbers[1, 6 + 3], 0)  # tempo=0 is missing.
        self.assertFalse(present[2].any())
        self.assertEqual(restored.transform([{"lang": "never_seen"}])[0][0, 0], 1)

    def test_lyrics_bounded_empty_and_pooling(self):
        self.assertEqual(lyric_chunks(None), [])
        text = " ".join(f"word{i}" for i in range(1500))
        chunks = lyric_chunks(text)
        self.assertEqual(len(chunks), 8)
        self.assertTrue(chunks[0].startswith("word0 "))
        self.assertTrue(chunks[-1].endswith("word1499"))
        result = pool_lyric_chunks(np.array([[1, 0], [0, 1]], dtype=np.float32), [0, 0], 2)
        np.testing.assert_allclose(result[0], [2**-.5, 2**-.5])
        np.testing.assert_array_equal(result[1], [0, 0])

    def test_missing_blocks_cannot_change_embedding(self):
        altered = self.features.clone()
        for i, field in enumerate(("audio", "card", "lyrics", "categories", "numbers")):
            absent = self.features[:, self.tower.layout["present"]][:, i].eq(0)
            altered[absent, self.tower.layout[field]] = 1 if field == "categories" else 900
        with torch.inference_mode():
            expected, actual = self.tower(self.features), self.tower(altered)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_finite_unit_vectors_empty_and_batch_independence(self):
        empty = torch.zeros_like(self.features[:1])
        with torch.inference_mode():
            result = self.tower(self.features)
            single = torch.cat([self.tower(row[None]) for row in self.features])
            fallback = self.tower(empty)
        self.assertEqual(tuple(result.shape), (3, 256))
        self.assertTrue(torch.isfinite(fallback).all())
        torch.testing.assert_close(result.norm(dim=1), torch.ones(3))
        torch.testing.assert_close(single, result, atol=2e-6, rtol=2e-6)

    def test_category_order_has_no_effect(self):
        altered = self.features.clone()
        altered[:, self.tower.layout["categories"]] = altered[:, self.tower.layout["categories"]].flip(1)
        with torch.inference_mode():
            torch.testing.assert_close(self.tower(altered), self.tower(self.features))

    def test_joint_loss_reaches_attention_categories_and_query(self):
        model = TransformerTwoTower(384, 516, self.features.shape[1], 512, 256, .1,
                                    len(self.prep.vocabulary))
        query = model.query(torch.randn(3, 384), torch.randn(3, 516))
        tracks = model.item(self.features)
        loss = torch.nn.functional.cross_entropy(query @ tracks.T / .08, torch.arange(3))
        loss.backward()
        for weight in (model.query_net[1].weight, model.item_net.audio[1].weight,
                       model.item_net.category_embedding.weight,
                       model.item_net.transformer.layers[0].self_attn.in_proj_weight):
            self.assertTrue(torch.isfinite(weight.grad).all())
            self.assertGreater(float(weight.grad.abs().sum()), 0)
        first, second = model.item_net.transformer.layers
        self.assertFalse(torch.equal(first.self_attn.in_proj_weight, second.self_attn.in_proj_weight))

    def test_torchscript_reload_variable_batch_and_masks(self):
        script = export_item_tower(self.tower, self.features)
        stream = io.BytesIO()
        torch.jit.save(script, stream)
        stream.seek(0)
        restored = torch.jit.load(stream).eval()
        probe = torch.cat([self.features, torch.zeros_like(self.features), self.features[:1]])
        with torch.inference_mode():
            torch.testing.assert_close(restored(probe), self.tower(probe), atol=2e-5, rtol=2e-5)

    def test_invalid_features_fail_before_attention(self):
        values = self.features.numpy()
        layout = self.tower.layout
        parts = [values[:, layout[k]] for k in ("audio", "card", "lyrics", "categories", "numbers", "present")]
        parts[3] = parts[3].copy(); parts[3][0, 0] = .5
        with self.assertRaisesRegex(ValueError, "integers"):
            pack_features(*parts)


if __name__ == "__main__":
    unittest.main()
