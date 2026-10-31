"""Deterministic vector-cache boundaries; no model, GPU or actual player data."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time

import numpy as np
import pytest

from support_copilot.knowledge import HybridRetriever


class Encoder(HybridRetriever):
    def __init__(self, capacity=2, ttl=300):
        super().__init__("invented.json", query_cache_size=capacity, query_cache_ttl_seconds=ttl)
        self.ready = True
        self._vectors = np.eye(2, dtype=np.float32)
        self._encoder_identity = "invented-v1"
        self.calls = []
        self.invalid = False

    def _embed(self, texts, prefix):
        self.calls.append((list(texts), prefix))
        if self.invalid:
            return np.zeros((len(texts), 2), dtype=np.float32)
        result = np.array([[hashlib.sha256(text.encode()).digest()[0]+1, 3] for text in texts], dtype=np.float32)
        return result / np.linalg.norm(result, axis=1, keepdims=True)


def test_identical_queries_share_one_vector_and_returned_arrays_cannot_poison_cache():
    encoder = Encoder()
    first = encoder.encode_queries(["do not ban"])
    original = first.copy()
    first[:] = 0
    assert np.array_equal(encoder.encode_queries(["do not ban"]), original)
    assert len(encoder.calls) == 1
    assert all(isinstance(key[2], bytes) for key in encoder._query_cache)


def test_exact_negation_case_and_encoder_identity_do_not_collide():
    encoder = Encoder(capacity=8)
    encoder.encode_queries(["ban", "do not ban", "BAN"])
    assert len(encoder.calls[0][0]) == 3
    encoder._encoder_identity = "different-encoder"
    encoder.encode_queries(["ban"])
    assert len(encoder.calls) == 2


def test_lru_eviction_ttl_and_explicit_clear_require_new_encoding():
    encoder = Encoder(ttl=1)
    now = [0.]
    encoder._query_clock = lambda: now[0]
    encoder.encode_queries(["first", "second"])
    encoder.encode_queries(["first"])
    encoder.encode_queries(["third"])
    encoder.encode_queries(["second"])
    assert len(encoder.calls) == 3 and len(encoder._query_cache) == 2
    now[0] = 2
    encoder.encode_queries(["second"])
    assert len(encoder.calls) == 4
    encoder.clear_query_cache()
    assert not encoder._query_cache
    encoder.encode_queries(["second"])
    assert len(encoder.calls) == 5


def test_duplicate_batch_rows_are_encoded_once_without_changing_order():
    encoder = Encoder()
    values = encoder.encode_queries(["one", "two", "one"])
    assert encoder.calls == [(["one", "two"], "query")]
    assert np.array_equal(values[0], values[2])


def test_disabled_cache_is_equivalent_but_keeps_no_vectors():
    enabled, disabled = Encoder(), Encoder(capacity=0)
    for _ in range(2):
        assert np.array_equal(enabled.encode_queries(["text"]), disabled.encode_queries(["text"]))
    assert len(enabled.calls) == 1 and len(disabled.calls) == 2 and not disabled._query_cache


def test_concurrent_requests_do_not_duplicate_encoder_work():
    encoder = Encoder()
    with ThreadPoolExecutor(max_workers=4) as pool:
        vectors = list(pool.map(lambda _: encoder.encode_queries(["identical"]), range(8)))
    assert len(encoder.calls) == 1
    assert all(np.array_equal(vector, vectors[0]) for vector in vectors)


def test_invalid_vectors_are_never_cached():
    encoder = Encoder()
    encoder.invalid = True
    with pytest.raises(RuntimeError):
        encoder.encode_queries(["text"])
    assert not encoder._query_cache
