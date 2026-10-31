"""Small disposable fixtures; no model download or private corpus required."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from support_copilot.knowledge import HybridRetriever, _normalize, _tokens


def section(sid, title, points, text=None):
    return {"id": sid, "category": title, "points": [{"ref": ref, "text": value} for ref, value in points],
            "text": text if text is not None else "\n".join(f"{ref} {value}" for ref, value in points)}


class FakeEncoderRetriever(HybridRetriever):
    """Deterministic encoder double, explicitly not a semantic model check."""
    embedding_calls = 0

    def _load_encoder(self):
        self._model = SimpleNamespace(config=SimpleNamespace(hidden_size=3))

    def _embed(self, texts, prefix):
        self.embedding_calls += 1
        self.last_prefix = prefix
        self.last_texts = texts
        vectors = np.array([[len(text) + 1, text.count("a") + 1, 2] for text in texts], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "rules.json"
        self.sections = [
            section(30, "Общие правила игры", [("2.1", "Запрещено оскорбление игрока и родственников."),
                    ("5.2", "Запрещено использовать читы и стороннее ПО.")]),
            section(31, "Правила жалоб", [("1.2", "ЖАЛОБА НЕ ПРИНИМАЕТСЯ, ЕСЛИ:"),
                    ("1.2.1", "Доказательства отсутствуют.")],
                    "I. ПОРЯДОК ЖАЛОБ\nВажное вступление.\n1.2 ЖАЛОБА НЕ ПРИНИМАЕТСЯ, ЕСЛИ:\n1.2.1 Доказательства отсутствуют."),
            section(36, "Правила семейных мероприятий", [("1.23", "Запрещено использовать бронежилеты на мероприятии.")]),
            section(41, "Правила войны за бизнес", [("2.8", "Запрещено использование бронежилетов и винтовки.")]),
            section(42, "Правила битвы за фургон", [("1.7", "Разрешены бронежилеты со склада организации.")]),
            section(43, "RP термины", [], "• DeathMatch (DM) — убийство без причины.\nПримечание: условие.\n• IC — игровой персонаж."),
        ]
        self.write()

    def write(self):
        self.path.write_text(json.dumps({"rules": self.sections}, ensure_ascii=False), encoding="utf-8")

    def lexical(self):
        retriever = HybridRetriever(self.path, dense_enabled=False)
        retriever.load()
        return retriever

    def dense(self):
        model = self.root / "model"
        model.mkdir(exist_ok=True)
        (model / "config.json").write_text('{"hidden_size":3}', encoding="utf-8")
        return FakeEncoderRetriever(self.path, model, self.root / "cache")

    def test_namespaced_collisions_parent_conditions_and_terms(self):
        self.sections.append(section(99, "Другой раздел", [("2.1", "Другое условие.")]))
        self.write()
        retriever = self.lexical()
        self.assertEqual(retriever.stats()["points"], 8)
        self.assertEqual(retriever.stats()["terms"], 2)
        self.assertNotEqual(retriever.get("rules:99:2.1"), retriever.get("rules:30:2.1"))
        child = retriever.get("rules:31:1.2.1")
        self.assertIn("НЕ ПРИНИМАЕТСЯ", child.text)
        self.assertIn("ПОРЯДОК ЖАЛОБ", child.text)
        self.assertEqual(retriever.get("internal:43:term-001").ref, None)
        self.assertIn("Примечание", retriever.get("internal:43:term-001").text)
        self.assertIn("Важное вступление", retriever.get("internal:31:context").text)
        self.assertIsNone(retriever.get("rules:43:1.1"))
        self.assertIsNone(retriever.get("../../private"))
        self.assertLess(len(child.text), 250)
        ambiguous = retriever.search_with_diagnostics("Оскорбление игрока", rule_refs=["2.1"])
        self.assertIn("ambiguous_rule_refs", ambiguous["warnings"])
        scoped = retriever.search_with_diagnostics("Оскорбление игрока", rule_refs=["rules:30:2.1"])
        self.assertNotIn("ambiguous_rule_refs", scoped["warnings"])

    def test_topic_and_event_context_beat_weak_claimed_reference(self):
        retriever = self.lexical()
        result = retriever.search("Оскорбление родственников игрока, он пишет пункт 5.2", ["5.2"])
        self.assertEqual(result[0].canonical_id, "rules:30:2.1")
        self.assertEqual(retriever.search("Игрок использует читы")[0].canonical_id, "rules:30:5.2")
        for query, canonical in [("бронежилеты на семейном мероприятии", "rules:36:1.23"),
                                 ("бронежилеты на войне за бизнес", "rules:41:2.8"),
                                 ("бронежилеты на битве за фургон", "rules:42:1.7")]:
            self.assertEqual(retriever.search(query)[0].canonical_id, canonical)

    def test_ready_empty_queries_and_duplicate_rejection(self):
        retriever = HybridRetriever(self.path, dense_enabled=False)
        with self.assertRaises(RuntimeError):
            retriever.search("query")
        retriever.load()
        self.assertEqual(retriever.search(""), [])
        self.assertEqual(retriever.search("несуществующееслово"), [])
        self.assertEqual(retriever.search("игрок", limit=0), [])
        self.sections[0]["points"].append(self.sections[0]["points"][0])
        self.write()
        with self.assertRaises(ValueError):
            retriever.load()
        self.assertFalse(retriever.ready)

    def test_dense_missing_model_fails_without_silent_fallback(self):
        retriever = HybridRetriever(self.path)
        with self.assertRaises(FileNotFoundError):
            retriever.load()
        self.assertFalse(retriever.ready)

    def test_public_query_encoder_reuses_loaded_model_and_checks_vectors(self):
        retriever = self.dense()
        retriever.load()
        texts = ["invented account question", "invented game question"]
        vectors = retriever.encode_queries(texts)
        self.assertEqual(vectors.shape, (2, 3))
        self.assertEqual(retriever.last_texts, texts)
        self.assertEqual(retriever.last_prefix, "query")
        self.assertTrue(np.allclose(np.linalg.norm(vectors, axis=1), 1))
        for bad in (np.zeros((2, 3)), np.full((2, 3), np.nan), np.ones((1, 3))):
            retriever.clear_query_cache()  # Force a fresh backend call for each invalid encoder.
            retriever._embed = lambda texts, prefix, value=bad: value
            with self.subTest(shape=bad.shape), self.assertRaises(RuntimeError):
                retriever.encode_queries(texts)

    def test_public_query_encoder_requires_dense_ready_and_nonempty_texts(self):
        retriever = self.dense()
        with self.assertRaises(RuntimeError):
            retriever.encode_queries(["text"])
        with self.assertRaises(RuntimeError):
            self.lexical().encode_queries(["text"])
        retriever.load()
        for texts in ([], [" "], [3], "text"):
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                retriever.encode_queries(texts)

    def test_unicode_words_and_uzbek_apostrophes_do_not_split_into_fragments(self):
        for spelling in ("O‘yinchi", "O’yinchi", "Oʻyinchi", "Oʼyinchi", "O'yinchi"):
            self.assertEqual(_tokens(spelling), ["o'yinchi"])
        self.assertEqual(_tokens("құпиясөз әдепті үміт"), ["құпиясөз", "әдепті", "үміт"])
        # No blind legacy-Kazakh substitution of valid Ukrainian characters.
        self.assertEqual(_normalize("ї ґ є"), "ї ґ є")

    def test_multilingual_query_recall_hints_retrieve_original_source(self):
        retriever = self.lexical()
        cases = [
            ("Ойыншы мені балағаттады", "rules:30:2.1"),
            ("O‘yinchi meni haqorat qildi", "rules:30:2.1"),
            ("Ойыншы бөгде бағдарлама қолданды", "rules:30:5.2"),
            ("O'yinchi taqiqlangan dastur ishlatyapti", "rules:30:5.2"),
            ("Отбасылық іс-шарада сауыт киюге бола ма?", "rules:36:1.23"),
            ("Отбасы шайқасында сауыт киюге бола ма?", "rules:36:1.23"),
            ("Бизнес үшін шайқаста сауыт киюге бола ма?", "rules:41:2.8"),
            ("Фургон үшін шайқаста сауыт киюге бола ма?", "rules:42:1.7"),
            ("Oilaviy tadbirda bronejilet mumkinmi?", "rules:36:1.23"),
            ("Biznes uchun jangda bronejilet mumkinmi?", "rules:41:2.8"),
            ("Furgon jangida bronejilet mumkinmi?", "rules:42:1.7"),
        ]
        for query, expected in cases:
            with self.subTest(query=query):
                diagnostics = retriever.search_with_diagnostics(query)
                result = diagnostics["results"][0]
                self.assertEqual(result.canonical_id, expected)
                self.assertEqual(result.text, retriever.get(expected).text)
                self.assertTrue(diagnostics["query_cues"])
                self.assertIn("uncalibrated_retrieval_scores", diagnostics["warnings"])

    def test_ambiguous_event_queries_are_diagnosed_not_decided(self):
        retriever = self.lexical()
        for query in ("Можно использовать бронежилет?", "Bronejilet mumkinmi?", "Сауыт киюге бола ма?"):
            diagnostics = retriever.search_with_diagnostics(query)
            self.assertIn("ambiguous_event_scope", diagnostics["warnings"])
            self.assertEqual(diagnostics["event_sections"], [])
        conflicting = retriever.search_with_diagnostics("Бронежилет на семейном мероприятии и битве за фургон")
        self.assertIn("multiple_event_sections", conflicting["warnings"])
        self.assertIn("ambiguous_event_scope", conflicting["warnings"])
        self.assertEqual(conflicting["event_sections"], [36, 42])

    def test_general_topic_paraphrases_retrieve_sources_without_asserting_violation(self):
        self.sections[0]["points"].extend([
            {"ref": "4.1", "text": "Запрещен обман при передаче и обмене игрового имущества."},
            {"ref": "4.4", "text": "Запрещены продажа и покупка игровой валюты за реальные деньги."},
        ])
        self.write()
        retriever = self.lexical()
        examples = [
            ("Қарсыласым жаман сөз айтты", "rules:30:2.1", "insult"),
            ("U doim yomon gap aytadi", "rules:30:2.1", "insult"),
            ("Только подозреваю чит, доказательства отсутствуют", "rules:30:5.2", "cheat"),
            ("Мүлікті айырбастау туралы сұрағым бар", "rules:30:4.1", "trade_dispute"),
            ("Mulkni almashish haqida savol", "rules:30:4.1", "trade_dispute"),
            ("Какое правило про валюту за реальные деньги?", "rules:30:4.4", "rmt"),
            ("Ойын валютасы нақты ақшаға сатылады", "rules:30:4.4", "rmt"),
            ("Haqiqiy pulga o'yin valyutasi taklif qilindi", "rules:30:4.4", "rmt"),
        ]
        for query, wanted, cue in examples:
            with self.subTest(query=query):
                diagnostic = retriever.search_with_diagnostics(query, limit=5)
                self.assertIn(wanted, [rule.canonical_id for rule in diagnostic["results"]])
                self.assertIn(cue, diagnostic["query_cues"])
                self.assertNotIn("verdict", diagnostic)
                self.assertNotIn("answer", diagnostic)

    def test_armor_inflections_keep_conflicting_sources_and_scope_warning(self):
        retriever = self.lexical()
        for query in ("Броню на бизнес или семейное мероприятие брать?",
                      "Бронь на семейном мероприятии и фургоне"):
            with self.subTest(query=query):
                diagnostic = retriever.search_with_diagnostics(query, limit=10)
                self.assertIn("armor", diagnostic["query_cues"])
                self.assertIn("ambiguous_event_scope", diagnostic["warnings"])
                self.assertGreater(len(diagnostic["event_sections"]), 1)
                for point in ("rules:36:1.23", "rules:41:2.8", "rules:42:1.7"):
                    if int(point.split(":")[1]) in diagnostic["event_sections"]:
                        self.assertIn(point, [rule.canonical_id for rule in diagnostic["results"]])

    def test_real_money_cue_is_not_created_by_generic_money_or_gift(self):
        retriever = self.lexical()
        for query in ("Мне подарили деньги в игре", "Игровую валюту можно обменять?"):
            self.assertNotIn("rmt", retriever.search_with_diagnostics(query)["query_cues"])

    def test_no_partial_reference_matching_or_reference_only_retrieval(self):
        retriever = self.lexical()
        unknown = retriever.search_with_diagnostics("оскорбление родственников", rule_refs=["2.10"])
        self.assertIn("unknown_rule_refs", unknown["warnings"])
        self.assertEqual(unknown["results"][0].canonical_id, "rules:30:2.1")
        unsupported = retriever.search_with_diagnostics("несуществующееслово", rule_refs=["2.1"])
        self.assertEqual(unsupported["results"], [])
        self.assertIn("no_lexical_support", unsupported["warnings"])
        self.assertIn("no_retrieval_candidates", unsupported["warnings"])

    def test_support_faq_gap_is_explicit_without_inventing_answer_or_source(self):
        retriever = self.lexical()
        for query in ("Оплатил донат, бонус не пришел", "To‘lov qildim, bonus kelmadi", "Құпиясөз қате, аккаунтыма кіре алмаймын"):
            diagnostics = retriever.search_with_diagnostics(query)
            self.assertIn("support_faq_not_verified", diagnostics["warnings"])
            self.assertNotIn("answer", diagnostics)
            for result in diagnostics["results"]:
                self.assertEqual(result.text, retriever.get(result.canonical_id).text)

    def test_diagnostics_preserve_original_dense_query_and_source_fingerprint(self):
        retriever = self.dense()
        retriever.load()
        query = "O‘yinchi meni haqorat qildi"
        diagnostics = retriever.search_with_diagnostics(query)
        self.assertEqual(retriever.last_texts, [query])
        self.assertEqual(retriever.last_prefix, "query")
        self.assertEqual(diagnostics["dense_enabled"], True)
        self.assertIn("uzbek_lexical_hint", diagnostics["query_language_signals"])
        self.assertTrue(diagnostics["lexical_hits"])
        self.assertEqual(len(retriever.stats()["source_sha256"]), 64)
        self.assertEqual(retriever.stats()["score_type"], "reciprocal_rank_fusion_not_probability")

    def test_strong_lexical_match_survives_noisy_dense_nearest_neighbour(self):
        # A synthetic role-specific point shares one word, while the general
        # point matches all translated terms. Dense noise prefers the weak point.
        self.sections.append(section(99, "Отдельная роль", [("2.1", "Оскорбление — отдельное условие роли.")]))
        self.write()
        retriever = self.dense()
        retriever.load()
        retriever._embed = lambda texts, prefix: np.tile(np.array([[1, 0, 0]], dtype=np.float32), (len(texts), 1))
        retriever._vectors = np.array([[-1, 0, 0] if doc.canonical_id == "rules:30:2.1" else [1, 0, 0]
                                       for doc in retriever._docs], dtype=np.float32)
        actual = retriever.search("O‘yinchi meni haqorat qildi")
        self.assertEqual(actual[0].canonical_id, "rules:30:2.1")

    def test_cache_reuse_source_and_model_invalidation(self):
        first = self.dense()
        first.load()
        self.assertFalse(first.stats()["cache_hit"])
        self.assertEqual(first.last_prefix, "passage")
        second = self.dense()
        second.load()
        self.assertTrue(second.stats()["cache_hit"])
        self.assertEqual(second.embedding_calls, 0)
        second.search("test")
        self.assertEqual(second.last_prefix, "query")
        self.sections[0]["points"][0]["text"] += " Обновление."
        self.write()
        third = self.dense()
        third.load()
        self.assertFalse(third.stats()["cache_hit"])
        (third.embedding_model_path / "weights.safetensors").write_bytes(b"new identity")
        fourth = FakeEncoderRetriever(self.path, third.embedding_model_path, third.index_dir)
        fourth.load()
        self.assertFalse(fourth.stats()["cache_hit"])

    def test_corrupt_cache_rebuilt_and_cache_paths_not_trusted(self):
        first = self.dense()
        first.load()
        cache = first.index_dir
        for bad in (np.zeros((1, 3), dtype=np.float32),
                    np.full((len(first._docs), 3), np.nan, dtype=np.float32),
                    np.ones((len(first._docs), 1), dtype=np.float32),
                    np.array([{"unsafe": True}], dtype=object)):
            np.save(cache / "vectors.npy", bad)
            next_index = self.dense()
            next_index.load()
            self.assertFalse(next_index.stats()["cache_hit"])
        metadata = json.loads((cache / "metadata.json").read_text())
        metadata["vectors_path"] = "../sensitive.npy"
        (cache / "metadata.json").write_text(json.dumps(metadata))
        next_index = self.dense()
        next_index.load()
        self.assertFalse(next_index.stats()["cache_hit"])


if __name__ == "__main__":
    unittest.main()
