"""Data-boundary and safety checks; no model packages or inference required."""
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_soup_data", ROOT / "scripts" / "prepare_soup_data.py")
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


class BoundaryTests(unittest.TestCase):
    def test_contact_and_nickname_scrubbing(self):
        text = "Пишите me@example.com https://example.org/a example.ru t.me/name +7 (701) 123-45-67 Boris_Scott29"
        clean = data.scrub(text)
        for secret in ("example", "t.me", "701", "Boris_Scott", "@", "https"):
            self.assertNotIn(secret, clean)
        self.assertIn("[EMAIL]", clean)
        self.assertIn("[PHONE]", clean)
        self.assertIn("[PLAYER]", clean)

    def test_normalization_handles_obvious_leakage_variants(self):
        self.assertEqual(data.normalize(" Ёж_ОСКОРБИЛ!!! меня "), data.normalize("еж оскорбил меня"))
        keys = data.text_keys({"messages": [{"role": "player", "text": "Первое"},
                                           {"role": "admin", "text": "SECRET"},
                                           {"role": "player", "text": "Второе"}]})
        self.assertIn("первое", keys)
        self.assertIn("второе", keys)
        self.assertNotIn("secret", keys)

    def test_sensitive_topics_override_lexical_rules_and_appeals_rejected(self):
        self.assertEqual(data.classify("Меня взломали, я вышел с аккаунта"), "account")
        self.assertEqual(data.classify("Донат не пришел после оплаты"), "payment")
        self.assertIsNone(data.classify("Меня забанили за читы, игрок был без читов"))
        self.assertIsNone(data.classify("Помогите пожалуйста с проблемой"))
        self.assertIsNone(data.classify("Игрок оскорбляет меня и использует читы"))
        for ambiguous in ("Как в меню опг взять заказ на взлом авто", "Админ взломал и сажает всех подряд",
                          "Не могу оплатить квартиру", "В кейсах донат сняли а призы не пришли",
                          "Игрок оскорблял и заскамил меня на 300к", "Игрок угрожает что взломает мой аккаунт",
                          "Я никого не оскорблял, прошу пересмотреть жалобу", "Выдали мут за оскорбление игрока"):
            self.assertIsNone(data.classify(ambiguous), ambiguous)

    def test_output_cannot_leave_private_owned_directory(self):
        with self.assertRaisesRegex(ValueError, "Artifacts must remain"):
            data.prepare(Path("does-not-need-to-exist.zip"), ROOT / "public-dataset")

    def test_qa_appeals_admissions_and_account_action_requests_are_excluded(self):
        # Short, identity-free regression cases for patterns found by QA.
        cases = (
            "Меня забвнили после встречи с читером, игрок передал оружие",
            "Можете уточнить где я использавал постороние програмы (читы)",
            "Можно амнистацию я был не прав что оскорблял родню",
            "Повторно ЖБ так как я его оскорбил за жалобу на меня",
            "Оскорблял его в нрп чат, прошу выгнать и дать выговор",
            "Срочно кик или бан, акк украли",
            "Я сам оскорбил игрока и прошу прощения",
            "Почему меня забонили за читы, игрок был рядом",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertIsNone(data.classify(text))
        self.assertEqual(data.classify("Игрок оскорбил меня в общем чате"), "insult")
        self.assertEqual(data.classify("Игрок использует читы и летает над дорогой"), "cheats")
        self.assertEqual(data.classify("Мой аккаунт украли, не могу войти"), "account")


class GeneratedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (data.DEFAULT_OUTPUT / "manifest.json").exists() or not data.DEFAULT_SOURCE.exists():
            raise unittest.SkipTest("Run scripts/prepare_soup_data.py with local handout first")
        cls.manifest = json.loads((data.DEFAULT_OUTPUT / "manifest.json").read_text(encoding="utf-8"))
        cls.splits = {s: [json.loads(line) for line in (data.DEFAULT_OUTPUT / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()]
                      for s in ("train", "heldout")}
        selected = {e["ticket_id"] for values in cls.manifest["provenance"].values() for e in values}
        cls.source_rows = {}
        with ZipFile(data.DEFAULT_SOURCE) as archive:
            cls.rules = data.rule_index(json.loads(archive.read(data.RULE_MEMBER)))
            cls.valid = list(data.rows(archive, data.VALID_MEMBER))
            for row in data.rows(archive, data.TRAIN_MEMBER):
                if row["ticket_id"] in selected:
                    cls.source_rows[row["ticket_id"]] = row

    def test_actual_source_exclusion_and_group_disjointness(self):
        official_ids = {r["ticket_id"] for r in self.valid}
        official_reporters = {r["reporter_hash"] for r in self.valid}
        official_texts = set().union(*(data.text_keys(r) | {data.normalize(data.scrub(data.player_text(r), r))} for r in self.valid))
        all_reporters, all_texts, all_ids = set(), set(), set()
        self.assertEqual(len(self.source_rows), 192)
        for split in ("heldout", "train"):
            for evidence in self.manifest["provenance"][split]:
                row = self.source_rows[evidence["ticket_id"]]
                keys = data.text_keys(row) | {data.normalize(data.scrub(data.player_text(row), row))}
                self.assertNotIn(row["ticket_id"], official_ids | all_ids)
                self.assertNotIn(row["reporter_hash"], official_reporters | all_reporters)
                self.assertFalse(keys & (official_texts | all_texts))
                all_ids.add(row["ticket_id"])
                all_reporters.add(row["reporter_hash"])
                all_texts.update(keys)

    def test_training_input_rebuilt_from_player_only_despite_poisoned_outcomes(self):
        for split, records in self.splits.items():
            for evidence, record in zip(self.manifest["provenance"][split], records):
                source = dict(self.source_rows[evidence["ticket_id"]])
                source.update({"verdict": "POISON", "punishment": "POISON", "status": "POISON", "violation": "POISON"})
                source["messages"] = [dict(m, text="ADMIN_SECRET") if m["role"] != "player" else m
                                      for m in source["messages"]]
                candidate = {"topic": evidence["topic"], "text": data.scrub(data.player_text(source), source)}
                rebuilt = data.make_record(candidate, self.rules, evidence["template_variant"])
                self.assertEqual(record, rebuilt)
                self.assertNotIn("POISON", json.dumps(record))
                self.assertNotIn("ADMIN_SECRET", json.dumps(record))
                self.assertEqual(set(record), {"messages"})
                self.assertEqual([m["role"] for m in record["messages"]], ["system", "user", "assistant"])

    def test_every_selected_source_still_passes_current_lexical_policy(self):
        for split in self.splits:
            for evidence in self.manifest["provenance"][split]:
                row = self.source_rows[evidence["ticket_id"]]
                self.assertEqual(data.classify(data.scrub(data.player_text(row), row)), evidence["topic"])

    def test_targets_safe_and_refs_are_grounded_in_exact_source_sections(self):
        unsafe = re.compile(r"рассмотрена|проверка завершена|по скриншотам|на скриншотах видно|будет наказан|накажем|верн[её]м|снимем бан|разбаним|телепортировали|уже помогли", re.I)
        variants = {topic: set() for topic in data.TOPICS}
        for records in self.splits.values():
            for record in records:
                target = json.loads(record["messages"][2]["content"])
                self.assertTrue(target["needs_human"])
                self.assertIsNone(unsafe.search(target["draft"]))
                self.assertEqual(target["rule_refs"], data.REFS[target["topic"]])
                variants[target["topic"]].add(target["draft"])
                for ref in target["rule_refs"]:
                    self.assertIn(ref, self.rules)
                    self.assertIn(ref, record["messages"][1]["content"])
                    self.assertIn(self.rules[ref].split(". ")[0] + ".", record["messages"][1]["content"])
                if target["topic"] in ("payment", "account"):
                    self.assertFalse(target["rule_refs"])
        totals = {topic: sum(json.loads(r["messages"][2]["content"])["topic"] == topic
                             for records in self.splits.values() for r in records) for topic in data.TOPICS}
        for topic, values in variants.items():
            self.assertEqual(len(values), min(4, totals[topic]))

    def test_manifest_hashes_and_holdout_qa_contract(self):
        self.assertEqual(self.manifest["counts"], {"train": 160, "heldout": 32})
        self.assertIn("NOT human gold", self.manifest["disclaimer"])
        self.assertTrue(all(v == 0 for v in self.manifest["split_checks"].values()))
        for name, evidence in self.manifest["artifacts"].items():
            blob = (data.DEFAULT_OUTPUT / name).read_bytes()
            self.assertEqual(data.digest(blob), evidence["sha256"])
            self.assertEqual(len(blob.splitlines()), evidence["rows"])
        qa = [json.loads(line) for line in (data.DEFAULT_OUTPUT / "heldout-qa.jsonl").read_text(encoding="utf-8").splitlines()]
        for case, record in zip(qa, self.splits["heldout"]):
            self.assertEqual(case["input_messages"], record["messages"][:2])
            self.assertEqual(case["expected_weak_target"], json.loads(record["messages"][2]["content"]))
            self.assertTrue(all(v is None for v in case["review"].values()))

    def test_same_seed_reproduces_all_artifact_bytes(self):
        with tempfile.TemporaryDirectory(dir=data.DEFAULT_OUTPUT, prefix="repro-") as temporary:
            manifest = data.prepare(data.DEFAULT_SOURCE, Path(temporary))
            self.assertEqual(manifest, self.manifest)
            for name in manifest["artifacts"]:
                self.assertEqual((Path(temporary) / name).read_bytes(), (data.DEFAULT_OUTPUT / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
