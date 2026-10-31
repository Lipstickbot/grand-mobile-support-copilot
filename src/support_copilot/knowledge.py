"""Offline, CPU-only retrieval over the supplied rule corpus.

The JSON source is authoritative. Cache files contain only numeric arrays and
identity metadata and can always be deleted and rebuilt from that source.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import unicodedata

import numpy as np

from .coretypes import RetrievedRule


_VERSION = 4
_POINT = re.compile(r"(?m)^\s*(\d+(?:\.\d+)+)\s+([^\n]*)")
_HEADING = re.compile(r"^[IVX]+[.)]\s+", re.I)
_STOP = set("и в во на по за из для от до с со а но или как это что мне меня у он его все при к не ли же о об the a an of in to is and".split())
_APOSTROPHES = str.maketrans({char: "'" for char in "‘’ʻʼ`"})
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", re.UNICODE)
# Query-only translations are narrow recall hints, never a source document or a
# decision that a player violated a rule. Dense E5 still sees the original query.
_QUERY_CUES = (
    (r"\binsult\w*|call\w*.{0,35}(?:disgusting|names)|abuse\w*.{0,25}(?:family|parents)|оскорб\w*|нецензур\w*|(?:реплик\w*|слов\w*|высказыван\w*).{0,30}(?:родн\w*|родител\w*|родствен\w*)|балағат\w*|қорла\w*|тіл\s+тигіз\w*|жаман\s+сөз\w*|\bhaqorat\w*|\bso'k\w*|\byomon\s+gap\w*", "оскорбление игрока родственников", "insult"),
    (r"\bчит\w*|макрос\w*|автоклик\w*|\b(?:macro|autoclick|makros|avtoklik)\w*|\bchit\w*|\baimbot\w*|бөгде\s+бағдарлам\w*|тыйым\s+салынған\s+бағдарлам\w*|\b(?:begona|taqiqlangan)\s+dastur\w*", "читы стороннее программное обеспечение", "cheat"),
    (r"бронежилет\w*|\bброн(?:я|ю|е|ей|ёй|ью|ь|и)\b|\b(?:bronejilet|bronijilet|zirh)\w*|\b(?:body\s+)?armou?r\b|сауыт\w*|қорғаныш\s+кеудеше\w*", "бронежилет", "armor"),
    (r"\boilaviy\s+tadbir\w*|\bfamily\s+(?:event|battle|war)\w*|\boila(?:lar)?\s+(?:jang|urush)\w*|отбасылық\s+(?:іс[ -]?шара|шайқас|соғыс)\w*|отбас(?:ы|ылар)\s+шайқас\w*", "семейное мероприятие", "family_event"),
    (r"\bbiznes\w*|\bbusiness\s+(?:battle|war)\w*", "бизнес", "business"),
    (r"\bfurgon\w*|\bvan\b", "фургон", "van_event"),
    (r"жасыл\s+(?:аймақ|аумақ)\w*|\byashil\s+(?:hudud|zona)\w*", "зеленая зона", "green_zone"),
    (r"мошен\w*|обман\w*|алаяқ\w*|алдап\w*|\bfiribgar\w*|\baldadi\b|\baldab\b", "мошенничество обман развод имущество", "scam"),
    # An exchange/trade mention is only a recall hint. Delays and disputes are
    # allegations, not evidence of fraud; preserve this distinct diagnostic cue.
    (r"обмен\w*.{0,30}имуществ\w*|имуществ\w*.{0,30}обмен\w*|мүлік\w*.{0,25}айырбаста\w*|\bmulk\w*.{0,25}almash\w*", "обмен имущество обман развод", "trade_dispute"),
    (r"(?:реальн\w*.{0,18}(?:деньг\w*|валют\w*)|деньг\w*\s+вне\s+игры|за\s+реал\b|\brmt\b|нақты\s+ақша\w*|\b(?:haqiqiy|real)\s+pul\w*)", "продажа покупка игровая валюта имущество аккаунт реальные деньги", "rmt"),
)
_ARMOR = re.compile(r"бронежилет\w*|\bброн(?:я|ю|е|ей|ёй|ью|ь|и)\b|сауыт\w*|қорғаныш\s+кеудеше\w*|\b(?:bronejilet|bronijilet|zirh)\w*|\b(?:body\s+)?armou?r\b")
_EVENTS = {
    36: re.compile(r"семейн\w*\s+(?:мероприят|битв|войн|порт|мп)|битв\w*\s+сем|отбасылық\s+(?:іс[ -]?шара|шайқас|соғыс)\w*|отбас(?:ы|ылар)\s+шайқас\w*|\boilaviy\s+tadbir\w*|\bfamily\s+(?:event|battle|war)\w*|\boila(?:lar)?\s+(?:jang|urush)\w*"),
    41: re.compile(r"бизнес|\bbizwar\b|\bbiznes\w*|\bbusiness\s+(?:battle|war)\w*|стрел[ауыке]"),
    42: re.compile(r"фургон|\bfurgon\w*|\bvan\b"),
    32: re.compile(r"зелен\w*\s+зон|\bgz\b|\bgreen\s*zone|жасыл\s+(?:аймақ|аумақ)\w*|\byashil\s+(?:hudud|zona)\w*"),
}
_SUPPORT_FAQ = re.compile(
    r"плат[её]ж\w*|оплат\w*|донат\w*|төлем\w*|төлед\w*|\bto'lov\w*|\bpayment\w*|"
    r"взлом\w*|восстанов\w*\s+аккаунт\w*|кіре\s+алмай\w*|құпиясөз\w*|\bparol\w*|\bhisobim\w*")


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold().translate(_APOSTROPHES).replace("ё", "е")


def _tokens(text: str) -> list[str]:
    # Conservative prefixes make common Russian inflections share a lexical
    # signal; E5 provides semantic recall beyond these literal matches.
    result = []
    for word in _WORD.findall(_normalize(text)):
        if word in _STOP:
            continue
        result.append(word[:6] if re.fullmatch(r"[а-я]{7,}", word) else word)
    return result


def _query_terms(query: str) -> tuple[str, list[str]]:
    normalized = _normalize(query)
    matches = [(alias, name) for pattern, alias, name in _QUERY_CUES if re.search(pattern, normalized)]
    entry_question = re.search(r"вход|зайти|войти|забеж|вбеж|забега|вбега|\benter\w*|\brun\w*\s+into|кіру|\bkirish", normalized)
    if not entry_question and _EVENTS[32].search(normalized) and re.search(r"огонь|стрел|стрельб|убий|урон|оқ\s*ат|атыс|шабуыл|\b(?:fire|shoot\w*|attack\w*)\b|o'q|otish|hujum", normalized):
        matches.append(("ответный огонь сотрудники силовых структур исполнение служебных обязанностей", "green_zone_combat"))
    return normalized + " " + " ".join(alias for alias, _ in matches), [name for _, name in matches]


def _language_signals(query: str) -> list[str]:
    normalized = _normalize(query)
    signals = []
    if re.search(r"[а-я]", normalized):
        signals.append("cyrillic_script")
    if re.search(r"[әғқңөұүһі]", normalized):
        signals.append("kazakh_script")
    if re.search(r"[a-z]", normalized):
        signals.append("latin_script")
    if re.search(r"\b(?:o'\w*|g'\w*|to'lov\w*|haqorat\w*|yordam\w*|oilaviy\w*)", normalized):
        signals.append("uzbek_lexical_hint")
    return signals


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _documents(payload: dict) -> list[RetrievedRule]:
    if not isinstance(payload, dict) or not isinstance(payload.get("rules"), list):
        raise ValueError("Rules source must contain a rules array")
    documents = []
    seen_sections = set()
    for section in payload["rules"]:
        sid = section["id"]
        if isinstance(sid, bool) or not isinstance(sid, int) or sid in seen_sections:
            raise ValueError("Section IDs must be unique integers")
        seen_sections.add(sid)
        title, full, points = section["category"], section["text"], section["points"]
        if not isinstance(title, str) or not isinstance(full, str) or not isinstance(points, list):
            raise ValueError("Invalid rule section")
        by_ref = {}
        for point in points:
            ref, text = point["ref"], point["text"]
            if not isinstance(ref, str) or not re.fullmatch(r"\d+(?:\.\d+)+", ref):
                raise ValueError("Invalid rule reference")
            if ref in by_ref or not isinstance(text, str) or not text.strip():
                raise ValueError("Duplicate or empty numbered point")
            by_ref[ref] = text.strip()

        # Capture headings and material outside numbered blocks, while avoiding
        # the duplicated full-section text supplied alongside points.
        heading = ""
        headings = {}
        nonpoint = []
        active_point = False
        for line in full.splitlines():
            stripped = line.strip()
            if _HEADING.match(stripped):
                heading = stripped
                nonpoint.append(stripped)
                active_point = False
            elif match := _POINT.match(line):
                if match.group(1) in by_ref:
                    headings[match.group(1)] = heading
                    active_point = True
                elif not active_point:
                    nonpoint.append(stripped)
            elif not active_point and stripped:
                nonpoint.append(stripped)

        for ref, text in by_ref.items():
            context = []
            if headings.get(ref):
                context.append(headings[ref])
            components = ref.split(".")
            for depth in range(2, len(components)):
                parent = ".".join(components[:depth])
                if parent in by_ref:
                    context.append(f"Условие родительского пункта {parent}: {by_ref[parent]}")
            context.append(text)
            documents.append(RetrievedRule(f"rules:{sid}:{ref}", sid, title, ref, "\n".join(context)))

        if not points:
            chunks = [x.strip() for x in re.split(r"(?m)^\s*•\s*", full) if x.strip()]
            for index, text in enumerate(chunks, 1):
                documents.append(RetrievedRule(f"internal:{sid}:term-{index:03d}", sid, title, None, text, source_type="term"))
        elif nonpoint:
            documents.append(RetrievedRule(f"internal:{sid}:context", sid, title, None, "\n".join(nonpoint), source_type="context"))
    if not documents:
        raise ValueError("Rules source is empty")
    return documents


class HybridRetriever:
    def __init__(self, rules_path: Path, embedding_model_path: Path | None = None,
                 index_dir: Path | None = None, dense_enabled: bool = True,
                 query_cache_size: int = 128, query_cache_ttl_seconds: float = 300.):
        if not 0 <= query_cache_size <= 1024 or not 0 < query_cache_ttl_seconds <= 3600:
            raise ValueError("Invalid bounded query cache settings")
        self.rules_path = Path(rules_path)
        self.embedding_model_path = Path(embedding_model_path) if embedding_model_path is not None else None
        self.index_dir = Path(index_dir) if index_dir is not None else None
        self.dense_enabled = dense_enabled
        self.ready = False
        self._docs: list[RetrievedRule] = []
        self._by_id = {}
        self._model = self._tokenizer = None
        self._vectors = None
        self._cache_hit = False
        self._source_sha256 = None
        self._query_cache = OrderedDict()
        self._query_lock = threading.RLock()
        self._query_cache_size = query_cache_size
        self._query_cache_ttl = query_cache_ttl_seconds
        self._query_clock = time.monotonic
        self._query_hits = self._query_misses = 0

    def clear_query_cache(self):
        with self._query_lock:
            self._query_cache.clear()
            self._query_hits = self._query_misses = 0

    def _model_identity(self) -> str:
        path = self.embedding_model_path
        if path is None or not path.is_dir():
            raise FileNotFoundError("Dense retrieval requires a local embedding model directory")
        # Content hashes invalidate the cache even when a model is replaced in
        # place. No cache-supplied paths are ever opened.
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in {".json", ".txt", ".model", ".safetensors", ".bin"})
        if not files:
            raise ValueError("Local embedding model directory is empty")
        identity = [(p.relative_to(path).as_posix(), _sha_file(p)) for p in files]
        return hashlib.sha256(json.dumps(identity).encode()).hexdigest()

    def _load_encoder(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.embedding_model_path), local_files_only=True, trust_remote_code=False)
        self._model = AutoModel.from_pretrained(str(self.embedding_model_path), local_files_only=True, trust_remote_code=False, use_safetensors=True).to("cpu").eval()

    def _embed(self, texts: list[str], prefix: str) -> np.ndarray:
        import torch
        self._load_encoder()
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), 16):
                inputs = self._tokenizer([f"{prefix}: {x}" for x in texts[start:start + 16]], padding=True, truncation=True, max_length=512, return_tensors="pt")
                output = self._model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(output.dtype)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                batches.append(pooled.cpu().float().numpy())
        return np.concatenate(batches).astype(np.float32, copy=False)

    def _cached_vectors(self, expected: dict) -> np.ndarray | None:
        if self.index_dir is None:
            return None
        try:
            metadata = json.loads((self.index_dir / "metadata.json").read_text(encoding="utf-8"))
            if metadata != expected:
                return None
            # Fixed filenames, no pickle and bounded header preclude cache path
            # traversal and object deserialization.
            vector_path = self.index_dir / "vectors.npy"
            if vector_path.stat().st_size > len(self._docs) * expected["embedding_dimension"] * 4 + 8192:
                return None
            vectors = np.load(vector_path, allow_pickle=False, max_header_size=1024)
            if not isinstance(vectors, np.ndarray):
                vectors.close()
                return None
            if (vectors.dtype != np.float32 or vectors.ndim != 2 or vectors.shape[0] != len(self._docs)
                    or not 1 <= vectors.shape[1] <= 4096 or not np.isfinite(vectors).all()
                    or vectors.shape[1] != expected["embedding_dimension"]
                    or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)):
                return None
            return vectors
        except (OSError, ValueError, TypeError, EOFError):
            return None

    def _save_vectors(self, metadata: dict) -> None:
        if self.index_dir is None:
            return
        self.index_dir.mkdir(parents=True, exist_ok=True)
        # Replace metadata last: interruption leaves an invalidatable cache.
        for filename, writer in (("vectors.npy", lambda f: np.save(f, self._vectors, allow_pickle=False)),
                                 ("metadata.json", lambda f: f.write(json.dumps(metadata).encode("utf-8")))):
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.index_dir, delete=False) as stream:
                    temporary = Path(stream.name)
                    writer(stream)
                os.replace(temporary, self.index_dir / filename)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def load(self) -> None:
        self.ready = False
        self.clear_query_cache()
        self._cache_hit = False
        self._vectors = None
        raw = self.rules_path.read_bytes()
        self._source_sha256 = hashlib.sha256(raw).hexdigest()
        self._docs = _documents(json.loads(raw.decode("utf-8-sig")))
        self._by_id = {doc.canonical_id: doc for doc in self._docs}
        self._counts = [Counter(_tokens(f"{doc.section_title} {doc.text}")) for doc in self._docs]
        self._lengths = np.array([sum(c.values()) for c in self._counts], dtype=np.float64)
        self._average_length = max(float(self._lengths.mean()), 1.0)
        frequencies = Counter(token for counts in self._counts for token in counts)
        self._idf = {token: math.log(1 + (len(self._docs) - n + .5) / (n + .5)) for token, n in frequencies.items()}
        if self.dense_enabled:
            model_identity = self._model_identity()
            if getattr(self, "_encoder_identity", model_identity) != model_identity:
                self._model = self._tokenizer = None
            self._encoder_identity = model_identity
            identity = {"version": _VERSION, "source_sha256": self._source_sha256,
                        "model_sha256": model_identity, "ids": [doc.canonical_id for doc in self._docs],
                        "pooling": "masked-mean-l2", "max_length": 512, "prefix": "e5-query-passage"}
            # Loading the encoder is required even for a cache hit: otherwise a
            # broken or incomplete model would appear ready until first query.
            self._load_encoder()
            identity["embedding_dimension"] = int(self._model.config.hidden_size)
            self._vectors = self._cached_vectors(identity)
            self._cache_hit = self._vectors is not None
            if self._vectors is None:
                self._vectors = self._embed([f"{doc.section_title}\n{doc.text}" for doc in self._docs], "passage")
                self._save_vectors(identity)
        self.ready = True

    def get(self, canonical_id: str) -> RetrievedRule | None:
        if not self.ready:
            raise RuntimeError("Retriever is not loaded")
        return self._by_id.get(canonical_id)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Reuse the loaded CPU E5 encoder for a local downstream classifier.

        Callers coordinate concurrent access, just as for search. This performs
        no download, changes no index and does not convert similarities into
        category probabilities. Dense-disabled sessions fail explicitly.
        """
        if not self.ready or not self.dense_enabled or self._vectors is None:
            raise RuntimeError("Loaded dense retrieval is required for query encoding")
        if not isinstance(texts, list) or not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Query encoding requires a nonempty list of nonempty texts")
        dimension = self._vectors.shape[1]
        with self._query_lock:
            now = self._query_clock()
            output = np.empty((len(texts), dimension), dtype=np.float32)
            missing = OrderedDict()
            for i, text in enumerate(texts):
                # Hash exact input, not a casefolded/expanded BM25 query. Only
                # vectors and digests live in this bounded in-memory cache.
                key = (getattr(self, "_encoder_identity", None), "e5-query-mean-l2-512-v1",
                       hashlib.sha256(text.encode("utf8")).digest())
                entry = self._query_cache.get(key)
                if entry is not None and now-entry[0] < self._query_cache_ttl:
                    self._query_cache.move_to_end(key)
                    output[i] = entry[1]
                    self._query_hits += 1
                else:
                    self._query_cache.pop(key, None)
                    if key not in missing:
                        missing[key] = (text, [])
                    missing[key][1].append(i)
            if missing:
                vectors = self._embed([entry[0] for entry in missing.values()], "query")
                if (not isinstance(vectors, np.ndarray) or vectors.shape != (len(missing), dimension)
                        or not np.isfinite(vectors).all()
                        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)):
                    raise RuntimeError("Query encoder returned invalid vectors")
                self._query_misses += len(missing)
                stored_at = self._query_clock()
                for (key, (_, indices)), vector in zip(missing.items(), vectors):
                    output[indices] = vector
                    if self._query_cache_size:
                        self._query_cache[key] = (stored_at, vector.astype(np.float32, copy=True))
                        while len(self._query_cache) > self._query_cache_size:
                            self._query_cache.popitem(last=False)
            # Never return a mutable view of cached data to a classifier/caller.
            return output

    def _bm25(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self._docs), dtype=np.float64)
        for token in set(_tokens(query)):
            idf = self._idf.get(token, 0)
            tf = np.array([counts.get(token, 0) for counts in self._counts])
            scores += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * self._lengths / self._average_length))
        return scores

    def search(self, query: str, rule_refs: list[str] | None = None, limit: int = 5) -> list[RetrievedRule]:
        return self.search_with_diagnostics(query, rule_refs=rule_refs, limit=limit)["results"]

    def search_with_diagnostics(self, query: str, rule_refs: list[str] | None = None, limit: int = 5) -> dict:
        """Return ranked candidates and explicit limits, never an answer or confidence.

        The complete original source text remains in every result. RRF scores are
        ranks rather than calibrated relevance. Even dense search with zero lexical
        support can return nearest neighbours for an unsupported question.
        """
        if not self.ready:
            raise RuntimeError("Retriever is not loaded")
        if not isinstance(query, str):
            raise TypeError("Query must be text")
        warnings = ["uncalibrated_retrieval_scores"]
        response = {"results": [], "lexical_hits": 0, "dense_enabled": self.dense_enabled,
                    "warnings": warnings, "query_language_signals": _language_signals(query),
                    "query_cues": [], "event_sections": [], "lexical_strength_bonus_applied": False}
        if limit <= 0 or not query.strip():
            warnings.append("empty_query_or_limit")
            return response
        # Numeric references from an allegation add no topical evidence.
        lexical_query, cues = _query_terms(query)
        lexical = self._bm25(lexical_query)
        response["query_cues"] = cues
        response["lexical_hits"] = int(np.count_nonzero(lexical > 0))
        if not response["lexical_hits"]:
            warnings.append("no_lexical_support")
        scores = np.zeros(len(self._docs), dtype=np.float64)
        candidates = set()
        for rank, index in enumerate(np.argsort(-lexical, kind="stable")):
            if lexical[index] > 0:
                scores[index] += 1 / (60 + rank + 1)
                candidates.add(int(index))
        # RRF discards score magnitude: a weak shared word may otherwise beat
        # a translated multi-term cue when low-resource dense recall is noisy.
        # Do not alter the established fusion for unexpanded original queries.
        # Add at most one top-rank vote, scaled by BM25 strength. This bounded
        # heuristic is not a relevance probability or proof of applicability.
        lexical_max = float(lexical.max())
        if lexical_max > 0 and cues:
            scores += (lexical / lexical_max) / 61
        response["lexical_strength_bonus_applied"] = bool(lexical_max > 0 and cues)
        if self.dense_enabled:
            vector = self.encode_queries([query])[0]
            similarities = self._vectors @ vector
            for rank, index in enumerate(np.argsort(-similarities, kind="stable")):
                scores[index] += 1 / (60 + rank + 1)
                candidates.add(int(index))

        # Context-sensitive section priors resolve rules with opposite meaning
        # across events. These are section hints, never a punishment decision.
        normalized = _normalize(query)
        event_sections = {sid for sid, pattern in _EVENTS.items() if pattern.search(normalized)}
        response["event_sections"] = sorted(event_sections)
        armor_events = event_sections & {36, 41, 42}
        if len(event_sections) > 1:
            warnings.append("multiple_event_sections")
        if _ARMOR.search(normalized) and len(armor_events) != 1:
            warnings.append("ambiguous_event_scope")
        if _SUPPORT_FAQ.search(normalized):
            # The known handout's source is rules, not an approved operational FAQ.
            warnings.append("support_faq_not_verified")
        known_refs = {doc.ref for doc in self._docs if doc.ref is not None} | set(self._by_id)
        if rule_refs and any(ref not in known_refs for ref in rule_refs):
            warnings.append("unknown_rule_refs")
        if rule_refs and any(sum(doc.ref == ref for doc in self._docs) > 1 for ref in rule_refs):
            warnings.append("ambiguous_rule_refs")
        for index in candidates:
            doc = self._docs[index]
            # Conflicting event names do not license arbitrarily choosing one.
            if len(event_sections) == 1:
                scores[index] *= 1.5 if doc.section_id in event_sections else .8
            # A weak reference can only break a tie among lexically supported
            # candidates. It never introduces an unrelated rule into results.
            if rule_refs and lexical[index] > 0 and (doc.canonical_id in rule_refs or doc.ref in rule_refs):
                scores[index] *= 1.01
            if doc.source_type == "context":
                scores[index] *= .5
        ranked = sorted(candidates, key=lambda i: (-scores[i], self._docs[i].canonical_id))[:limit]
        response["results"] = [replace(self._docs[i], score=float(scores[i])) for i in ranked]
        if not ranked:
            warnings.append("no_retrieval_candidates")
        return response

    def stats(self) -> dict:
        return {"ready": self.ready, "documents": len(self._docs),
                "sections": len({doc.section_id for doc in self._docs}),
                "points": sum(doc.ref is not None for doc in self._docs),
                "terms": sum(doc.source_type == "term" for doc in self._docs),
                "dense_enabled": self.dense_enabled, "dense_backend": "e5-cpu-exact-numpy" if self.dense_enabled else None,
                "embedding_dimension": int(self._vectors.shape[1]) if self._vectors is not None else None,
                "cache_hit": self._cache_hit, "source_sha256": self._source_sha256,
                "score_type": "reciprocal_rank_fusion_not_probability",
                "lexical_tokenizer": "unicode_nfc_apostrophe_v2",
                "query_cache": {"entries": len(self._query_cache), "capacity": self._query_cache_size,
                                "ttl_seconds": self._query_cache_ttl, "hits": self._query_hits,
                                "encoded_unique_misses": self._query_misses, "persistent": False},
                "lexical_strength_bonus": "query_cue_expansion_only_bounded_to_one_top_rank_vote"}
