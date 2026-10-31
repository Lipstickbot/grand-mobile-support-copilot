"""Triage v3: learned classifier + learned/lexical rule retrieval + grounded drafts.

Contract: POST /triage (see TZ). ``cited`` holds plain point refs ("2.1"), every one
of which exists in rules.json; the canonical id ("rules:30:2.1") is in ``reasons``.
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from ..intents import critical_intents
from ..knowledge import HybridRetriever
from ..policy import topics_for
from ..text import normalize_retrieval_query
from . import lang as langmod
from . import model as modelmod
from .labels import THREAT_OUTSIDE, MASS_BUG, EMPTY, norm, player_topic
from .rulekb import load_rule_ids, parse_refs, taxonomy_prior
from .templates import render, rule_phrase
from .sanctions import sanction_text
from .multilang import keyword_category

PROMISE = re.compile(
    r"разбан|будет (?:снят|разбан|возвращ|наказан|заблокир)|получит (?:бан|мут|блок|варн|наказ|джайл)|"
    r"верн[её]м|возврат(?:им)? (?:средств|денег)|компенсир|восстановим|"
    r"will be (?:unbanned|refunded|punished|banned)|we will refund|"
    r"banni olib tashlaymiz|pulingiz qaytariladi|jazolanadi|"
    r"бан алынады|ақшаңыз қайтарылады|жазаланады", re.I)
EVIDENCE_URL = re.compile(r"<url>|https?://|imgur|youtu|fraps|фрапс", re.I)
NICK_RE = re.compile(r"\b[A-Z][a-z]+_[A-Z][a-z]+\d*\b")
SERVER_RE = re.compile(r"(?:сервер\w*|server|сервере)\s*(?:№|#|:)?\s*(\d{1,3})\b|\b(\d{1,3})\s*(?:сервер\w*|server)", re.I)
PAYMENT_EXTRA = re.compile(r"(?:купил\w*|купив|оплатил\w*|оплатив|задонатил\w*|задонатив|поповнив|пополнил\w*|сатып алдым|to.?ladim|sotib oldim|"
                           r"paid|bought|pagu[eé]|compr[eé]|paguei|comprei).{0,60}(?:не\s*(?:приш|пришл|появ|зачисл|получ)|"
                           r"не\s*з.?явил|немає|нема|не прийш|түспеді|келмеді|kelmadi|tushmadi|not (?:received|arrived|credited)|no (?:lleg|recib)|n[aã]o (?:cheg|receb|apareceu))", re.I)
WHERE_Q = re.compile(r"\bгде\b|\bкуда\b|қайда|qayer|\bwhere\b|d[oó]nde|\bonde\b|как (?:найти|доехать|попасть)", re.I)
SENSITIVE_HINTS = re.compile(r"взлом|украл\w* акк|разбан|верните|возврат|refund|unban|оплат|донат|списал", re.I)


class DuplicateTracker:
    """Same reporter + same kind + same target/text within a window -> duplicate."""

    def __init__(self, window_minutes: int = 60, max_keys: int = 50000):
        self.window = window_minutes * 60
        self.max_keys = max_keys
        self._seen = defaultdict(list)
        self._lock = threading.Lock()

    @staticmethod
    def _sig(text: str) -> set[str]:
        t = re.sub(r"[^\w]+", " ", norm(text))
        return {t[i:i + 4] for i in range(max(1, len(t) - 3))}

    def check(self, reporter, created_at, kind, target, text) -> bool:
        if not reporter:
            return False
        ts = created_at.timestamp() if isinstance(created_at, datetime) else (
            datetime.fromisoformat(created_at).timestamp() if isinstance(created_at, str) and created_at else time.time())
        sig = self._sig(text)
        key = (reporter, kind)
        with self._lock:
            history = [h for h in self._seen[key] if abs(ts - h[0]) <= self.window]
            dup = False
            for _, h_target, h_sig in history:
                jacc = len(sig & h_sig) / max(1, len(sig | h_sig))
                if (target and h_target == target and kind == "complaint") or jacc >= 0.8:
                    dup = True
                    break
            history.append((ts, target, sig))
            self._seen[key] = history[-20:]
            if len(self._seen) > self.max_keys:
                self._seen.pop(next(iter(self._seen)))
        return dup


class TriageV3:
    def __init__(self, data_dir: Path, model_dir: Path, llm=None):
        self.data_dir, self.model_dir = Path(data_dir), Path(model_dir)
        self.llm = llm
        self.duplicates = DuplicateTracker()
        self.ready = False

    def load(self):
        self.bundle = modelmod.load(self.model_dir)
        self.taxonomy = json.loads((self.data_dir / "taxonomy.json").read_text(encoding="utf8"))
        self.known = load_rule_ids(self.data_dir / "rules.json")
        self.prior = taxonomy_prior(self.taxonomy, self.known)
        self.always_human = set(self.taxonomy.get("needs_human_always", []))
        self.retriever = HybridRetriever(self.data_dir / "rules.json", None, None, dense_enabled=False, query_cache_size=0)
        self.retriever.load()
        self._lock = threading.Lock()
        self.ready = True
        return self

    # ---------------- classification ----------------
    def classify(self, kind: str, text: str, refs_raw: str | None):
        X = self.bundle["vectorizer"].transform([modelmod.feature_text(kind, text, refs_raw)])
        head = self.bundle["heads"][kind]
        probs = modelmod.softmax(head.decision_function(X)[0])
        order = np.argsort(-probs)
        category, conf = str(head.classes_[order[0]]), float(probs[order[0]])
        margin = conf - float(probs[order[1]])
        return X, category, conf, margin

    def priority(self, category: str, text: str) -> tuple[str, list[str]]:
        t = norm(text)
        reasons = []
        crit = critical_intents(text)
        if category in ("report.account", "report.payment") or crit or THREAT_OUTSIDE.search(t) or MASS_BUG.search(t):
            if crit: reasons.append("p0_cue=" + ",".join(sorted(crit)))
            return "p0", reasons + ["priority=taxonomy_p0"]
        if category in ("complaint.cheat", "complaint.scam", "complaint.rmt_ads"):
            return "p1", ["priority=taxonomy_p1"]
        if category in ("complaint.invalid", "report.offtopic", "report.question"):
            return "p3", ["priority=taxonomy_p3"]
        return "p2", ["priority=taxonomy_p2"]

    # ---------------- retrieval ----------------
    def retrieve(self, X, category: str, text: str, refs_raw: str | None, k: int = 5):
        scores = defaultdict(float)
        why = defaultdict(list)
        if category.startswith("complaint."):
            head = self.bundle["rule_head"]
            p = modelmod.softmax(head.decision_function(X)[0], scale=3.0)
            for cid, pp in zip(head.classes_, p):
                scores[cid] += float(pp)
                if pp > .05: why[cid].append(f"learned={pp:.2f}")
        query, _ = normalize_retrieval_query(text)
        with self._lock:
            bm = self.retriever.search_with_diagnostics(query, limit=10)["results"]
        for rank, doc in enumerate(bm):
            scores[doc.canonical_id] += .15 / (rank + 1)
            why[doc.canonical_id].append(f"bm25_rank={rank + 1}")
        for cid in parse_refs((refs_raw or "") + " " + text, self.known):
            scores[cid] += .25
            why[cid].append("player_ref")
        for cid in self.prior.get(category, []):
            scores[cid] += .15
            why[cid].append("taxonomy_prior")
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [(self.retriever.get(cid), s, why[cid]) for cid, s in ranked if self.retriever.get(cid) is not None]

    def _pick_rule(self, hits, category):
        """A rule is cited only if it is strongly supported AND consistent with the category."""
        allowed = set(self.prior.get(category, []))
        for doc, score, _ in hits:
            if doc.ref is None:
                continue
            if category == "complaint.other" and score >= .45 and doc.section_id != 31:
                return doc
            if doc.canonical_id in allowed and score >= .2:
                return doc
            if score >= .6 and doc.section_id != 31:
                return doc
        return None

    # ---------------- drafting ----------------
    def draft(self, category, lang, hits, nick, has_evidence, duplicate, text=""):
        cited = []
        get = self.retriever.get
        def cite(doc):
            cited.append(doc)
            return rule_phrase(doc, lang)
        if duplicate and category.startswith("complaint."):
            return render("duplicate", lang, rule=cite(get("rules:31:1.2.9"))), cited
        if category == "complaint.admin":
            return render("admin", lang, rule=cite(get("rules:30:3.4"))), cited
        if category == "complaint.invalid":
            doc = get("rules:31:1.2.3") if has_evidence else get("rules:31:1.2.2")
            return render("invalid", lang, rule=cite(doc)), cited
        if category.startswith("complaint."):
            doc = self._pick_rule(hits, category)
            text = render("complaint", lang, rule=cite(doc), nick=nick) if doc else render("complaint_norule", lang, nick=nick)
            sanction = sanction_text(doc, lang) if doc is not None else ""
            if sanction:
                text += render("sanction", lang, sanction=sanction)
            if not has_evidence:
                text += render("need_evidence", lang, rule2=cite(get("rules:31:1.3.1")))
            return text, cited
        if category == "report.account":
            return render("report.account", lang, rule=cite(get("rules:30:1.4"))), cited
        if category == "report.complaint":
            doc = self._pick_rule([h for h in hits if h[0].section_id in (30, 32)], "complaint.insult")
            if doc is not None and hits and hits[0][1] >= .3:
                return render("report.complaint_rule", lang, rule=cite(doc)), cited
            return render("report.complaint", lang), cited
        if category == "report.question" and WHERE_Q.search(text):
            return render("report.question_where", lang), cited
        return render(category, lang), cited

    def exemplars(self, text: str, category: str, server_lang: str | None, k: int = 3):
        mem = self.bundle["memory"]
        q = mem["vectorizer"].transform([text])
        sims = (mem["matrix"] @ q.T).toarray().ravel()
        out = []
        for i in np.argsort(-sims)[:40]:
            meta = mem["meta"][i]
            if meta["category"] == category and (server_lang is None or meta["server_lang"] == server_lang or len(out) < 1):
                out.append({**meta, "similarity": float(sims[i])})
            if len(out) >= k:
                break
        return out

    # ---------------- main ----------------
    def triage(self, req: dict) -> dict:
        started = time.monotonic()
        kind = req["kind"]
        messages = req.get("messages", [])
        text = "\n".join(m.get("text", "") for m in messages if m.get("role") == "player")
        refs_raw = req.get("rule_refs_raw")
        images = int(req.get("images_count") or 0) + sum(int(m.get("attachments") or 0) for m in messages if m.get("role") != "admin")
        lang, lang_reasons = langmod.detect(text, req.get("lang"), req.get("server_lang"))
        reasons = list(lang_reasons)

        X, category, conf, margin = self.classify(kind, text, refs_raw)
        reasons.append(f"classifier=tfidf_svm_weak_labels conf={conf:.2f} margin={margin:.2f}")
        topics = topics_for(text)
        crit = critical_intents(text)
        # safety overrides independent of the learned model
        if PAYMENT_EXTRA.search(text):
            crit = crit | {"payment"}
        if kind == "report" and ("payment" in crit or ("payment" in topics and category != "report.account")):
            if category != "report.payment": reasons.append("override=payment_cue")
            category = "report.payment"
        elif kind == "report" and "account" in crit and category != "report.account":
            reasons.append("override=account_cue"); category = "report.account"
        if lang in ("kk", "en", "es", "pt") or "lang=ukrainian_answered_in_ru" in lang_reasons:
            kw = keyword_category(kind, text)
            if kw and kw != category:
                reasons.append(f"multilang_keywords={kw} (model={category})")
                category = kw
        if category == "complaint.other":
            topic = player_topic(text + " " + (refs_raw or ""))
            if topic:
                category = "complaint." + topic
                reasons.append("backoff=player_keywords_" + topic)
        if kind == "complaint" and not re.search(r"[^\W\d_]{3,}", re.sub(r"<\w+>", " ", text)) and not (refs_raw or "").strip(".… "):
            if category != "complaint.invalid": reasons.append("override=empty_description")
            category = "complaint.invalid"
        priority, preasons = self.priority(category, text)
        reasons += preasons

        hits = self.retrieve(X, category, text, refs_raw)
        reasons.append("retrieved=" + ",".join(f"{d.canonical_id}:{s:.2f}" for d, s, _ in hits))

        nick = req.get("reported_nick") or (NICK_RE.search(text).group(0) if NICK_RE.search(text) else None)
        has_evidence = images > 0 or bool(EVIDENCE_URL.search(text))
        target = req.get("reported_hash") or nick
        duplicate = self.duplicates.check(req.get("reporter_hash"), req.get("created_at"), kind, target, text)
        stripped = re.sub(r"[^\w\s+?!.]", "", norm(text))
        spam = kind == "report" and category == "report.offtopic" and bool(EMPTY.match(stripped) or not re.search(r"[^\W\d_]{2,}", stripped))

        draft_text, cited_docs = self.draft(category, lang, hits, nick, has_evidence, duplicate, text)
        draft_source = "template"
        if self.llm is not None and category not in ("report.offtopic",) and not duplicate:
            allowed = {d.canonical_id: d for d, _, _ in hits if d.ref} | {d.canonical_id: d for d in cited_docs}
            polished = self.llm.polish(text=text, lang=lang, category=category, template=draft_text,
                                       rules=list(allowed.values()), exemplars=self.exemplars(text, category, req.get("server_lang")))
            if polished is not None:
                new_text, used = polished
                if PROMISE.search(new_text) or not used <= set(allowed):
                    reasons.append("llm_draft_rejected=promise_or_unknown_rule")
                elif cited_docs and not used:
                    reasons.append("llm_draft_rejected=dropped_rule_citation")
                else:
                    draft_text, cited_docs, draft_source = new_text, [allowed[c] for c in used], "llm"
        if PROMISE.search(draft_text):  # belt and braces
            draft_text, cited_docs = render("report.question" if kind == "report" else "complaint_norule", lang, nick=nick), []
            reasons.append("draft_guard=promise_removed")
        reasons.append("draft=" + draft_source)
        cited = []
        for d in cited_docs:
            if d is not None and d.ref and d.ref not in cited and d.canonical_id in self.known:
                cited.append(d.ref)
                reasons.append("cited_id=" + d.canonical_id)

        # ---- needs_human policy ----
        human = []
        if category in self.always_human: human.append("taxonomy_needs_human_always")
        if priority == "p0": human.append("p0_critical")
        if crit or SENSITIVE_HINTS.search(text): human.append("payment_account_or_appeal_topic")
        if kind == "complaint" and category not in ("complaint.invalid",) and not duplicate:
            human.append("punishment_decision_requires_admin_and_screenshots")
        if category in ("report.help", "report.complaint"): human.append("in_game_action_by_admin")
        if category == "report.question" and draft_source != "llm": human.append("factual_answer_needed")
        if conf < .5 or margin < .15: human.append("low_classifier_confidence")
        needs_human = bool(human)
        reasons += ["needs_human=" + r for r in human]
        reasons.append(f"triage_ms={(time.monotonic() - started) * 1000:.1f}")
        server = SERVER_RE.search(text)
        return {
            "lang": lang,
            "category": category,
            "priority": priority,
            "is_spam_or_duplicate": bool(duplicate or spam),
            "extracted": {"nickname": nick,
                          "rule_refs": sorted({c.split(":")[2] for c in parse_refs((refs_raw or "") + " " + text, self.known)}),
                          "has_evidence": has_evidence,
                          "server": (server.group(1) or server.group(2)) if server else None},
            "draft_reply": {"lang": lang, "text": draft_text, "cited": cited},
            "needs_human": needs_human,
            "confidence": round(conf, 3),
            "reasons": reasons,
        }
