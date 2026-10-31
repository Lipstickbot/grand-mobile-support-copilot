"""Local RAG integration. Admin messages/outcome fields never enter classification or generation."""
from __future__ import annotations

import re
import json
import hashlib
from dataclasses import asdict
import threading
import time

from .config import Settings
from .policy import DuplicateTracker, FALLBACKS, LANGUAGE_CLARIFICATION, classify, detect_language, guard_draft, select_relevant_rules, topics_for
from .intake import intake_reply
from .text import restore_kazakh, analyze_language, normalize_retrieval_query
from .schemas import (DraftReply, Extracted, TriageRequest, TriageResponse,
                      RagSearchRequest, RagSearchResponse, KnowledgeHit)


class TriageEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.retriever = None
        self.generator = None
        self.classifier = None
        self._generation_health = None
        self.ready = False
        self._categories = None
        self._taxonomy_version = None
        self._taxonomy = {}
        self._corpus_version = "unloaded"
        self._retrieval_lock = threading.RLock()
        self.duplicates = DuplicateTracker()

    def load(self):
        if self.ready:
            return
        if not 0 < self.settings.generation_timeout_seconds <= 45:
            raise ValueError("Serving deadline must be at most 45 seconds")
        taxonomy = json.loads(self.settings.taxonomy_path.read_text(encoding="utf8"))
        self._taxonomy = taxonomy
        self._categories = {item["code"] for item in taxonomy["categories"]}
        if not self._categories or len(self._categories) != len(taxonomy["categories"]):
            raise ValueError("Invalid taxonomy category codes")
        self._taxonomy_version = taxonomy.get("version")
        from .knowledge import HybridRetriever
        if self.settings.dense_enabled or self.settings.generation_enabled:
            import torch
            torch.set_num_threads(self.settings.num_threads)
        self.retriever = HybridRetriever(self.settings.rules_path, self.settings.embedding_path,
                                         self.settings.index_dir, self.settings.dense_enabled,
                                         query_cache_size=128 if self.settings.query_cache_enabled else 0)
        self.retriever.load()
        self._corpus_version = hashlib.sha256(self.settings.rules_path.read_bytes()).hexdigest()
        if self.settings.classifier_enabled and self.settings.dense_enabled:
            from .classifier import TextClassifier
            self.classifier = TextClassifier(self.settings.classifier_path, self._encode_classifier,
                expected_encoder_sha256=self.retriever._encoder_identity,
                expected_taxonomy_sha256=hashlib.sha256(self.settings.taxonomy_path.read_bytes()).hexdigest())
            self.classifier.load()
            if set(self.classifier.manifest["category_labels"]) != self._categories:
                raise ValueError("Learned classifier taxonomy does not match the service")
        if self.settings.generation_enabled:
            from .providers import create_generator
            self.generator = create_generator(self.settings)
            self.generator.load()
            self._generation_health = self.generator.stats()
        self.ready = True

    def health(self):
        generation = self._generation_health
        # Native process liveness is nonblocking; torch backends' stats take the GPU lock.
        if self.generator is not None and self.settings.inference_backend == "llamacpp":
            generation = self.generator.stats()
        ready = self.ready and (self.generator is None or bool(generation and generation.get("ready")))
        return {"ready": ready, "mode": "local_llm_rag" if self.generator else "explicit_template_baseline",
                "classification": "e5_cpu_weak_supervision" if self.classifier else "uncalibrated_keyword_heuristic",
                "classifier": self.classifier.stats() if self.classifier else {"ready": False, "mode": "explicit_keyword_fallback"},
                "all_cases_require_human": True,
                "taxonomy_version": self._taxonomy_version,
                "corpus_version": self._corpus_version,
                "retrieval": self.retriever.stats() if self.retriever else None,
                "generation": generation}

    def metadata(self):
        return {"service": "grand-mobile-support-copilot", "contract_version": "2",
                "taxonomy_version": self._taxonomy_version, "corpus_version": self._corpus_version,
                "categories": [{key: item[key] for key in ("code", "kind", "title") if key in item}
                               for item in self._taxonomy.get("categories", [])],
                "priorities": [{key: item[key] for key in ("code", "title") if key in item}
                               for item in self._taxonomy.get("priorities", [])],
                "generation_languages": ["ru", "kk", "uz", "en"], "all_cases_require_human": True,
                "unknown_language_policy": "und_requests_clarification_without_llm_or_e5",
                "extracted_semantics": {"rule_refs": "unverified_player_search_hints",
                    "has_evidence": "reported_attachment_counts_only_no_content",
                    "nickname_and_server": "reported_not_verified"},
                "confidence_type": "uncalibrated_model_score" if self.classifier else "uncalibrated_heuristic", "retrieval_score_type": "uncalibrated_rank",
                "classifier": self.classifier.stats() if self.classifier else {"ready": False, "mode": "explicit_keyword_fallback"},
                "source_id_format": "rules:<section_id>:<ref>",
                "limitations": ["no_image_content_available", "no_verified_payment_or_account_faq",
                                "no_gold_category_priority", "source_presence_is_not_entailment"]}

    def _encode_classifier(self, texts):
        with self._retrieval_lock:
            return self.retriever.encode_queries(texts)

    def _classify(self, player, kind):
        category, priority, reasons = classify(player, kind)
        if self.classifier is None:
            return category, priority, reasons + ["classifier=keyword_fallback_explicit"], .55, True
        prediction = self.classifier.predict(player, kind)
        selected = category if prediction["uncertain"] else prediction["category"]
        if prediction["uncertain"]:
            reasons.append("classifier_category=uncertain_keyword_fallback")
        if category in {"report.account", "report.payment", "complaint.admin"}:
            selected = category
            reasons.append("classifier_category=mandatory_topic_override")
        proposed_priority = priority if prediction["priority_uncertain"] else prediction["priority"]
        # The learned head may raise urgency, never lower a recognized safety floor.
        selected_priority = min(priority, proposed_priority, key=lambda x: int(x[1]))
        if selected in {"report.account", "report.payment"}:
            selected_priority = "p0"
            reasons.append("priority=mandatory_category_safety_floor")
        reasons.extend(prediction["reasons"])
        reasons.append("priority=learned_with_independent_keyword_safety_floor")
        reasons.append("classifier_input_window=e5_512_tokens")
        return selected, selected_priority, reasons, prediction["confidence"], prediction["uncertain"]

    def classify_request(self, request: TriageRequest):
        if not self.ready:
            raise RuntimeError("Engine not loaded")
        player = "\n".join(m.text for m in request.messages if m.role == "player")
        analysis = analyze_language(player, request.server_lang, request.region)
        query, flags = normalize_retrieval_query(analysis["normalized_text"])
        if analysis["language"] == "und":
            category, priority, reasons = classify(query, request.kind)
            confidence, uncertain = 0., True
            reasons.append("classifier=skipped_unknown_language")
        else:
            category, priority, reasons, confidence, uncertain = self._classify(query, request.kind)
        reasons.extend(self._language_reasons(analysis, flags))
        return {"lang": analysis["language"], "category": category,
                "priority": priority, "confidence": confidence,
                "confidence_type": "uncalibrated_model_score" if self.classifier else "uncalibrated_heuristic",
                "uncertain": uncertain, "needs_human": True, "reasons": reasons}

    @staticmethod
    def _language_reasons(analysis, flags):
        return ["language_candidates=" + ",".join(analysis["candidates"]),
                "language_mixed=" + str(analysis["mixed"]).lower(),
                "language_ambiguous=" + str(analysis["ambiguous"]).lower(),
                "language_detection=lexical_heuristic_not_calibrated"] + [
                    "input_normalization=" + flag for flag in analysis["normalization_flags"] + flags]

    def _retrieve(self, query, refs=None, limit=10):
        with self._retrieval_lock:
            if hasattr(self.retriever, "search_with_diagnostics"):
                return self.retriever.search_with_diagnostics(query, rule_refs=refs, limit=limit)
            return {"results": self.retriever.search(query, rule_refs=refs, limit=limit),
                    "warnings": [], "query_cues": [], "event_sections": []}

    def search_knowledge(self, payload: RagSearchRequest) -> RagSearchResponse:
        if not self.ready or self.retriever is None:
            raise RuntimeError("Knowledge engine not ready")
        started = time.monotonic()
        analysis = analyze_language(payload.query, payload.lang)
        query, flags = normalize_retrieval_query(analysis["normalized_text"])
        result = (self._retrieve(query, payload.rule_refs, max(10, payload.limit)) if analysis["language"] != "und"
                  else {"results": [], "warnings": ["language_und_needs_clarification"]})
        selected = select_relevant_rules(query, result["results"],
                                         cues=result.get("query_cues"), events=result.get("event_sections"))[:payload.limit]
        warnings = list(result.get("warnings", []))
        warnings.append("retrieval_only_not_a_verdict")
        warnings.extend(analysis["normalization_flags"] + flags)
        if analysis["mixed"]:
            warnings.append("mixed_language_input")
        if not selected:
            warnings.append("no_verified_applicable_rule")
        return RagSearchResponse(query_lang=analysis["language"],
                                 results=[KnowledgeHit(**asdict(rule)) for rule in selected],
                                 warnings=sorted(set(warnings)), corpus_version=self._corpus_version,
                                 elapsed_ms=(time.monotonic()-started)*1000)

    def get_rule(self, section_id: int, ref: str):
        if not self.ready or self.retriever is None:
            raise RuntimeError("Knowledge engine not ready")
        with self._retrieval_lock:
            rule = self.retriever.get(f"rules:{section_id}:{ref}")
        return KnowledgeHit(**asdict(rule)) if rule is not None else None

    def triage(self, request: TriageRequest) -> TriageResponse:
        if not self.ready:
            raise RuntimeError("Engine not loaded")
        started = time.monotonic()
        # Reported player's statements remain unverified context; admin replies are target leakage.
        player = "\n".join(m.text for m in request.messages if m.role == "player")
        context = "\n".join(f"{m.role}: {m.text}" for m in request.messages if m.role != "admin")
        analysis = analyze_language(player, request.server_lang, request.region)
        player = analysis["normalized_text"]
        query, flags = normalize_retrieval_query(player)
        context, _ = restore_kazakh(context, request.server_lang, request.region)
        lang = analysis["language"]
        if lang == "und":
            category, priority, reasons = classify(query, request.kind)
            category_confidence, uncertain = 0., True
            reasons.append("classifier=skipped_unknown_language")
        else:
            category, priority, reasons, category_confidence, uncertain = self._classify(query, request.kind)
        reasons.extend(self._language_reasons(analysis, flags))
        if self._categories is not None and category not in self._categories:
            raise RuntimeError("Classifier does not match supplied taxonomy")
        refs = re.findall(r"\b\d+(?:\.\d+){1,3}\b", request.rule_refs_raw or "")
        refs += re.findall(r"(?:пункт|правил|п\.|rule)\s*(\d+(?:\.\d+){1,3})", player, re.I)
        refs = sorted(set(refs))
        diagnostic = (self._retrieve(query, refs, 10) if lang != "und"
                      else {"results": [], "warnings": ["language_und_no_rule_selection"]})
        retrieved = diagnostic["results"]
        rules = select_relevant_rules(query, retrieved, cues=diagnostic.get("query_cues"),
                                     events=diagnostic.get("event_sections"))[:5]
        reasons += ["retrieved=" + ",".join(r.canonical_id for r in retrieved),
                    "topic_gate_selected=" + ",".join(r.canonical_id for r in rules),
                    "attachments=presence_only_not_reviewed", "admin_messages_and_outcomes_excluded",
                    "attachment_content_available=false", "reported_facts=unverified_player_claims",
                    "rule_refs=unverified_player_search_hints",
                    f"reported_images_count={request.images_count}",
                    "reported_message_attachments=" + str(sum(m.attachments for m in request.messages if m.role != "admin"))]
        reasons.extend("retrieval_warning=" + warning for warning in diagnostic.get("warnings", []))
        if not rules:
            reasons.append("no_verified_applicable_rule;ask_for_details_not_invented_faq")
        has_attachments = request.images_count > 0 or any(m.attachments > 0 for m in request.messages if m.role != "admin")
        intake_topics = topics_for(query)
        source_absent_intake = self.generator is not None and lang in {"ru", "kk", "uz", "en"} and not rules
        raw = ""
        if source_absent_intake:
            raw = intake_reply(lang, intake_topics, has_attachments=has_attachments)
            reasons += ["generation=skipped_no_verified_rule", "draft=source_absent_intake_template"]
        elif self.generator is not None and lang in {"ru", "kk", "uz", "en"}:
            try:
                raw = self.generator.generate(context, lang, rules, category)
            except Exception as exc:
                from .generation import GenerationError
                if not isinstance(exc, GenerationError):
                    raise
                reasons.append("generation_error=" + exc.code)
        else:
            reasons.append("generation=disabled_or_language_unsupported")
        if lang == "und":
            draft, guard_reasons = LANGUAGE_CLARIFICATION, []
            reasons.append("draft=language_clarification")
        else:
            draft, guard_reasons = guard_draft(raw, rules, lang, player_text=None if source_absent_intake else player)
        reasons.extend(guard_reasons)
        if guard_reasons:
            reasons.append("draft=localized_template_fallback")
            # Counted attachments are not visible proof. Avoid requesting the
            # same screenshot again after rejecting an unsafe draft.
            if self.generator is not None:
                draft = intake_reply(lang, intake_topics, has_attachments=has_attachments)
        cited_ids = set(re.findall(r"rules:\d+:[\d.]+", draft))
        # Citations are surfaced only if the model actually used a full retrieved rule ID.
        cited = [r.canonical_id for r in rules if r.ref is not None and r.canonical_id in cited_ids]
        if not cited:
            reasons.append("draft_has_no_validated_rule_citation")
        duplicate = self.duplicates.check(request.reporter_hash, request.created_at, request.kind, player)
        if duplicate:
            reasons.append("duplicate=exact_same_reporter_and_kind_within_1h")
        else:
            reasons.append("spam_detection=not_implemented;duplicate_requires_identity_and_timezone")
        nickname = request.reported_nick
        if not nickname:
            match = re.search(r"(?:ник(?:нейм)?|nickname)\s*[:=]\s*([\w.-]{2,32})", player, re.I)
            nickname = match.group(1).rstrip(".,;:") if match else None
        server = re.search(r"(?:сервер|server)\s*(?:№|#|:)?\s*(\d{1,3})\b", player, re.I)
        reasons.append(f"triage_seconds={time.monotonic() - started:.3f}")
        return TriageResponse(
            lang=lang, category=category, priority=priority,
            is_spam_or_duplicate=duplicate,
            extracted=Extracted(nickname=nickname, rule_refs=refs,
                                has_evidence=request.images_count > 0 or any(m.attachments > 0 for m in request.messages if m.role != "admin"),
                                server=server.group(1) if server else None),
            draft_reply=DraftReply(lang=lang if lang in FALLBACKS else "ru", text=draft, cited=cited),
            needs_human=True, confidence=0. if lang == "und" else 0.35 if guard_reasons else category_confidence,
            reasons=reasons)

    def close(self):
        self.ready = False
        if self.generator is not None:
            self.generator.close()
        self.generator = None
        self._generation_health = None
        if self.retriever is not None and hasattr(self.retriever, "clear_query_cache"):
            self.retriever.clear_query_cache()
        self.retriever = None
        self.classifier = None
