from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["player", "reported_player", "admin"]
    text: str = Field(max_length=8000)
    attachments: int = Field(default=0, ge=0, le=1000)


class TriageRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    kind: Literal["complaint", "report"]
    messages: list[Message] = Field(min_length=1, max_length=50)
    reporter_hash: str | None = Field(default=None, max_length=128)
    created_at: datetime | None = None
    reported_nick: str | None = Field(default=None, max_length=128)
    rule_refs_raw: str | None = Field(default=None, max_length=256)
    images_count: int = Field(default=0, ge=0, le=1000)
    server_lang: str | None = Field(default=None, max_length=12)
    region: str | None = Field(default=None, max_length=32)

    @model_validator(mode="after")
    def validate_input(self):
        if not any(m.role == "player" and m.text.strip() for m in self.messages):
            raise ValueError("A nonempty player message is required")
        if sum(len(m.text) for m in self.messages if m.role != "admin") > 16000:
            raise ValueError("Player conversation exceeds 16000 characters")
        return self


class Extracted(BaseModel):
    nickname: str | None
    rule_refs: list[str]
    has_evidence: bool
    server: str | None


class DraftReply(BaseModel):
    lang: str
    text: str
    cited: list[str]


class TriageResponse(BaseModel):
    lang: str
    category: str
    priority: Literal["p0", "p1", "p2", "p3"]
    is_spam_or_duplicate: bool
    extracted: Extracted
    draft_reply: DraftReply
    needs_human: bool
    confidence: float = Field(ge=0, le=1)
    reasons: list[str]


class ClassificationResponse(BaseModel):
    lang: str
    category: str
    priority: Literal["p0", "p1", "p2", "p3"]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    confidence_type: str
    uncertain: bool
    needs_human: bool
    reasons: list[str]


class RagSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=4000)
    rule_refs: list[str] = Field(default_factory=list, max_length=20)
    limit: int = Field(default=5, ge=1, le=10)
    lang: Literal["ru", "kk", "uz", "en", "es", "pt", "und"] | None = None

    @model_validator(mode="after")
    def validate_refs(self):
        if any(not ref or len(ref) > 32 for ref in self.rule_refs):
            raise ValueError("Each rule reference must have 1 to 32 characters")
        return self


class KnowledgeHit(BaseModel):
    canonical_id: str
    section_id: int
    section_title: str
    ref: str | None
    text: str
    score: float = Field(allow_inf_nan=False, description="Uncalibrated retrieval rank score, not probability")
    source_type: str


class RagSearchResponse(BaseModel):
    query_lang: str
    results: list[KnowledgeHit]
    warnings: list[str]
    corpus_version: str
    elapsed_ms: float = Field(ge=0, allow_inf_nan=False)
