from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedRule:
    canonical_id: str
    section_id: int
    section_title: str
    ref: str | None
    text: str
    score: float = 0.0
    source_type: str = "rule"
