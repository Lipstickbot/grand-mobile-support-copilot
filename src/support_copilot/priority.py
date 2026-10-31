"""Independent urgency policy; an LLM cannot lower urgency or approve actions.

Taxonomy p0: access/compromise, payments, mass bug, real-world threats. Ban appeals
require a human but do not automatically mean p0 in the supplied taxonomy.
"""
from dataclasses import dataclass
import re

PRIORITIES = ("p0", "p1", "p2", "p3")


@dataclass(frozen=True)
class PriorityDecision:
    priority: str
    needs_human: bool
    reasons: tuple[str, ...]


def priority_floor(text: str, topics: set[str]) -> str:
    if topics & {"account", "payment"} or re.search(r"угрож.*(?:адрес|жизн|реальн)|массовый баг", text.casefold()):
        return "p0"
    if topics & {"cheat", "scam", "rmt_ads"}:
        return "p1"
    if topics & {"admin", "insult", "dm"}:
        return "p2"
    return "p3"


def decide_priority(text: str, category: str, *, topics: set[str],
                    learned_priority: str | None = None, learned_uncertain: bool = True) -> PriorityDecision:
    if learned_priority is not None and learned_priority not in PRIORITIES:
        raise ValueError("Unknown learned priority")
    floor = priority_floor(text, topics)
    reasons = []
    priority = floor if learned_priority is None or learned_uncertain else min(floor, learned_priority, key=PRIORITIES.index)
    if category in {"report.account", "report.payment"}:
        priority = "p0"
        reasons.append("priority=mandatory_category_safety_floor")
    mandatory = topics & {"account", "payment", "admin"}
    if mandatory or category in {"report.account", "report.payment", "complaint.admin"}:
        reasons.append("human_escalation=mandatory_sensitive_topic")
    # Every prototype draft is reviewed. This never claims an external handoff occurred.
    return PriorityDecision(priority, True, tuple(reasons))
