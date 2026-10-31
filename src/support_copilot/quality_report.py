"""Numeric-only public quality report; private request/response rows stay private."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QualityStats(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    cases: int = Field(ge=1, le=2000)
    category_correct: int = Field(ge=0, le=2000)
    category_cases: int = Field(ge=1, le=2000)
    category_macro_f1: float = Field(ge=0, le=1, allow_inf_nan=False)
    priority_macro_f1: float = Field(ge=0, le=1, allow_inf_nan=False)
    p0_recall: float = Field(ge=0, le=1, allow_inf_nan=False)
    required_citation_hits: int = Field(ge=0, le=2000)
    required_citation_cases: int = Field(ge=0, le=2000)
    pseudo_citation_cases: int = Field(ge=0, le=2000)
    intake_template_cases: int = Field(ge=0, le=2000)
    fallback_cases: int = Field(ge=0, le=2000)
    latency_p50_seconds: float = Field(ge=0, allow_inf_nan=False)
    latency_p95_seconds: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def counts(self):
        if (self.category_correct > self.category_cases
                or self.required_citation_hits > self.required_citation_cases
                or self.required_citation_cases > self.cases
                or any(x > self.cases for x in (self.pseudo_citation_cases, self.intake_template_cases, self.fallback_cases))
                or self.latency_p50_seconds > self.latency_p95_seconds):
            raise ValueError("Inconsistent aggregate counts")
        return self


class QualityComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    before: QualityStats
    after: QualityStats


class QualityStress(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    cases: int = Field(ge=1, le=2000)
    required_citation_hits: int = Field(ge=0, le=2000)
    required_citation_cases: int = Field(ge=0, le=2000)
    pseudo_citation_cases: int = Field(ge=0, le=2000)
    critical_correct: int = Field(ge=0, le=2000)
    critical_priority_correct: int | None = Field(default=None, ge=0, le=2000)
    critical_cases: int = Field(ge=0, le=2000)
    language_matches: int = Field(ge=0, le=2000)

    @model_validator(mode="after")
    def counts(self):
        if not (self.required_citation_hits <= self.required_citation_cases <= self.cases
                and self.critical_correct <= self.critical_cases <= self.cases
                and self.language_matches <= self.cases and self.pseudo_citation_cases <= self.cases):
            raise ValueError("Inconsistent stress counts")
        if self.critical_priority_correct is not None and not self.critical_correct <= self.critical_priority_correct <= self.critical_cases:
            raise ValueError("Inconsistent critical priority counts")
        return self


class PublicQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1]
    label_kind: Literal["authored_synthetic_quality_comparison_not_human_gold"]
    human_gold: Literal[False]
    status: Literal["complete"]
    comparison: QualityComparison
    extra_stress: QualityStress | None = None
    # These fixed labels expose no free-form private text, local paths or examples.
    limitations: list[Literal["synthetic_not_human_gold", "no_native_language_review",
        "templates_are_not_model_training", "same_cases_used_for_development",
        "semantic_accuracy_unmeasured", "single_sequential_run_not_SLO"]] = Field(max_length=6)
