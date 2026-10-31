"""Artificial data only: outcome leakage and misleading aggregate boundaries."""
import importlib.util
from pathlib import Path
import pytest
from pydantic import ValidationError
from support_copilot.quality_report import QualityStress


def script(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).parents[1]/"scripts"/(name+".py"))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_data_audit_player_view_never_contains_admin_or_outcomes():
    m=script("analyze_handout")
    row={"messages":[{"role":"player","text":"застрял"},{"role":"admin","text":"SENTINEL"}],
         "verdict":"SENTINEL","punishment":"SENTINEL","violation":"SENTINEL"}
    assert m.player_view(row)=="застрял"


def test_priority_only_recall_cannot_be_lower_than_combined_category_and_priority():
    d={"cases":16,"required_citation_hits":0,"required_citation_cases":0,"pseudo_citation_cases":0,
       "critical_correct":11,"critical_priority_correct":15,"critical_cases":16,"language_matches":16}
    assert QualityStress.model_validate(d).critical_priority_correct==15
    for bad in (10,17):
        with pytest.raises(ValidationError): QualityStress.model_validate({**d,"critical_priority_correct":bad})
