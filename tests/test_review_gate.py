import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location("review_gate",Path(__file__).parents[1]/"scripts/validate_review_packet.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def reviewed():
    a={"reviewer_id":"human-a","human_attestation":True,"category":"report.help","priority":"p3","language":"ru"}
    return {"kind":"report","category":"report.help","priority":"p3","verified_language":"ru",
        "approved_for_evaluation":True,"human_gold":True,"reviewer_1":a,"reviewer_2":{**a,"reviewer_id":"human-b"}}


def test_blinded_unfilled_packet_cannot_become_gold():
    with pytest.raises(ValueError):m.validate_record({"approved_for_evaluation":False}, {"report.help"})


def test_two_agreeing_humans_are_required():
    row=reviewed()
    assert m.validate_record(row,{"report.help"})
    row["reviewer_2"]["reviewer_id"]="human-a"
    with pytest.raises(ValueError):m.validate_record(row,{"report.help"})


def test_disagreement_is_not_silently_majority_or_model_labeled():
    row=reviewed();row["reviewer_2"]["priority"]="p0"
    with pytest.raises(ValueError):m.validate_record(row,{"report.help"})
    row["adjudicator"]={**row["reviewer_1"],"reviewer_id":"human-c"}
    assert m.validate_record(row,{"report.help"})


def test_no_human_attestation_cannot_be_filled_with_model_labels():
    row=reviewed();row["reviewer_1"]["human_attestation"]=False
    with pytest.raises(ValueError):m.validate_record(row,{"report.help"})


def selection():
    row={"review_id":"fixed","kind":"report","messages":[{"role":"player","text":"invented question"}],
         "server_lang":"ru","region":None,"images_count":0,"source_language_heuristic":"ru"}
    return row,{"schema_version":1,"request_sha256_by_review_id":{"fixed":m.selection_hash(row)}}


def test_gold_labels_may_change_but_frozen_player_text_cannot():
    row,manifest=selection()
    assert m.validate_selection([{**row,"category":"report.help"}],manifest) is None
    row["messages"][0]["text"]="replacement to cherry pick"
    with pytest.raises(ValueError):m.validate_selection([row],manifest)


def test_evaluation_duplicates_or_omissions_are_not_extra_independent_evidence():
    row,manifest=selection()
    for rows in ([row,row],[]):
        with pytest.raises(ValueError):m.validate_selection(rows,manifest)
