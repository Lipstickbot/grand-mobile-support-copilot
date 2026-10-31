from support_copilot.text import restore_kazakh


def test_documented_legacy_mapping_preserves_case_in_kazakh_context():
    assert restore_kazakh("ќґѕєјўї ЌҐЅЄЈЎЇ", "kk") == ("қғңәөұү ҚҒҢӘӨҰҮ", True)


def test_distinctive_legacy_input_and_region_can_restore_without_language_hint():
    assert restore_kazakh("ќорлады")[0] == "қорлады"
    assert restore_kazakh("маґан", region="kz")[0] == "маған"


def test_ambiguous_letters_are_not_globally_rewritten():
    text = "Це є повідомлення українською"
    assert restore_kazakh(text) == (text, False)
    assert restore_kazakh("Игрок оскорбил меня") == ("Игрок оскорбил меня", False)
