"""Privacy: secrets never stored, sensitive data opt-in, masking, no leaks in logs."""

from __future__ import annotations

import logging

import pytest

from era_memory import SecretContentError, SensitiveContentError

SECRETS = [
    "my api key is AIzaSyA1234567890abcdefghijklmnopqrstuvw",
    "the password is hunter2hunter2",
    "token: ghp_abcdefghijklmnopqrstuvwxyz0123456789",
    "here is my key sk-abcdefghijklmnopqrstuvwxyz",
    "Authorization: Bearer abcdefghijklmnop1234567890",
    "BRIDGE_API_KEY is 0123456789abcdef0123456789abcdef",
]


@pytest.mark.parametrize("text", SECRETS)
def test_secrets_are_never_saved_automatically(mm, text):
    r = mm.ingest_turn(f"Remember that {text}", conversation_id="c")
    assert r.secrets_blocked >= 1 and r.saved == []
    assert mm.list(include_history=True) == []
    assert all("hunter2" not in t["content"] and "AIza" not in t["content"] for t in mm.store.get_turns("c"))


@pytest.mark.parametrize("text", SECRETS)
def test_secrets_are_refused_even_on_explicit_save(mm, text):
    with pytest.raises(SecretContentError):
        mm.save(text)
    assert mm.list(include_history=True) == []


SENSITIVE = {
    "I was diagnosed with asthma last year.": "health",
    "My credit card number is 4111 1111 1111 1111.": "financial",
    "My salary is 12 lakhs per year.": "financial",
    "My Aadhaar number is 1234 5678 9012.": "government_id",
    "My PAN is ABCDE1234F.": "government_id",
}


@pytest.mark.parametrize("text,category", SENSITIVE.items())
def test_sensitive_info_is_not_saved_automatically_and_only_category_is_reported(mm, text, category):
    r = mm.ingest_turn(text, conversation_id="c")
    assert category in r.held_back and r.saved == []
    assert mm.list(include_history=True) == []
    payload = str(r.to_dict())
    assert "4111" not in payload and "asthma" not in payload and "1234 5678" not in payload   # response has no values
    assert all(text not in t["content"] for t in mm.store.get_turns("c"))                       # not kept in the turn buffer


def test_sensitive_explicit_save_requires_opt_in(mm):
    with pytest.raises(SensitiveContentError) as e:
        mm.save("I take medication for hypertension")
    assert "health" in e.value.categories
    out = mm.save("I take medication for hypertension", allow_sensitive=True)
    assert out.memory.is_sensitive and out.memory.share_with_model is False


def test_sensitive_memory_is_masked_in_public_views_unless_revealed(mm):
    m = mm.save("My blood group is O positive", allow_sensitive=True).memory
    assert "O positive" not in str(m.to_dict()) and "hidden" in m.to_dict()["content"]
    assert m.to_dict(reveal=True)["content"] == "My blood group is O positive"


def test_sensitive_memory_is_not_sent_to_the_model_by_default(mm):
    mm.save("My blood group is O positive", allow_sensitive=True)
    pkg = mm.build_context("what is my blood group", "c")
    assert "O positive" not in pkg.text and pkg.withheld_sensitive == 1          # user is told something was withheld


def test_sensitive_memory_can_be_opted_in_for_the_model(mm):
    mm.save("My blood group is O positive", allow_sensitive=True, share_with_model=True)
    assert "O positive" in mm.build_context("what is my blood group", "c").text


def test_logs_never_contain_memory_content(mm, caplog):
    caplog.set_level(logging.DEBUG)
    mm.ingest_turn("My name is Zarathustra Quillfeather and I live in Atlantis.", conversation_id="c")
    mm.save("Secret hobby: underwater basket weaving")
    mm.build_context("what is my name", "d")
    mm.ingest_turn("My PAN is ABCDE1234F", conversation_id="c")
    log = caplog.text
    for needle in ("Zarathustra", "Atlantis", "basket", "ABCDE1234F"):
        assert needle not in log


def test_turn_buffer_scrubs_sensitive_sentences_but_keeps_the_rest(mm):
    mm.ingest_turn("I was diagnosed with asthma. My editor is VS Code.", "Noted.", conversation_id="c")
    stored = " ".join(t["content"] for t in mm.store.get_turns("c"))
    assert "asthma" not in stored and "VS Code" in stored


def test_hedged_sensitive_words_in_normal_chat_do_not_block_normal_facts(mm):
    r = mm.ingest_turn("My favorite editor is Neovim.", conversation_id="c")
    assert r.saved and not r.held_back
