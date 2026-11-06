"""Retrieval: relevance, new-conversation recall, cross-conversation linking, budgets."""

from __future__ import annotations

from era_memory import CallableEmbedding, EmbeddingService, MemoryConfig, MemoryManager, MemoryType
from era_memory.memory_retriever import HashingEmbedding


def test_retrieves_relevant_info_in_a_new_conversation(mm):
    mm.ingest_turn("My favorite color is teal.", conversation_id="chat-A")
    pkg = mm.build_context("What's my favorite color?", conversation_id="chat-B")   # different conversation
    assert "teal" in pkg.text and pkg.enabled and pkg.items


def test_irrelevant_memories_are_not_returned(mm):
    mm.ingest_turn("My favorite color is teal.", conversation_id="a")
    pkg = mm.build_context("How do I bake sourdough bread?", conversation_id="b")
    assert "teal" not in pkg.text


def test_no_memory_means_no_claims(mm):
    pkg = mm.build_context("What's my dog's name?", conversation_id="x")
    assert pkg.text == "" and pkg.items == []          # nothing reliable -> nothing to "remember"


def test_connects_project_memories_across_different_dates(mm, clock):
    mm.ingest_turn("I'm working on Atlas, the inventory app.", conversation_id="mon")
    mm.ingest_turn("We decided to use PostgreSQL for the Atlas database.", conversation_id="mon")
    clock.advance(days=9)
    mm.ingest_turn("Atlas blocker is the barcode scanner driver.", conversation_id="wed")
    clock.advance(days=3)
    summ = mm.project_summary("Atlas")
    assert summ["memory_count"] >= 2 and {"mon", "wed"} <= set(summ["conversations"])
    assert "PostgreSQL" in summ["summary"] and "barcode" in summ["summary"]
    pkg = mm.build_context("Any update on Atlas?", conversation_id="fri")             # third, later chat
    assert "PostgreSQL" in pkg.text and "barcode" in pkg.text and pkg.project_id == "atlas"


def test_continue_unfinished_task_from_a_previous_session(mm, clock):
    mm.ingest_turn("I'm working on the invoice parser.", conversation_id="yesterday")
    clock.advance(days=1)
    pkg = mm.build_context("Let's continue where we left off", conversation_id="today")
    assert pkg.continuation and "invoice parser" in pkg.text


def test_next_steps_and_blockers_surface_for_the_project(mm):
    mm.ingest_turn("I'm building Atlas. Next step is wiring the scanner.", conversation_id="c1")
    mm.ingest_turn("I'm stuck on the label printer for Atlas.", conversation_id="c1")
    pkg = mm.build_context("what's left on Atlas", conversation_id="c2")
    assert "wiring the scanner" in pkg.text and "label printer" in pkg.text


def test_communication_preferences_always_apply(mm):
    mm.ingest_turn("Please keep your answers short.", conversation_id="a")
    pkg = mm.build_context("Explain how photosynthesis works", conversation_id="b")
    assert "short" in pkg.text


def test_recency_and_importance_break_ties_but_never_replace_relevance(mm, clock):
    mm.save("Backups run nightly on the NAS", importance=0.3)
    clock.advance(days=200)
    mm.save("Totally unrelated: the cat likes tuna", importance=1.0)
    hits = mm.search("when do backups run")
    assert hits and "Backups" in hits[0].memory.content
    assert all("tuna" not in h.memory.content for h in hits)


def test_context_respects_the_character_budget(make_manager):
    mm = make_manager(max_context_chars=500)
    for i in range(30):
        mm.save(f"Atlas requirement number {i}: the scanner must support format {i} barcodes")
    pkg = mm.build_context("atlas scanner requirement barcodes", conversation_id="c")
    assert 0 < pkg.char_count <= 500 and pkg.truncated


def test_retrieval_limit_is_configurable(make_manager):
    mm = make_manager(max_retrieved=2, max_context_chars=5000)
    for i in range(6):
        mm.save(f"Server rack {i} hosts the database replica {i}")
    assert len(mm.build_context("database replica server rack", "c").items) <= 2


def test_uncertain_and_confirmed_are_labelled_differently(mm):
    mm.ingest_turn("I think my deadline is Friday.", conversation_id="c")
    text = mm.build_context("what is my deadline", "d").text
    assert "Friday" in text and "uncertain" in text
    mm2_text = mm.build_context("what is my deadline", "d").text
    assert "confirmed" not in mm2_text.split("Relevant memories:")[1]


def test_working_memory_of_other_conversations_is_not_leaked_normally(mm):
    mm.ingest_turn("I'm working on the graph exporter.", conversation_id="c1")
    assert "graph exporter" not in mm.build_context("tell me about graphs exporter", "c2").text
    assert "graph exporter" in mm.build_context("tell me about graphs exporter", "c1").text


def test_unavailable_primary_embedding_provider_falls_back(db_path, clock):
    def boom(_t):
        raise RuntimeError("embedding API is down")

    emb = EmbeddingService([CallableEmbedding("remote-embed", boom)])
    mm = MemoryManager(MemoryConfig(db_path=db_path), embedder=emb, clock=clock)
    mm.save("ERA stores memories in a local SQLite file")
    hits = mm.search("where are memories stored")
    assert hits and "SQLite" in hits[0].memory.content
    assert mm.stats()["embedding_providers"][-1] == HashingEmbedding.name


def test_hashing_embedding_matches_spelling_variants(mm):
    mm.save("The colour scheme is dark blue")
    assert mm.search("colours scheme")               # morphological/spelling similarity works offline


def test_semantic_only_match_without_shared_keywords(db_path, clock):
    """A pluggable provider can supply true semantics; the retriever must use it."""
    vocab = {"car": [1, 0, 0], "automobile": [1, 0, 0], "banana": [0, 1, 0]}

    def fake(text):
        for w, v in vocab.items():
            if w in text.lower():
                return v
        return [0, 0, 1]

    mm = MemoryManager(MemoryConfig(db_path=db_path), embedder=EmbeddingService([CallableEmbedding("fake-sem", fake)]), clock=clock)
    mm.save("I drive an automobile to work")
    hits = mm.search("my car")
    assert hits and "automobile" in hits[0].memory.content and hits[0].keyword == 0.0


def test_single_shared_word_is_not_enough_to_count_as_relevant(mm):
    """A lone overlapping word (here 'firmware') must not drag in an otherwise unrelated memory."""
    mm.save("Atlas scanner uses the USB HID protocol with custom driver firmware")
    q = "how should I configure the firmware update settings for my printer at home"
    assert mm.search(q) == []
    assert mm.build_context(q, "c").text == ""
    # while a genuinely related question still finds it
    assert mm.search("which protocol does the Atlas scanner use")


def test_compound_question_retrieves_every_relevant_memory(mm):
    """Regression: a multi-part question dilutes keyword overlap (and every memory starts with the generic
    word 'User'); each part must still find its memory. The database part is answered through the project
    summary because 'database' vs 'SQLite' is a synonym link only a real embedding provider can make."""
    mm.ingest_turn("I live in Agra.", conversation_id="a")
    mm.ingest_turn("I'm building ERA.", conversation_id="a")
    mm.ingest_turn("We decided to use SQLite for ERA memory.", conversation_id="a")
    question = "where do I live and what database is ERA using?"
    text = mm.build_context(question, "b").text
    assert "Agra" in text and "SQLite" in text
    mm.ingest_turn("Actually I moved to Pune.", conversation_id="b")
    text = mm.build_context(question, "c").text
    assert "Pune" in text and "Agra" not in text and "SQLite" in text


def test_synonym_gap_is_closed_by_a_real_embedding_provider(db_path, clock):
    """Documented limit of the built-in fallback: with a semantic provider plugged in, the link is made."""
    from era_memory import CallableEmbedding, EmbeddingService, MemoryConfig, MemoryManager

    def sem(text):
        t = text.lower()
        return [1.0 if ("database" in t or "sqlite" in t) else 0.0, 1.0 if "agra" in t or "live" in t else 0.0, 0.05]

    mm = MemoryManager(MemoryConfig(db_path=db_path), embedder=EmbeddingService([CallableEmbedding("sem", sem)]), clock=clock)
    mm.ingest_turn("We decided to use SQLite for ERA memory.", conversation_id="a")     # no project registered
    assert "SQLite" in mm.build_context("what database is ERA using?", "b").text
