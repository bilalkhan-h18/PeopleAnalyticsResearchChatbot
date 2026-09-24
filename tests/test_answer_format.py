import json
from types import SimpleNamespace as NS

from pa_chatbot.assistant import build_user_content, format_answer
from pa_chatbot.metadata import cite_label, reference_entry
from pa_chatbot.prompts import build_system_prompt
from pa_chatbot.store import Hit


def _hit(id, pid, authors, year, title):
    paper = {"title": title, "authors": authors, "year": year, "venue": "J. Test", "doi": "10.1/x"}
    meta = {"paper_id": pid, "label": cite_label(paper), "title": title, "page_start": 3, "page_end": 4,
            "paper_json": json.dumps(paper)}
    return Hit(id=id, text=f"text of {id}", metadata=meta, score=1.0)


def test_cite_label_variants():
    assert cite_label({"authors": ["Jane Smith"], "year": "2020"}) == "Smith (2020)"
    assert cite_label({"authors": ["Marler, Janet", "Boudreau, John"], "year": "2017"}) == "Marler & Boudreau (2017)"
    assert cite_label({"authors": ["A B", "C D", "E F"], "year": ""}) == "B et al. (n.d.)"


def test_reference_entry_has_doi_link():
    assert "https://doi.org/10.1/x" in reference_entry({"authors": ["A"], "year": "2020", "title": "T", "doi": "10.1/x"})


def test_user_content_carries_search_results_then_question():
    hits = [_hit("c1", "p1", ["Ann Lee"], "2020", "Paper one")]
    content = build_user_content("Why do people leave?", "Hypothesis builder", hits)
    assert content[0]["type"] == "search_result"
    assert content[0]["source"] == "c1"
    assert content[0]["citations"] == {"enabled": True}
    assert content[-1]["type"] == "text" and "Why do people leave?" in content[-1]["text"]
    assert "hypotheses" in content[-1]["text"].lower()


def test_format_answer_numbers_papers_by_first_citation():
    hits = [
        _hit("c1", "p1", ["Ann Lee"], "2020", "Paper one"),
        _hit("c2", "p2", ["Bo Kim"], "2018", "Paper two"),
        _hit("c3", "p1", ["Ann Lee"], "2020", "Paper one"),
        _hit("c4", "p3", ["Cy Park"], "2010", "Paper three"),
    ]
    cite = lambda src: NS(source=src, cited_text="quoted words")  # noqa: E731
    blocks = [
        NS(type="thinking", thinking=""),
        NS(type="text", text="Intro. ", citations=None),
        NS(type="text", text="Kim found X.", citations=[cite("c2")]),
        NS(type="text", text=" Lee found Y.", citations=[cite("c1"), cite("c3")]),
    ]
    md, plain, sources = format_answer(blocks, hits)
    assert plain == "Intro. Kim found X. Lee found Y."
    assert "Kim found X.<sup>[1]</sup>" in md
    assert "Lee found Y.<sup>[2]</sup>" in md
    assert "[1] Kim" not in md  # references list uses full entries
    assert "**References**" in md and "Paper two" in md
    assert "Also retrieved" in sources and "Paper three" in sources
    assert sources.count('"quoted words"') == 2  # deduped per paper


def test_system_prompt_policy_toggle():
    assert "general knowledge - not from your library" in build_system_prompt(True)
    assert "Answer only from the retrieved excerpts" in build_system_prompt(False)
