import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

from pa_chatbot import screening
from pa_chatbot.screening import KEEP, REMOVE, UNSURE, keyword_verdict, screen

ROOT = Path(__file__).resolve().parent.parent


def test_keyword_rules():
    assert keyword_verdict("Cancer statistics, 2019", "CA A Cancer Journal for Clinicians").decision == REMOVE
    assert keyword_verdict("Why people stay: using job embeddedness to predict voluntary turnover", "AMJ").decision == KEEP
    assert keyword_verdict("Dynamic capabilities: what are they?", "SMJ").decision == UNSURE


def test_claude_screening_maps_ids_and_falls_back(monkeypatch):
    calls = []

    def fake_call(**kw):
        calls.append(kw)
        # Answer only the first paper; the second must fall back to keywords.
        return {"decisions": [{"id": 0, "decision": "remove", "reason": "oncology"}]}

    monkeypatch.setattr(screening, "structured_call", fake_call)
    papers = [{"title": "Cancer statistics", "venue": "CA"}, {"title": "Employee turnover", "venue": "JAP"}]
    verdicts = screen(papers)
    assert [v.decision for v in verdicts] == [REMOVE, KEEP]
    assert [v.method for v in verdicts] == ["claude", "keywords"]
    assert "0. Cancer statistics | CA" in calls[0]["prompt"]


def _run(corpus, *args):
    env = dict(os.environ, CORPUS_DIR=str(corpus), DATA_DIR=str(corpus.parent / "data"))
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "clean_corpus.py"), *args],
                          env=env, capture_output=True, text=True, check=True).stdout


def test_review_then_apply(tmp_path):
    corpus = tmp_path / "corpus"
    fields = ["topic", "status", "file", "title", "year", "authors", "venue", "doi", "cited_by", "pdf_url", "openalex_id"]
    rows = [
        {"topic": "hr", "status": "downloaded", "file": "a.pdf", "title": "Cancer statistics, 2019", "venue": "The Lancet", "openalex_id": "W1"},
        {"topic": "hr", "status": "downloaded", "file": "b.pdf", "title": "Employee turnover meta-analysis", "venue": "JAP", "openalex_id": "W2"},
        {"topic": "hr", "status": "downloaded", "file": "c.pdf", "title": "Dynamic capabilities", "venue": "SMJ", "openalex_id": "W3"},
    ]
    (corpus / "hr").mkdir(parents=True)
    (corpus / "mine").mkdir()
    for r in rows:
        (corpus / "hr" / r["file"]).write_bytes(b"%PDF")
    (corpus / "mine" / "own.pdf").write_bytes(b"%PDF")
    with open(corpus / "_found_papers.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    out = _run(corpus, "--no-llm")
    assert "Nothing has been deleted" in out
    assert all((corpus / "hr" / r["file"]).exists() for r in rows)

    # The user overrides a decision in Excel: remove the "unsure" one too.
    review = list(csv.DictReader(open(corpus / "_cleanup_review.csv", encoding="utf-8-sig")))
    for r in review:
        if r["title"] == "Dynamic capabilities":
            r["decision"] = "remove"
    with open(corpus / "_cleanup_review.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(review[0]))
        w.writeheader()
        w.writerows(review)

    _run(corpus, "--apply")
    assert not (corpus / "hr" / "a.pdf").exists() and (corpus / "_removed" / "hr" / "a.pdf").exists()
    assert not (corpus / "hr" / "c.pdf").exists()
    assert (corpus / "hr" / "b.pdf").exists()
    assert (corpus / "mine" / "own.pdf").exists()
    log = {r["openalex_id"]: r["status"] for r in csv.DictReader(open(corpus / "_found_papers.csv", encoding="utf-8-sig"))}
    assert log == {"W1": "removed", "W2": "downloaded", "W3": "removed"}
