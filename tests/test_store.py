import json

import numpy as np
import pytest
from chromadb.api.types import EmbeddingFunction

from pa_chatbot.store import LiteratureStore, reciprocal_rank_fusion, tokenize

VOCAB = ["turnover", "engagement", "burnout", "network", "pay", "survival", "leadership", "demands"]


class BagOfWords(EmbeddingFunction):
    """Deterministic offline embedding for tests."""

    def __init__(self):
        pass

    def __call__(self, input):
        vecs = []
        for text in input:
            toks = tokenize(text)
            v = np.array([toks.count(w) for w in VOCAB], dtype=float) + 1e-3
            vecs.append(v / np.linalg.norm(v))
        return vecs

    @staticmethod
    def name():
        return "bag-of-words-test"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return BagOfWords()


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert [d for d, _ in fused][:2] in (["a", "b"], ["b", "a"])
    assert dict(fused)["d"] < dict(fused)["b"]


def _meta(pid, label, year):
    paper = {"title": f"{label} paper", "authors": [label], "year": str(year)}
    return {
        "paper_id": pid, "file": f"{pid}.pdf", "collection": "", "title": paper["title"], "label": f"{label} ({year})",
        "year_int": year, "paper_type": "other", "paper_json": json.dumps(paper), "page_start": 1, "page_end": 1,
    }


@pytest.fixture
def store(tmp_path):
    s = LiteratureStore(path=tmp_path / "chroma", embedding_function=BagOfWords(), collection_name="test")
    docs = {
        "p1": ("Smith", 2015, ["turnover turnover embeddedness predicts turnover", "turnover intention and pay", "turnover again"]),
        "p2": ("Lee", 2021, ["engagement and burnout under job demands", "demands resources engagement"]),
        "p3": ("Kim", 2019, ["survival analysis of turnover timing", "network analysis of collaboration"]),
    }
    for pid, (label, year, chunks) in docs.items():
        s.add_chunks(
            ids=[f"{pid}-{i}" for i in range(len(chunks))],
            texts=chunks,
            search_texts=chunks,
            metadatas=[_meta(pid, label, year) for _ in chunks],
        )
    return s


def test_hybrid_search_finds_relevant_papers(store):
    hits = store.search(["turnover"], k=4)
    assert hits[0].paper_id in {"p1", "p3"}
    assert "p2" not in {h.paper_id for h in hits[:2]}


def test_max_chunks_per_paper(store):
    hits = store.search(["turnover"], k=10, max_per_paper=1)
    ids = [h.paper_id for h in hits]
    assert len(ids) == len(set(ids))


def test_year_filter(store):
    hits = store.search(["turnover engagement"], k=10, year_min=2020)
    assert {h.paper_id for h in hits} == {"p2"}


def test_papers_and_delete(store):
    assert {p["authors"][0] for p in store.papers()} == {"Smith", "Lee", "Kim"}
    store.delete_paper("p1")
    assert "p1" not in store.paper_ids()
    assert all(h.paper_id != "p1" for h in store.search(["turnover"], k=10))
