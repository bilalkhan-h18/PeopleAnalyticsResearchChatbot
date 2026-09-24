"""Hybrid (semantic + keyword) index over the literature chunks.

- Semantic: Chroma, embedded locally (no OpenAI key needed; Anthropic has no
  embeddings endpoint). Chroma's default ONNX MiniLM model is used unless
  EMBEDDING_MODEL names a sentence-transformers model (e.g. BAAI/bge-base-en-v1.5).
- Keyword: BM25, which matters for exact terms like "UWES", "JD-R" or
  "turnover intention" that embeddings blur together.
- The two rankings are merged with reciprocal rank fusion.
"""

import json
import os
import re
from dataclasses import dataclass

from .config import settings

_STOPWORDS = set(
    "a an and are as at be but by for from has have in into is it its of on or "
    "that the their there these this to was were which with we our they not can "
    "may also than between more such been how what does do".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower()) if t not in _STOPWORDS]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict
    score: float

    @property
    def paper_id(self) -> str:
        return self.metadata["paper_id"]

    @property
    def pages(self) -> str:
        a, b = self.metadata.get("page_start"), self.metadata.get("page_end")
        return f"p. {a}" if a == b else f"pp. {a}-{b}"


def _embedding_function():
    from chromadb.utils import embedding_functions

    model = os.getenv("EMBEDDING_MODEL")
    if model:
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)
    return embedding_functions.DefaultEmbeddingFunction()


class LiteratureStore:
    def __init__(self, path=None, embedding_function=None, collection_name=None):
        import chromadb

        path = path or settings.chroma_dir
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name or settings.collection_name,
            embedding_function=embedding_function or _embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._bm25_meta: list[dict] = []

    # ---- writing -------------------------------------------------------

    def paper_ids(self) -> set[str]:
        result = self._collection.get(include=["metadatas"])
        return {m["paper_id"] for m in result["metadatas"]}

    def delete_paper(self, paper_id: str) -> None:
        self._collection.delete(where={"paper_id": paper_id})
        self._bm25 = None

    def add_chunks(self, ids: list[str], texts: list[str], search_texts: list[str], metadatas: list[dict]) -> None:
        """`texts` is what Claude reads; `search_texts` adds a paper-level header
        ("contextual chunk") so a passage is findable by its paper's framework,
        method and title even when the passage itself doesn't name them."""
        batch = 200
        for i in range(0, len(ids), batch):
            metas = [dict(m, display_text=t) for m, t in zip(metadatas[i:i + batch], texts[i:i + batch])]
            self._collection.upsert(
                ids=ids[i:i + batch],
                documents=search_texts[i:i + batch],
                metadatas=metas,
            )
        self._bm25 = None

    # ---- reading -------------------------------------------------------

    def count(self) -> int:
        return self._collection.count()

    def papers(self) -> list[dict]:
        result = self._collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for m in result["metadatas"]:
            if m["paper_id"] not in seen:
                seen[m["paper_id"]] = json.loads(m["paper_json"])
        return list(seen.values())

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi

        result = self._collection.get(include=["documents", "metadatas"])
        self._bm25_ids = result["ids"]
        self._bm25_meta = result["metadatas"]
        corpus = [tokenize(d) for d in result["documents"]]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def _keyword_search(self, query: str, n: int, year_min: int | None) -> list[str]:
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order:
            if scores[i] <= 0 or len(out) >= n:
                break
            if year_min and (self._bm25_meta[i].get("year_int") or 0) < year_min:
                continue
            out.append(self._bm25_ids[i])
        return out

    def _vector_search(self, query: str, n: int, year_min: int | None) -> list[str]:
        total = self.count()
        if total == 0:
            return []
        where = {"year_int": {"$gte": year_min}} if year_min else None
        result = self._collection.query(query_texts=[query], n_results=min(n, total), where=where)
        return result["ids"][0]

    def search(
        self,
        queries: list[str],
        k: int = settings.default_sources,
        year_min: int | None = None,
        per_query: int = settings.candidates_per_query,
        max_per_paper: int = settings.max_chunks_per_paper,
    ) -> list[Hit]:
        rankings: list[list[str]] = []
        for q in queries:
            rankings.append(self._vector_search(q, per_query, year_min))
            rankings.append(self._keyword_search(q, per_query, year_min))
        fused = reciprocal_rank_fusion(rankings)
        if not fused:
            return []

        # Fetch a generous pool, then cap chunks per paper so one long book
        # can't crowd out everything else.
        pool_ids = [doc_id for doc_id, _ in fused[: k * 4]]
        score_of = dict(fused)
        got = self._collection.get(ids=pool_ids, include=["metadatas"])
        by_id = dict(zip(got["ids"], got["metadatas"]))

        hits, per_paper = [], {}
        for doc_id in pool_ids:
            meta = by_id.get(doc_id)
            if meta is None:
                continue
            pid = meta["paper_id"]
            if per_paper.get(pid, 0) >= max_per_paper:
                continue
            per_paper[pid] = per_paper.get(pid, 0) + 1
            hits.append(Hit(id=doc_id, text=meta["display_text"], metadata=meta, score=score_of[doc_id]))
            if len(hits) >= k:
                break
        return hits
