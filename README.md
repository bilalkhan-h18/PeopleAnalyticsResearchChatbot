# People Analytics Research Assistant

A research partner for people analytics work. Describe a workforce problem or a working thesis, and it searches **your own library** of academic and practitioner literature, then gives you:

- the relevant research, with page-level citations you can check,
- theories and frameworks that could structure the problem (e.g. JD-R, Job Embeddedness, Social Exchange),
- testable hypotheses,
- analytical approaches suited to typical HR data, and the threats to validity,
- practical, ethical and stakeholder considerations.

It evolved from `legacy/litgen_rag_model.py` (a university Colab RAG notebook on OpenAI + LangChain). This version runs locally on **Claude** with a **Gradio 5** UI.

## How it works

```
question ─► Claude plans 3-5 academic search queries
         ─► hybrid search over your library (semantic + BM25 keyword, fused)
         ─► top excerpts sent to Claude as citable search results
         ─► streamed answer with [n] citations + quoted passages in the side panel
```

| Piece | Choice | Why |
|---|---|---|
| PDF parsing | PyMuPDF, page-aware, bibliography stripped | Page numbers survive into citations; reference lists don't pollute search |
| Paper metadata | Claude tags each paper once (authors, year, frameworks, constructs, methods, topics) | Proper references; chunks are searchable by their paper's framework/method |
| Embeddings | Local (Chroma's MiniLM by default, or any sentence-transformers model) | Free, private, no second API key |
| Search | Vector + BM25 with reciprocal rank fusion, max 3 excerpts per paper | Exact terms ("UWES", "JD-R") and meaning both count; no single book dominates |
| Answering | Claude with search-result citations, adaptive thinking, 6 answer modes | Citations are tied to the actual passages, not invented |

**Answer modes:** Research brief (default), Frameworks & theories, Hypothesis builder, Methods & measurement, Evidence check, Open question.

A "general knowledge" toggle lets Claude go beyond your library. When it does, it marks those points *(general knowledge - not from your library)*.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then paste your API key into .env
```

### 1. Build your library

Put PDFs in `corpus/` (sub-folders optional). To find open-access papers automatically:

```bash
python scripts/find_papers.py                          # built-in people analytics topic list
python scripts/find_papers.py --query "psychological safety teams" --topic teams
```

This searches [OpenAlex](https://openalex.org), downloads open-access PDFs into `corpus/<topic>/`, and logs everything, including paywalled papers you may want to get through a library, to `corpus/_found_papers.csv`.

### 2. Index it

The first run downloads a small local embedding model (~80 MB).

```bash
python -m pa_chatbot.ingest            # new/changed PDFs only; safe to re-run
python -m pa_chatbot.ingest --no-llm   # skip Claude tagging (free, weaker references)
python -m pa_chatbot.ingest --rebuild  # start over (e.g. after changing EMBEDDING_MODEL)
```

### 3. Chat

```bash
python app.py                          # http://127.0.0.1:7860
```

### Checking quality

```bash
python scripts/evaluate.py             # retrieval-only, free: which papers come back for each eval question
python scripts/evaluate.py --answer    # full answers to eval/results.md (costs one chat turn per question)
pytest                                 # unit tests (no API calls), needs requirements-dev.txt
```

Edit `eval/questions.txt` with questions from your own work and the papers you expect to see.

## Cost and your API key

The app uses your personal Anthropic API key. It's pay-as-you-go, billed per token.

**Estimated cost per question** (12 excerpts, default settings):

| Answer model | Per question | 200 questions/month |
|---|---|---|
| `claude-opus-5` (default, $5 / $25 per M input/output tokens) | ~$0.15-0.30 | ~$30-60 |
| `claude-sonnet-5` ($2 / $10) | ~$0.06-0.12 | ~$12-25 |

**Indexing** costs roughly $0.04 per paper on Opus 5, or $0.02 on Sonnet 5. It's a one-off cost per paper: tags are cached in `data/paper_metadata.json`. Use `--no-llm` to index for free. Embeddings run locally and cost nothing.

To cut costs, set `ANSWER_MODEL=claude-sonnet-5` and/or `UTILITY_MODEL=claude-sonnet-5` in `.env`. You can also lower `ANSWER_EFFORT` to `medium` or retrieve fewer excerpts.

**Getting a key:**
1. Sign in at the Claude Console ([platform.claude.com](https://platform.claude.com)).
2. In the billing settings, buy prepaid credits. $20-25 is a sensible start. Leave auto-reload **off** until you know your usage.
3. Set a monthly spend limit, so a runaway script can't surprise you.
4. Create an API key and paste it into `.env`. Never commit `.env` (it's git-ignored).
5. Check the Console's usage page after a week or two and adjust.

**Data:** questions and retrieved excerpts go to the Anthropic API under your personal account. Don't enter employee-level data, names or confidential company information; describe the problem instead. Check your employer's policy on external AI tools.

## Project layout

```
app.py                  Gradio 5 UI
pa_chatbot/
  config.py             settings from .env
  chunking.py           PDF → pages → overlapping, page-tracked chunks
  metadata.py           Claude paper tagging + reference formatting
  store.py              Chroma + BM25 hybrid index
  prompts.py            system prompt and answer modes (edit these to tune behaviour)
  assistant.py          plan → retrieve → cited streaming answer
  ingest.py             indexing CLI
scripts/find_papers.py  OpenAlex open-access paper finder
scripts/evaluate.py     retrieval/answer checks on eval/questions.txt
legacy/                 the original university notebook, for reference
```

## Ideas for later

- A cross-encoder reranker (e.g. `BAAI/bge-reranker-base`) between search and answer.
- GROBID parsing for cleaner section structure in two-column journal PDFs.
- A "save to project" button that collects answers into a running literature review.
