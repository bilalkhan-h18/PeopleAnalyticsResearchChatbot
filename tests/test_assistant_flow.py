from types import SimpleNamespace as NS

from pa_chatbot import assistant as assistant_mod
from pa_chatbot.assistant import AnswerResult, Event, ResearchAssistant

from .fakes import FakeClient
from .test_store import store  # noqa: F401  (fixture)


def _answer(kw):
    results = [b for b in kw["messages"][-1]["content"] if b["type"] == "search_result"]
    first = results[0]["source"]
    return [
        NS(type="text", text="Embeddedness lowers exit risk.", citations=[NS(source=first, cited_text="turnover")]),
        NS(type="text", text=" Consider a survival model (general knowledge - not from your library).", citations=None),
    ]


def test_ask_end_to_end(store, monkeypatch):  # noqa: F811
    fake = FakeClient(_answer)
    monkeypatch.setattr(assistant_mod, "get_client", lambda: fake)
    monkeypatch.setattr("pa_chatbot.llm.get_client", lambda: fake)

    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    items = list(ResearchAssistant(store).ask("Why do people quit?", history, mode="Hypothesis builder", n_sources=4))

    assert isinstance(items[0], Event) and items[0].kind == "status"
    assert any(isinstance(i, Event) and i.kind == "delta" for i in items)
    result = items[-1]
    assert isinstance(result, AnswerResult)
    assert result.queries[0] == "Why do people quit?"
    assert "<sup>[1]</sup>" in result.markdown
    assert "Search queries used" in result.sources_md

    planner = [kw for kind, kw in fake.requests if kind == "create"][0]
    assert planner["output_config"]["format"]["type"] == "json_schema"
    answer = [kw for kind, kw in fake.requests if kind == "stream"][0]
    assert answer["model"] == "claude-opus-5"
    assert answer["fallbacks"] == "default"
    assert answer["betas"] == ["server-side-fallback-2026-07-01"]
    assert answer["thinking"] == {"type": "adaptive"}
    assert answer["messages"][:2] == history
    assert "Why do people quit?" in answer["messages"][-1]["content"][-1]["text"]
