"""A stand-in for anthropic.Anthropic that records requests and returns canned output."""

import json
from types import SimpleNamespace as NS


class FakeStream:
    def __init__(self, blocks, stop_reason="end_turn"):
        self._blocks = blocks
        self._stop = stop_reason

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for b in self._blocks:
            if b.type == "text":
                for word in b.text.split(" "):
                    yield NS(type="text", text=word + " ")

    def get_final_message(self):
        return NS(content=self._blocks, stop_reason=self._stop)


class FakeClient:
    def __init__(self, answer_fn):
        self.requests = []
        self._answer_fn = answer_fn
        outer = self
        self.messages = NS(create=self._create)
        self.beta = NS(messages=NS(stream=lambda **kw: outer._stream(**kw)))

    def _create(self, **kw):
        self.requests.append(("create", kw))
        payload = {"queries": ["job embeddedness turnover", "survival analysis voluntary turnover"]}
        return NS(stop_reason="end_turn", content=[NS(type="text", text=json.dumps(payload))])

    def _stream(self, **kw):
        self.requests.append(("stream", kw))
        return FakeStream(self._answer_fn(kw))
