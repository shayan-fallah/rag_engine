"""Test doubles: a bag-of-words embedding model and a scripted chat model."""

import hashlib
import math
import re

from django.conf import settings

from core.llm import LLM


def fake_embed(text: str) -> list[float]:
    """Words hashed into the vector: texts sharing words are close."""
    vec = [0.0] * settings.EMBEDDING_DIMENSIONS
    for word in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % len(vec)] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec] if any(vec) else [1.0 / math.sqrt(len(vec))] * len(vec)


class EmbeddingCounter:
    def __init__(self):
        self.texts: list[str] = []

    def embed_texts(self, texts):
        self.texts.extend(texts)
        return [fake_embed(t) for t in texts]

    def embed_text(self, text):
        return fake_embed(text)


class ScriptedLLM(LLM):
    """Answers with a function of (system, messages). Records every call.

    reply(system, messages) -> str           for stream/complete
    choose(system, messages) -> str          category for parse()
    tool_plan(system, messages) -> [(tool, args)]   tool calls run_tools() makes before answering
    """

    def __init__(self, reply=None, choose=None, tool_plan=None):
        super().__init__("scripted", None, None)
        self.reply = reply or (lambda system, messages: "OK")
        self.choose = choose
        self.tool_plan = tool_plan or (lambda system, messages: [])
        self.calls: list[dict] = []

    def stream(self, system, messages):
        self.calls.append({"kind": "stream", "system": system, "messages": messages})
        text = self.reply(system, messages)
        yield from re.findall(r"\S+\s*", text)

    def parse(self, system, messages, schema):
        self.calls.append({"kind": "parse", "system": system, "messages": messages})
        return schema(category=self.choose(system, messages))

    def run_tools(self, system, messages, tools, max_rounds=5, on_text=None):
        self.calls.append({"kind": "tools", "system": system, "messages": messages})
        by_name = {t.name: t for t in tools}
        results = [by_name[name].run(args) for name, args in self.tool_plan(system, messages)]
        final_messages = messages + [{"role": "tool", "content": "\n".join(results)}]
        self.calls.append({"kind": "tools_answer", "system": system, "messages": final_messages})
        text = self.reply(system, final_messages)
        if on_text:
            on_text(text)
        return text
