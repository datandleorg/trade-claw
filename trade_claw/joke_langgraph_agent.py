"""LangGraph joke agent: single LLM node (ready to extend with tools / multi-step flows)."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

DEFAULT_JOKE_MODEL = "gpt-5.4-mini"

_JOKE_SYSTEM = (
    "You are a brief comedy writer. Reply with exactly one short original joke, "
    "no preamble or explanation."
)


class JokeState(TypedDict, total=False):
    """Graph state: populated by the joke_agent node."""

    joke: str


def build_joke_agent_graph(*, api_key: str, model: str) -> CompiledStateGraph:
    llm = ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=0.9,
        max_completion_tokens=200,
    )

    def joke_agent(_state: JokeState) -> JokeState:
        msg = llm.invoke(
            [
                SystemMessage(content=_JOKE_SYSTEM),
                HumanMessage(content="Tell me a new joke."),
            ]
        )
        text = (msg.content or "").strip()
        return {"joke": text}

    builder = StateGraph(JokeState)
    builder.add_node("joke_agent", joke_agent)
    builder.add_edge(START, "joke_agent")
    builder.add_edge("joke_agent", END)
    return builder.compile()
