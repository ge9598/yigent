"""Web search tool — Tavily API (preferred), with DuckDuckGo fallback."""

from __future__ import annotations

import logging

import httpx

from src.core.types import PermissionLevel, ToolContext, ToolDefinition, ToolSchema

from .registry import register

logger = logging.getLogger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"
_DDG_URL = "https://api.duckduckgo.com/"


async def _tavily_search(api_key: str, query: str, max_results: int, timeout: int) -> str:
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": True,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(_TAVILY_URL, json=payload)
        if resp.status_code != 200:
            return f"Error: Tavily returned {resp.status_code}: {resp.text[:300]}"
        data = resp.json()

    lines: list[str] = []
    if data.get("answer"):
        lines.append(f"Answer: {data['answer']}\n")
    for i, r in enumerate(data.get("results", []), 1):
        lines.append(f"{i}. {r.get('title', 'untitled')}")
        lines.append(f"   {r.get('url', '')}")
        content = (r.get("content") or "").strip().replace("\n", " ")
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"   {content}")
        lines.append("")
    return "\n".join(lines) if lines else f"No results for: {query}"


async def _duckduckgo_search(query: str, timeout: int) -> str:
    params = {"q": query, "format": "json", "no_html": "1"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(_DDG_URL, params=params)
        if resp.status_code != 200:
            return f"Error: DuckDuckGo returned {resp.status_code}"
        data = resp.json()

    lines: list[str] = []
    if data.get("AbstractText"):
        lines.append(f"Abstract: {data['AbstractText']}")
        if data.get("AbstractURL"):
            lines.append(f"Source: {data['AbstractURL']}\n")
    for topic in data.get("RelatedTopics", [])[:5]:
        if "Text" in topic:
            lines.append(f"- {topic['Text']}")
            if topic.get("FirstURL"):
                lines.append(f"  {topic['FirstURL']}")
    if not lines:
        return (
            f"No results from DuckDuckGo for: {query}\n"
            "(DDG instant-answer API is limited. Configure Tavily for real search.)"
        )
    return "\n".join(lines)


async def _web_search_handler(ctx: ToolContext, query: str, num_results: int | None = None) -> str:
    cfg = ctx.config.search
    max_results = num_results or cfg.max_results
    timeout = cfg.timeout

    if cfg.provider == "tavily":
        if not cfg.tavily_api_key:
            return (
                "Error: Tavily provider selected but tavily_api_key not set. "
                "Add it to configs/local.yaml under search.tavily_api_key."
            )
        try:
            return await _tavily_search(cfg.tavily_api_key, query, max_results, timeout)
        except httpx.HTTPError as e:
            # Log only the exception class (and HTTP status when available) —
            # str(e) for httpx errors includes the request URL, which can leak
            # query string secrets or token-bearing redirects. See audit A4/#6.
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None:
                logger.warning(
                    "Tavily failed (%s status=%s), falling back to DuckDuckGo",
                    type(e).__name__, status,
                )
            else:
                logger.warning(
                    "Tavily failed (%s), falling back to DuckDuckGo",
                    type(e).__name__,
                )
            return await _duckduckgo_search(query, timeout)

    return await _duckduckgo_search(query, timeout)


register(ToolDefinition(
    name="web_search",
    description="Search the web via Tavily (preferred) or DuckDuckGo.",
    handler=_web_search_handler,
    schema=ToolSchema(
        name="web_search",
        description="Search the web and return top results with titles, URLs, and snippets.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "num_results": {"type": "integer", "description": "Max results (default: config)."},
            },
            "required": ["query"],
        },
        permission_level=PermissionLevel.READ_ONLY,
        timeout=20,
    ),
    needs_context=True,
))
