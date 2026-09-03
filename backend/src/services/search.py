"""Search dispatch helpers leveraging agent search tools."""

from __future__ import annotations

import logging
import os
import random
import time
from threading import Lock
from typing import Any, Optional, Tuple

import requests

from hello_agents.tools import SearchTool
from hello_agents.tools.builtin.search_tool import _fetch_raw_content, _limit_text

from config import Configuration
from utils import (
    deduplicate_and_format_sources,
    format_sources,
    get_config_value,
)

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_SOURCE = 2000

# Semantic Scholar 限流策略（依据 paper-agent 实测经验：
# docs/Semantic Scholar 使用文档.md，2026-09-02 晚间高峰实测）：
# - 带 Key 节流 1.2s/请求（官方许可上限 1 req/s），匿名 3s/请求；
# - 429/5xx 尊重 Retry-After 头；缺失时指数退避 + 0~50% 抖动
#   （1s→2s→4s→8s 封顶），最多 5 次尝试；
# - 其他 4xx（如 403 Key 无效）立即失败，不重试；
# - 查询过长（>80 字符）会被截断/无结果，客户端主动截断并准备短查询变体。
_S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
_S2_INTERVAL_KEYED = 1.2
_S2_INTERVAL_ANON = 3.0
_S2_MAX_ATTEMPTS = 5
_S2_BACKOFF_BASE = 1.0
_S2_BACKOFF_CAP = 8.0
_S2_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_S2_QUERY_MAX_CHARS = 80
_S2_QUERY_MAX_TERMS = 4  # 词条过多会过严匹配导致 0 结果，二次尝试时截断

# OpenAlex 兜底：免费、无 key、限流宽松，学术文献质量与 S2 相当
_OPENALEX_URL = "https://api.openalex.org/works"
_OPENALEX_SELECT = "title,doi,publication_year,cited_by_count,abstract_inverted_index"
_S2_LOCK = Lock()
_s2_last_call = 0.0

# ddgs 自带的 bing 引擎默认被禁用（disabled=True），而其 duckduckgo 引擎
# 请求 html.duckduckgo.com，在部分网络环境（如国内）不可直达。这里手动
# 启用并注册 bing 引擎，作为本地网络下的可用搜索后端。
def _enable_ddgs_bing_engine() -> None:
    try:
        from ddgs.engines import ENGINES
        from ddgs.engines.bing import Bing

        Bing.disabled = False
        ENGINES["text"]["bing"] = Bing
    except Exception as exc:  # noqa: BLE001 - 可选能力，失败不影响启动
        logger.warning("无法启用 ddgs bing 引擎: %s", exc)


_enable_ddgs_bing_engine()

# 这些后端在未配置 Tavily/SerpApi 时最终都会落到 DuckDuckGo（不可达），
# 统一改走 ddgs 的 bing 引擎（直连可达）。
_BING_FALLBACK_BACKENDS = {"duckduckgo", "advanced", "hybrid"}

_GLOBAL_SEARCH_TOOL = SearchTool(backend="hybrid")


def _search_via_bing(
    query: str,
    fetch_full_page: bool,
    max_results: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Search through ddgs bing engine and normalise results.

    Mirrors the payload shape of ``SearchTool._search_duckduckgo`` so the
    downstream context builders keep working unchanged.
    """
    from ddgs import DDGS

    results: list[dict[str, Any]] = []
    notices: list[str] = []

    try:
        with DDGS(timeout=15) as client:  # type: ignore[call-arg]
            search_results = list(
                client.text(query, backend="bing", max_results=max_results)
            )
    except Exception as exc:  # pragma: no cover - 网络异常
        raise RuntimeError(f"Bing 搜索失败: {exc}") from exc

    for entry in search_results:
        url = entry.get("href") or entry.get("url")
        title = entry.get("title") or url or ""
        content = entry.get("body") or entry.get("content") or ""

        if not url or not title:
            notices.append(f"忽略不完整的 Bing 结果: {entry}")
            continue

        raw_content = content
        if fetch_full_page and url:
            fetched = _fetch_raw_content(url)
            if fetched:
                raw_content = _limit_text(fetched, max_tokens)

        results.append(
            {
                "title": title,
                "url": url,
                "content": content,
                "raw_content": raw_content,
            }
        )

    return {
        "results": results,
        "backend": "bing",
        "answer": None,
        "notices": notices,
    }


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Parse Retry-After header (seconds; HTTP-date 暂不支持，退回默认退避)."""

    if not value:
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


def _s2_backoff_wait(attempt: int, retry_after: Optional[float]) -> float:
    """退避等待：尊重 Retry-After；否则指数退避 1s→2s→4s→8s 封顶 + 0~50% 抖动."""

    if retry_after is not None:
        return max(1.0, retry_after)
    base = min(_S2_BACKOFF_CAP, _S2_BACKOFF_BASE * (2 ** attempt))
    return base * (1 + random.uniform(0.0, 0.5))


def _search_via_semanticscholar(
    query: str,
    max_results: int,
) -> dict[str, Any]:
    """Search Semantic Scholar for academic papers and normalise results.

    结果字段对齐 SearchTool 的 `_normalized_result` 形状，下游
    prepare_research_context / SourceRegistry 无需改动。
    """
    global _s2_last_call

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    headers = {"x-api-key": api_key} if api_key else {}
    interval = _S2_INTERVAL_KEYED if api_key else _S2_INTERVAL_ANON

    terms = query.split()
    queries: list[str] = [query[:_S2_QUERY_MAX_CHARS]]
    if len(terms) > _S2_QUERY_MAX_TERMS:
        # 词条过多时 S2 全词匹配可能 0 结果，二次尝试用前几个关键词
        queries.append(" ".join(terms[:_S2_QUERY_MAX_TERMS]))
    params: dict[str, Any] = {
        "fields": "title,abstract,year,venue,citationCount,url,openAccessPdf",
        "limit": max_results,
    }

    # 外层遍历查询变体（原查询 → 截断查询），内层处理限流/网络重试
    last_error: Exception | None = None
    papers: list[dict[str, Any]] = []

    for query_index, search_query in enumerate(queries):
        query_params = {**params, "query": search_query}
        for attempt in range(_S2_MAX_ATTEMPTS):
            with _S2_LOCK:
                elapsed = time.monotonic() - _s2_last_call
                if elapsed < interval:
                    time.sleep(interval - elapsed)
                try:
                    response = requests.get(
                        _S2_SEARCH_URL, headers=headers, params=query_params, timeout=30
                    )
                    _s2_last_call = time.monotonic()
                except Exception as exc:  # pragma: no cover - 网络异常
                    last_error = exc
                    if attempt + 1 < _S2_MAX_ATTEMPTS:
                        wait = _s2_backoff_wait(attempt, None)
                        logger.warning(
                            "Semantic Scholar 请求失败(第%d次): %s，%.1fs 后重试",
                            attempt + 1, exc, wait,
                        )
                        time.sleep(wait)
                        continue
                    raise RuntimeError(f"Semantic Scholar 搜索失败: {exc}") from exc

            status = response.status_code
            if status in _S2_RETRYABLE_STATUSES:
                last_error = RuntimeError(f"Semantic Scholar HTTP {status}")
                if attempt + 1 < _S2_MAX_ATTEMPTS:
                    wait = _s2_backoff_wait(
                        attempt, _parse_retry_after(response.headers.get("Retry-After"))
                    )
                    logger.warning(
                        "Semantic Scholar HTTP %d(第%d次尝试)，%.1fs 后重试",
                        status, attempt + 1, wait,
                    )
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"Semantic Scholar 搜索失败: HTTP {status}（重试耗尽）"
                ) from last_error

            if status != 200:
                # 403（Key 无效）/400（参数错误）等重试无意义，立即失败
                raise RuntimeError(
                    f"Semantic Scholar 搜索失败: HTTP {status}: {response.text[:200]}"
                )

            # 注意：S2 bulk 端点实测无视 limit 参数（请求 limit=5 返回全部匹配），
            # 必须在客户端截断，防止证据/引用池爆炸。
            papers = list(response.json().get("data") or [])[:max_results]
            break

        if papers:
            break
        if query_index + 1 < len(queries):
            logger.info(
                "Semantic Scholar 查询「%s」无结果，改用截断查询重试", search_query[:60]
            )

    results: list[dict[str, Any]] = []
    notices: list[str] = []

    if not papers:
        notices.append("Semantic Scholar 未返回任何匹配论文")

    for paper in papers:
        title = str(paper.get("title") or "").strip()
        paper_url = str(paper.get("url") or "").strip()
        if not title or not paper_url:
            notices.append(f"忽略不完整的 Semantic Scholar 结果: {paper.get('paperId')}")
            continue

        meta_bits: list[str] = []
        if paper.get("year"):
            meta_bits.append(f"年份: {paper['year']}")
        if paper.get("venue"):
            meta_bits.append(f"发表: {paper['venue']}")
        if paper.get("citationCount") is not None:
            meta_bits.append(f"引用数: {paper['citationCount']}")

        abstract = str(paper.get("abstract") or "").strip()
        content = abstract or "（无摘要）"
        if meta_bits:
            content = f"[元信息] {'；'.join(meta_bits)}\n\n{content}"

        results.append(
            {
                "title": title,
                "url": paper_url,
                "content": content,
                "raw_content": content,
            }
        )

    return {
        "results": results,
        "backend": "semanticscholar",
        "answer": None,
        "notices": notices,
    }


def _reconstruct_openalex_abstract(
    inverted: dict[str, list[int]] | None,
) -> str:
    """OpenAlex 摘要以倒排索引给出，按词位置还原为文本."""

    if not inverted:
        return ""

    positions: list[tuple[int, str]] = []
    for word, indexes in inverted.items():
        for position in indexes:
            positions.append((position, word))
    positions.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positions)


def _search_via_openalex(query: str, max_results: int) -> dict[str, Any]:
    """Search OpenAlex for academic papers (Semantic Scholar 的兜底后端)."""

    params: dict[str, Any] = {
        "search": query[:200],
        "per-page": max_results,
        "select": _OPENALEX_SELECT,
    }

    try:
        response = requests.get(_OPENALEX_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # pragma: no cover - 网络异常
        raise RuntimeError(f"OpenAlex 搜索失败: {exc}") from exc

    results: list[dict[str, Any]] = []
    notices: list[str] = []

    for work in list(payload.get("results") or [])[:max_results]:
        title = str(work.get("title") or "").strip()
        work_id = str(work.get("id") or "")
        url = work.get("doi") or (
            f"https://openalex.org/{work_id.split('/')[-1]}" if work_id else ""
        )
        if not title or not url:
            continue

        meta_bits: list[str] = []
        if work.get("publication_year"):
            meta_bits.append(f"年份: {work['publication_year']}")
        if work.get("cited_by_count") is not None:
            meta_bits.append(f"引用数: {work['cited_by_count']}")

        abstract = _reconstruct_openalex_abstract(work.get("abstract_inverted_index"))
        content = abstract or "（无摘要）"
        if meta_bits:
            content = f"[元信息] {'；'.join(meta_bits)}\n\n{content}"

        results.append(
            {
                "title": title,
                "url": url,
                "content": content,
                "raw_content": content,
            }
        )

    if not results:
        notices.append("OpenAlex 未返回任何匹配论文")

    return {
        "results": results,
        "backend": "openalex",
        "answer": None,
        "notices": notices,
    }


def dispatch_search(
    query: str,
    config: Configuration,
    loop_count: int,
) -> Tuple[dict[str, Any] | None, list[str], Optional[str], str]:
    """Execute configured search backend and normalise response payload."""

    search_api = get_config_value(config.search_api)

    try:
        if search_api.lower() == "semanticscholar":
            try:
                raw_response = _search_via_semanticscholar(
                    query=query,
                    max_results=5,
                )
            except RuntimeError as exc:
                logger.warning("Semantic Scholar 不可用，回退 OpenAlex: %s", exc)
                raw_response = _search_via_openalex(query=query, max_results=5)
                raw_response.setdefault("notices", []).append(
                    f"Semantic Scholar 不可用（{exc}），结果来自 OpenAlex"
                )

            if not raw_response.get("results"):
                logger.info("Semantic Scholar 无结果，尝试 OpenAlex 补充检索")
                try:
                    supplement = _search_via_openalex(query=query, max_results=5)
                except RuntimeError as exc:
                    logger.warning("OpenAlex 补充检索也失败: %s", exc)
                else:
                    if supplement.get("results"):
                        raw_response["results"] = supplement["results"]
                        raw_response["backend"] = "openalex"
                        raw_response.setdefault("notices", []).append(
                            "Semantic Scholar 无结果，结果来自 OpenAlex"
                        )
        elif search_api.lower() in _BING_FALLBACK_BACKENDS:
            raw_response = _search_via_bing(
                query=query,
                fetch_full_page=config.fetch_full_page,
                max_results=5,
                max_tokens=MAX_TOKENS_PER_SOURCE,
            )
        else:
            raw_response = _GLOBAL_SEARCH_TOOL.run(
                {
                    "input": query,
                    "backend": search_api,
                    "mode": "structured",
                    "fetch_full_page": config.fetch_full_page,
                    "max_results": 5,
                    "max_tokens_per_source": MAX_TOKENS_PER_SOURCE,
                    "loop_count": loop_count,
                }
            )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Search backend %s failed: %s", search_api, exc)
        raise

    if isinstance(raw_response, str):
        notices = [raw_response]
        logger.warning("Search backend %s returned text notice: %s", search_api, raw_response)
        payload: dict[str, Any] = {
            "results": [],
            "backend": search_api,
            "answer": None,
            "notices": notices,
        }
    else:
        payload = raw_response
        notices = list(payload.get("notices") or [])

    backend_label = str(payload.get("backend") or search_api)
    answer_text = payload.get("answer")
    results = payload.get("results", [])

    if notices:
        for notice in notices:
            logger.info("Search notice (%s): %s", backend_label, notice)

    logger.info(
        "Search backend=%s resolved_backend=%s answer=%s results=%s",
        search_api,
        backend_label,
        bool(answer_text),
        len(results),
    )

    return payload, notices, answer_text, backend_label


def prepare_research_context(
    search_result: dict[str, Any] | None,
    answer_text: Optional[str],
    config: Configuration,
) -> tuple[str, str]:
    """Build structured context and source summary for downstream agents."""

    sources_summary = format_sources(search_result)
    context = deduplicate_and_format_sources(
        search_result or {"results": []},
        max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
        fetch_full_page=config.fetch_full_page,
    )

    if answer_text:
        context = f"AI直接答案：\n{answer_text}\n\n{context}"

    return sources_summary, context
