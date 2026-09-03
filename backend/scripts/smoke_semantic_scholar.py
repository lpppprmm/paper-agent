"""Semantic Scholar 可用性自检脚本（参照 paper-agent 使用文档 §5）。

检查 4 项：
  1. /graph/v1/paper/search（带 Key）——项目主用端点；
  2. /graph/v1/paper/search（匿名对照）——验证"为什么必须配 Key"；
  3. /recommendations/v1（带 Key）——独立限流池，饱和时可兜底；
  4. 项目客户端端到端（节流 + 自动重试 + 查询变体）——产品路径真实可用性。

退出码：0 = 可用；1 = Key 无效等硬错误；2 = 当前被限流/5xx（外部环境）。

用法（在 backend 目录下）：
    ../.venv/bin/python scripts/smoke_semantic_scholar.py [--attempts 3] [--interval 1.2]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BACKEND_DIR / ".env")

GRAPH_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
GRAPH_SEARCH_PARAMS = {"query": "generative AI", "limit": 2, "fields": "title,year"}
RECOMMENDATIONS_URL = (
    "https://api.semanticscholar.org/recommendations/v1/papers/forpaper/arXiv:2211.17192"
)

API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()


def mask_key(key: str) -> str:
    """只打印 Key 前后 4 位，不输出明文."""

    if not key:
        return "（未配置，匿名模式）"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}…{key[-4:]}"


def probe(
    url: str,
    *,
    with_key: bool,
    attempts: int,
    interval: float,
    params: dict[str, str] | None = None,
) -> list[int]:
    """探测端点 attempts 次，返回 HTTP 状态码列表."""

    headers = {"x-api-key": API_KEY} if with_key else {}
    statuses: list[int] = []
    for _ in range(attempts):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            statuses.append(response.status_code)
        except requests.RequestException as exc:
            print(f"  请求异常: {exc}")
            statuses.append(0)
        time.sleep(interval)
    return statuses


def summarize(label: str, statuses: list[int]) -> None:
    """按状态码聚合输出."""

    counts: dict[int, int] = {}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    detail = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    print(f"[{label}] {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic Scholar 可用性自检")
    parser.add_argument("--attempts", type=int, default=3, help="每类探测次数（默认 3）")
    parser.add_argument(
        "--interval", type=float, default=None, help="探测间隔秒（默认：带 Key 1.2 / 匿名 3）"
    )
    args = parser.parse_args()

    interval = args.interval or (1.2 if API_KEY else 3.0)

    print("=== Semantic Scholar 可用性自检 ===")
    print(f"时间      = {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"api_key   = {mask_key(API_KEY)}")
    print(f"attempts  = {args.attempts}")
    print()

    keyed = probe(
        GRAPH_SEARCH_URL,
        with_key=True,
        attempts=args.attempts,
        interval=interval,
        params=GRAPH_SEARCH_PARAMS,
    )
    summarize("keyed search ", keyed)

    anon = probe(
        GRAPH_SEARCH_URL,
        with_key=False,
        attempts=args.attempts,
        interval=interval,
        params=GRAPH_SEARCH_PARAMS,
    )
    summarize("anon search  ", anon)

    reco = probe(RECOMMENDATIONS_URL, with_key=True, attempts=args.attempts, interval=interval)
    summarize("recommendations", reco)

    # 项目客户端端到端：节流 + 自动重试 + 查询变体
    print("[client e2e   ] 调用 services.search._search_via_semanticscholar ...")
    from services.search import _search_via_semanticscholar

    started = time.monotonic()
    try:
        payload = _search_via_semanticscholar(query="speculative decoding LLM", max_results=3)
        results = payload.get("results") or []
        print(f"[client e2e   ] OK，{len(results)} 条，{time.monotonic() - started:.1f}s")
        client_ok = bool(results)
    except RuntimeError as exc:
        print(f"[client e2e   ] FAIL: {exc}")
        client_ok = False

    print()
    hard_errors = [status for status in keyed + anon + reco if status in (400, 403)]
    if hard_errors:
        print("结论：存在硬错误（403=Key 无效 / 400=参数错误），检查 .env 配置。")
        return 1

    if client_ok:
        print("结论：当前可正常使用（客户端节流与重试兜得住间歇限流）。")
        return 0

    print("结论：当前被限流/5xx（外部环境），稍后或换时段重试；可依赖 OpenAlex 兜底。")
    return 2


if __name__ == "__main__":
    sys.exit(main())
