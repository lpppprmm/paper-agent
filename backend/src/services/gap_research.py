"""GapResearch: plan targeted search queries for evidence-deficient arguments."""

from __future__ import annotations

import logging
from typing import Any, Optional

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from models import ResearchArgument, SummaryState
from prompts import gap_query_planner_instructions
from services.json_parsing import parse_structured_json

logger = logging.getLogger(__name__)

FALLBACK_QUERY_CHARS = 80


class GapResearchService:
    """为证据不足的论点规划补检索查询（LLM 批量生成，失败降级为论点原文）。"""

    def __init__(self, agent: ToolAwareSimpleAgent, config: Configuration) -> None:
        self._agent = agent
        self._config = config

    def plan_queries(
        self,
        state: SummaryState,
        deficient: list[ResearchArgument],
    ) -> dict[int, str]:
        """Return {argument_id: query} for every deficient argument."""

        prompt = gap_query_planner_instructions.format(
            research_topic=state.research_topic or "",
            deficient_block=self._build_deficient_block(deficient),
        )

        try:
            response = self._agent.run(prompt)
        except Exception:
            logger.exception("GapQueryPlanner LLM call failed")
            response = ""
        finally:
            self._agent.clear_history()

        planned = self._parse_response(response or "", deficient)
        if planned is None:
            logger.warning("GapQueryPlanner output unusable; using fallback queries")
            return self._fallback_queries(deficient)

        return planned

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------
    @staticmethod
    def _build_deficient_block(deficient: list[ResearchArgument]) -> str:
        return "\n".join(
            f"- 论点[{arg.id}]：{arg.claim}（现有证据编号：{arg.evidence_refs or '无'}）"
            for arg in deficient
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _parse_response(
        self,
        response: str,
        deficient: list[ResearchArgument],
    ) -> Optional[dict[int, str]]:
        payload = parse_structured_json(response, self._config)
        if not payload:
            return None

        raw_queries = payload.get("queries")
        if not isinstance(raw_queries, list):
            return None

        queries: dict[int, str] = {}
        for item in raw_queries:
            if not isinstance(item, dict):
                continue
            try:
                argument_id = int(item.get("argument_id"))
            except (TypeError, ValueError):
                continue
            query = str(item.get("query") or "").strip()
            if query and any(arg.id == argument_id for arg in deficient):
                queries[argument_id] = query[:FALLBACK_QUERY_CHARS]

        return queries or None

    @staticmethod
    def _fallback_queries(deficient: list[ResearchArgument]) -> dict[int, str]:
        """确定性兜底：直接用论点原文（截断）作为查询."""

        return {
            arg.id: arg.claim[:FALLBACK_QUERY_CHARS]
            for arg in deficient
        }
