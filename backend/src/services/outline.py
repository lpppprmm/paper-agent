"""OutlineBuilder: derive paper sections bound to arguments."""

from __future__ import annotations

import logging
from typing import Any, Optional

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from models import PaperOutline, PaperSection, SummaryState
from prompts import outline_builder_instructions
from services.json_parsing import parse_structured_json

logger = logging.getLogger(__name__)


class OutlineBuilderService:
    """Wraps the outline agent to produce sections bound to research arguments."""

    def __init__(self, agent: ToolAwareSimpleAgent, config: Configuration) -> None:
        self._agent = agent
        self._config = config

    def build(self, state: SummaryState) -> PaperOutline:
        """Build the outline: fixed 引言 + LLM body sections + fixed 结论."""

        prompt = outline_builder_instructions.format(
            research_topic=state.research_topic or "",
            arguments_block=self._build_arguments_block(state),
        )

        try:
            response = self._agent.run(prompt)
        except Exception:
            logger.exception("OutlineBuilder LLM call failed")
            return self._build_fallback(state)

        finally:
            self._agent.clear_history()

        outline = self._parse_response(response or "", state)
        if outline is None:
            logger.warning("OutlineBuilder output unusable; falling back")
            return self._build_fallback(state)

        return outline

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------
    @staticmethod
    def _build_arguments_block(state: SummaryState) -> str:
        if not state.paper_arguments:
            return "（无论点素材）"
        return "\n".join(
            f"[{arg.id}] {arg.claim}（证据编号：{arg.evidence_refs}）"
            for arg in state.paper_arguments
        )

    # ------------------------------------------------------------------
    # Parsing and validation
    # ------------------------------------------------------------------
    def _parse_response(
        self,
        response: str,
        state: SummaryState,
    ) -> Optional[PaperOutline]:
        payload = parse_structured_json(response, self._config)
        if not payload:
            return None

        title = str(payload.get("title") or "").strip()
        raw_sections = payload.get("sections")
        if not isinstance(raw_sections, list):
            return None

        argument_ids = {arg.id for arg in state.paper_arguments}
        body_sections: list[PaperSection] = []

        for item in raw_sections:
            if not isinstance(item, dict):
                continue
            section_title = str(item.get("title") or "").strip()
            if not section_title:
                continue

            goal = str(item.get("goal") or f"论证：{section_title}").strip()
            bound_ids = [
                arg_id
                for arg_id in self._to_int_list(item.get("argument_ids"))
                if arg_id in argument_ids
            ]

            body_sections.append(
                PaperSection(
                    id=0,  # renumbered below (2..n)
                    title=section_title,
                    goal=goal,
                    argument_ids=bound_ids,
                )
            )

        if not body_sections:
            return None

        return self._finalize_outline(title, body_sections, state)

    def _finalize_outline(
        self,
        title: str,
        body_sections: list[PaperSection],
        state: SummaryState,
    ) -> PaperOutline:
        """补全引言/结论、重编号，并保证每个论点至少出现在一个章节。"""

        sections = [
            PaperSection(
                id=1,
                title="引言",
                goal="阐述研究背景、提出研究问题、说明研究缺口与本论文的论证思路",
            )
        ]

        for index, section in enumerate(body_sections, start=2):
            section.id = index
            sections.append(section)

        # 未被任何章节覆盖的论点，追加「综合讨论」节承接
        covered = {
            arg_id
            for section in sections
            for arg_id in section.argument_ids
        }
        missing = [arg.id for arg in state.paper_arguments if arg.id not in covered]
        if missing:
            sections.append(
                PaperSection(
                    id=len(sections) + 1,
                    title="综合讨论",
                    goal=f"补充论证未单独成节的论点（编号 {missing}）",
                    argument_ids=missing,
                )
            )

        sections.append(
            PaperSection(
                id=len(sections) + 1,
                title="结论",
                goal="总结全文论点、回应研究问题、指出研究局限与未来方向",
            )
        )

        return PaperOutline(title=title or f"{state.research_topic or '研究主题'}研究", sections=sections)

    @staticmethod
    def _to_int_list(value: Any) -> list[int]:
        if not isinstance(value, list):
            return []
        result: list[int] = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _build_fallback(state: SummaryState) -> PaperOutline:
        """确定性 fallback：引言 + 每论点一节 + 结论。"""

        sections = [
            PaperSection(
                id=1,
                title="引言",
                goal="阐述研究背景、提出研究问题、说明研究缺口与本论文的论证思路",
            )
        ]

        for arg in state.paper_arguments:
            sections.append(
                PaperSection(
                    id=len(sections) + 1,
                    title=(arg.claim[:30] + "…") if len(arg.claim) > 30 else arg.claim,
                    goal=f"论证：{arg.claim[:100]}",
                    argument_ids=[arg.id],
                )
            )

        sections.append(
            PaperSection(
                id=len(sections) + 1,
                title="结论",
                goal="总结全文论点、回应研究问题、指出研究局限与未来方向",
            )
        )

        return PaperOutline(
            title=f"{state.research_topic or '研究主题'}研究",
            sections=sections,
        )
