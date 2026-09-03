"""PaperWriter: per-section generation, abstract, and deterministic assembly."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator
from typing import Optional, Tuple

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from models import (
    AbstractResult,
    PaperOutline,
    PaperSection,
    PaperSectionContent,
    SummaryState,
)
from prompts import abstract_generator_instructions, paper_writer_section_instructions
from services.json_parsing import parse_structured_json
from services.references import SourceRegistry
from services.text_processing import strip_tool_calls
from utils import strip_thinking_tokens

logger = logging.getLogger(__name__)

PREVIOUS_SECTION_CHARS = 800
ROLLING_SUMMARY_CHARS = 3000
ABSTRACT_BODY_CHARS = 6000
ABSTRACT_FALLBACK_CHARS = 200
MAX_EVIDENCE_ENTRIES = 8  # 每节证据块条数上限，防止补检索后引用池过大撑爆 prompt


class PaperWriterService:
    """Generates paper sections one by one, then assembles the final document."""

    def __init__(
        self,
        agent_factory: Callable[[], ToolAwareSimpleAgent],
        abstract_agent: ToolAwareSimpleAgent,
        config: Configuration,
    ) -> None:
        self._agent_factory = agent_factory
        self._abstract_agent = abstract_agent
        self._config = config

    # ------------------------------------------------------------------
    # Section writing
    # ------------------------------------------------------------------
    def stream_section(
        self,
        state: SummaryState,
        registry: SourceRegistry,
        section: PaperSection,
        previous: list[PaperSectionContent],
    ) -> Tuple[Iterator[str], Callable[[], str]]:
        """Stream one section's text while collecting the full output."""

        prompt = self._build_section_prompt(state, registry, section, previous)
        remove_thinking = self._config.strip_thinking_tokens
        raw_buffer = ""
        visible_output = ""
        emit_index = 0
        agent = self._agent_factory()

        def flush_visible() -> Iterator[str]:
            nonlocal emit_index, raw_buffer
            while True:
                start = raw_buffer.find("<think>", emit_index)
                if start == -1:
                    if emit_index < len(raw_buffer):
                        segment = raw_buffer[emit_index:]
                        emit_index = len(raw_buffer)
                        if segment:
                            yield segment
                    break

                if start > emit_index:
                    segment = raw_buffer[emit_index:start]
                    emit_index = start
                    if segment:
                        yield segment

                end = raw_buffer.find("</think>", start)
                if end == -1:
                    break
                emit_index = end + len("</think>")

        def generator() -> Iterator[str]:
            nonlocal raw_buffer, visible_output, emit_index
            try:
                for chunk in agent.stream_run(prompt):
                    raw_buffer += chunk
                    if remove_thinking:
                        for segment in flush_visible():
                            visible_output += segment
                            if segment:
                                yield segment
                    else:
                        visible_output += chunk
                        if chunk:
                            yield chunk
            finally:
                if remove_thinking:
                    for segment in flush_visible():
                        visible_output += segment
                        if segment:
                            yield segment
                agent.clear_history()

        def get_section_text() -> str:
            if remove_thinking:
                cleaned = strip_thinking_tokens(visible_output)
            else:
                cleaned = visible_output

            return strip_tool_calls(cleaned).strip()

        return generator(), get_section_text

    def write_section(
        self,
        state: SummaryState,
        registry: SourceRegistry,
        section: PaperSection,
        previous: list[PaperSectionContent],
    ) -> str:
        """Non-streaming variant for the sync API path."""

        prompt = self._build_section_prompt(state, registry, section, previous)

        agent = self._agent_factory()
        try:
            response = agent.run(prompt)
        finally:
            agent.clear_history()

        text = (response or "").strip()
        if self._config.strip_thinking_tokens:
            text = strip_thinking_tokens(text)

        return strip_tool_calls(text).strip()

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------
    def _build_section_prompt(
        self,
        state: SummaryState,
        registry: SourceRegistry,
        section: PaperSection,
        previous: list[PaperSectionContent],
    ) -> str:
        """Build the per-section prompt with goal, arguments, evidence and citations."""

        outline: Optional[PaperOutline] = state.paper_outline
        paper_title = (
            outline.title if outline and outline.title else (state.research_topic or "研究主题")
        )

        arguments_by_id = {arg.id: arg for arg in state.paper_arguments}
        bound_arguments = [
            arguments_by_id[arg_id]
            for arg_id in section.argument_ids
            if arg_id in arguments_by_id
        ]
        arguments_block = "\n".join(
            f"- 论点[{arg.id}]：{arg.claim}" for arg in bound_arguments
        ) or "（本章无显式绑定论点，仅作背景或过渡）"

        # 证据块：从 registry 拉取绑定论点的证据来源（去重、保持顺序）
        evidence_ids: list[int] = []
        for arg in bound_arguments:
            evidence_ids.extend(arg.evidence_refs)
        seen: set[int] = set()
        unique_evidence_ids = [
            sid for sid in evidence_ids if not (sid in seen or seen.add(sid))
        ][:MAX_EVIDENCE_ENTRIES]

        evidence_lines: list[str] = []
        for source_id in unique_evidence_ids:
            ref = registry.get(source_id)
            if ref is None:
                continue
            evidence_lines.append(
                f"[{ref.id}] {ref.title}\nURL: {ref.url}\n摘录: {ref.content_excerpt}"
            )
        evidence_block = (
            "\n\n".join(evidence_lines)
            or "（本章暂无绑定证据，请基于论点与前文论证展开）"
        )

        # 引用池：全部已注册来源，允许跨章节引用
        all_refs = registry.all_refs()
        citation_pool_block = (
            "\n".join(f"[{ref.id}] {ref.title} — {ref.url}" for ref in all_refs)
            if all_refs
            else "（引用池为空）"
        )

        # 前文滚动摘要
        if previous:
            rolling = "\n\n".join(item.content[:PREVIOUS_SECTION_CHARS] for item in previous)
            if len(rolling) > ROLLING_SUMMARY_CHARS:
                rolling = f"{rolling[:ROLLING_SUMMARY_CHARS]}... [truncated]"
        else:
            rolling = "（本节为论文首节）"

        return paper_writer_section_instructions.format(
            research_topic=state.research_topic or "",
            paper_title=paper_title,
            section_title=section.title,
            section_goal=section.goal,
            arguments_block=arguments_block,
            evidence_block=evidence_block,
            citation_pool_block=citation_pool_block,
            previous_sections_summary=rolling,
        )

    # ------------------------------------------------------------------
    # Abstract and keywords
    # ------------------------------------------------------------------
    def generate_abstract(
        self,
        state: SummaryState,
        key_findings: list[str],
    ) -> AbstractResult:
        """Generate 摘要/关键词 from key findings and the assembled body."""

        body = "\n\n".join(
            f"{item.title}\n{item.content}" for item in state.paper_sections
        )
        body_excerpt = body[:ABSTRACT_BODY_CHARS]
        key_findings_block = (
            "\n".join(f"- {finding}" for finding in key_findings)
            if key_findings
            else "（无）"
        )

        prompt = abstract_generator_instructions.format(
            research_topic=state.research_topic or "",
            key_findings=key_findings_block,
            body_excerpt=body_excerpt,
        )

        try:
            response = self._abstract_agent.run(prompt)
        except Exception:
            logger.exception("Abstract generation failed")
            response = ""
        finally:
            self._abstract_agent.clear_history()

        payload = parse_structured_json(response or "", self._config)
        if payload:
            abstract = str(payload.get("abstract") or "").strip()
            raw_keywords = payload.get("keywords")
            keywords = (
                [str(k).strip() for k in raw_keywords if str(k).strip()]
                if isinstance(raw_keywords, list)
                else []
            )
            if abstract:
                return AbstractResult(abstract=abstract, keywords=keywords)

        logger.warning("Abstract output unusable; using fallback")
        outline = state.paper_outline
        thesis = state.paper_arguments[0].claim if state.paper_arguments else ""
        fallback = (outline.title if outline and outline.title else "") or thesis or body_excerpt
        return AbstractResult(
            abstract=fallback[:ABSTRACT_FALLBACK_CHARS],
            keywords=[],
        )

    # ------------------------------------------------------------------
    # Assembly (pure Python)
    # ------------------------------------------------------------------
    def assemble_paper(
        self,
        state: SummaryState,
        registry: SourceRegistry,
        sections_content: list[PaperSectionContent],
        abstract_result: AbstractResult,
    ) -> str:
        """Assemble title + abstract + sections + Python-built bibliography."""

        outline: Optional[PaperOutline] = state.paper_outline
        title = outline.title if outline and outline.title else (state.research_topic or "研究主题")

        parts: list[str] = [f"# {title}", "", "## 摘要", "", abstract_result.abstract]

        if abstract_result.keywords:
            parts.extend(["", "## 关键词", "", "；".join(abstract_result.keywords)])

        parts.append("")
        for item in sections_content:
            parts.extend(["", f"## {item.section_id}. {item.title}", "", item.content])

        body_text = "\n".join(parts)
        cited_ids = {int(match) for match in re.findall(r"\[(\d+)\]", body_text)}
        bibliography = registry.build_bibliography(cited_ids or None)

        return f"{body_text}\n\n{bibliography}\n"
