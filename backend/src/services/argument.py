"""ArgumentBuilder: distill research summaries into an argumentative skeleton."""

from __future__ import annotations

import logging
from typing import Any, Optional

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from models import ArgumentBundle, ResearchArgument, SummaryState, TodoItem
from prompts import argument_builder_instructions
from services.json_parsing import parse_structured_json
from services.references import SourceRegistry

logger = logging.getLogger(__name__)

TASK_SUMMARY_CHARS = 1200


class ArgumentBuilderService:
    """Wraps the argument-building agent to produce a structured ArgumentBundle."""

    def __init__(self, agent: ToolAwareSimpleAgent, config: Configuration) -> None:
        self._agent = agent
        self._config = config

    def build(self, state: SummaryState, registry: SourceRegistry) -> ArgumentBundle:
        """Extract research question / thesis / gap / arguments from task summaries."""

        prompt = argument_builder_instructions.format(
            research_topic=state.research_topic or "",
            tasks_block=self._build_tasks_block(state.todo_items),
            reference_pool_block=self._build_reference_pool_block(registry),
        )

        try:
            response = self._agent.run(prompt)
        except Exception:
            logger.exception("ArgumentBuilder LLM call failed")
            return self._build_fallback(state)

        finally:
            self._agent.clear_history()

        bundle = self._parse_response(response or "", state, registry)
        if bundle is None:
            logger.warning("ArgumentBuilder output unusable; falling back")
            return self._build_fallback(state)

        return bundle

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------
    @staticmethod
    def _build_tasks_block(tasks: list[TodoItem]) -> str:
        if not tasks:
            return "（无任务素材）"

        blocks: list[str] = []
        for task in tasks:
            summary = (task.summary or "").strip()
            if len(summary) > TASK_SUMMARY_CHARS:
                summary = f"{summary[:TASK_SUMMARY_CHARS]}... [truncated]"
            summary = summary or "（该任务无有效总结）"

            blocks.append(
                f"任务 {task.id}: {task.title}\n"
                f"- 目标：{task.intent}\n"
                f"- 检索查询：{task.query}\n"
                f"- 总结：{summary}\n"
                f"- 来源编号：{task.source_ref_ids}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _build_reference_pool_block(registry: SourceRegistry) -> str:
        refs = registry.all_refs()
        if not refs:
            return "（引用池为空）"
        return "\n".join(f"[{ref.id}] {ref.title} — {ref.url}" for ref in refs)

    # ------------------------------------------------------------------
    # Parsing and validation
    # ------------------------------------------------------------------
    def _parse_response(
        self,
        response: str,
        state: SummaryState,
        registry: SourceRegistry,
    ) -> Optional[ArgumentBundle]:
        payload = parse_structured_json(response, self._config)
        if not payload:
            return None

        valid_source_ids = {ref.id for ref in registry.all_refs()}
        valid_task_ids = {task.id for task in state.todo_items}

        raw_arguments = payload.get("arguments")
        if not isinstance(raw_arguments, list):
            return None

        arguments: list[ResearchArgument] = []
        for idx, item in enumerate(raw_arguments, start=1):
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim") or "").strip()
            if not claim:
                continue

            evidence_refs = [
                source_id
                for source_id in self._to_int_list(item.get("evidence_source_ids"))
                if source_id in valid_source_ids
            ]
            related_task_ids = [
                task_id
                for task_id in self._to_int_list(item.get("related_task_ids"))
                if task_id in valid_task_ids
            ]

            arguments.append(
                ResearchArgument(
                    id=idx,
                    claim=claim,
                    evidence_refs=evidence_refs,
                    related_task_ids=related_task_ids,
                )
            )

        if not arguments:
            return None

        key_findings = [
            str(finding).strip()
            for finding in payload.get("key_findings", [])
            if isinstance(finding, (str, int, float)) and str(finding).strip()
        ]

        return ArgumentBundle(
            research_question=str(payload.get("research_question") or state.research_topic or "").strip(),
            thesis=str(payload.get("thesis") or "").strip(),
            gap=str(payload.get("gap") or "").strip(),
            arguments=arguments,
            key_findings=key_findings,
            fallback=False,
        )

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
    def _build_fallback(state: SummaryState) -> ArgumentBundle:
        """确定性 fallback：每个已完成任务映射为一个论点，管线不中断。"""

        topic = state.research_topic or "研究主题"
        arguments: list[ResearchArgument] = []
        for task in state.todo_items:
            summary = (task.summary or "").strip() or task.intent
            arguments.append(
                ResearchArgument(
                    id=len(arguments) + 1,
                    claim=summary[:200],
                    evidence_refs=list(task.source_ref_ids),
                    related_task_ids=[task.id],
                )
            )

        return ArgumentBundle(
            research_question=topic,
            thesis="",
            gap=f"现有研究尚未对「{topic}」形成系统化梳理",
            arguments=arguments,
            key_findings=[],
            fallback=True,
        )
