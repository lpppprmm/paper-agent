"""State models used by the deep research workflow."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import List, Optional

from typing_extensions import Annotated


@dataclass(kw_only=True)
class TodoItem:
    """单个待办任务项。"""

    id: int
    title: str
    intent: str
    query: str
    status: str = field(default="pending")
    summary: Optional[str] = field(default=None)
    sources_summary: Optional[str] = field(default=None)
    notices: list[str] = field(default_factory=list)
    note_id: Optional[str] = field(default=None)
    note_path: Optional[str] = field(default=None)
    stream_token: Optional[str] = field(default=None)
    source_ref_ids: list[int] = field(default_factory=list)  # SourceRegistry ids


@dataclass(kw_only=True)
class SourceRef:
    """一条已注册到 SourceRegistry 的引用来源。"""

    id: int
    title: str
    url: str
    content_excerpt: str  # 注册时截断的正文摘要
    task_id: int


@dataclass(kw_only=True)
class ResearchArgument:
    """论文中的一个分论点及其证据绑定。"""

    id: int
    claim: str
    evidence_refs: list[int] = field(default_factory=list)  # SourceRegistry ids
    related_task_ids: list[int] = field(default_factory=list)


@dataclass(kw_only=True)
class PaperSection:
    """论文大纲中的一个章节（1=引言，末位=结论）。"""

    id: int
    title: str
    goal: str
    argument_ids: list[int] = field(default_factory=list)


@dataclass(kw_only=True)
class PaperOutline:
    """论文大纲：标题 + 有序章节。"""

    title: str = ""
    sections: list[PaperSection] = field(default_factory=list)


@dataclass(kw_only=True)
class PaperSectionContent:
    """单个章节的成文内容。"""

    section_id: int
    title: str
    content: str


@dataclass(kw_only=True)
class ArgumentBundle:
    """ArgumentBuilder 的输出（含 fallback 标志）。"""

    research_question: str
    thesis: str
    gap: str
    arguments: list[ResearchArgument] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    fallback: bool = False  # True 表示由确定性 fallback 生成而非 LLM


@dataclass(kw_only=True)
class AbstractResult:
    """摘要与关键词生成结果。"""

    abstract: str
    keywords: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class SummaryState:
    research_topic: str = field(default=None)  # Report topic
    search_query: str = field(default=None)  # Deprecated placeholder
    web_research_results: Annotated[list, operator.add] = field(default_factory=list)
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    research_loop_count: int = field(default=0)  # Research loop count
    running_summary: str = field(default=None)  # Legacy summary field
    todo_items: Annotated[list, operator.add] = field(default_factory=list)
    structured_report: Optional[str] = field(default=None)
    report_note_id: Optional[str] = field(default=None)
    report_note_path: Optional[str] = field(default=None)
    paper_arguments: list[ResearchArgument] = field(default_factory=list)
    paper_outline: Optional[PaperOutline] = field(default=None)
    paper_sections: list[PaperSectionContent] = field(default_factory=list)
    paper_markdown: Optional[str] = field(default=None)
    paper_note_id: Optional[str] = field(default=None)
    paper_note_path: Optional[str] = field(default=None)


@dataclass(kw_only=True)
class SummaryStateInput:
    research_topic: str = field(default=None)  # Report topic


@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = field(default=None)  # Backward-compatible文本
    report_markdown: Optional[str] = field(default=None)
    paper_markdown: Optional[str] = field(default=None)
    todo_items: List[TodoItem] = field(default_factory=list)

