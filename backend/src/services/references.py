"""Thread-safe citation registry mapping gathered sources to stable IDs."""

from __future__ import annotations

from threading import Lock
from typing import Any, Optional
from urllib.parse import urlsplit

from models import SourceRef

CONTENT_EXCERPT_CHARS = 800


class SourceRegistry:
    """给每条搜索来源分配稳定编号（1 基、全局递增），供引用与参考文献使用。

    注册发生在任务 worker 线程中（run_stream 每个任务一个线程），
    因此所有读写都必须在锁内进行。
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._sources: list[SourceRef] = []
        self._url_to_id: dict[str, int] = {}

    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalise a URL for dedup: lowercase host, drop fragment/trailing slash."""

        parts = urlsplit(url.strip())
        netloc = parts.netloc.lower()
        path = parts.path
        if path.endswith("/"):
            path = path[:-1]
        return f"{parts.scheme.lower()}://{netloc}{path}{parts.query}"

    def register(
        self,
        *,
        title: str,
        url: str,
        content: str,
        task_id: int,
    ) -> SourceRef:
        """Register a source; return the (new or existing) ref. First occurrence wins."""

        key = self.normalize_url(url)

        with self._lock:
            existing_id = self._url_to_id.get(key)
            if existing_id is not None:
                return self._sources[existing_id - 1]

            ref = SourceRef(
                id=len(self._sources) + 1,
                title=title or url,
                url=url,
                content_excerpt=(content or "")[:CONTENT_EXCERPT_CHARS],
                task_id=task_id,
            )
            self._sources.append(ref)
            self._url_to_id[key] = ref.id
            return ref

    def register_results(self, results: list[dict[str, Any]], task_id: int) -> list[SourceRef]:
        """Register every result in order; skip entries without a URL.

        task_id=0 表示来源来自论文链路的补充检索（More Research），
        不属于任何研究任务。
        """

        refs: list[SourceRef] = []
        for result in results:
            url = result.get("url")
            if not url:
                continue
            refs.append(
                self.register(
                    title=result.get("title") or url,
                    url=url,
                    content=result.get("content") or "",
                    task_id=task_id,
                )
            )
        return refs

    def get(self, source_id: int) -> Optional[SourceRef]:
        """Fetch a ref by id (1-based)."""

        with self._lock:
            if 1 <= source_id <= len(self._sources):
                return self._sources[source_id - 1]
            return None

    def all_refs(self) -> list[SourceRef]:
        """Copy of all registered refs in id order."""

        with self._lock:
            return list(self._sources)

    def by_task(self, task_id: int) -> list[SourceRef]:
        """Refs first registered by a given task."""

        with self._lock:
            return [ref for ref in self._sources if ref.task_id == task_id]

    def build_citation_header(self, refs: list[SourceRef]) -> str:
        """引用池头部，注入 summarizer 上下文，要求引用必须使用 [编号] 格式."""

        lines = ["引用池（引用来源时必须使用 [编号] 格式）："]
        lines.extend(f"[{ref.id}] {ref.title} — {ref.url}" for ref in refs)
        return "\n".join(lines)

    def build_bibliography(self, cited_ids: Optional[set[int]] = None) -> str:
        """生成「## 参考文献」段落（Python 确定性生成，编号与 registry 一致）。"""

        with self._lock:
            refs = [ref for ref in self._sources if cited_ids is None or ref.id in cited_ids]

        if not refs:
            return "## 参考文献\n\n暂无引用来源。"

        lines = ["## 参考文献", ""]
        for ref in refs:
            lines.append(f"[{ref.id}] {ref.title}. {ref.url}")
        return "\n".join(lines)

    def snapshot(self) -> list[dict[str, Any]]:
        """SSE 事件用的结构化快照."""

        with self._lock:
            return [
                {"id": ref.id, "title": ref.title, "url": ref.url, "task_id": ref.task_id}
                for ref in self._sources
            ]
