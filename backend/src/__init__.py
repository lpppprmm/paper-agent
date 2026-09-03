"""Paper Agent Deep Research - A deep research assistant."""

__version__ = "0.0.1"

from .agent import DeepResearchAgent
from .config import Configuration, SearchAPI
from .models import SummaryState, SummaryStateInput, SummaryStateOutput, TodoItem

__all__ = [
    "DeepResearchAgent",
    "Configuration",
    "SearchAPI",
    "SummaryState",
    "SummaryStateInput",
    "SummaryStateOutput",
    "TodoItem",
]

