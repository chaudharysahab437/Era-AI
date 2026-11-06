"""Concise conversation and project summaries.

Extractive by default (no model needed): keeps the most informative user
sentences. An optional ``llm_hook(text) -> str`` may produce a better
summary; if it raises or returns nothing, the extractive result is used.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional

from .memory_models import Category, Memory
from .privacy import classify_sensitive, find_secrets
from .text_utils import split_sentences, tokens, truncate

logger = logging.getLogger(__name__)

_SIGNAL = re.compile(r"\b(decid\w*|need|goal|plan|blocker|stuck|next|deadline|prefer|using|switch\w*|finish\w*|built|working on)\b", re.I)


class MemorySummarizer:
    def __init__(self, llm_hook: Optional[Callable[[str], str]] = None, max_chars: int = 600) -> None:
        self.llm_hook = llm_hook
        self.max_chars = max_chars

    def summarize_conversation(self, turns: Iterable[Dict[str, str]], existing: Optional[str] = None) -> str:
        user_text = [t["content"] for t in turns if t.get("role") == "user"]
        pool: List[str] = []
        if existing:
            pool += [s for s in split_sentences(existing.replace("Discussed: ", "")) if s]
        for t in user_text:
            pool += split_sentences(t)
        pool = [s for s in pool if len(s.split()) >= 4 and not s.endswith("?") and not find_secrets(s) and not classify_sensitive(s)]
        if not pool:
            return existing or ""

        if self.llm_hook is not None:
            try:
                out = (self.llm_hook("\n".join(pool)) or "").strip()
                if out and not find_secrets(out) and not classify_sensitive(out):
                    return truncate(out, self.max_chars)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Summarizer hook failed (%s); using extractive summary.", type(exc).__name__)

        freq = Counter(t for s in pool for t in tokens(s))
        ranked = []
        for idx, s in enumerate(pool):
            toks = tokens(s)
            if not toks:
                continue
            score = sum(freq[t] for t in toks) / math.sqrt(len(toks)) + (2.0 if _SIGNAL.search(s) else 0.0)
            ranked.append((score, idx, s))
        ranked.sort(reverse=True)
        picked, used = [], 0
        for _, idx, s in ranked:
            if used + len(s) + 1 > self.max_chars - 12 and picked:
                continue
            picked.append((idx, s))
            used += len(s) + 1
            if len(picked) >= 5:
                break
        picked.sort()
        return truncate("Discussed: " + " ".join(s for _, s in picked), self.max_chars)

    def summarize_project(self, name: str, memories: List[Memory], max_chars: int = 900) -> str:
        groups: Dict[str, List[Memory]] = {"Decisions": [], "Progress": [], "Blockers": [], "Next steps": [], "Other": []}
        for m in memories:
            if m.is_sensitive:
                continue
            key = {
                Category.DECISION: "Decisions",
                Category.PROJECT_UPDATE: "Progress",
                Category.BLOCKER: "Blockers",
                Category.NEXT_STEP: "Next steps",
            }.get(m.category, "Other")
            groups[key].append(m)
        lines = [f"Project {name}:"]
        last = ""
        for title, limit in (("Decisions", 3), ("Progress", 3), ("Blockers", 2), ("Next steps", 3), ("Other", 2)):
            items = sorted(groups[title], key=lambda x: x.updated_at, reverse=True)[:limit]
            if not items:
                continue
            lines.append(f"{title}: " + " | ".join(f"{truncate(re.sub(r'^(Decision|Progress|Blocker|Next step): ', '', i.content), 110)} ({i.updated_at[:10]})" for i in items))
            last = max(last, items[0].updated_at)
        if len(lines) == 1:
            return ""
        if last:
            lines.append(f"Last activity: {last[:10]}")
        return truncate("\n".join(lines), max_chars)
