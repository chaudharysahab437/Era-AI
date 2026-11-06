"""Small, dependency-free text helpers shared by the memory modules."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Set

STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are aren't as at be because been before being below
    between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each
    few for from further had hadn't has hasn't have haven't having he her here hers herself him himself his how
    i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most my myself no nor not of off on
    once only or other ought our ours ourselves out over own same she should shouldn't so some such than that
    the their theirs them themselves then there these they this those through to too under until up very was
    wasn't we we'd we'll we're we've were weren't what when where which while who whom why will with won't would
    wouldn't you you'd you'll you're you've your yours yourself yourselves s t d ll m re ve
    please tell know remember let ok okay hey hi hello yeah yes really just like get got
    user
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def stem(word: str) -> str:
    """Very light suffix stripping -- enough to match plurals and simple verb forms."""
    w = word
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


def tokens(text: str, keep_stop: bool = False) -> List[str]:
    """Lower-cased, stop-word-free, stemmed tokens (order preserved)."""
    raw = _TOKEN_RE.findall((text or "").lower().replace("'", ""))
    if keep_stop:
        return [stem(t) for t in raw]
    # one-character tokens are noise EXCEPT digits: 'rack 3' and 'rack 4' are different facts
    return [stem(t) for t in raw if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def token_set(text: str) -> Set[str]:
    return set(tokens(text))


def content_hash(text: str) -> str:
    """Stable hash of the *meaning-bearing* tokens, so trivial rewordings of
    punctuation/case/plurals hash the same."""
    norm = " ".join(tokens(text, keep_stop=True))
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def overlap(query: Set[str], memory: Set[str]) -> float:
    """|q ∩ m| / min(|q|, |m|) -- 1.0 when the shorter side is fully contained."""
    if not query or not memory:
        return 0.0
    return len(query & memory) / min(len(query), len(memory))


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return s or "project"


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def first(iterable: Iterable):
    for x in iterable:
        return x
    return None
