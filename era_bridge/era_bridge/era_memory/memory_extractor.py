"""Rule-based extraction of memories from a conversation turn.

No network, no model: deterministic regex rules that pull out facts,
preferences, decisions, goals, events and project updates from what the
USER said. (The assistant's own words are never a source of facts.)

Safeguards built in:
* questions and one-shot device commands ("open notepad") are skipped, so
  ordinary chatter is not saved;
* clauses containing secrets are dropped (only a count is reported);
* clauses with sensitive data (health / financial / government IDs) are
  dropped and reported by *category only* -- see ``privacy.py``;
* hedged statements ("I think", "maybe") get low confidence;
* "actually / no longer / not X anymore" wording flags a *correction*, so
  the manager can supersede the older value.

An optional ``llm_hook`` can add candidates later (e.g. a local model);
its output goes through the same privacy filter, and if the hook fails the
rules still work.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Optional

from .memory_models import Candidate, Category, ExtractionResult, MemoryType
from .privacy import classify_sensitive, find_secrets
from .text_utils import split_sentences, truncate

logger = logging.getLogger(__name__)

LLMHook = Callable[[str, Optional[str]], List[Dict]]

_QUESTION_START = re.compile(
    r"^(what|who|when|where|why|how|which|can|could|would|will|do|does|did|is|are|am|should|shall|may|tell me|show me)\b", re.I
)
_COMMAND_VERBS = {
    "open", "close", "launch", "shutdown", "restart", "play", "search", "show", "take", "send", "call", "turn",
    "stop", "remind", "look", "check", "know", "ask", "see", "hear", "talk", "say", "understand", "learn",
}
_HEDGE = re.compile(r"\b(i think|maybe|probably|perhaps|i guess|not sure|might|possibly|i believe|i suppose)\b", re.I)
_CORR_START = re.compile(
    r"^\s*(?:actually|correction|to correct|scratch that|wait|no[,!.]|nope|sorry|oops|update[:,]|i meant|"
    r"that'?s (?:wrong|not (?:right|correct)|incorrect))\b",
    re.I,
)
_CORR_ANY = re.compile(
    r"\b(?:no longer|not anymore|anymore|instead of|i(?:'ve| have)? changed|changed my mind|switched (?:to|from)|"
    r"is now|are now|moved to)\b",
    re.I,
)
_TRAIL = re.compile(r"\s+(?:now|anymore|these days|currently|right now|though|instead|at the moment)\s*$", re.I)
_NOT_TAIL = re.compile(r"\s*,?\s+(?:and\s+)?not\s+.+$", re.I)
_COMM_CUES = re.compile(
    r"\b(answer|reply|respond|response|speak|talk|explain|call me|tone|language|hindi|english|hinglish|short|brief|"
    r"detailed|concise|formal|casual|sassy|step by step|slowly|simple)\w*",
    re.I,
)
_LANGS = re.compile(r"\b(hindi|english|hinglish|marathi|tamil|telugu|bengali|urdu|punjabi|gujarati|spanish|french|german)\b", re.I)
_LENGTH = re.compile(r"\b(short|brief|concise|detailed|long|lengthy|verbose)\b", re.I)
_ATTR_SKIP = {
    "question", "point", "guess", "opinion", "thought", "idea", "answer", "request", "message", "command",
    "query", "mistake", "problem", "issue", "code", "password", "pin", "token", "key", "secret", "otp", "name",
}
_ROLE_SKIP = re.compile(r"^(bit|little|lot|big fan|huge fan|good|bad|fast|slow|very|really|not|kind of|sort of|way)\b", re.I)
_TIME_CUE = re.compile(r"\b(yesterday|today|tonight|tomorrow|last (?:week|month|year|night)|this (?:morning|week)|ago|on (?:mon|tues|wednes|thurs|fri|satur|sun)day)\b", re.I)
_NOT_PROJECT = {"I", "It", "The", "This", "That", "My", "We", "A", "An", "Some", "Python", "Google", "Windows"}


def _clean(value: str, correction: bool = False) -> str:
    v = value.strip().strip("\"'").strip()
    if correction:
        v = _NOT_TAIL.sub("", v)
    v = _TRAIL.sub("", v)
    v = re.sub(r"\s+", " ", v).strip(" ,;:-")
    return truncate(v, 160)


def _absolutize(text: str, today: date) -> str:
    """Replace relative day words with real dates so the memory stays true later."""
    rep = {
        "yesterday": (today - timedelta(days=1)).isoformat(),
        "today": today.isoformat(),
        "tonight": today.isoformat(),
        "tomorrow": (today + timedelta(days=1)).isoformat(),
    }
    return re.sub(r"\b(yesterday|today|tonight|tomorrow)\b", lambda m: rep[m.group(1).lower()], text, flags=re.I)


class MemoryExtractor:
    def __init__(self, llm_hook: Optional[LLMHook] = None) -> None:
        self.llm_hook = llm_hook

    # ------------------------------------------------------------------ public
    def extract(
        self,
        user_message: str,
        assistant_message: Optional[str] = None,
        *,
        project_known: bool = False,
        today: Optional[date] = None,
    ) -> ExtractionResult:
        today = today or datetime.now().date()
        result = ExtractionResult()
        msg_correction = bool(_CORR_START.match(user_message or ""))
        seen_hashes = set()

        # project / active task hints come from the whole message
        result.project_name = self._project_name(user_message)
        project_ctx = project_known or bool(result.project_name)

        for sentence in split_sentences(user_message):
            clauses = re.split(r"\s+and\s+(?=(?:i|my|we|our)\b)", sentence, flags=re.I)
            for clause in clauses:
                clause = clause.strip()
                if len(clause.split()) < 3:
                    continue
                if find_secrets(clause):
                    result.secrets_blocked += 1
                    continue
                cats = classify_sensitive(clause)
                if cats:
                    for c in cats:
                        if c not in result.held_back:
                            result.held_back.append(c)
                    continue
                if clause.endswith("?") or _QUESTION_START.match(clause):
                    continue
                correction = msg_correction or bool(_CORR_START.match(clause)) or bool(_CORR_ANY.search(clause))
                hedged = bool(_HEDGE.search(clause))
                for cand in self._rules(clause, correction, hedged, project_ctx, today, result):
                    key = cand.content.lower()
                    if key not in seen_hashes:
                        seen_hashes.add(key)
                        result.candidates.append(cand)

        if self.llm_hook is not None:
            self._run_hook(user_message, assistant_message, result, seen_hashes)
        return result

    # ----------------------------------------------------------------- helpers
    def _run_hook(self, user_message, assistant_message, result: ExtractionResult, seen) -> None:
        try:
            extra = self.llm_hook(user_message, assistant_message) or []
        except Exception as exc:  # noqa: BLE001 - optional component must never break memory
            logger.warning("LLM extraction hook failed (%s); using rules only.", type(exc).__name__)
            return
        for item in extra:
            try:
                content = str(item["content"]).strip()
                if not content or content.lower() in seen:
                    continue
                if find_secrets(content):
                    result.secrets_blocked += 1
                    continue
                cats = classify_sensitive(content)
                if cats:
                    result.held_back += [c for c in cats if c not in result.held_back]
                    continue
                result.candidates.append(
                    Candidate(
                        content=truncate(content, 300),
                        type=MemoryType(item.get("type", "semantic")),
                        category=Category(item.get("category", "fact")),
                        slot=item.get("slot"),
                        confidence=min(float(item.get("confidence", 0.6)), 0.7),  # model output is never 'confirmed'
                        importance=float(item.get("importance", 0.5)),
                        source_kind="chat",
                    )
                )
                seen.add(content.lower())
            except (KeyError, ValueError, TypeError):
                continue

    @staticmethod
    def _project_name(text: str) -> Optional[str]:
        for pat in (
            r"\b(?:project|app|assistant|system|website|bot|engine)\s+(?:called|named)\s+([A-Za-z][\w\-]*)",
            r"\bmy\s+([A-Z][\w\-]+)\s+(?:project|app|assistant|system|website|bot)\b",
            r"\b(?:working on|building|developing|debugging|fixing|designing|continuing|upgrading)\s+(?:the |my |our |a |an )?([A-Z][\w\-]*)",
        ):
            m = re.search(pat, text or "")
            if m and m.group(1) not in _NOT_PROJECT:
                return m.group(1)
        return None

    def _rules(self, clause: str, correction: bool, hedged: bool, project_ctx: bool, today: date, result: ExtractionResult) -> List[Candidate]:
        out: List[Candidate] = []
        conf = 0.4 if hedged else 0.9

        def mk(content, mtype, cat, importance, slot=None, confidence=None, **meta) -> Candidate:
            return Candidate(
                content=truncate(_absolutize(content, today), 300),
                type=mtype,
                category=cat,
                slot=slot,
                confidence=conf if confidence is None else confidence,
                importance=max(0.1, importance - (0.15 if hedged else 0.0)) + (0.05 if correction else 0.0),
                is_correction=correction,
                metadata=dict(meta),
            )

        low = clause.strip()

        # --- identity -------------------------------------------------------
        m = re.search(r"\bmy name is\s+([A-Za-z][\w'\-]*)(?:\s+([A-Za-z][\w'\-]*))?", low, re.I)
        if m:
            name = m.group(1)
            if m.group(2) and m.group(2)[0].isupper() and m.group(2).lower() not in {"and", "but", "so", "i", "from"}:
                name += " " + m.group(2)
            out.append(mk(f"User's name is {name}.", MemoryType.SEMANTIC, Category.FACT, 0.8, "user:name"))
            return out

        m = re.search(r"\b(?:call me|address me as|refer to me as)\s+([\w'\-]+)", low, re.I)
        if m:
            out.append(mk(f"User wants to be called {m.group(1)}.", MemoryType.PREFERENCE, Category.COMMUNICATION, 0.75, "user:nickname"))
            return out

        m = re.search(r"\bi\s+(?:currently\s+)?(?:live|stay|reside|am based|am living|moved|have moved)\s+(?:in|to|at)\s+([^,.;!?]+)", low, re.I)
        if m:
            v = _clean(m.group(1), correction)
            if v:
                out.append(mk(f"User lives in {v}.", MemoryType.SEMANTIC, Category.FACT, 0.7, "user:location"))
                return out

        m = re.search(r"\bi(?:'m| am)\s+(?:currently\s+)?(?:a|an)\s+([^,.;!?]+)", low, re.I)
        if m and not _ROLE_SKIP.match(m.group(1)):
            v = _clean(m.group(1), correction)
            if v:
                out.append(mk(f"User is a {v}.", MemoryType.SEMANTIC, Category.FACT, 0.65, "user:role"))
                return out

        m = re.search(r"\bi(?:'m| am)?\s*(?:currently\s+)?(?:study|studying)\s+([^,.;!?]+)", low, re.I)
        if m:
            v = _clean(m.group(1), correction)
            if v:
                out.append(mk(f"User studies {v}.", MemoryType.SEMANTIC, Category.FACT, 0.65, "user:studies"))
                return out

        # --- project structure: blockers / next steps / decisions --------------
        m = re.search(r"\b(?:the |my |our )?(?:current |main )?blocker is\s+([^.;!?]+)", low, re.I) or re.search(
            r"\b(?:i(?:'m| am)|we(?:'re| are))\s+(?:currently\s+)?(?:blocked|stuck)\s+(?:on|by|with|at)\s+([^.;!?]+)", low, re.I
        ) or re.search(r"\bthe (?:problem|issue|bug) is\s+([^.;!?]+)", low, re.I)
        if m:
            v = _clean(m.group(1), correction)
            if v:
                out.append(mk(f"Blocker: {v}.", MemoryType.PROJECT, Category.BLOCKER, 0.7))
                return out

        m = re.search(r"\b(?:next steps?|todo|to-do|next up)\s*(?:is|are|:)\s*([^.;!?]+)", low, re.I)
        if m:
            v = _clean(m.group(1), correction)
            if v:
                out.append(mk(f"Next step: {v}.", MemoryType.PROJECT, Category.NEXT_STEP, 0.65))
                return out

        m = re.search(r"\b(?:we|i)\s+(?:have\s+|'ve\s+)?(?:decided|chose|agreed)\s+(?:to|on|that)\s+([^.;!?]+)", low, re.I) or re.search(
            r"\b(?:we|i)(?:'ll|\s+will|\s+are going to|\s+am going to|'re going to)\s+((?:use|go with|switch to|stick with)\s+[^.;!?]+)", low, re.I
        ) or re.search(r"\b(?:switched|moved|migrated)\s+(to\s+[^.;!?]+)", low, re.I)
        if m:
            v = _clean(m.group(1), correction)
            if v:
                out.append(mk(f"Decision: {v}.", MemoryType.PROJECT, Category.DECISION, 0.75))
                return out

        m = re.search(r"\b(?:i|we)\s+(?:still\s+)?(?:need|have)\s+to\s+([^.;!?]+)", low, re.I)
        if m:
            v = _clean(m.group(1), correction)
            if v and v.split()[0].lower() not in _COMMAND_VERBS:
                out.append(mk(f"Next step: {v}.", MemoryType.PROJECT, Category.NEXT_STEP, 0.6))
                return out

        m = re.search(
            r"\b(?:i|we)(?:\s+have|'ve)?\s+(?:just\s+)?(finished|completed|built|implemented|fixed|added|deployed|shipped|integrated|created|wired|set up)\s+([^.;!?]+)",
            low, re.I,
        )
        if m and project_ctx:
            v = _clean(m.group(2), correction)
            if v:
                out.append(mk(f"Progress: {m.group(1).lower()} {v}.", MemoryType.PROJECT, Category.PROJECT_UPDATE, 0.55))
                return out

        # --- active task (working memory) ---------------------------------------
        m = re.search(
            r"\b(?:i(?:'m| am)|we(?:'re| are))\s+(?:currently\s+)?(?:working on|building|developing|debugging|fixing|writing|designing|making)\s+([^.;!?]+)",
            low, re.I,
        ) or re.search(r"\blet'?s\s+(?:work on|build|fix|debug|write|continue|design|make)\s+([^.;!?]+)", low, re.I)
        if m:
            v = _clean(m.group(1), correction)
            if v:
                result.active_task = v
                if project_ctx:
                    out.append(mk(f"Progress: working on {v}.", MemoryType.PROJECT, Category.PROJECT_UPDATE, 0.5))
                return out

        # --- goals --------------------------------------------------------------
        m = re.search(r"\b(?:my goal is|i(?:'m| am) trying|i want|i plan|i aim|i(?:'d| would) like)\s+(?:to\s+)?([^.;!?]+)", low, re.I)
        if m:
            v = _clean(m.group(1), correction)
            if v and v.split()[0].lower() not in _COMMAND_VERBS:
                out.append(mk(f"User's goal: {v}.", MemoryType.SEMANTIC, Category.GOAL, 0.7))
                return out

        # --- generic "my X is Y" ---------------------------------------------------
        m = re.search(r"\bmy\s+([a-z][a-z\-]*(?:\s[a-z\-]+){0,2}?)\s+(?:is|are|was|will be)\s+(?:now\s+)?([^,.;!?]+)", low, re.I)
        if m:
            attr = m.group(1).strip().lower()
            if not (set(attr.split()) & _ATTR_SKIP):
                v = _clean(m.group(2), correction)
                if v:
                    fav = attr.startswith(("favorite", "favourite"))
                    out.append(
                        mk(
                            f"User's {attr} is {v}.",
                            MemoryType.PREFERENCE if fav else MemoryType.SEMANTIC,
                            Category.PREFERENCE if fav else Category.FACT,
                            0.6,
                            f"user:{attr}",
                        )
                    )
                    return out

        # --- communication / general preferences -------------------------------------
        m = re.search(r"^(?:please\s+)?(?:always|never|from now on|going forward|in future|from here on)\b\s*(.+)", low, re.I) or (
            re.search(r"\b(?:keep|make|reply|respond|answer|speak|talk|explain)\b.*\b(?:short|brief|concise|detailed|simple|formal|casual|hindi|english|hinglish|slowly|step by step)\b", low, re.I)
        )
        if m and _COMM_CUES.search(low):
            slot = "comm:language" if _LANGS.search(low) else ("comm:length" if _LENGTH.search(low) else None)
            body = _clean(re.sub(r"^(?:please\s+)?", "", low, flags=re.I), correction)
            out.append(mk(f"Communication preference: {body[0].lower() + body[1:] if body else body}.", MemoryType.PREFERENCE, Category.COMMUNICATION, 0.75, slot))
            return out

        m = re.search(r"\bi\s+(?:really\s+|strongly\s+)?(prefer|like|love|enjoy|hate|dislike|can't stand|am a fan of)\s+([^.;!?]+)", low, re.I)
        if m:
            verb = {"prefer": "prefers", "like": "likes", "love": "loves", "enjoy": "enjoys", "hate": "hates",
                    "dislike": "dislikes", "can't stand": "can't stand", "am a fan of": "is a fan of"}[m.group(1).lower()]
            obj = _clean(m.group(2), correction)
            if obj and obj.lower() not in {"it", "that", "this", "them", "you", "the idea"}:
                cat = Category.COMMUNICATION if _COMM_CUES.search(obj) else Category.PREFERENCE
                out.append(mk(f"User {verb} {obj}.", MemoryType.PREFERENCE, cat, 0.6))
                return out

        # --- events ----------------------------------------------------------------
        m = re.search(
            r"\bi\s+(?:have|got|'ve got)\s+(?:an?\s+|the\s+|my\s+)?(exam|test|interview|meeting|presentation|deadline|appointment|viva|submission|trip|flight)\b([^.;!?]*)",
            low, re.I,
        )
        if m:
            out.append(mk(f"User has a {m.group(1).lower()}{m.group(2).rstrip()}.", MemoryType.EPISODIC, Category.EVENT, 0.55))
            return out
        m = re.search(r"\bi\s+(went|met|attended|joined|visited|passed|failed|won|submitted|started|completed|graduated)\s+([^.;!?]+)", low, re.I)
        if m and _TIME_CUE.search(low):
            out.append(mk(f"User {m.group(1).lower()} {_clean(m.group(2))}.", MemoryType.EPISODIC, Category.EVENT, 0.45))
            return out

        return out
