"""Bridges capture + context into a provider call and back into a
validated ScreenState. This is the only place that constructs the prompt
sent to a vision provider."""

from __future__ import annotations

import logging

from vision.capture import Frame
from vision.models import ScreenState, WindowInfo
from vision.providers.base import ProviderError
from vision.providers.gemini_provider import GeminiProvider

logger = logging.getLogger("era_vision.analyzer")


def _build_prompt(window: WindowInfo, task_context: str | None, trigger: str) -> str:
    lines = [
        f"Active window title: {window.title!r}" if window.title else "Active window title: (unavailable)",
        f"Active process: {window.process_name or '(unknown)'}",
        f"Trigger for this observation: {trigger}",
    ]
    if task_context:
        lines.append(f"Current user task (for context only, do not assume it is complete): {task_context}")
    lines.append(
        "Describe the application, the important visible UI elements, any visible "
        "error messages, and the important visible text. Return JSON matching the "
        "provided schema."
    )
    return "\n".join(lines)


class Analyzer:
    def __init__(self, provider: GeminiProvider) -> None:
        self._provider = provider

    @property
    def provider_configured(self) -> bool:
        return self._provider.is_configured()

    @property
    def provider_name(self) -> str:
        return self._provider.name

    def analyze(
        self,
        frame: Frame,
        window: WindowInfo,
        *,
        trigger: str,
        roi_bbox: tuple[int, int, int, int] | None = None,
        task_context: str | None = None,
    ) -> ScreenState:
        prompt = _build_prompt(window, task_context, trigger)
        roi_image = frame.crop(roi_bbox).image if roi_bbox is not None else None
        try:
            return self._provider.analyze(frame.image, prompt=prompt, roi_image=roi_image)
        except ProviderError:
            # Let the caller (service.py) decide how to log/surface this;
            # re-raising keeps analyzer.py provider-agnostic.
            raise
