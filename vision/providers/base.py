"""Vision provider abstraction.

All provider-specific request/response/auth logic must live inside a
provider module (e.g. `gemini_provider.py`) behind this interface. Nothing
outside `vision/providers/` should know how a specific provider's REST API
is shaped -- `analyzer.py` only talks to `VisionProvider`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from vision.models import ScreenState


class ProviderError(RuntimeError):
    """Base class for provider failures."""


class ProviderAuthError(ProviderError):
    """The provider rejected our credentials."""


class ProviderRateLimitError(ProviderError):
    """The provider is rate-limiting us."""


class ProviderTimeoutError(ProviderError):
    """The provider did not respond in time."""


class ProviderResponseError(ProviderError):
    """The provider responded, but the response could not be used
    (malformed JSON, missing fields, safety block, etc.)."""


class VisionProvider(ABC):
    name: str = "base"

    @abstractmethod
    def analyze(
        self,
        image: Image.Image,
        *,
        prompt: str,
        roi_image: Image.Image | None = None,
    ) -> ScreenState:
        """Analyze a screenshot (and optional region-of-interest crop) and
        return a structured ScreenState. Must raise a ProviderError subclass
        on failure rather than returning a fabricated/default result, so
        callers can distinguish "nothing to report" from "the call failed"."""

    @abstractmethod
    def is_configured(self) -> bool:
        """True if this provider has the credentials it needs to be called.
        Must not make a network call."""
