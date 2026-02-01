"""Shared types for EchoX."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


@dataclass
class Word:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    score: Optional[float] = None


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    segments: list[Segment]
    metadata: dict[str, Any] = field(default_factory=dict)

    def iter_words(self) -> Iterable[Word]:
        for segment in self.segments:
            for word in segment.words:
                yield word