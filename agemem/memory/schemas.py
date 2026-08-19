from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

@dataclass
class MemoryEntry:
    id: str
    content: str
    kind: str = "fact"
    importance: float = 0.5
    confidence: float = 1.0
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        content: str,
        *,
        kind: str = "fact",
        importance: float = 0.5,
        confidence: float = 1.0,
        source: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> "MemoryEntry":
        return cls(
            id=str(uuid4()),
            content=content,
            kind=kind,
            importance=importance,
            confidence=confidence,
            source=source,
            metadata=metadata or {},
        )

@dataclass
class ContextSegment:
    id: str
    content: str
    role: str
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        content: str,
        *,
        role: str,
        metadata: dict[str, Any] | None = None,
    ) -> "ContextSegment":
        return cls(
            id=str(uuid4()),
            content=content,
            role=role,
            metadata=metadata or {},
        )

@dataclass
class ToolResult:
    success: bool
    tool_name: str
    changed_state: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None