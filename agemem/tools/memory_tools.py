from __future__ import annotations

from agemem.memory.ltm_store import LTMStore
from agemem.memory.schemas import ContextSegment, ToolResult
from agemem.memory.stm_buffer import STMBuffer

class MemoryTools:
    def __init__(self, ltm: LTMStore, stm: STMBuffer) -> None:
        self.ltm = ltm
        self.stm = stm
        
    def add(
        self,
        content: str,
        *,
        kind: str = "fact",
        importance: float = 0.5,
        confidence: float = 1.0,
        source: str = "agent",
        metadata: dict | None = None,
    ) -> ToolResult:
        entry = self.ltm.add(
            content=content,
            kind=kind,
            importance=importance,
            confidence=confidence,
            source=source,
            metadata=metadata,
        )
        
        return ToolResult(
            success=True,
            tool_name="ADD",
            changed_state=True,
            result={
                "memory_id": entry.id,
                "content": entry.content,
                "kind": entry.kind,
                "importance": entry.importance,
                "confidence": entry.confidence,
            },
        )