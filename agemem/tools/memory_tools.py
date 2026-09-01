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
    
    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        kind: str | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        source: str | None = None,
        metadata: dict | None = None,
    ) -> ToolResult:
        entry = self.ltm.update(
            memory_id=memory_id,
            content=content,
            kind=kind,
            importance=importance,
            confidence=confidence,
            source=source,
            metadata=metadata,
        )
        
        if entry is None:
            return ToolResult(
                success=False,
                tool_name="UPDATE",
                changed_state=False,
                error=f"Memory entry not found: {memory_id}",
            )
        
        return ToolResult(
            success=True,
            tool_name="UPDATE",
            changed_state=True,
            result={
                "memory_id": entry.id,
                "content": entry.content,
                "kind": entry.kind,
                "importance": entry.importance,
                "confidence": entry.confidence,
            },
        )
    
    def delete(self, memory_id: str) -> ToolResult:
        entry = self.ltm.delete(memory_id)
        
        if entry is None:
            return ToolResult(
                success=False,
                tool_name="DELETE",
                changed_state=False,
                error=f"Memory entry not found: {memory_id}",
            )
        
        return ToolResult(
            success=True,
            tool_name="DELETE",
            changed_state=True,
            result={
                "memory_id": entry.id,
                "content": entry.content,
            }
        )
    
    def retrieve (self, query: str, *, top_k: int = 5) -> ToolResult:
        entries = self.ltm.retrieve(query, top_k=top_k)
        
        segments: list[ContextSegment] = []
        
        for entry in entries:
            segment = ContextSegment.create(
                content=entry.content,
                role="retrieved_memory",
                metadata={
                    "memory_id": entry.id,
                    "kind": entry.kind,
                    "importance": entry.importance,
                    "confidence": entry.confidence,
                },
            )
            segments.append(segment)
            
        self.stm.extend(segments)
        
        return ToolResult(
            success=True,
            tool_name="RETRIEVE",
            changed_state=True,
            result={
                "query": query,
                "retrieved_count": len(entries),
                "segment_ids": [segment.id for segment in segments],
                "memory_ids": [entry.id for entry in entries],
            },
        )
    
    def summary(
        self,
        segment_ids: set[str],
        *,
        summary_text: str | None = None,
    ) -> ToolResult:
        segments = [
            segment 
            for segment in self.stm.list_segments()
            if segment.id in segment_ids
        ]
        
        if not segments:
            return ToolResult(
                success=False,
                tool_name="SUMMARY",
                changed_state=False,
                error="No matching STM segments found to summarize.",
            )
        
        if summary_text is None:
            summary_text = " ".join(segment.content for segment in segments)
        
        replacement = ContextSegment.create(
            content=summary_text,
            role="summary",
            metadata={
                "summarized_segment_ids": [segment.id for segment in segments],
                "summarized_count": len(segments),
            },
        )
        
        removed = self.stm.replace_segments(segment_ids, replacement)
        
        return ToolResult(
            success=True,
            tool_name="SUMMARY",
            changed_state=True,
            result={
                "summary_segment_id": replacement.id,
                "removed_segment_ids": [segment.id for segment in removed],
                "summarized_text": replacement.content,
            },
        )