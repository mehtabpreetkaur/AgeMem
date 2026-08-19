from __future__ import annotations

from collections.abc import Iterable

from agemem.memory.schemas import ContextSegment

class STMBuffer:
    def __init__(self, max_segments: int = 20) -> None:
        if max_segments <= 0:
            raise ValueError("max_segments must be greater than 0")
        
        self.max_segments = max_segments
        self._segments: list[ContextSegment] = []
    
    def add_segment(self, segment: ContextSegment) -> None:
        self._segments.append(segment)
        self._enforce_budget()
    
    def add_message(
        self, 
        content: str,
        *,
        role: str,
        metadata: dict | None = None,
    ) -> ContextSegment:
        # Creates and inserts a new STM Context Segment
        segment = ContextSegment.create(
            content=content,
            role=role,
            metadata=metadata,
        )
        self.add_segment(segment)
        return segment
    
    def extend(self, segments: Iterable[ContextSegment]) -> None:
        self._segments.extend(segments)
        self._enforce_budget()
    
    def list_segments(self) -> list[ContextSegment]:
        # Returns a copy of the current STM contents
        return list(self._segments)
    
    def get_segment(self, segment_id: str) -> ContextSegment | None:
        for segment in self._segments:
            if segment.id == segment_id:
                return segment
        return None
    
    def remove_segments(self, segment_ids: set[str]) -> list[ContextSegment]:
        # This function will power the FILTER tool
        removed: list[ContextSegment] = []
        kept: list[ContextSegment] = []
        
        for segment in self._segments:
            if segment.id in segment_ids:
                removed.append(segment)
            else:
                kept.append(segment)
        
        self._segments = kept
        return removed
    
    def replace_segment(
        self,
        segment_ids: set[str],
        replacement: ContextSegment,
    ) -> list[ContextSegment]:
        # This function will power the SUMMARY tool
        removed = self.remove_segments(segment_ids)
        self.add_segment(replacement)
        return removed
    
    def clear(self) -> None:
        self._segments.clear()
        
    def _enforce_budget(self) -> None:
        # Simulated context-window limit - currently 20 segments 
        overflow = len(self._segments) - self.max_segments
        if overflow > 0:
            self._segments = self._segments[overflow:]