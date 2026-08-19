# In this version LTM is an in-memory Python store
# It works while the program is running, but it does not persist after the process exits.

from __future__ import annotations

from agemem.memory.schemas import MemoryEntry, utc_now

class LTMStore:
    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        
    def add(
        self,
        content: str,
        *,
        kind: str = "fact",
        importance: float = 0.5,
        confidence: float = 1.0,
        source: str = "unknown",
        metadata: dict | None = None,
    ) -> MemoryEntry:
        # Create a new LTM Memory Entry 
        # This will map our memory tool ADD
        entry = MemoryEntry.create(
            content=content,
            kind=kind,
            importance=importance,
            confidence=confidence,
            source=source,
            metadata=metadata,
        )
        self._entries[entry.id] = entry
        return entry
    
    def get(self, memory_id: str) -> MemoryEntry | None:
        return self._entries.get(memory_id)
    
    def list_entries(self) -> list[MemoryEntry]:
        return list(self._entries.values())
    
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
    ) -> MemoryEntry | None:
        # Modifies existing LTM Memory Entry
        # This will map to our memory tool UPDATE
        entry = self.get(memory_id)
        if entry is None:
            return None
        
        if content is not None:
            entry.content = content
        if kind is not None:
            entry.kind = kind
        if importance is not None:
            entry.importance = importance
        if confidence is not None:
            entry.confidence = confidence
        if source is not None:
            entry.source = source
        if metadata is not None:
            entry.metadata.update(metadata)
        
        entry.updated_at = utc_now()
        return entry
    
    def delete(self, memory_id: str) -> MemoryEntry | None:
        # Removes an existing LTM Memory Entry
        # This will map to our memory tool DELETE
        return self._entries.pop(memory_id, None)
    
    def search(self, query: str, *, top_k: int = 5) -> list[MemoryEntry]:
        # Primitive retrieval mechanism - uses simle keyword matching
        # This will map to our memory tool RETRIEVE
        query_terms = self._normalize(query)
        
        scored_entries: list[tuple[int, MemoryEntry]] = []
        
        for entry in self._entries.values():
            content_terms = self._normalize(entry.content)
            score = len(query_terms.intersection(content_terms))
            
            if score > 0:
                scored_entries.append((score, entry))
        
        # Memory prioritization based on importance and confidence
        scored_entries.sort(
            key=lambda item: (
                item[0],
                item[1].importance,
                item[1].confidence,
            ),
            reverse=True,
        )
        
        return [entry for _, entry in scored_entries[:top_k]]
    
    def clear(self) -> None:
        self._entries.clear()
        
    def _normalize(self, text: str) -> set[str]:
        return {
            token.strip(".,!?;:()[]{}\"'").lower()
            for token in text.split()
            if token.strip(".,!?;:()[]{}\"'")
        }