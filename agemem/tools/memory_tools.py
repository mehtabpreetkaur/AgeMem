from __future__ import annotations

from agemem.memory.ltm_store import LTMStore
from agemem.memory.schemas import ContextSegment, ToolResult
from agemem.memory.stm_buffer import STMBuffer

class MemoryTools:
    def __init__(self, ltm: LTMStore, stm: STMBuffer) -> None:
        self.ltm = ltm
        self.stm = stm