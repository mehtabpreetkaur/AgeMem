from agemem.memory.ltm_store import LTMStore
from agemem.memory.stm_buffer import STMBuffer
from agemem.tools.memory_tools import MemoryTools

def test_add_and_retrieve_memory() -> None:
    ltm = LTMStore()
    stm = STMBuffer(max_segments=10)
    tools = MemoryTools(ltm, stm)
    
    add_result = tools.add(
        "The user is building an AgeMem prototype.",
        importance=0.9,
    )

    assert add_result.success is True
    assert add_result.tool_name == "ADD"
    assert add_result.changed_state is True
    assert add_result.result["memory_id"] is not None

    retrieve_result = tools.retrieve("AgeMem prototype")

    assert retrieve_result.success is True
    assert retrieve_result.tool_name == "RETRIEVE"
    assert retrieve_result.changed_state is True
    assert retrieve_result.result["retrieved_count"] == 1

    stm_segments = stm.list_segments()

    assert len(stm_segments) == 1
    assert stm_segments[0].role == "retrieved_memory"
    assert stm_segments[0].content == "The user is building an AgeMem prototype."
    assert stm_segments[0].metadata["memory_id"] == add_result.result["memory_id"]
    
def test_update_memory() -> None:
    ltm = LTMStore()
    stm = STMBuffer(max_segments=10)
    tools = MemoryTools(ltm, stm)
    
    add_result = tools.add("The user likes tea.", importance=0.3)
    memory_id = add_result.result["memory_id"]
    
    update_result = tools.update(
        memory_id,
        content="The user likes black tea.",
        importance=0.8,
    )
    
    assert update_result.success is True
    assert update_result.tool_name == "UPDATE"
    assert update_result.changed_state is True
    assert update_result.result["memory_id"] == memory_id
    assert update_result.result["content"] == "The user likes black tea."
    assert update_result.result["importance"] == 0.8
    
    stored_entry = ltm.get(memory_id)
    
    assert stored_entry is not None
    assert stored_entry.content == "The user likes black tea."
    assert stored_entry.importance == 0.8

def test_delete_memory() -> None:
    ltm = LTMStore()
    stm = STMBuffer(max_segments=10)
    tools = MemoryTools(ltm, stm)
    
    add_result = tools.add("Temporary fact.")
    memory_id = add_result.result["memory_id"]
    
    delete_result = tools.delete(memory_id)
    
    assert delete_result.success is True
    assert delete_result.tool_name == "DELETE"
    assert delete_result.changed_state is True
    assert delete_result.result["memory_id"] == memory_id
    assert ltm.get(memory_id) is None

def test_summary_replaces_stm_segments() -> None:
    ltm = LTMStore()
    stm = STMBuffer(max_segments=10)
    tools = MemoryTools(ltm, stm)
    
    first = stm.add_message("The user is building memory tools.", role="user")
    second = stm.add_message("The agent should use actions.", role="assistant")
    
    summary_result = tools.summary(
        {first.id, second.id},
        summary_text="The project is adding agent-facing memory actions.",
    )
    
    assert summary_result.success is True
    assert summary_result.tool_name == "SUMMARY"
    assert summary_result.changed_state is True
    assert summary_result.result["removed_segment_ids"] == [first.id, second.id]
    
    stm_segments = stm.list_segments()
    
    assert len(stm_segments) == 1
    assert stm_segments[0].role == "summary"
    assert stm_segments[0].content == "The project is adding agent-facing memory actions."
    assert stm_segments[0].metadata["summarized_count"] == 2

def test_filter_removes_stm_segments() -> None:
    ltm = LTMStore()
    stm = STMBuffer(max_segments=10)
    tools = MemoryTools(ltm, stm)
    
    keep = stm.add_message("Important active context.", role="user")
    remove = stm.add_message("Irrelevant active context.", role="user")
    
    filter_result = tools.filter({remove.id})
    
    assert filter_result.success is True
    assert filter_result.tool_name == "FILTER"
    assert filter_result.changed_state is True
    assert filter_result.result["removed_segment_ids"] == [remove.id]
    assert filter_result.result["removed_count"] == 1
    
    stm_segments = stm.list_segments()
    
    assert len(stm_segments) == 1
    assert stm_segments[0].id == keep.id