"""Prueba aislada de persistencia del checkpointer SQLite asíncrono."""
from typing import TypedDict
import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, START, END


class EstadoPrueba(TypedDict):
    valor: int


async def incrementar(state: EstadoPrueba) -> dict:
    return {"valor": state["valor"] + 1}


@pytest.mark.asyncio
async def test_checkpoint_persiste_estado(tmp_path):
    db_path = tmp_path / "checkpoints.sqlite"
    builder = StateGraph(EstadoPrueba)
    builder.add_node("eco", incrementar)
    builder.add_edge(START, "eco")
    builder.add_edge("eco", END)
    config = {"configurable": {"thread_id": "test-persistencia"}}
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        graph = builder.compile(checkpointer=saver)
        resultado = await graph.ainvoke({"valor": 1}, config=config)
        assert resultado["valor"] == 2
        estado = await graph.aget_state(config)
        assert estado.values["valor"] == 2
    assert db_path.exists()
