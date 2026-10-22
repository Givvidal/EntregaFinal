
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from state import AgentState
from nodes import nodo_investigador, nodo_analista
from supervisor import nodo_supervisor
from synthesis import nodo_sintesis
from aclaracion import (
    nodo_evaluar_aclaracion,
    nodo_solicitar_aclaracion,
    decidir_aclaracion,
)


def enrutar(
    state: AgentState,
) -> Literal["investigador", "analista", "sintesis"]:
    if state["next_agent"] == "FINISH":
        return "sintesis"

    return state["next_agent"]


def nodo_aprobacion(state: AgentState):
    aprobacion = interrupt({
        "tipo": "aprobacion",
        "mensaje": "La tarea requiere aprobación antes de continuar.",
        "job_id": state.get("job_id"),
    })

    return {"aprobado": aprobacion}


def decidir_aprobacion(state: AgentState):
    if state.get("requires_approval", False):
        return "aprobacion"

    return "FINISH"


grafo = StateGraph(AgentState)

grafo.add_node("evaluar_aclaracion", nodo_evaluar_aclaracion)
grafo.add_node("solicitar_aclaracion", nodo_solicitar_aclaracion)
grafo.add_node("supervisor", nodo_supervisor)
grafo.add_node("investigador", nodo_investigador)
grafo.add_node("analista", nodo_analista)
grafo.add_node("sintesis", nodo_sintesis)
grafo.add_node("aprobacion", nodo_aprobacion)

grafo.add_edge(START, "evaluar_aclaracion")

grafo.add_conditional_edges(
    "evaluar_aclaracion",
    decidir_aclaracion,
    {
        "solicitar_aclaracion": "solicitar_aclaracion",
        "supervisor": "supervisor",
    },
)

grafo.add_edge("solicitar_aclaracion", "supervisor")

grafo.add_conditional_edges(
    "supervisor",
    enrutar,
    {
        "investigador": "investigador",
        "analista": "analista",
        "sintesis": "sintesis",
    },
)

grafo.add_edge("investigador", "supervisor")
grafo.add_edge("analista", "supervisor")

grafo.add_conditional_edges(
    "sintesis",
    decidir_aprobacion,
    {
        "aprobacion": "aprobacion",
        "FINISH": END,
    },
)

grafo.add_edge("aprobacion", END)


def crear_app(checkpointer):
    return grafo.compile(checkpointer=checkpointer)
