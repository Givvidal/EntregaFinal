from langchain_core.messages import HumanMessage, AIMessage

from state import AgentState, obtener_consulta
from agents.research_agent import agente_investigador
from agents.analyst_agent import agente_analista


async def nodo_investigador(state: AgentState) -> dict:
    tarea = obtener_consulta(state)

    resultado = await agente_investigador.ainvoke(
        {"messages": [HumanMessage(content=tarea)]}
    )

    respuesta = extraer_texto(resultado["messages"][-1].content)

    return {
        "messages": [AIMessage(content=respuesta, name="investigador")],
        "contribuciones": [
            {"agente": "investigador", "aporte": respuesta}
        ],
        "pasos": state.get("pasos", 0) + 1, 
    }


async def nodo_analista(state: AgentState) -> dict:
    pregunta_original = obtener_consulta(state)
    aporte_investigador = state["contribuciones"][-1]["aporte"]

    tarea = (
        f"Pregunta original:\n{pregunta_original}\n\n"
        f"Datos obtenidos por el investigador:\n{aporte_investigador}\n\n"
        "Realizá los cálculos necesarios utilizando la calculadora."
    )

    resultado = await agente_analista.ainvoke(
        {"messages": [HumanMessage(content=tarea)]}
    )

    respuesta = extraer_texto(resultado["messages"][-1].content)

    return {
        "messages": [AIMessage(content=respuesta, name="analista")],
        "contribuciones": [
            {"agente": "analista", "aporte": respuesta}
        ],
        "pasos": state.get("pasos", 0) + 1,
    }
    
def extraer_texto(contenido):
    if isinstance(contenido, list):
        return "\n".join(
            parte.get("text", "")
            for parte in contenido
            if isinstance(parte, dict)
        )
    return str(contenido)