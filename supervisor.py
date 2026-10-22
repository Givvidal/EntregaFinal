import os
from typing import Literal

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from state import AgentState, obtener_consulta


MAX_PASOS = 6


class DecisionSupervisor(BaseModel):
    next: Literal["investigador", "analista", "FINISH"] = Field(
        description="Próximo agente a invocar, o FINISH si la tarea ya está completa"
    )
    razon: str = Field(
        description="Breve justificación de la decisión"
    )


SUPERVISOR_PROMPT = """Sos el Supervisor de un equipo con dos especialistas:

- investigador: busca datos concretos exclusivamente en el RAG.
- analista: realiza cálculos utilizando los datos obtenidos.

Reglas obligatorias:

1. Si todavía no existe ningún aporte del investigador, mandá a 'investigador'.

2. Si ya existe un aporte del investigador y la tarea original requiere
   un cálculo, promedio, diferencia, porcentaje o comparación numérica,
   mandá a 'analista'.

3. Una vez que el investigador realizó su aporte, NO vuelvas a mandarlo
   a investigar. El analista debe trabajar con los datos obtenidos.

4. Si ya existe un aporte del analista con el cálculo solicitado,
   respondé 'FINISH'.

5. No mandes nuevamente al analista si ya realizó su cálculo.

6. No inventes datos.

Antes de finalizar verificá:
- que exista información del investigador;
- que el analista haya realizado los cálculos necesarios;
- que los aportes sean suficientes para responder la pregunta.

Contribuciones hasta ahora:
{contribuciones}
"""


llm_supervisor = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
    temperature=0
).with_structured_output(DecisionSupervisor)


async def nodo_supervisor(state: AgentState) -> dict:
    if state.get("pasos", 0) >= MAX_PASOS:
        return {"next_agent": "FINISH", "task_completed": True}

    contribuciones = state.get("contribuciones", [])
    agentes_realizados = {c["agente"] for c in contribuciones}

    # La investigación es siempre el primer paso y se ejecuta una sola vez.
    if "investigador" not in agentes_realizados:
        return {"next_agent": "investigador", "task_completed": False}

    # Si el analista ya aportó, no se repite ningún agente.
    if "analista" in agentes_realizados:
        return {"next_agent": "FINISH", "task_completed": True}

    contribuciones_texto = "\n".join(
        f"- {c['agente']}: {c['aporte'][:200]}" for c in contribuciones
    ) or "(ninguna todavía)"
    pregunta_original = obtener_consulta(state)
    prompt_sistema = SUPERVISOR_PROMPT.format(contribuciones=contribuciones_texto)

    decision = await llm_supervisor.ainvoke([
        {"role": "system", "content": prompt_sistema},
        {"role": "user", "content": f"Tarea original: {pregunta_original}"},
    ])

    # Después de investigar, el supervisor solo puede analizar o finalizar.
    siguiente = "FINISH" if decision.next == "investigador" else decision.next
    return {
        "next_agent": siguiente,
        "task_completed": siguiente == "FINISH",
    }
