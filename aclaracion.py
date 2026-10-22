
import os

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import interrupt

from state import AgentState


class DecisionAclaracion(BaseModel):
    necesita_aclaracion: bool = Field(
        description=(
            "True solamente si falta información esencial para "
            "responder correctamente; False si la consulta es respondible."
        )
    )
    pregunta: str = Field(
        description=(
            "Pregunta breve y concreta para resolver la ambigüedad. "
            "Si no hace falta aclaración, dejar vacía."
        )
    )


PROMPT_ACLARACION = """
Sos un evaluador de consultas para un sistema de investigación multiagente.

Tu tarea es determinar si se puede investigar y responder la consulta
sin adivinar información esencial.

Pedí una aclaración cuando:
- La consulta se refiere a "eso", "esa situación", "lo anterior" u otro
  referente que no está identificado en el texto disponible.
- Falta identificar el tema, objeto o caso sobre el que se pregunta.
- Hay varias interpretaciones materialmente diferentes y elegir una
  cambiaría la respuesta.

NO pidas aclaración cuando:
- La consulta es general pero se puede responder de forma útil.
- Solo hay errores ortográficos que no impiden entenderla.
- La respuesta puede explicar razonablemente el tema sin más datos.

Si necesitás aclaración, hacé UNA pregunta breve que permita continuar.
No respondas la consulta ni inventes el contexto faltante.
"""


llm_aclaracion = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
    temperature=0,
).with_structured_output(DecisionAclaracion)


async def nodo_evaluar_aclaracion(state: AgentState) -> dict:
    mensajes = state.get("messages", [])
    consulta = str(mensajes[0].content) if mensajes else ""

    decision = await llm_aclaracion.ainvoke([
        {"role": "system", "content": PROMPT_ACLARACION},
        {"role": "user", "content": f"Consulta a evaluar:\n{consulta}"},
    ])

    pregunta = decision.pregunta.strip()

    # Evita quedar bloqueado si el modelo devuelve una decisión incompleta.
    necesita = decision.necesita_aclaracion and bool(pregunta)

    return {
        "requiere_aclaracion": necesita,
        "pregunta_aclaracion": pregunta if necesita else None,
    }


def decidir_aclaracion(state: AgentState) -> str:
    if state.get("requiere_aclaracion", False):
        return "solicitar_aclaracion"

    return "supervisor"


async def nodo_solicitar_aclaracion(state: AgentState) -> dict:
    respuesta = interrupt({
        "tipo": "clarificacion",
        "job_id": state.get("job_id"),
        "pregunta": state.get("pregunta_aclaracion"),
    })

    if not isinstance(respuesta, str) or not respuesta.strip():
        raise ValueError("La aclaración recibida está vacía.")

    return {
        "respuesta_aclaracion": respuesta.strip(),
        "requiere_aclaracion": False,
    }
