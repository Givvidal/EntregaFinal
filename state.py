
import operator
from typing import Annotated, Optional, List, Dict

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    next_agent: Optional[str]
    contribuciones: Annotated[List[Dict[str, str]], operator.add]
    pasos: int
    task_completed: bool
    requires_approval: bool
    job_id: str
    aprobado: bool

    # Aclaración humana
    requiere_aclaracion: bool
    pregunta_aclaracion: Optional[str]
    respuesta_aclaracion: Optional[str]


def obtener_consulta(state: AgentState) -> str:
    """Combina la consulta original con la aclaración humana, si existe."""
    mensajes = state.get("messages", [])
    original = mensajes[0].content if mensajes else ""

    if not isinstance(original, str):
        original = str(original)

    aclaracion = state.get("respuesta_aclaracion")

    if aclaracion and aclaracion.strip():
        return (
            f"{original}\n\n"
            f"Aclaración del usuario:\n{aclaracion.strip()}"
        )

    return original