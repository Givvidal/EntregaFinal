"""Contratos Pydantic de entrada y salida de la API."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing import Annotated
from schemas.schemas import ChatMessage, ModelResponse, Provider
from schemas.entidades import AnalisisVideojuego

Consulta = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
TextoAnalisis = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50000)]


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: Consulta
    requires_approval: bool = False


class TaskAccepted(BaseModel):
    job_id: str
    status: Literal["PENDING"]



class TaskStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: Literal[
        "PENDING",
        "RUNNING",
        "RETRYABLE",
        "WAITING_CLARIFICATION",
        "WAITING_APPROVAL",
        "DONE",
        "FAILED",
        "REJECTED",
    ]
    query: Consulta
    requires_approval: bool = False
    result: str | None = None
    error: str | None = None
    approved: bool | None = None
    clarification_question: str | None = None


class ClarificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: Consulta


class ClarificationResponse(BaseModel):
    job_id: str
    status: Literal[
        "WAITING_CLARIFICATION",
        "WAITING_APPROVAL",
        "DONE",
        "RETRYABLE",
        "FAILED",
    ]
    clarification_question: str | None = None
    result: str | None = None



class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved: bool = True


class ApprovalResponse(BaseModel):
    job_id: str
    status: Literal["DONE", "REJECTED"]
    approved: bool


class HealthResponse(BaseModel):
    estado: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Provider
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: TextoAnalisis
    provider: Provider = Provider.GEMINI


class AnalysisResponse(BaseModel):
    resultado: AnalisisVideojuego
