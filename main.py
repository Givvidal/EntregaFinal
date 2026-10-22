"""API FastAPI: tareas multiagente, clientes LLM y cadena estructurada."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path as ApiPath
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from redis.asyncio import Redis

load_dotenv()

from app.graph import crear_app
from error_handling import es_error_recuperable, mensaje_error
from app.observability import configurar_phoenix
from app.worker import REDIS_URL, QUEUE_NAME, STATUS_PREFIX, guardar_estado, recuperar_tareas_interrumpidas, worker
from clientes.factory import crear_cliente
from chain.chain import process_text
from schemas.api import (
    AnalysisRequest,
    AnalysisResponse,
    ApprovalRequest,
    ApprovalResponse,
    ChatRequest,
    HealthResponse,
    TaskAccepted,
    TaskRequest,
    TaskStatusResponse,
    ClarificationRequest,
    ClarificationResponse,
)
from schemas.schemas import ModelResponse

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DB_PATH = Path(os.getenv("CHECKPOINT_DB_PATH", str(BASE_DIR / "checkpoints.sqlite")))

def registrar_fin_worker(task: asyncio.Task) -> None:
                if task.cancelled():
                    return

                error = task.exception()

                if error is not None:
                    logger.error(
                        "El worker terminó inesperadamente.",
                        exc_info=(type(error), error, error.__traceback__),
                    )
                else:
                    logger.error("El worker terminó sin una cancelación solicitada.")
                    
def _texto_contenido(contenido) -> str:
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        return "\n".join(
            str(parte.get("text", "")) for parte in contenido if isinstance(parte, dict)
        )
    return str(contenido)


def _payload_interrupcion(resultado: dict) -> dict:
    interrupciones = resultado.get("__interrupt__", [])

    if not interrupciones:
        return {}

    interrupcion = interrupciones[0]
    valor = getattr(interrupcion, "value", interrupcion)

    return valor if isinstance(valor, dict) else {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    worker_task = None
    try:
        await redis.ping()
        CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as checkpointer:
            app.state.redis = redis
            app.state.checkpointer = checkpointer
            app.state.graph = crear_app(checkpointer)
            recuperadas = await recuperar_tareas_interrumpidas(redis)
            if recuperadas:
                logger.warning("Se marcaron %s tareas interrumpidas como RETRYABLE.", recuperadas)
            worker_task = asyncio.create_task(worker(app), name="proyecto-final-worker")
            worker_task.add_done_callback(registrar_fin_worker)
            
            logger.info("API iniciada. Redis=cola/estados; SQLite=checkpoints (%s)", CHECKPOINT_DB_PATH)
            yield
            if worker_task:
                worker_task.cancel()
                try:
                    await worker_task
                except asyncio.CancelledError:
                    pass
    finally:
        await redis.aclose()


try:
    configurar_phoenix()
except Exception:
    logger.exception("No se pudo inicializar Phoenix. La API seguirá sin trazas visibles.")

app = FastAPI(
    title="ProyectoFinal - Sistema multiagente",
    version="1.0.0",
    description="API asíncrona con supervisor, agentes, RAG, Redis y checkpoints SQLite.",
    lifespan=lifespan,
)


@app.get("/", response_model=HealthResponse, tags=["salud"])
async def inicio() -> HealthResponse:
    return HealthResponse(estado="API funcionando")


@app.post("/tasks", response_model=TaskAccepted, status_code=202, tags=["tareas"])
async def crear_tarea(task: TaskRequest) -> TaskAccepted:
    job_id = str(uuid.uuid4())
    estado = TaskStatusResponse(
        job_id=job_id,
        status="PENDING",
        query=task.query,
        requires_approval=task.requires_approval,
    )
    try:
        async with app.state.redis.pipeline(transaction=True) as pipe:
            pipe.set(f"{STATUS_PREFIX}{job_id}", estado.model_dump_json())
            pipe.rpush(QUEUE_NAME, job_id)
            await pipe.execute()
    except Exception as exc:
        logger.exception("No se pudo encolar la tarea")
        raise HTTPException(status_code=503, detail="No se pudo encolar la tarea") from exc
    return TaskAccepted(job_id=job_id, status="PENDING")


@app.get("/tasks/{job_id}", response_model=TaskStatusResponse, tags=["tareas"])
async def consultar_tarea(
    job_id: Annotated[str, ApiPath(pattern=r"^[0-9a-fA-F-]{36}$")]
) -> TaskStatusResponse:
    datos = await app.state.redis.get(f"{STATUS_PREFIX}{job_id}")
    if datos is None:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    try:
        return TaskStatusResponse.model_validate(json.loads(datos))
    except Exception as exc:
        logger.exception("Estado inválido almacenado para %s", job_id)
        raise HTTPException(status_code=500, detail="El estado almacenado de la tarea es inválido") from exc


@app.post(
    "/tasks/{job_id}/clarify",
    response_model=ClarificationResponse,
    tags=["tareas"],
)
async def aclarar_tarea(
    job_id: Annotated[str, ApiPath(pattern=r"^[0-9a-fA-F-]{36}$")],
    aclaracion: ClarificationRequest,
) -> ClarificationResponse:
    datos = await app.state.redis.get(f"{STATUS_PREFIX}{job_id}")

    if datos is None:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")

    tarea = TaskStatusResponse.model_validate(json.loads(datos))

    if tarea.status != "WAITING_CLARIFICATION":
        raise HTTPException(
            status_code=400,
            detail="La tarea no está esperando una aclaración",
        )

    # Evita que la tarea siga figurando como pendiente mientras se reanuda.
    estado_ejecucion = tarea.model_copy(update={
        "status": "RUNNING",
        "clarification_question": None,
        "error": None,
    })
    await guardar_estado(
        app.state.redis,
        job_id,
        estado_ejecucion.model_dump(mode="json"),
    )

    try:
        resultado = await app.state.graph.ainvoke(
            Command(resume=aclaracion.answer),
            config={"configurable": {"thread_id": job_id}},
        )

        mensajes = resultado.get("messages", [])
        respuesta = (
            _texto_contenido(mensajes[-1].content)
            if mensajes
            else None
        )

        if resultado.get("__interrupt__"):
            payload = _payload_interrupcion(resultado)

            if payload.get("tipo") == "clarificacion":
                nuevo_estado = tarea.model_copy(update={
                    "status": "WAITING_CLARIFICATION",
                    "clarification_question": payload.get("pregunta"),
                    "result": respuesta,
                    "error": None,
                })
            else:
                nuevo_estado = tarea.model_copy(update={
                    "status": "WAITING_APPROVAL",
                    "clarification_question": None,
                    "result": respuesta,
                    "error": None,
                })
        else:
            nuevo_estado = tarea.model_copy(update={
                "status": "DONE",
                "clarification_question": None,
                "result": respuesta or "El grafo terminó sin respuesta.",
                "error": None,
            })

        await guardar_estado(
            app.state.redis,
            job_id,
            nuevo_estado.model_dump(mode="json"),
        )

        return ClarificationResponse(
            job_id=job_id,
            status=nuevo_estado.status,
            clarification_question=nuevo_estado.clarification_question,
            result=nuevo_estado.result,
        )

    except Exception as exc:
        logger.exception("No se pudo reanudar la tarea %s", job_id)
        recuperable = es_error_recuperable(exc)
        estado_error = tarea.model_copy(update={
            "status": "RETRYABLE" if recuperable else "FAILED",
            "error": mensaje_error(exc),
            # Conservamos la pregunta para que el estado siga siendo auditable.
            "clarification_question": tarea.clarification_question,
        })
        await guardar_estado(
            app.state.redis,
            job_id,
            estado_error.model_dump(mode="json"),
        )
        if recuperable:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Error temporal al continuar la tarea. El checkpoint se conservó.",
                    "job_id": job_id,
                    "retry_endpoint": f"/tasks/{job_id}/retry",
                    "error": mensaje_error(exc),
                },
            ) from exc
        raise HTTPException(
            status_code=500,
            detail="No se pudo reanudar la tarea con la aclaración; revisar el estado y los logs",
        ) from exc


@app.post("/tasks/{job_id}/retry", response_model=TaskStatusResponse, tags=["tareas"])
async def reintentar_tarea(
    job_id: Annotated[str, ApiPath(pattern=r"^[0-9a-fA-F-]{36}$")],
) -> TaskStatusResponse:
    """Reanuda manualmente una tarea recuperable desde el checkpoint existente.

    No vuelve a encolar la consulta original ni repite todo el grafo desde cero.
    Para errores de cuota, usar únicamente cuando el proveedor vuelva a permitir
    solicitudes.
    """
    datos = await app.state.redis.get(f"{STATUS_PREFIX}{job_id}")
    if datos is None:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    tarea = TaskStatusResponse.model_validate(json.loads(datos))
    if tarea.status != "RETRYABLE":
        raise HTTPException(
            status_code=409,
            detail=f"La tarea no está disponible para reanudación (estado: {tarea.status})",
        )

    await guardar_estado(app.state.redis, job_id, tarea.model_copy(update={
        "status": "RUNNING", "error": None,
    }).model_dump(mode="json"))
    try:
        resultado = await app.state.graph.ainvoke(
            None,
            config={"configurable": {"thread_id": job_id}},
        )
        if resultado.get("__interrupt__"):
            payload = _payload_interrupcion(resultado)
            if payload.get("tipo") == "clarificacion":
                nuevo_estado = tarea.model_copy(update={
                    "status": "WAITING_CLARIFICATION",
                    "clarification_question": payload.get("pregunta"),
                    "error": None,
                })
            else:
                nuevo_estado = tarea.model_copy(update={
                    "status": "WAITING_APPROVAL",
                    "clarification_question": None,
                    "error": None,
                })
        else:
            mensajes = resultado.get("messages", [])
            respuesta = _texto_contenido(mensajes[-1].content) if mensajes else None
            nuevo_estado = tarea.model_copy(update={
                "status": "DONE",
                "result": respuesta or tarea.result or "El grafo terminó sin respuesta.",
                "clarification_question": None,
                "error": None,
            })
        await guardar_estado(app.state.redis, job_id, nuevo_estado.model_dump(mode="json"))
        return nuevo_estado
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Falló la reanudación de la tarea %s", job_id)
        recuperable = es_error_recuperable(exc)
        estado_error = tarea.model_copy(update={
            "status": "RETRYABLE" if recuperable else "FAILED",
            "error": mensaje_error(exc),
        })
        await guardar_estado(app.state.redis, job_id, estado_error.model_dump(mode="json"))
        if recuperable:
            raise HTTPException(
                status_code=503,
                detail={"message": "La tarea sigue siendo recuperable.", "job_id": job_id,
                        "retry_endpoint": f"/tasks/{job_id}/retry", "error": mensaje_error(exc)},
            ) from exc
        raise HTTPException(status_code=500, detail="La reanudación falló por un error no recuperable") from exc


@app.post("/tasks/{job_id}/approve", response_model=ApprovalResponse, tags=["tareas"])
async def aprobar_tarea(
    job_id: Annotated[str, ApiPath(pattern=r"^[0-9a-fA-F-]{36}$")],
    decision: ApprovalRequest | None = None,
) -> ApprovalResponse:
    datos = await app.state.redis.get(f"{STATUS_PREFIX}{job_id}")
    if datos is None:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    tarea = TaskStatusResponse.model_validate(json.loads(datos))
    if tarea.status != "WAITING_APPROVAL":
        raise HTTPException(status_code=400, detail="La tarea no está esperando aprobación")

    aprobado = True if decision is None else decision.approved
    try:
        resultado = await app.state.graph.ainvoke(
            Command(resume=aprobado),
            config={"configurable": {"thread_id": job_id}},
        )
        mensajes = resultado.get("messages", [])
        respuesta = _texto_contenido(mensajes[-1].content) if mensajes else tarea.result
        nuevo_estado = tarea.model_copy(update={
            "status": "DONE" if aprobado else "REJECTED",
            "result": respuesta,
            "approved": aprobado,
            "error": None,
        })
        await guardar_estado(app.state.redis, job_id, nuevo_estado.model_dump(mode="json"))
        return ApprovalResponse(job_id=job_id, status=nuevo_estado.status, approved=aprobado)
    except Exception as exc:
        logger.exception("No se pudo reanudar la tarea %s", job_id)
        raise HTTPException(status_code=500, detail="No se pudo reanudar la tarea") from exc


@app.post("/llm/chat", response_model=ModelResponse, tags=["clientes M1"])
async def chat_llm(request: ChatRequest) -> ModelResponse:
    """Endpoint de prueba de los clientes asíncronos del Módulo 1."""
    try:
        cliente = crear_cliente(request.provider)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        return await cliente.generate(request.messages)
    finally:
        await cliente.aclose()


@app.post("/analysis", response_model=AnalysisResponse, tags=["cadena M2"])
async def analizar_texto(request: AnalysisRequest) -> AnalysisResponse:
    """Ejecuta la cadena LCEL asíncrona con salida Pydantic estructurada."""
    resultado = await process_text(request.text, provider=request.provider)
    return AnalysisResponse(resultado=resultado)
