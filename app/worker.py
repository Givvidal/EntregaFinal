"""Worker asíncrono que procesa la cola Redis y ejecuta LangGraph."""
import asyncio
import json
import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from redis.asyncio import Redis
from schemas.api import TaskStatusResponse
from utils import serializar_traza
from error_handling import es_error_recuperable, mensaje_error

load_dotenv()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.getenv("REDIS_QUEUE_NAME", "tasks_queue")
STATUS_PREFIX = os.getenv("REDIS_STATUS_PREFIX", "task:")
logger = logging.getLogger(__name__)


def _texto(contenido) -> str:
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        return "\n".join(str(x.get("text", "")) for x in contenido if isinstance(x, dict))
    return str(contenido)


def _payload_interrupcion(resultado: dict) -> dict:
    interrupciones = resultado.get("__interrupt__", [])

    if not interrupciones:
        return {}

    interrupcion = interrupciones[0]
    valor = getattr(interrupcion, "value", interrupcion)

    return valor if isinstance(valor, dict) else {}


async def guardar_estado(redis: Redis, job_id: str, estado: dict) -> None:
    # Validar antes de persistir evita estados incompletos o inconsistentes.
    validado = TaskStatusResponse.model_validate(estado)
    await redis.set(f"{STATUS_PREFIX}{job_id}", validado.model_dump_json())




async def recuperar_tareas_interrumpidas(redis: Redis) -> int:
    """Marca RUNNING antiguos como reanudables tras reiniciar el proceso.

    Solo se llama durante el arranque, antes de crear el worker nuevo.
    La ejecución se reanuda manualmente con el endpoint /retry.
    """
    recuperadas = 0
    patron = f"{STATUS_PREFIX}*"
    async for clave in redis.scan_iter(match=patron):
        datos = await redis.get(clave)
        if not datos:
            continue
        try:
            tarea = TaskStatusResponse.model_validate_json(datos)
        except Exception:
            logger.exception("No se pudo recuperar el estado almacenado %s", clave)
            continue
        if tarea.status != "RUNNING":
            continue
        estado = tarea.model_copy(update={
            "status": "RETRYABLE",
            "error": (
                "La ejecución anterior se interrumpió durante un reinicio. "
                "El checkpoint se conserva; revisá la tarea y usá /retry para reanudarla."
            ),
        })
        await redis.set(clave, estado.model_dump_json())
        recuperadas += 1
        logger.warning("Tarea interrumpida recuperable: %s", tarea.job_id)
    return recuperadas


async def worker(app: FastAPI) -> None:
    redis: Redis = app.state.redis
    graph = app.state.graph
    logger.info("Worker iniciado; cola Redis: %s", QUEUE_NAME)
    while True:
        try:
            elemento = await redis.blpop(QUEUE_NAME, timeout=1)

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception(
                "Error consumiendo la cola Redis '%s'. Se reintentará.",
                QUEUE_NAME,
            )
            await asyncio.sleep(1)
            continue

        if elemento is None:
            pendientes = await redis.llen(QUEUE_NAME)

            if pendientes > 0:
                logger.warning(
                    "BLPOP devolvió None, pero la cola '%s' tiene %s tareas pendientes.",
                    QUEUE_NAME,
                    pendientes,
                )
            continue

        logger.info("Worker recibió una tarea de Redis: %s", elemento)
        _, job_id = elemento
        datos = await redis.get(f"{STATUS_PREFIX}{job_id}")
        if datos is None:
            logger.warning("Job %s descartado: no tiene estado", job_id)
            continue
        try:
            tarea = TaskStatusResponse.model_validate(json.loads(datos))
        except Exception:
            logger.exception("Estado inválido para job %s", job_id)
            continue
        if tarea.status != "PENDING":
            logger.warning("Job %s descartado con estado %s", job_id, tarea.status)
            continue
        await guardar_estado(redis, job_id, tarea.model_copy(update={"status": "RUNNING"}).model_dump(mode="json"))
        try:
            resultado = await graph.ainvoke(
                {
                    "job_id": job_id,
                    "messages": [HumanMessage(content=tarea.query)],
                    "requires_approval": tarea.requires_approval,
                    "contribuciones": [],
                    "pasos": 0,
                    "task_completed": False,
                },
                config={"configurable": {"thread_id": job_id}},
            )
            logger.info("Traza de mensajes para job %s: %s", job_id, json.dumps(serializar_traza(resultado.get("messages", [])), ensure_ascii=False))
            
            if resultado.get("__interrupt__"):
                payload = _payload_interrupcion(resultado)
                tipo = payload.get("tipo")

                if tipo == "clarificacion":
                    estado = tarea.model_copy(update={
                        "status": "WAITING_CLARIFICATION",
                        "clarification_question": payload.get("pregunta"),
                        "error": None,
                    })
                else:
                    estado = tarea.model_copy(update={
                        "status": "WAITING_APPROVAL",
                        "clarification_question": None,
                        "error": None,
                    })
            else:
                mensajes = resultado.get("messages", [])
                respuesta = (
                    _texto(mensajes[-1].content)
                    if mensajes
                    else "El grafo terminó sin respuesta."
                )
                estado = tarea.model_copy(update={
                    "status": "DONE",
                    "result": respuesta,
                    "clarification_question": None,
                    "error": None,
                })

            await guardar_estado(redis, job_id, estado.model_dump(mode="json"))
        except asyncio.CancelledError:
            # Evita marcar una tarea como fallida por el apagado intencional del servidor.
            raise
        except Exception as exc:
            logger.exception("Falló la tarea %s", job_id)
            recuperable = es_error_recuperable(exc)
            estado_error = tarea.model_copy(update={
                "status": "RETRYABLE" if recuperable else "FAILED",
                "error": mensaje_error(exc),
            })
            await guardar_estado(redis, job_id, estado_error.model_dump(mode="json"))
            if recuperable:
                logger.warning(
                    "La tarea %s conserva su checkpoint y puede reanudarse manualmente.",
                    job_id,
                )
