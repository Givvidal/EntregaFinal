"""Clasificación conservadora de errores para tareas LangGraph."""
from __future__ import annotations

import asyncio


def es_error_recuperable(exc: BaseException) -> bool:
    """True si tiene sentido permitir una reanudación manual desde checkpoint.

    No reintenta automáticamente: evita consumir cuota en bucle cuando Gemini
    devuelve 429/RESOURCE_EXHAUSTED.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return True

    partes = [str(exc), type(exc).__name__]
    for attr in ("status_code", "code"):
        valor = getattr(exc, attr, None)
        if valor is not None:
            partes.append(str(valor))
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if status is not None:
            partes.append(str(status))

    texto = " ".join(partes).lower()
    marcadores = (
        "429", "resource_exhausted", "rate limit", "ratelimit",
        "quota exceeded", "quota_failure", "too many requests",
        "503", "502", "504", "service unavailable", "bad gateway",
        "gateway timeout", "temporarily unavailable", "temporary failure",
        "connection reset", "connection refused", "connection error",
        "timeout", "timed out",
    )
    return any(marcador in texto for marcador in marcadores)


def mensaje_error(exc: BaseException) -> str:
    """Mensaje útil para el usuario sin ocultar la causa del proveedor."""
    texto = str(exc).strip() or type(exc).__name__
    normalizado = texto.lower()
    if "resource_exhausted" in normalizado or "quota" in normalizado or "429" in normalizado:
        return (
            f"Cuota o límite del proveedor agotado: {texto}. "
            "La tarea conserva su checkpoint; esperá a que se restablezca la cuota "
            "y luego usa POST /tasks/{job_id}/retry. No se reintenta automáticamente."
        )
    return texto
