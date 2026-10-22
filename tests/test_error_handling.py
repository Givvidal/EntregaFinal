"""Pruebas unitarias del manejo de fallos y del estado reanudable."""
import asyncio

import pytest
from error_handling import es_error_recuperable, mensaje_error


def test_gemini_429_resource_exhausted_es_recuperable():
    exc = RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
    assert es_error_recuperable(exc) is True


def test_timeout_es_recuperable():
    assert es_error_recuperable(asyncio.TimeoutError("request timed out")) is True


def test_error_de_validacion_no_se_reintenta_automaticamente():
    assert es_error_recuperable(ValueError("entrada inválida")) is False


def test_mensaje_de_cuota_indica_reanudacion_manual():
    mensaje = mensaje_error(RuntimeError("429 RESOURCE_EXHAUSTED quota"))
    assert "checkpoint" in mensaje.lower()
    assert "no se reintenta automáticamente" in mensaje.lower()
    assert "/retry" in mensaje




def test_http_503_es_recuperable():
    class ServicioNoDisponible(Exception):
        status_code = 503

    assert es_error_recuperable(ServicioNoDisponible("servicio no disponible")) is True
