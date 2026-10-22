"""Envía una traza sintética a Phoenix para verificar el colector OTLP.

Requisitos: Phoenix escuchando en localhost:6006 y dependencias del proyecto.
Después de ejecutarlo, buscar el span `verificacion_phoenix_proyecto_final`
 en el proyecto `ProyectoFinal` de la interfaz web.
"""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv
from opentelemetry import trace
from phoenix.otel import register

load_dotenv()
endpoint = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
project = os.getenv("PHOENIX_PROJECT_NAME", "ProyectoFinal")
ui_url = endpoint.removesuffix("/v1/traces")

try:
    request = urllib.request.Request(ui_url, method="GET")
    with urllib.request.urlopen(request, timeout=5) as response:
        if response.status >= 400:
            raise RuntimeError(f"Phoenix respondió HTTP {response.status}")
except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
    print(f"ERROR: Phoenix no responde en {ui_url}: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    provider = register(project_name=project, endpoint=endpoint)
    tracer = trace.get_tracer("proyecto-final.verificacion")
    with tracer.start_as_current_span("verificacion_phoenix_proyecto_final") as span:
        span.set_attribute("test.tipo", "observabilidad")
        span.set_attribute("test.proyecto", "ProyectoFinal")
        span.set_attribute("test.resultado", "span_emitido")
    if hasattr(provider, "force_flush"):
        provider.force_flush(timeout_millis=10000)
except Exception as exc:
    print(f"ERROR: no se pudo emitir la traza: {exc}", file=sys.stderr)
    sys.exit(1)

print("Traza de verificación emitida.")
print(f"Phoenix UI: {ui_url}")
print(f"Proyecto: {project}")
print("Busca el span: verificacion_phoenix_proyecto_final")
print("Importante: confirma visualmente en Phoenix que el span aparece; el envío por sí solo no demuestra que quedó almacenado.")
