import os
import logging
from dotenv import load_dotenv
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

load_dotenv()
logger = logging.getLogger(__name__)

def configurar_phoenix():
    endpoint = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
    tracer_provider = register(
        project_name=os.getenv("PHOENIX_PROJECT_NAME", "ProyectoFinal"),
        endpoint=endpoint,
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    logger.info("Instrumentación Phoenix configurada para %s", endpoint)
    return tracer_provider
