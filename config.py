"""Configuración: no falla al importar para permitir tests sin credenciales."""
import os
from dotenv import load_dotenv
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "videojuegos-rag")


def validar_configuracion_rag() -> None:
    faltantes = [nombre for nombre, valor in (("PINECONE_API_KEY", PINECONE_API_KEY), ("GOOGLE_API_KEY", GOOGLE_API_KEY)) if not valor]
    if faltantes:
        raise RuntimeError("Faltan variables de entorno: " + ", ".join(faltantes) + ". Revisá el archivo .env.")
