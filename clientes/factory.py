"""Factory validada para los clientes asíncronos del Módulo 1."""
import os
from schemas.schemas import Provider
from clientes.openAIClient import OpenAIClient
from clientes.anthropicClient import AnthropicClient
from clientes.geminiClient import GeminiClient


def _api_key(nombre: str) -> str:
    valor = os.getenv(nombre, "").strip()
    if not valor:
        raise ValueError(f"Falta configurar {nombre} en el archivo .env")
    return valor


def crear_cliente(provider: Provider):
    provider = Provider(provider)
    if provider == Provider.OPENAI:
        return OpenAIClient(_api_key("OPENAI_API_KEY"), os.getenv("OPENAI_MODEL", "gpt-4o-mini"), 0.2, 1024)
    if provider == Provider.ANTHROPIC:
        return AnthropicClient(_api_key("ANTHROPIC_API_KEY"), os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"), 0.2, 1024)
    if provider == Provider.GEMINI:
        return GeminiClient(_api_key("GOOGLE_API_KEY"), os.getenv("GEMINI_MODEL", "gemini-flash-latest"), 0.2, 1024)
    raise ValueError(f"Proveedor no soportado: {provider}")
