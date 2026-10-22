from abc import ABC, abstractmethod
from typing import AsyncGenerator, List
from schemas.schemas import ChatMessage, ModelResponse


class BaseLLMClient(ABC):
    """Contrato asíncrono común para los clientes de proveedores LLM."""

    @abstractmethod
    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        raise NotImplementedError

    @abstractmethod
    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        raise NotImplementedError
        yield  # pragma: no cover

    async def aclose(self) -> None:
        """Cierra el cliente subyacente si el SDK ofrece un método de cierre."""
        import inspect
        cliente = getattr(self, "_client", None)
        if cliente is None:
            return
        async_client = getattr(cliente, "aio", None)
        close = getattr(async_client, "aclose", None) or getattr(async_client, "close", None)
        if close is None:
            close = getattr(cliente, "aclose", None) or getattr(cliente, "close", None)
        if close is not None:
            resultado = close()
            if inspect.isawaitable(resultado):
                await resultado
