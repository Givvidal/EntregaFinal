# OpenAIClient
from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError
from clientes.baseClient import BaseLLMClient
from schemas.schemas import ChatMessage,ModelResponse,Provider
from typing import List,AsyncGenerator

class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[m.model_dump() for m in messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return ModelResponse(
                provider=Provider.OPENAI,
                model=self.model,
                content=response.choices[0].message.content or "",
            )
        except RateLimitError as e:
            return ModelResponse(
                provider=Provider.OPENAI,
                model=self.model,
                content="",
                error=f"Límite de cuota excedido: {e}",
            )
        except APIConnectionError as e:
            return ModelResponse(
                provider=Provider.OPENAI,
                model=self.model,
                content="",
                error=f"Error de conexión: {e}",
            )
        except APIError as e:
            return ModelResponse(
                provider=Provider.OPENAI,
                model=self.model,
                content="",
                error=f"Error de la API de OpenAI: {e}",
            )

    async def generate_stream(
        self, messages: List[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=[m.model_dump() for m in messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except (RateLimitError, APIConnectionError, APIError) as e:
            yield f"\n[⚠️ Error durante el streaming: {e}]"