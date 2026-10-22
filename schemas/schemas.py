from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class ModelResponse(BaseModel):
    provider: Provider
    model: str
    content: str
    error: Optional[str] = None
