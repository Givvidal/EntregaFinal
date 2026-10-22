import logging
import os
from dotenv import load_dotenv

load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from schemas.entidades import AnalisisVideojuego


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Sos un analista especializado en videojuegos. "
        "Analizá el texto proporcionado por el usuario e identificá "
        "las mecánicas, elementos o conceptos relacionados con el juego. "
        "Determiná qué tipo de contenido predomina y generá un resumen breve."
    ),
    ("human", "{texto}"),
])


def get_model(provider: str):

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Falta configurar OPENAI_API_KEY en el archivo .env")
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            api_key=api_key,
        )

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Falta configurar ANTHROPIC_API_KEY en el archivo .env")
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            temperature=0,
            api_key=api_key,
        )

    elif provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Falta configurar GOOGLE_API_KEY en el archivo .env")
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
            temperature=1.0,
            google_api_key=api_key,
        )

    raise ValueError(f"Proveedor no soportado: {provider}")


def build_chain(provider: str = "openai"):

    model = get_model(provider)

    structured_model = model.with_structured_output(
        AnalisisVideojuego
    )

    chain = (prompt | structured_model).with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

    return chain


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline_extraccion")


async def process_text(
    text: str,
    provider: str = "openai"
) -> AnalisisVideojuego:

    chain = build_chain(provider)

    logger.info(
        f"[{provider}] Procesando texto ({len(text)} caracteres)..."
    )

    try:
        resultado = await chain.ainvoke({"texto": text})

        logger.info(
            f"[{provider}] Extracción validada: "
            f"{resultado.model_dump()}"
        )

        return resultado

    except Exception as e:
        logger.error(
            f"[{provider}] Falló tras reintentos: {e}"
        )
        raise