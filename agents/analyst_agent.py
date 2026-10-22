import os
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import calculadora


llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
    temperature=0,
)

agente_analista = create_agent(
    model=llm,
    tools=[calculadora],
    system_prompt=(
        "Sos un agente de análisis matemático. Tu única función es realizar "
        "cálculos utilizando exclusivamente la herramienta calculadora a "
        "partir de los datos proporcionados por otros agentes. "
        "No busques información nueva ni inventes datos. "
        "Si faltan datos necesarios para calcular, indicá cuáles faltan."
    ),
)