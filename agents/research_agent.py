import os
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import buscar_en_rag


llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-flash-latest")
)

agente_investigador = create_agent(
    model=llm,
    tools=[buscar_en_rag],
    system_prompt=(
        "Sos un agente de investigación. "
        "Tu única función es buscar información en el RAG usando "
        "exclusivamente la herramienta buscar_en_rag. "
        "No respondas la pregunta original. "
        "No hagas cálculos, promedios, diferencias, porcentajes ni "
        "comparaciones numéricas, aunque puedas resolverlos. "
        "Devolvé únicamente los datos encontrados y sus fuentes."
    ),
)