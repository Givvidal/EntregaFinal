"""Interfaz compatible para consultar el RAG híbrido utilizado por los agentes."""
from pydantic import BaseModel
from rag_system import RAGSystem


class RAGResponse(BaseModel):
    respuesta: str
    fuentes: list[str]
    fragmentos_recuperados: int


async def get_rag_response(query: str) -> RAGResponse:
    rag = RAGSystem()
    resultados = await rag.obtener_top_k(query)
    if not resultados:
        return RAGResponse(
            respuesta="No encontré información relevante en los documentos disponibles.",
            fuentes=[],
            fragmentos_recuperados=0,
        )
    contexto = "\n\n".join(
        f"[Fuente: {item['fuente']}]\n{item['contenido']}" for item in resultados
    )
    return RAGResponse(
        respuesta=contexto,
        fuentes=sorted({item["fuente"] for item in resultados}),
        fragmentos_recuperados=len(resultados),
    )
