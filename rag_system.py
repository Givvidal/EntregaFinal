from typing import List, Dict

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_pinecone import PineconeVectorStore

from ingest import cargar_documentos, dividir_documentos, generar_embeddings
from pinecone_setup import obtener_indice, NAMESPACE


class RAGSystem:
    """Sistema de recuperación híbrida usando BM25 + Pinecone."""

    def __init__(self, k: int = 5):
        self.k = k

        # Cargamos los documentos para BM25
        documentos = cargar_documentos()
        self.chunks = dividir_documentos(documentos)

        # Embeddings para la búsqueda semántica
        embeddings = generar_embeddings()

        # Conexión con Pinecone
        obtener_indice()

        self.vectorstore = PineconeVectorStore(
            index_name=self._obtener_nombre_indice(),
            embedding=embeddings,
            namespace=NAMESPACE,
        )

        # Retriever léxico
        self.retriever_bm25 = BM25Retriever.from_documents(self.chunks)
        self.retriever_bm25.k = k

        # Retriever semántico
        self.retriever_vectorial = self.vectorstore.as_retriever(
            search_kwargs={
                "k": k,
                "namespace": NAMESPACE,
            }
        )

        # Retriever híbrido
        self.retriever_hibrido = EnsembleRetriever(
            retrievers=[
                self.retriever_bm25,
                self.retriever_vectorial,
            ],
            weights=[0.5, 0.5],
        )

    def _obtener_nombre_indice(self):
        """Obtiene el nombre del índice configurado."""

        from config import INDEX_NAME

        return INDEX_NAME

    async def obtener_top_k(self, query: str) -> List[Dict]:
        """Devuelve los k documentos más relevantes."""

        documentos = await self.retriever_hibrido.ainvoke(query)

        documentos = documentos[: self.k]

        return [
            {
                "contenido": documento.page_content,
                "fuente": documento.metadata.get("source", "desconocida"),
                "documento_id": documento.metadata.get(
                    "documento_id", documento.metadata.get("source", "desconocido")
                ),
                "categoria": documento.metadata.get("categoria", "desconocida"),
                "chunk_id": documento.metadata.get("chunk_id", "desconocido"),
            }
            for documento in documentos
        ]
