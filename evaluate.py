import asyncio
import json
from typing import List, Dict

from rag_system import RAGSystem


def cargar_golden_set(ruta: str = "golden_set.json") -> List[Dict]:
    """Carga las preguntas y documentos esperados del Golden Set."""

    with open(ruta, "r", encoding="utf-8") as archivo:
        return json.load(archivo)


async def evaluar(rag_system: RAGSystem, golden_set: List[Dict]) -> Dict:
    """Calcula Recall@5 y Precision@5."""

    resultados = []

    for caso in golden_set:

        pregunta = caso["pregunta"]
        documento_esperado = caso["documento_id_esperado"]

        top_k = await rag_system.obtener_top_k(pregunta)

        fuentes_recuperadas = [resultado["documento_id"] for resultado in top_k]

        # Recall@5
        recall = 1.0 if documento_esperado in fuentes_recuperadas else 0.0

        # Precision@5
        coincidencias = sum(
            1 for fuente in fuentes_recuperadas if fuente == documento_esperado
        )

        precision = (
            coincidencias / len(fuentes_recuperadas) if fuentes_recuperadas else 0.0
        )

        resultados.append(
            {
                "pregunta": pregunta,
                "esperado": documento_esperado,
                "recuperados": fuentes_recuperadas,
                "recall@5": recall,
                "precision@5": precision,
            }
        )

    if not resultados:
        return {"detalle": [], "recall@5_promedio": 0.0, "precision@5_promedio": 0.0}

    recall_promedio = sum(r["recall@5"] for r in resultados) / len(resultados)

    precision_promedio = sum(r["precision@5"] for r in resultados) / len(resultados)

    return {
        "detalle": resultados,
        "recall@5_promedio": recall_promedio,
        "precision@5_promedio": precision_promedio,
    }


def mostrar_reporte(reporte: Dict) -> None:
    """Muestra los resultados de la evaluación."""

    print("=" * 80)
    print("📊 EVALUACIÓN DEL SISTEMA RAG")
    print("=" * 80)

    for resultado in reporte["detalle"]:

        estado = "✅" if resultado["recall@5"] == 1.0 else "❌"

        print(f"\n{estado} {resultado['pregunta']}")
        print(f"   Esperado: {resultado['esperado']}")
        print(f"   Recuperados: {resultado['recuperados']}")
        print(f"   Recall@5: " f"{resultado['recall@5']:.0%}")
        print(f"   Precision@5: " f"{resultado['precision@5']:.0%}")

    print("\n" + "=" * 80)

    print(f"📈 RECALL@5 PROMEDIO: " f"{reporte['recall@5_promedio']:.1%}")

    print(f"📈 PRECISION@5 PROMEDIO: " f"{reporte['precision@5_promedio']:.1%}")


if __name__ == "__main__":

    rag_system = RAGSystem(k=5)

    golden_set = cargar_golden_set()

    reporte = asyncio.run(evaluar(rag_system, golden_set))

    mostrar_reporte(reporte)
