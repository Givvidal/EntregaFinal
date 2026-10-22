"""Herramientas validadas para el agente investigador y el analista."""
import ast
import asyncio
import math
import operator as op
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing import Annotated
from langchain_core.tools import tool
from rag_system import RAGSystem


class RAGSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pregunta: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class CalculatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expresion: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=250)]


class RAGSearchOutput(BaseModel):
    respuesta: str
    fuentes: list[str] = Field(default_factory=list)
    cantidad_fragmentos: int = Field(ge=0)


class CalculatorOutput(BaseModel):
    exito: bool
    resultado: str | None = None
    error: str | None = None


_rag_system = None
_rag_lock = asyncio.Lock()


async def _obtener_rag_system() -> RAGSystem:
    global _rag_system
    if _rag_system is None:
        async with _rag_lock:
            if _rag_system is None:
                _rag_system = await asyncio.to_thread(RAGSystem)
    return _rag_system


@tool(args_schema=RAGSearchInput)
async def buscar_en_rag(pregunta: str) -> str:
    """Busca información en los documentos indexados y devuelve fragmentos con fuentes."""
    rag_system = await _obtener_rag_system()
    resultados = await rag_system.obtener_top_k(pregunta)
    if not resultados:
        salida = RAGSearchOutput(
            respuesta="No se encontró información relevante en el RAG.",
            fuentes=[],
            cantidad_fragmentos=0,
        )
        return salida.model_dump_json()
    texto = "\n\n".join(f"[Fuente: {r['fuente']}] {r['contenido']}" for r in resultados)
    salida = RAGSearchOutput(
        respuesta=texto,
        fuentes=sorted({str(r["fuente"]) for r in resultados}),
        cantidad_fragmentos=len(resultados),
    )
    return salida.model_dump_json()


_OPERADORES_BINARIOS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow}
_OPERADORES_UNARIOS = {ast.UAdd: op.pos, ast.USub: op.neg}


def _evaluar_nodo(nodo):
    if isinstance(nodo, ast.Constant) and type(nodo.value) in (int, float):
        if not math.isfinite(float(nodo.value)):
            raise ValueError("Solo se permiten números finitos")
        return nodo.value
    if isinstance(nodo, ast.BinOp) and type(nodo.op) in _OPERADORES_BINARIOS:
        izquierda = _evaluar_nodo(nodo.left)
        derecha = _evaluar_nodo(nodo.right)
        if isinstance(nodo.op, ast.Pow) and abs(derecha) > 100:
            raise ValueError("El exponente máximo permitido es 100")
        resultado = _OPERADORES_BINARIOS[type(nodo.op)](izquierda, derecha)
        if isinstance(resultado, (int, float)) and not math.isfinite(float(resultado)):
            raise ValueError("El resultado no es finito")
        return resultado
    if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _OPERADORES_UNARIOS:
        return _OPERADORES_UNARIOS[type(nodo.op)](_evaluar_nodo(nodo.operand))
    raise ValueError("Expresión no permitida. Usá solo números, +, -, *, /, ** y paréntesis")


@tool(args_schema=CalculatorInput)
def calculadora(expresion: str) -> str:
    """Evalúa una expresión matemática segura. Ejemplo: '5*14 + 5*21 + 2*28'."""
    try:
        arbol = ast.parse(expresion, mode="eval")
        resultado = _evaluar_nodo(arbol.body)
        salida = CalculatorOutput(exito=True, resultado=str(resultado))
        return salida.model_dump_json()
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        salida = CalculatorOutput(exito=False, error=str(exc))
        return salida.model_dump_json()
