import json
import pytest
from pydantic import ValidationError
from schemas.api import TaskRequest, ChatRequest, AnalysisRequest
from schemas.schemas import ChatMessage, Provider
from tools import calculadora


def test_consulta_vacia_es_rechazada():
    with pytest.raises(ValidationError):
        TaskRequest(query="   ")


def test_consulta_demasiado_larga_es_rechazada():
    with pytest.raises(ValidationError):
        TaskRequest(query="x" * 9000)


def test_chat_exige_al_menos_un_mensaje():
    with pytest.raises(ValidationError):
        ChatRequest(provider=Provider.GEMINI, messages=[])


def test_analysis_rechaza_texto_vacio():
    with pytest.raises(ValidationError):
        AnalysisRequest(text="   ")


def test_calculadora_evalua_expresion_valida():
    resultado = json.loads(calculadora.invoke({"expresion": "5*14 + 1"}))
    assert resultado["exito"] is True
    assert resultado["resultado"] == "71"


def test_calculadora_rechaza_llamada_a_funcion():
    salida = json.loads(calculadora.invoke({"expresion": "__import__('os').system('whoami')"}))
    assert salida["exito"] is False
    assert salida["error"]
