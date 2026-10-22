from langchain_core.messages import AIMessage,ToolMessage

def extraer_texto(mensaje) -> str:
    """Normaliza el content de un mensaje (Gemini a veces devuelve una lista de bloques)."""
    contenido = mensaje.content
    if isinstance(contenido,str):
        return contenido
    if isinstance(contenido,list):
        return "".join(b.get("text", "") for b in contenido if isinstance(b,dict))
    return str(contenido)

def serializar_traza(mensajes) -> list:
    """Convierte los mensajes de LangGraph a un formato JSON simple, para dejar
    la traza ReAct como log dentro del repositorio"""
    traza = []
    for m in mensajes:
        entrada = {"tipo":type(m).__name__,"contenido": extraer_texto(m)}
        if isinstance(m, AIMessage) and m.tool_calls:
            entrada["tool_calls"] = [
                {"nombre": tc["name"],"argumentos":tc["args"]} for tc in m.tool_calls
            ]
        if isinstance(m, ToolMessage):
            entrada["herramienta"] = m.name
        traza.append(entrada)
    return traza