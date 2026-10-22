from pydantic import BaseModel, Field


class AnalisisVideojuego(BaseModel):
    mecanicas: list[str] = Field(default_factory=list, description="Mecánicas identificadas en el texto")
    elementos: list[str] = Field(default_factory=list, description="Elementos o conceptos relacionados con el juego")
    tipo_contenido: str = Field(description="Tipo de contenido predominante")
    resumen: str = Field(description="Resumen breve del texto analizado")
