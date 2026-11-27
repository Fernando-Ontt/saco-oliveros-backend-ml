from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class PrediccionRequest(BaseModel):
    id_matricula: int


class PrediccionResponse(BaseModel):
    id_matricula: int
    probabilidad: float
    nivel: str
    umbral: float
    algoritmo: str
    factores_principales: list[str]


class EvidenciaCreate(BaseModel):
    id_docente: Optional[int] = None
    fecha: Optional[str] = None
    tipo: Literal["Positiva", "Leve", "Grave", "Socioemocional"]
    deriva_a_tutor: bool = False


class TutoriaCreate(BaseModel):
    id_docente: Optional[int] = None
    fecha: Optional[str] = None
    motivo: str = Field(..., min_length=1)
    accion: str = Field(..., min_length=1)
    derivacion: Literal["Ninguna", "Psicologia", "Coordinacion"] = "Ninguna"


class DerivacionCreate(BaseModel):
    id_matricula: int
    id_docente: int
    id_curso: Optional[int] = None
    id_observacion: Optional[int] = None
    fecha: Optional[str] = None
    motivo: str = Field(..., min_length=1)
    id_tutor: Optional[int] = None


class DerivacionUpdate(BaseModel):
    estado: Optional[Literal["Pendiente", "Revisada", "Atendida"]] = None
    id_tutor: Optional[int] = None
    fecha_revision: Optional[str] = None
    accion_tutor: Optional[str] = None


class ContactoCreate(BaseModel):
    id_apoderado: Optional[int] = None
    fecha: Optional[str] = None
    tipo: Literal["Llamada", "Citacion", "Mensaje"]
    respondio: bool = False


class SimulacionRequest(BaseModel):
    cambios: dict[str, Any] = Field(..., min_length=1)


class RecalcularRiesgoRequest(BaseModel):
    id_periodo: Optional[int] = None


class PrediccionLoteRequest(BaseModel):
    anio: Optional[int] = None
    grado: Optional[int] = Field(default=None, ge=1, le=5)
    id_seccion: Optional[int] = None
    id_periodo: Optional[int] = None


class PrediccionManualRequest(BaseModel):
    variables: dict[str, Any] = Field(..., min_length=1)
