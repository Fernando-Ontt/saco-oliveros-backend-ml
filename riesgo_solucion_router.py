import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from riesgo_solucion_service import RiesgoSolucionResponse, RiesgoSolucionService

DB_PATH = Path(__file__).parent / "bd_piloto.db"

router = APIRouter(prefix="/estudiantes", tags=["Riesgo y Soluciones"])


def get_riesgo_service():
    """Abre conexión por request, inyecta la clase de servicio y cierra al terminar."""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield RiesgoSolucionService(conn)
    finally:
        conn.close()


@router.get(
    "/{id_matricula}/riesgo-solucion",
    response_model=RiesgoSolucionResponse,
)
def obtener_riesgo_solucion(
    id_matricula: int,
    service: RiesgoSolucionService = Depends(get_riesgo_service),
):
    resultado = service.obtener_diagnostico_y_soluciones(id_matricula)
    if resultado is None:
        raise HTTPException(status_code=404, detail="Matrícula no encontrada")
    return resultado