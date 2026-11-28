from fastapi import APIRouter, Depends, HTTPException
import sqlite3

from riesgo_solucion_service import RiesgoSolucionResponse, RiesgoSolucionService

router = APIRouter(tags=["Riesgo y Soluciones"])
DB_PATH = "bd_piloto.db"


def get_riesgo_service():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield RiesgoSolucionService(conn)
    finally:
        conn.close()


@router.get(
    "/estudiantes/{id_matricula}/riesgo-solucion",
    response_model=RiesgoSolucionResponse,
)
def obtener_riesgo_solucion(
    id_matricula: int,
    service: RiesgoSolucionService = Depends(get_riesgo_service),
):
    resultado = service.obtener_diagnostico_y_soluciones(id_matricula)
    if not resultado:
        raise HTTPException(
            status_code=404,
            detail=f"Matrícula {id_matricula} no encontrada",
        )
    return resultado