from fastapi import APIRouter, HTTPException
import sqlite3

from tutoria_seguimiento_service import (
    TutoriaCreateRequest,
    TutoriaItemResponse,
    TutoriaRegistroResponse,
    TutoriaSeguimientoService,
)

router = APIRouter(tags=["Tutorías y Seguimiento"])
DB_PATH = "bd_piloto.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _verificar_matricula(conn: sqlite3.Connection, id_matricula: int) -> None:
    row = conn.execute(
        "SELECT 1 FROM matricula WHERE id_matricula = ?", (id_matricula,)
    ).fetchone()
    if not row:
        raise HTTPException(
            status_code=404, detail=f"Matrícula {id_matricula} no encontrada"
        )


@router.get(
    "/estudiantes/{id_matricula}/tutorias",
    response_model=list[TutoriaItemResponse],
)
def obtener_tutorias_estudiante(id_matricula: int):
    conn = get_db()
    try:
        _verificar_matricula(conn, id_matricula)
        service = TutoriaSeguimientoService(conn)
        return service.listar_por_estudiante(id_matricula)
    finally:
        conn.close()


@router.post(
    "/estudiantes/{id_matricula}/tutorias",
    response_model=TutoriaRegistroResponse,
)
def crear_tutoria_estudiante(id_matricula: int, payload: TutoriaCreateRequest):
    conn = get_db()
    try:
        _verificar_matricula(conn, id_matricula)
        service = TutoriaSeguimientoService(conn)
        return service.registrar_seguimiento(id_matricula, payload)
    finally:
        conn.close()