"""Endpoints de secciones, leídos desde bd_piloto.db."""
import sqlite3
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(tags=["secciones"])

DB_PATH = Path(__file__).parent / "bd_piloto.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@router.get("/secciones")
def listar_secciones():
    """Las secciones con su tutor y cantidad de estudiantes."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT s.id_seccion, s.grado, s.letra,
                   s.grado || s.letra AS codigo,
                   d.nombres || ' ' || d.apellidos AS tutor,
                   COUNT(m.id_matricula) AS total_estudiantes
            FROM seccion s
            LEFT JOIN docente d ON d.id_docente = s.id_tutor
            LEFT JOIN matricula m ON m.id_seccion = s.id_seccion
            GROUP BY s.id_seccion
            ORDER BY s.grado, s.letra
        """).fetchall()
    return [dict(r) for r in rows]


@router.get("/matriculas-seccion")
def matriculas_seccion(seccion: str | None = None):
    """Mapa {id_matricula: "3A"}. Con ?seccion=A filtra por letra."""
    query = """
        SELECT m.id_matricula, s.grado || s.letra AS seccion
        FROM matricula m
        JOIN seccion s ON s.id_seccion = m.id_seccion
    """
    params: tuple = ()
    if seccion:
        query += " WHERE s.letra = ?"
        params = (seccion.upper(),)
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return {str(r["id_matricula"]): r["seccion"] for r in rows}
