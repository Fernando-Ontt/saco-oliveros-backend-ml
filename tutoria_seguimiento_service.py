from datetime import date
import sqlite3
from typing import Optional
from pydantic import BaseModel, Field


class TutoriaCreateRequest(BaseModel):
    fecha: Optional[str] = None
    motivo: str = Field(..., min_length=2)
    accion: str = Field(..., min_length=2)
    derivacion: str = "Ninguna"
    nota: Optional[str] = None
    tutor: Optional[str] = "Tutor"
    mejoro_sim: Optional[bool] = None
    id_derivacion: Optional[int] = None


class TutoriaItemResponse(BaseModel):
    id_tutoria: Optional[int] = None
    id_matricula: int
    fecha: str
    motivo: str
    accion: str
    derivacion: str
    mejoro_sim: Optional[bool] = None
    nota: Optional[str] = None
    tutor: Optional[str] = None


class TutoriaRegistroResponse(BaseModel):
    ok: bool
    duplicado_hoy: bool = False
    mensaje: str
    registro: TutoriaItemResponse
    ya_aplicadas: list[TutoriaItemResponse]


class TutoriaSeguimientoService:
    """Aísla la persistencia y consulta de acciones de tutoría por estudiante en SQLite."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        self._asegurar_esquema()

    def _columnas_tabla(self, tabla: str) -> set[str]:
        rows = self.conn.execute(f"PRAGMA table_info({tabla})").fetchall()
        return {r["name"] for r in rows}

    def _asegurar_esquema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tutoria (
                id_tutoria INTEGER PRIMARY KEY AUTOINCREMENT,
                id_matricula INTEGER NOT NULL,
                fecha TEXT NOT NULL,
                motivo TEXT NOT NULL,
                accion TEXT NOT NULL,
                derivacion TEXT DEFAULT 'Ninguna',
                mejoro_sim INTEGER,
                nota TEXT,
                tutor TEXT
            )
            """
        )
        # Si la tabla ya existía de bd_piloto.db con menos columnas, agrega las faltantes sin romper datos
        cols = self._columnas_tabla("tutoria")
        columnas_extra = {
            "derivacion": "TEXT DEFAULT 'Ninguna'",
            "mejoro_sim": "INTEGER",
            "nota": "TEXT",
            "tutor": "TEXT",
        }
        for col, tipo in columnas_extra.items():
            if col not in cols:
                self.conn.execute(f"ALTER TABLE tutoria ADD COLUMN {col} {tipo}")
        self.conn.commit()

    def listar_por_estudiante(self, id_matricula: int) -> list[TutoriaItemResponse]:
        cols = self._columnas_tabla("tutoria")
        pk_col = "id_tutoria" if "id_tutoria" in cols else "rowid"

        rows = self.conn.execute(
            f"""
            SELECT {pk_col} AS id_tutoria, id_matricula, fecha, motivo, accion,
                   COALESCE(derivacion, 'Ninguna') AS derivacion,
                   mejoro_sim, nota, tutor
            FROM tutoria
            WHERE id_matricula = ?
            ORDER BY fecha DESC, {pk_col} DESC
            """,
            (id_matricula,),
        ).fetchall()

        resultado: list[TutoriaItemResponse] = []
        for r in rows:
            raw_mejoro = r["mejoro_sim"]
            mejoro_bool = None if raw_mejoro is None else bool(int(raw_mejoro))
            resultado.append(
                TutoriaItemResponse(
                    id_tutoria=r["id_tutoria"],
                    id_matricula=r["id_matricula"],
                    fecha=r["fecha"],
                    motivo=r["motivo"],
                    accion=r["accion"],
                    derivacion=r["derivacion"],
                    mejoro_sim=mejoro_bool,
                    nota=r["nota"],
                    tutor=r["tutor"],
                )
            )
        return resultado

    def registrar_seguimiento(
        self, id_matricula: int, payload: TutoriaCreateRequest
    ) -> TutoriaRegistroResponse:
        fecha_registro = (payload.fecha or date.today().isoformat()).strip()
        accion_limpia = payload.accion.strip()
        motivo_limpio = payload.motivo.strip()

        # 1. Evitar duplicados exactos en el mismo día para el mismo alumno y acción
        existente = self.conn.execute(
            """
            SELECT rowid AS id_tutoria, id_matricula, fecha, motivo, accion,
                   COALESCE(derivacion, 'Ninguna') AS derivacion,
                   mejoro_sim, nota, tutor
            FROM tutoria
            WHERE id_matricula = ?
              AND fecha = ?
              AND LOWER(TRIM(accion)) = LOWER(?)
            LIMIT 1
            """,
            (id_matricula, fecha_registro, accion_limpia),
        ).fetchone()

        if existente:
            raw_m = existente["mejoro_sim"]
            item_existente = TutoriaItemResponse(
                id_tutoria=existente["id_tutoria"],
                id_matricula=existente["id_matricula"],
                fecha=existente["fecha"],
                motivo=existente["motivo"],
                accion=existente["accion"],
                derivacion=existente["derivacion"],
                mejoro_sim=None if raw_m is None else bool(int(raw_m)),
                nota=existente["nota"],
                tutor=existente["tutor"],
            )
            return TutoriaRegistroResponse(
                ok=True,
                duplicado_hoy=True,
                mensaje=f"La acción «{accion_limpia}» ya estaba registrada hoy ({fecha_registro}) para este estudiante.",
                registro=item_existente,
                ya_aplicadas=self.listar_por_estudiante(id_matricula),
            )

        # 2. Insertar nuevo seguimiento individualizado en SQLite
        mejoro_db = None if payload.mejoro_sim is None else (1 if payload.mejoro_sim else 0)
        cur = self.conn.execute(
            """
            INSERT INTO tutoria (id_matricula, fecha, motivo, accion, derivacion, mejoro_sim, nota, tutor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                id_matricula,
                fecha_registro,
                motivo_limpio,
                accion_limpia,
                payload.derivacion.strip() or "Ninguna",
                mejoro_db,
                (payload.nota or "").strip() or None,
                (payload.tutor or "Tutor").strip(),
            ),
        )

        # 3. Si viene vinculada a una derivación docente, marcarla como Atendida
        if payload.id_derivacion:
            self.conn.execute(
                """
                UPDATE derivacion_docente
                SET estado = 'Atendida',
                    fecha_revision = COALESCE(fecha_revision, ?),
                    accion_tutor = ?,
                    derivacion_tutoria = ?,
                    nota_tutor = COALESCE(?, nota_tutor),
                    tutor = ?
                WHERE id_derivacion = ?
                """,
                (
                    fecha_registro,
                    accion_limpia,
                    payload.derivacion.strip() or "Ninguna",
                    (payload.nota or "").strip() or None,
                    (payload.tutor or "Tutor").strip(),
                    payload.id_derivacion,
                ),
            )

        self.conn.commit()

        nuevo_item = TutoriaItemResponse(
            id_tutoria=cur.lastrowid,
            id_matricula=id_matricula,
            fecha=fecha_registro,
            motivo=motivo_limpio,
            accion=accion_limpia,
            derivacion=payload.derivacion.strip() or "Ninguna",
            mejoro_sim=payload.mejoro_sim,
            nota=(payload.nota or "").strip() or None,
            tutor=(payload.tutor or "Tutor").strip(),
        )

        return TutoriaRegistroResponse(
            ok=True,
            duplicado_hoy=False,
            mensaje="Seguimiento registrado correctamente en bd_piloto.db.",
            registro=nuevo_item,
            ya_aplicadas=self.listar_por_estudiante(id_matricula),
        )