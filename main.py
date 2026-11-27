import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Literal, Optional

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from explainability import (
    construir_df_desde_dict,
    explicar_riesgo_matricula,
    obtener_importancia_global,
)
from features import FEATURES, UMBRAL_RIESGO, clasificar_nivel, extraer_features_matricula
from schemas import (
    ContactoCreate,
    DerivacionCreate,
    DerivacionUpdate,
    EvidenciaCreate,
    PrediccionLoteRequest,
    PrediccionManualRequest,
    PrediccionRequest,
    PrediccionResponse,
    RecalcularRiesgoRequest,
    SimulacionRequest,
    TutoriaCreate,
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "bd_piloto.db"
MODEL_PATH = BASE_DIR / "modelo_riesgo.joblib"
SOLUCIONES_PATH = BASE_DIR / "soluciones.json"

app = FastAPI(
    title="API Alerta Temprana y Patrones de Solución - Saco Oliveros",
    version="1.0.0",
)

# Habilitar CORS para conectar con React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo ML y catálogo de soluciones al iniciar
modelo = joblib.load(MODEL_PATH)
with open(SOLUCIONES_PATH, "r", encoding="utf-8") as f:
    catalogo_soluciones = json.load(f)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(r) for r in rows]


def _ensure_exists(conn: sqlite3.Connection, table: str, pk_col: str, pk_val: int, label: str) -> sqlite3.Row:
    row = conn.execute(f"SELECT * FROM {table} WHERE {pk_col} = ?", (pk_val,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"{label} ({pk_col}={pk_val}) no encontrado")
    return row


def _hoy() -> str:
    return date.today().isoformat()


def detectar_factores(df_row) -> list[str]:
    factores = []
    if df_row["prom_b12"] is not None and df_row["prom_b12"] < 12.5:
        factores.append(f"Promedio B1-B2 bajo ({df_row['prom_b12']:.2f})")
    if df_row["cursos_bajo_11_b2"] is not None and df_row["cursos_bajo_11_b2"] >= 1:
        factores.append(f"{int(df_row['cursos_bajo_11_b2'])} curso(s) desaprobado(s) en B2")
    if df_row["pct_asistencia_b12"] is not None and df_row["pct_asistencia_b12"] < 88:
        factores.append(f"Asistencia B1-B2 baja ({df_row['pct_asistencia_b12']:.1f}%)")
    if df_row["n_obs_graves"] >= 1 or df_row["n_obs_leves"] >= 2:
        factores.append(
            f"Observaciones de conducta ({int(df_row['n_obs_graves'])} graves, {int(df_row['n_obs_leves'])} leves)"
        )
    if df_row["meses_mora_b12"] >= 2:
        factores.append(f"Mora en pensiones ({int(df_row['meses_mora_b12'])} meses)")
    return factores or ["Sin alertas críticas destacadas en B1-B2"]


def inferir_motivo_matricula(conn: sqlite3.Connection, id_matricula: int) -> str:
    """Determina el motivo predominante de intervención para una matrícula."""
    row_tut = conn.execute(
        "SELECT motivo, COUNT(*) c FROM tutoria WHERE id_matricula = ? GROUP BY motivo ORDER BY c DESC LIMIT 1",
        (id_matricula,),
    ).fetchone()
    if row_tut and row_tut["motivo"]:
        return row_tut["motivo"]

    row = conn.execute(
        "SELECT prom_b12, pct_asistencia_b12, n_obs_graves, n_obs_socioemocional "
        "FROM vw_dataset_modelo WHERE id_matricula = ?",
        (id_matricula,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Matrícula {id_matricula} no encontrada")

    if row["prom_b12"] is not None and row["prom_b12"] < 12.0:
        return "Bajo rendimiento"
    if row["pct_asistencia_b12"] is not None and row["pct_asistencia_b12"] < 85.0:
        return "Inasistencias"
    if (row["n_obs_graves"] or 0) > 0:
        return "Conducta"
    if (row["n_obs_socioemocional"] or 0) > 0:
        return "Socioemocional"
    return "Bajo rendimiento"


def _buscar_recomendaciones(motivo: str) -> tuple[dict, list[dict], str]:
    perfiles = catalogo_soluciones.get("perfiles", [])
    advertencia = catalogo_soluciones.get("advertencia", "")
    perfil = next((p for p in perfiles if p["motivo"].lower() == motivo.lower()), None)
    if not perfil:
        raise HTTPException(status_code=404, detail=f"No se encontraron soluciones para el motivo '{motivo}'")
    acciones = sorted(perfil["acciones"], key=lambda x: x["pct_mejoro"], reverse=True)
    return perfil, acciones, advertencia


# ==============================================================================
# ENDPOINTS EXISTENTES
# ==============================================================================


@app.get("/")
def health():
    return {"status": "ok", "modelo": "Random Forest", "umbral": UMBRAL_RIESGO}


@app.post("/predecir", response_model=PrediccionResponse)
def predecir_riesgo(payload: PrediccionRequest):
    conn = get_conn()
    try:
        X = extraer_features_matricula(conn, payload.id_matricula)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        conn.close()

    prob = float(modelo.predict_proba(X)[0][1])
    nivel = clasificar_nivel(prob)
    factores = detectar_factores(X.iloc[0])

    return PrediccionResponse(
        id_matricula=payload.id_matricula,
        probabilidad=round(prob, 4),
        nivel=nivel,
        umbral=round(UMBRAL_RIESGO, 4),
        algoritmo="Random Forest",
        factores_principales=factores,
    )


@app.get("/soluciones/{id}")
def obtener_soluciones(id: str, motivo: Optional[str] = None):
    """
    Soporta dos modos:
    - GET /soluciones/12  -> interpreta 12 como id_matricula, detecta su motivo y devuelve acciones ordenadas.
    - GET /soluciones/Conducta -> devuelve directamente las acciones para ese motivo.
    """
    perfiles = catalogo_soluciones.get("perfiles", [])
    if id.isdigit():
        id_matricula = int(id)
        conn = get_conn()
        try:
            motivo_detectado = motivo or inferir_motivo_matricula(conn, id_matricula)
        finally:
            conn.close()
    else:
        id_matricula = None
        motivo_detectado = id

    perfil_encontrado, acciones_ordenadas, advertencia = _buscar_recomendaciones(motivo_detectado)

    return {
        "id_matricula": id_matricula,
        "motivo": perfil_encontrado["motivo"],
        "acciones_recomendadas": acciones_ordenadas,
        "todos_los_perfiles": perfiles,
        "advertencia": advertencia,
    }


# ==============================================================================
# 12 ENDPOINTS DE LA API DE ALERTA TEMPRANA Y GESTIÓN
# ==============================================================================

_BASE_ESTUDIANTES_SQL = """
FROM matricula m
JOIN estudiante e ON e.id_estudiante = m.id_estudiante
JOIN seccion s ON s.id_seccion = m.id_seccion
JOIN sede se ON se.id_sede = s.id_sede
LEFT JOIN riesgo r ON r.id_matricula = m.id_matricula
    AND r.id_periodo = (
        SELECT MAX(r2.id_periodo) FROM riesgo r2 WHERE r2.id_matricula = m.id_matricula
    )
"""


@app.get("/estudiantes")
def listar_estudiantes(
    anio: Optional[int] = None,
    grado: Optional[int] = Query(default=None, ge=1, le=5),
    id_seccion: Optional[int] = None,
    nivel: Optional[Literal["Bajo", "Medio", "Alto"]] = None,
    q: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Lista las matrículas de estudiantes con su sección y último nivel de riesgo."""
    filtros = []
    params: list = []

    if anio is not None:
        filtros.append("m.anio = ?")
        params.append(anio)
    if grado is not None:
        filtros.append("s.grado = ?")
        params.append(grado)
    if id_seccion is not None:
        filtros.append("m.id_seccion = ?")
        params.append(id_seccion)
    if nivel is not None:
        filtros.append("r.nivel = ?")
        params.append(nivel)
    if q:
        filtros.append(
            "(e.nombres LIKE ? OR e.apellidos LIKE ? OR e.id_anonimo LIKE ? OR e.documento_simulado LIKE ?)"
        )
        like_q = f"%{q.strip()}%"
        params.extend([like_q, like_q, like_q, like_q])

    where_sql = f"WHERE {' AND '.join(filtros)}" if filtros else ""

    conn = get_conn()
    try:
        total = conn.execute(f"SELECT COUNT(*) {_BASE_ESTUDIANTES_SQL} {where_sql}", params).fetchone()[0]
        rows = conn.execute(
            f"""
            SELECT m.id_matricula, m.id_estudiante, e.id_anonimo, e.nombres, e.apellidos,
                   e.documento_simulado, e.sexo, m.anio, s.id_seccion, s.grado, s.letra,
                   se.nombre AS sede, r.probabilidad, r.nivel, r.nivel_regla, r.fecha_prediccion
            {_BASE_ESTUDIANTES_SQL}
            {where_sql}
            ORDER BY COALESCE(r.probabilidad, 0) DESC, e.apellidos ASC, e.nombres ASC
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).fetchall()
    finally:
        conn.close()

    return {"total": total, "limit": limit, "offset": offset, "items": _rows_to_dicts(rows)}


@app.get("/estudiantes/{id_matricula}")
def obtener_estudiante(id_matricula: int):
    """Devuelve el detalle completo de un estudiante matriculado, su apoderado, riesgo e indicadores."""
    conn = get_conn()
    try:
        row = conn.execute(
            f"""
            SELECT m.*, e.id_anonimo, e.nombres, e.apellidos, e.documento_simulado, e.sexo,
                   e.anio_ingreso, e.tipo_ingreso, e.colegio_procedencia, e.nota_diagnostico_ingreso,
                   e.zona, e.distancia_km, e.tiempo_traslado_min, e.convive_con,
                   e.educacion_apoderado, e.apoyo_familiar, e.internet_casa, e.horas_estudio_sem,
                   e.aspira_universidad, e.relacion_familiar, e.salud, e.actividades_extra, e.nro_hermanos,
                   s.grado, s.letra, s.id_tutor, se.nombre AS sede, se.turno,
                   r.probabilidad, r.nivel, r.nivel_regla, r.factores AS factores_riesgo,
                   r.modelo, r.fecha_prediccion
            {_BASE_ESTUDIANTES_SQL}
            WHERE m.id_matricula = ?
            """,
            (id_matricula,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Matrícula {id_matricula} no encontrada")

        apoderado = conn.execute(
            "SELECT id_apoderado, nombres, apellidos, documento_simulado, parentesco "
            "FROM apoderado WHERE id_estudiante = ?",
            (row["id_estudiante"],),
        ).fetchone()

        indicadores = conn.execute(
            """
            SELECT prom_b1, prom_b2, prom_b12, tendencia_notas, min_curso_b2, cursos_bajo_11_b2,
                   pct_asistencia_b12, pct_asistencia_b1, pct_asistencia_b2, total_faltas,
                   faltas_consecutivas_max, n_obs_positivas, n_obs_leves, n_obs_graves,
                   n_obs_socioemocional, n_tutorias, n_contactos_familia, meses_mora_b12
            FROM vw_dataset_modelo WHERE id_matricula = ?
            """,
            (id_matricula,),
        ).fetchone()
    finally:
        conn.close()

    return {
        "estudiante": dict(row),
        "apoderado": dict(apoderado) if apoderado else None,
        "indicadores": dict(indicadores) if indicadores else None,
    }


@app.get("/estudiantes/{id_matricula}/historial")
def obtener_historial_estudiante(id_matricula: int):
    """Devuelve el historial académico, de asistencia, conducta, tutorías, contactos, derivaciones y pagos."""
    conn = get_conn()
    try:
        _ensure_exists(conn, "matricula", "id_matricula", id_matricula, "Matrícula")

        calificaciones = conn.execute(
            """
            SELECT c.id_calificacion, cu.nombre AS curso, cu.area, p.anio, p.bimestre,
                   c.tipo_evaluacion, c.fecha, c.nota
            FROM calificacion c
            JOIN curso cu ON cu.id_curso = c.id_curso
            JOIN periodo p ON p.id_periodo = c.id_periodo
            WHERE c.id_matricula = ?
            ORDER BY p.bimestre ASC, cu.nombre ASC, c.fecha ASC
            """,
            (id_matricula,),
        ).fetchall()

        asistencia_resumen = conn.execute(
            "SELECT estado, COUNT(*) AS total FROM asistencia WHERE id_matricula = ? GROUP BY estado",
            (id_matricula,),
        ).fetchall()

        asistencias_recientes = conn.execute(
            """
            SELECT id_asistencia, fecha, estado, hora_ingreso, medio
            FROM asistencia
            WHERE id_matricula = ? AND estado <> 'Presente'
            ORDER BY fecha DESC LIMIT 30
            """,
            (id_matricula,),
        ).fetchall()

        observaciones = conn.execute(
            """
            SELECT o.id_observacion, o.fecha, o.tipo, o.deriva_a_tutor, o.id_docente,
                   d.nombres || ' ' || d.apellidos AS docente
            FROM observacion_conducta o
            LEFT JOIN docente d ON d.id_docente = o.id_docente
            WHERE o.id_matricula = ?
            ORDER BY o.fecha DESC, o.id_observacion DESC
            """,
            (id_matricula,),
        ).fetchall()

        tutorias = conn.execute(
            """
            SELECT t.id_tutoria, t.fecha, t.motivo, t.accion, t.derivacion, t.id_docente,
                   d.nombres || ' ' || d.apellidos AS docente
            FROM tutoria t
            LEFT JOIN docente d ON d.id_docente = t.id_docente
            WHERE t.id_matricula = ?
            ORDER BY t.fecha DESC, t.id_tutoria DESC
            """,
            (id_matricula,),
        ).fetchall()

        contactos = conn.execute(
            """
            SELECT cf.id_contacto, cf.fecha, cf.tipo, cf.respondio,
                   a.nombres || ' ' || a.apellidos AS apoderado, a.parentesco
            FROM contacto_familia cf
            JOIN apoderado a ON a.id_apoderado = cf.id_apoderado
            WHERE cf.id_matricula = ?
            ORDER BY cf.fecha DESC, cf.id_contacto DESC
            """,
            (id_matricula,),
        ).fetchall()

        derivaciones = conn.execute(
            """
            SELECT dd.id_derivacion, dd.fecha, dd.motivo, dd.estado, dd.fecha_revision, dd.accion_tutor,
                   c.nombre AS curso, d.nombres || ' ' || d.apellidos AS docente
            FROM derivacion_docente dd
            JOIN docente d ON d.id_docente = dd.id_docente
            LEFT JOIN curso c ON c.id_curso = dd.id_curso
            WHERE dd.id_matricula = ?
            ORDER BY dd.fecha DESC, dd.id_derivacion DESC
            """,
            (id_matricula,),
        ).fetchall()

        pagos = conn.execute(
            "SELECT id_pago, mes, estado, dias_atraso FROM pago_pension WHERE id_matricula = ? ORDER BY mes ASC",
            (id_matricula,),
        ).fetchall()

        encuestas = conn.execute(
            """
            SELECT eb.id_encuesta, p.bimestre, eb.motivacion, eb.estres,
                   eb.relacion_pares, eb.relacion_familia, eb.seguridad
            FROM encuesta_bienestar eb
            JOIN periodo p ON p.id_periodo = eb.id_periodo
            WHERE eb.id_matricula = ?
            ORDER BY p.bimestre ASC
            """,
            (id_matricula,),
        ).fetchall()
    finally:
        conn.close()

    return {
        "id_matricula": id_matricula,
        "calificaciones": _rows_to_dicts(calificaciones),
        "asistencia_resumen": {r["estado"]: r["total"] for r in asistencia_resumen},
        "inasistencias_tardanzas_recientes": _rows_to_dicts(asistencias_recientes),
        "observaciones": _rows_to_dicts(observaciones),
        "tutorias": _rows_to_dicts(tutorias),
        "contactos_familia": _rows_to_dicts(contactos),
        "derivaciones": _rows_to_dicts(derivaciones),
        "pagos": _rows_to_dicts(pagos),
        "encuestas_bienestar": _rows_to_dicts(encuestas),
    }


@app.get("/estudiantes/{id_matricula}/recomendaciones")
def obtener_recomendaciones_estudiante(id_matricula: int, motivo: Optional[str] = None):
    """Devuelve las acciones recomendadas según el motivo predominante del estudiante."""
    conn = get_conn()
    try:
        _ensure_exists(conn, "matricula", "id_matricula", id_matricula, "Matrícula")
        motivo_detectado = motivo or inferir_motivo_matricula(conn, id_matricula)
        riesgo_row = conn.execute(
            "SELECT probabilidad, nivel FROM riesgo WHERE id_matricula = ? ORDER BY id_periodo DESC LIMIT 1",
            (id_matricula,),
        ).fetchone()
    finally:
        conn.close()

    perfil, acciones, advertencia = _buscar_recomendaciones(motivo_detectado)
    return {
        "id_matricula": id_matricula,
        "riesgo_actual": dict(riesgo_row) if riesgo_row else None,
        "motivo": perfil["motivo"],
        "acciones_recomendadas": acciones,
        "advertencia": advertencia,
    }


@app.get("/estudiantes/{id_matricula}/explicacion-riesgo")
def explicar_riesgo_estudiante(
    id_matricula: int,
    top_n: int = Query(default=8, ge=1, le=20),
):
    """Muestra qué variables aumentan o reducen la predicción de riesgo respecto a la cohorte."""
    conn = get_conn()
    try:
        return explicar_riesgo_matricula(conn, modelo, id_matricula, top_n=top_n)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        conn.close()


@app.get("/alertas")
def listar_alertas(
    nivel: Optional[Literal["Medio", "Alto"]] = None,
    anio: Optional[int] = None,
    grado: Optional[int] = Query(default=None, ge=1, le=5),
    id_seccion: Optional[int] = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Muestra automáticamente los alumnos con riesgo Medio o Alto ordenados por probabilidad."""
    filtros = ["r.nivel IN ('Medio', 'Alto')"]
    params: list = []

    if nivel is not None:
        filtros.append("r.nivel = ?")
        params.append(nivel)
    if anio is not None:
        filtros.append("m.anio = ?")
        params.append(anio)
    if grado is not None:
        filtros.append("s.grado = ?")
        params.append(grado)
    if id_seccion is not None:
        filtros.append("m.id_seccion = ?")
        params.append(id_seccion)

    where_sql = f"WHERE {' AND '.join(filtros)}"

    conn = get_conn()
    try:
        conteos = conn.execute(
            f"""
            SELECT r.nivel, COUNT(*) AS cnt
            {_BASE_ESTUDIANTES_SQL}
            {where_sql}
            GROUP BY r.nivel
            """,
            params,
        ).fetchall()
        resumen = {"Alto": 0, "Medio": 0}
        for c in conteos:
            resumen[c["nivel"]] = c["cnt"]

        rows = conn.execute(
            f"""
            SELECT m.id_matricula, m.id_estudiante, e.id_anonimo, e.nombres, e.apellidos,
                   m.anio, s.id_seccion, s.grado, s.letra, se.nombre AS sede,
                   r.probabilidad, r.nivel, r.nivel_regla, r.factores, r.fecha_prediccion
            {_BASE_ESTUDIANTES_SQL}
            {where_sql}
            ORDER BY r.probabilidad DESC, m.id_matricula ASC
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).fetchall()
    finally:
        conn.close()

    return {
        "total": resumen["Alto"] + resumen["Medio"],
        "resumen": resumen,
        "limit": limit,
        "offset": offset,
        "items": _rows_to_dicts(rows),
    }


@app.get("/derivaciones")
def listar_derivaciones(
    estado: Optional[Literal["Pendiente", "Revisada", "Atendida"]] = None,
    id_matricula: Optional[int] = None,
    id_docente: Optional[int] = None,
    id_tutor: Optional[int] = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Lista las derivaciones docentes con información del estudiante, docente, curso y tutor."""
    filtros = []
    params: list = []

    if estado is not None:
        filtros.append("dd.estado = ?")
        params.append(estado)
    if id_matricula is not None:
        filtros.append("dd.id_matricula = ?")
        params.append(id_matricula)
    if id_docente is not None:
        filtros.append("dd.id_docente = ?")
        params.append(id_docente)
    if id_tutor is not None:
        filtros.append("dd.id_tutor = ?")
        params.append(id_tutor)

    where_sql = f"WHERE {' AND '.join(filtros)}" if filtros else ""
    base_from = """
    FROM derivacion_docente dd
    JOIN matricula m ON m.id_matricula = dd.id_matricula
    JOIN estudiante e ON e.id_estudiante = m.id_estudiante
    JOIN seccion s ON s.id_seccion = m.id_seccion
    JOIN docente d ON d.id_docente = dd.id_docente
    LEFT JOIN curso c ON c.id_curso = dd.id_curso
    LEFT JOIN docente t ON t.id_docente = dd.id_tutor
    """

    conn = get_conn()
    try:
        total = conn.execute(f"SELECT COUNT(*) {base_from} {where_sql}", params).fetchone()[0]
        rows = conn.execute(
            f"""
            SELECT dd.*, e.nombres || ' ' || e.apellidos AS estudiante, e.id_anonimo,
                   s.grado, s.letra, d.nombres || ' ' || d.apellidos AS docente,
                   c.nombre AS curso, t.nombres || ' ' || t.apellidos AS tutor
            {base_from}
            {where_sql}
            ORDER BY dd.fecha DESC, dd.id_derivacion DESC
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).fetchall()
    finally:
        conn.close()

    return {"total": total, "limit": limit, "offset": offset, "items": _rows_to_dicts(rows)}


@app.post("/estudiantes/{id_matricula}/evidencias", status_code=201)
def registrar_evidencia(id_matricula: int, payload: EvidenciaCreate):
    """Registra una observación de conducta (evidencia) para un estudiante matriculado."""
    conn = get_conn()
    try:
        _ensure_exists(conn, "matricula", "id_matricula", id_matricula, "Matrícula")
        if payload.id_docente is not None:
            _ensure_exists(conn, "docente", "id_docente", payload.id_docente, "Docente")

        fecha = payload.fecha or _hoy()
        cur = conn.execute(
            """
            INSERT INTO observacion_conducta (id_matricula, id_docente, fecha, tipo, deriva_a_tutor)
            VALUES (?, ?, ?, ?, ?)
            """,
            (id_matricula, payload.id_docente, fecha, payload.tipo, int(payload.deriva_a_tutor)),
        )
        conn.commit()
        creado = conn.execute(
            "SELECT * FROM observacion_conducta WHERE id_observacion = ?", (cur.lastrowid,)
        ).fetchone()
    finally:
        conn.close()

    return dict(creado)


@app.post("/estudiantes/{id_matricula}/tutorias", status_code=201)
def registrar_tutoria(id_matricula: int, payload: TutoriaCreate):
    """Registra una sesión de tutoría para un estudiante matriculado."""
    conn = get_conn()
    try:
        _ensure_exists(conn, "matricula", "id_matricula", id_matricula, "Matrícula")
        if payload.id_docente is not None:
            _ensure_exists(conn, "docente", "id_docente", payload.id_docente, "Docente")

        fecha = payload.fecha or _hoy()
        cur = conn.execute(
            """
            INSERT INTO tutoria (id_matricula, id_docente, fecha, motivo, accion, derivacion)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (id_matricula, payload.id_docente, fecha, payload.motivo, payload.accion, payload.derivacion),
        )
        conn.commit()
        creado = conn.execute("SELECT * FROM tutoria WHERE id_tutoria = ?", (cur.lastrowid,)).fetchone()
    finally:
        conn.close()

    return dict(creado)


@app.post("/derivaciones", status_code=201)
def crear_derivacion(payload: DerivacionCreate):
    """Crea una nueva derivación docente en estado Pendiente."""
    conn = get_conn()
    try:
        _ensure_exists(conn, "matricula", "id_matricula", payload.id_matricula, "Matrícula")
        _ensure_exists(conn, "docente", "id_docente", payload.id_docente, "Docente")
        if payload.id_curso is not None:
            _ensure_exists(conn, "curso", "id_curso", payload.id_curso, "Curso")
        if payload.id_observacion is not None:
            _ensure_exists(conn, "observacion_conducta", "id_observacion", payload.id_observacion, "Observación")
        if payload.id_tutor is not None:
            _ensure_exists(conn, "docente", "id_docente", payload.id_tutor, "Tutor")

        fecha = payload.fecha or _hoy()
        cur = conn.execute(
            """
            INSERT INTO derivacion_docente
                (id_matricula, id_docente, id_curso, id_observacion, fecha, motivo, estado, id_tutor)
            VALUES (?, ?, ?, ?, ?, ?, 'Pendiente', ?)
            """,
            (
                payload.id_matricula,
                payload.id_docente,
                payload.id_curso,
                payload.id_observacion,
                fecha,
                payload.motivo,
                payload.id_tutor,
            ),
        )
        conn.commit()
        creado = conn.execute(
            "SELECT * FROM derivacion_docente WHERE id_derivacion = ?", (cur.lastrowid,)
        ).fetchone()
    finally:
        conn.close()

    return dict(creado)


@app.post("/estudiantes/{id_matricula}/contactos", status_code=201)
def registrar_contacto_familia(id_matricula: int, payload: ContactoCreate):
    """Registra un contacto con el apoderado de un estudiante matriculado."""
    conn = get_conn()
    try:
        mat = _ensure_exists(conn, "matricula", "id_matricula", id_matricula, "Matrícula")

        if payload.id_apoderado is not None:
            _ensure_exists(conn, "apoderado", "id_apoderado", payload.id_apoderado, "Apoderado")
            id_apoderado = payload.id_apoderado
        else:
            apo = conn.execute(
                "SELECT id_apoderado FROM apoderado WHERE id_estudiante = ?", (mat["id_estudiante"],)
            ).fetchone()
            if not apo:
                raise HTTPException(
                    status_code=404,
                    detail=f"No se encontró apoderado registrado para el estudiante {mat['id_estudiante']}",
                )
            id_apoderado = apo["id_apoderado"]

        fecha = payload.fecha or _hoy()
        cur = conn.execute(
            """
            INSERT INTO contacto_familia (id_matricula, id_apoderado, fecha, tipo, respondio)
            VALUES (?, ?, ?, ?, ?)
            """,
            (id_matricula, id_apoderado, fecha, payload.tipo, int(payload.respondio)),
        )
        conn.commit()
        creado = conn.execute(
            "SELECT * FROM contacto_familia WHERE id_contacto = ?", (cur.lastrowid,)
        ).fetchone()
    finally:
        conn.close()

    return dict(creado)


@app.patch("/derivaciones/{id_derivacion}")
def actualizar_derivacion(id_derivacion: int, payload: DerivacionUpdate):
    """Actualiza el estado, tutor asignado, fecha de revisión o acción tomada de una derivación."""
    cambios = payload.model_dump(exclude_unset=True)
    if not cambios:
        raise HTTPException(status_code=400, detail="Debe enviar al menos un campo para actualizar")

    conn = get_conn()
    try:
        _ensure_exists(conn, "derivacion_docente", "id_derivacion", id_derivacion, "Derivación")
        if payload.id_tutor is not None:
            _ensure_exists(conn, "docente", "id_docente", payload.id_tutor, "Tutor")

        if payload.estado in ("Revisada", "Atendida") and "fecha_revision" not in cambios:
            cambios["fecha_revision"] = _hoy()

        set_sql = ", ".join(f"{col} = ?" for col in cambios.keys())
        conn.execute(
            f"UPDATE derivacion_docente SET {set_sql} WHERE id_derivacion = ?",
            [*cambios.values(), id_derivacion],
        )
        conn.commit()
        actualizado = conn.execute(
            "SELECT * FROM derivacion_docente WHERE id_derivacion = ?", (id_derivacion,)
        ).fetchone()
    finally:
        conn.close()

    return dict(actualizado)


# ==============================================================================
# ENDPOINTS PREDICTIVOS ADICIONALES
# ==============================================================================


def _nivel_regla_desde_row(df_row) -> str:
    asist = df_row["pct_asistencia_b12"]
    prom = df_row["prom_b12"]
    if (asist is not None and asist < 80) or (prom is not None and prom < 11):
        return "Alto"
    if (asist is not None and asist < 90) or (prom is not None and prom < 13):
        return "Medio"
    return "Bajo"


def _resolver_id_periodo(conn: sqlite3.Connection, id_matricula: int, id_periodo: Optional[int]) -> int:
    if id_periodo is not None:
        _ensure_exists(conn, "periodo", "id_periodo", id_periodo, "Periodo")
        return id_periodo

    row_r = conn.execute(
        "SELECT id_periodo FROM riesgo WHERE id_matricula = ? ORDER BY id_periodo DESC LIMIT 1",
        (id_matricula,),
    ).fetchone()
    if row_r:
        return row_r["id_periodo"]

    row_p = conn.execute(
        """
        SELECT p.id_periodo
        FROM matricula m
        JOIN periodo p ON p.anio = m.anio AND p.bimestre = 2
        WHERE m.id_matricula = ?
        """,
        (id_matricula,),
    ).fetchone()
    if not row_p:
        raise HTTPException(status_code=404, detail=f"No se encontró periodo para la matrícula {id_matricula}")
    return row_p["id_periodo"]


def _recalcular_y_guardar(conn: sqlite3.Connection, id_matricula: int, id_periodo: Optional[int] = None) -> dict:
    X = extraer_features_matricula(conn, id_matricula)
    prob = float(modelo.predict_proba(X)[0][1])
    nivel = clasificar_nivel(prob)
    nivel_regla = _nivel_regla_desde_row(X.iloc[0])
    factores = "; ".join(detectar_factores(X.iloc[0]))
    periodo_objetivo = _resolver_id_periodo(conn, id_matricula, id_periodo)
    fecha_pred = datetime.now().isoformat()

    conn.execute(
        """
        INSERT INTO riesgo (id_matricula, id_periodo, nivel_regla, probabilidad, nivel, factores, modelo, fecha_prediccion)
        VALUES (?, ?, ?, ?, ?, ?, 'Random Forest_v1.0', ?)
        ON CONFLICT(id_matricula, id_periodo) DO UPDATE SET
            nivel_regla = excluded.nivel_regla,
            probabilidad = excluded.probabilidad,
            nivel = excluded.nivel,
            factores = excluded.factores,
            modelo = excluded.modelo,
            fecha_prediccion = excluded.fecha_prediccion
        """,
        (id_matricula, periodo_objetivo, nivel_regla, prob, nivel, factores, fecha_pred),
    )
    row = conn.execute(
        "SELECT * FROM riesgo WHERE id_matricula = ? AND id_periodo = ?",
        (id_matricula, periodo_objetivo),
    ).fetchone()
    return dict(row)


@app.get("/modelo")
def info_modelo(top_n: int = Query(default=15, ge=1, le=59)):
    """Devuelve metadatos del modelo ML, umbral calibrado e importancia global de variables."""
    return {
        "algoritmo": "Random Forest",
        "version": "1.0",
        "umbral_riesgo": round(UMBRAL_RIESGO, 4),
        "total_variables": len(FEATURES),
        "variables_importantes": obtener_importancia_global(modelo, top_n=top_n),
    }


@app.get("/estudiantes/{id_matricula}/tendencia-riesgo")
def obtener_tendencia_riesgo(id_matricula: int):
    """Devuelve la evolución histórica del riesgo del estudiante por periodo."""
    conn = get_conn()
    try:
        _ensure_exists(conn, "matricula", "id_matricula", id_matricula, "Matrícula")
        rows = conn.execute(
            """
            SELECT r.id_periodo, p.anio, p.bimestre, r.probabilidad, r.nivel,
                   r.nivel_regla, r.factores, r.modelo, r.fecha_prediccion
            FROM riesgo r
            JOIN periodo p ON p.id_periodo = r.id_periodo
            WHERE r.id_matricula = ?
            ORDER BY p.anio ASC, p.bimestre ASC
            """,
            (id_matricula,),
        ).fetchall()
    finally:
        conn.close()

    return {"id_matricula": id_matricula, "historial_riesgo": _rows_to_dicts(rows)}


@app.post("/estudiantes/{id_matricula}/simular-riesgo")
def simular_riesgo(id_matricula: int, payload: SimulacionRequest):
    """Simula cómo cambiaría la probabilidad de riesgo al modificar una o más variables del estudiante."""
    conn = get_conn()
    try:
        X_actual = extraer_features_matricula(conn, id_matricula)
        X_simulado, aplicadas = construir_df_desde_dict(conn, payload.cambios, base_df=X_actual)
    except ValueError as exc:
        status = 404 if "No se encontró la matrícula" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc))
    finally:
        conn.close()

    prob_actual = float(modelo.predict_proba(X_actual)[0][1])
    prob_simulada = float(modelo.predict_proba(X_simulado)[0][1])

    return {
        "id_matricula": id_matricula,
        "probabilidad_actual": round(prob_actual, 4),
        "nivel_actual": clasificar_nivel(prob_actual),
        "probabilidad_simulada": round(prob_simulada, 4),
        "nivel_simulado": clasificar_nivel(prob_simulada),
        "diferencia_probabilidad": round(prob_simulada - prob_actual, 4),
        "variables_modificadas": aplicadas,
        "factores_simulados": detectar_factores(X_simulado.iloc[0]),
    }


@app.post("/estudiantes/{id_matricula}/recalcular-riesgo")
def recalcular_riesgo_estudiante(id_matricula: int, payload: Optional[RecalcularRiesgoRequest] = None):
    """Recalcula el riesgo con los datos actuales de la BD y actualiza la tabla riesgo."""
    id_periodo = payload.id_periodo if payload else None
    conn = get_conn()
    try:
        actualizado = _recalcular_y_guardar(conn, id_matricula, id_periodo=id_periodo)
        conn.commit()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    finally:
        conn.close()

    return actualizado


@app.post("/predicciones/lote")
def recalcular_riesgo_lote(payload: PrediccionLoteRequest):
    """Recalcula y guarda el riesgo en lote para todas las matrículas que coincidan con los filtros."""
    filtros = []
    params: list = []
    if payload.anio is not None:
        filtros.append("m.anio = ?")
        params.append(payload.anio)
    if payload.grado is not None:
        filtros.append("s.grado = ?")
        params.append(payload.grado)
    if payload.id_seccion is not None:
        filtros.append("m.id_seccion = ?")
        params.append(payload.id_seccion)

    where_sql = f"WHERE {' AND '.join(filtros)}" if filtros else ""

    conn = get_conn()
    try:
        if payload.id_periodo is not None:
            _ensure_exists(conn, "periodo", "id_periodo", payload.id_periodo, "Periodo")

        ids = [
            r["id_matricula"]
            for r in conn.execute(
                f"SELECT m.id_matricula FROM matricula m JOIN seccion s ON s.id_seccion = m.id_seccion {where_sql}",
                params,
            ).fetchall()
        ]
        resumen = {"Alto": 0, "Medio": 0, "Bajo": 0}
        for id_mat in ids:
            reg = _recalcular_y_guardar(conn, id_mat, id_periodo=payload.id_periodo)
            resumen[reg["nivel"]] = resumen.get(reg["nivel"], 0) + 1
        conn.commit()
    finally:
        conn.close()

    return {"total_procesados": len(ids), "resumen": resumen}


@app.post("/predecir/manual")
def predecir_manual(payload: PrediccionManualRequest):
    """Evalúa el riesgo a partir de variables ingresadas manualmente (completa las faltantes con la cohorte)."""
    conn = get_conn()
    try:
        X, aplicadas = construir_df_desde_dict(conn, payload.variables, base_df=None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        conn.close()

    prob = float(modelo.predict_proba(X)[0][1])
    return {
        "probabilidad": round(prob, 4),
        "nivel": clasificar_nivel(prob),
        "umbral": round(UMBRAL_RIESGO, 4),
        "algoritmo": "Random Forest",
        "variables_enviadas": len(aplicadas),
        "variables_completadas_cohorte": len(FEATURES) - len(aplicadas),
        "factores_principales": detectar_factores(X.iloc[0]),
    }
