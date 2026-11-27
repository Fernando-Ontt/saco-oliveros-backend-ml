import json
import sqlite3
from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from features import UMBRAL_RIESGO, clasificar_nivel, extraer_features_matricula

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "bd_piloto.db"
MODEL_PATH = BASE_DIR / "modelo_riesgo.joblib"
SOLUCIONES_PATH = BASE_DIR / "soluciones.json"

app = FastAPI(
    title="API Alerta Temprana y Patrones de Solución - Saco Oliveros",
    version="1.0.0",
)
app = FastAPI(
    title="API Alerta Temprana y Patrones de Solución - Saco Oliveros",
    version="1.0.0",
)
from ia_explicacion import router as ia_explicacion_router
app.include_router(ia_explicacion_router)



# Aquí sigue tu CORSMiddleware y tus rutas anteriores intactas...

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
    return conn


class PrediccionRequest(BaseModel):
    id_matricula: int


class PrediccionResponse(BaseModel):
    id_matricula: int
    probabilidad: float
    nivel: str
    umbral: float
    algoritmo: str
    factores_principales: list[str]


def detectar_factores(df_row) -> list[str]:
    factores = []
    if df_row["prom_b12"] is not None and df_row["prom_b12"] < 12.5:
        factores.append(f"Promedio B1-B2 bajo ({df_row['prom_b12']:.2f})")
    if df_row["cursos_bajo_11_b2"] is not None and df_row["cursos_bajo_11_b2"] >= 1:
        factores.append(f"{int(df_row['cursos_bajo_11_b2'])} curso(s) desaprobado(s) en B2")
    if df_row["pct_asistencia_b12"] is not None and df_row["pct_asistencia_b12"] < 88:
        factores.append(f"Asistencia B1-B2 baja ({df_row['pct_asistencia_b12']:.1f}%)")
    if df_row["n_obs_graves"] >= 1 or df_row["n_obs_leves"] >= 2:
        factores.append(f"Observaciones de conducta ({int(df_row['n_obs_graves'])} graves, {int(df_row['n_obs_leves'])} leves)")
    if df_row["meses_mora_b12"] >= 2:
        factores.append(f"Mora en pensiones ({int(df_row['meses_mora_b12'])} meses)")
    return factores or ["Sin alertas críticas destacadas en B1-B2"]


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


def inferir_motivo_matricula(conn: sqlite3.Connection, id_matricula: int) -> str:
    """Determina el motivo predominante de intervención para una matrícula."""
    # 1. Revisar si ya tiene tutorías o derivaciones registradas
    row_tut = conn.execute(
        "SELECT motivo, COUNT(*) c FROM tutoria WHERE id_matricula = ? GROUP BY motivo ORDER BY c DESC LIMIT 1",
        (id_matricula,),
    ).fetchone()
    if row_tut and row_tut["motivo"]:
        return row_tut["motivo"]

    # 2. Inferir desde sus indicadores en vw_dataset_modelo
    row = conn.execute(
        "SELECT prom_b12, pct_asistencia_b12, n_obs_graves, n_obs_socioemocional FROM vw_dataset_modelo WHERE id_matricula = ?",
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


@app.get("/soluciones/{id}")
def obtener_soluciones(id: str, motivo: Optional[str] = None):
    """
    Soporta dos modos:
    - GET /soluciones/12  -> interpreta 12 como id_matricula, detecta su motivo y devuelve acciones ordenadas.
    - GET /soluciones/Conducta -> devuelve directamente las acciones para ese motivo.
    """
    perfiles = catalogo_soluciones.get("perfiles", [])
    advertencia = catalogo_soluciones.get("advertencia", "")

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

    perfil_encontrado = next(
        (p for p in perfiles if p["motivo"].lower() == motivo_detectado.lower()),
        None,
    )
    if not perfil_encontrado:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontraron soluciones para el motivo '{motivo_detectado}'",
        )

    acciones_ordenadas = sorted(
        perfil_encontrado["acciones"],
        key=lambda x: x["pct_mejoro"],
        reverse=True,
    )

    return {
        "id_matricula": id_matricula,
        "motivo": perfil_encontrado["motivo"],
        "acciones_recomendadas": acciones_ordenadas,
        "todos_los_perfiles": perfiles,
        "advertencia": advertencia,
    }