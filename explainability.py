import sqlite3
from typing import Any

import pandas as pd

from features import CAT_COLS, FEATURES, clasificar_nivel, extraer_features_matricula

FEATURE_LABELS: dict[str, str] = {
    "grado": "Grado escolar",
    "edad": "Edad",
    "trabaja": "Trabaja",
    "repitio_grado": "Repitió grado anteriormente",
    "cursos_desaprobados_previos": "Cursos desaprobados previos",
    "beca": "Cuenta con beca",
    "tipo_ingreso": "Tipo de ingreso",
    "colegio_procedencia": "Colegio de procedencia",
    "nota_diagnostico_ingreso": "Nota de diagnóstico de ingreso",
    "zona": "Zona de residencia",
    "distancia_km": "Distancia al colegio (km)",
    "tiempo_traslado_min": "Tiempo de traslado (min)",
    "convive_con": "Convive con",
    "educacion_apoderado": "Educación del apoderado",
    "apoyo_familiar": "Apoyo familiar",
    "internet_casa": "Internet en casa",
    "horas_estudio_sem": "Horas de estudio semanales",
    "aspira_universidad": "Aspira a la universidad",
    "relacion_familiar": "Relación familiar",
    "salud": "Estado de salud",
    "actividades_extra": "Actividades extracurriculares",
    "nro_hermanos": "Número de hermanos",
    "prom_b1": "Promedio Bimestre 1",
    "prom_b2": "Promedio Bimestre 2",
    "prom_b12": "Promedio acumulado B1-B2",
    "tendencia_notas": "Tendencia de notas (B2 - B1)",
    "min_curso_b2": "Nota mínima en curso B2",
    "cursos_bajo_11_b2": "Cursos desaprobados en B2",
    "prom_matematica": "Promedio en Matemática",
    "prom_comunicacion": "Promedio en Comunicación",
    "varianza_notas": "Variabilidad de notas",
    "prom_simulacros": "Promedio en simulacros",
    "pct_asistencia_b12": "Asistencia acumulada B1-B2 (%)",
    "pct_asistencia_b1": "Asistencia Bimestre 1 (%)",
    "pct_asistencia_b2": "Asistencia Bimestre 2 (%)",
    "tendencia_asistencia": "Tendencia de asistencia (B2 - B1)",
    "pct_tardanzas": "Porcentaje de tardanzas (%)",
    "pct_faltas_injustificadas": "Faltas injustificadas (%)",
    "total_faltas": "Total de faltas",
    "pct_faltas_lun_vie": "Faltas en lunes/viernes (%)",
    "faltas_consecutivas_max": "Máximo de faltas consecutivas",
    "n_obs_positivas": "Observaciones positivas",
    "n_obs_leves": "Observaciones leves",
    "n_obs_graves": "Observaciones graves",
    "n_obs_socioemocional": "Observaciones socioemocionales",
    "n_derivaciones_docente": "Derivaciones docentes a tutoría",
    "n_tutorias": "Cantidad de tutorías",
    "n_derivaciones_psicologia": "Derivaciones a psicología",
    "n_contactos_familia": "Contactos con la familia",
    "n_citaciones": "Citaciones al apoderado",
    "tasa_respuesta_familia": "Tasa de respuesta familiar",
    "meses_mora_b12": "Meses con retraso en pensión",
    "meses_moroso_b12": "Meses en estado moroso",
    "max_dias_atraso_b12": "Máximo días de atraso en pensión",
    "bienestar_motivacion": "Bienestar: motivación",
    "bienestar_estres": "Bienestar: estrés",
    "bienestar_pares": "Bienestar: relación con pares",
    "bienestar_familia": "Bienestar: relación familiar",
    "bienestar_seguridad": "Bienestar: percepción de seguridad",
}

_baseline_cache: dict[str, Any] | None = None


def _obtener_baseline(conn: sqlite3.Connection) -> dict[str, Any]:
    """Calcula la mediana/moda de cada feature en la cohorte (con caché en memoria)."""
    global _baseline_cache
    if _baseline_cache is not None:
        return _baseline_cache

    cols_sql = ", ".join(FEATURES)
    df = pd.read_sql_query(f"SELECT {cols_sql} FROM vw_dataset_modelo", conn)
    baseline: dict[str, Any] = {}
    for col in FEATURES:
        if col in CAT_COLS:
            moda = df[col].dropna().mode()
            baseline[col] = str(moda.iloc[0]) if not moda.empty else None
        else:
            serie = pd.to_numeric(df[col], errors="coerce").dropna()
            baseline[col] = float(serie.median()) if not serie.empty else 0.0

    _baseline_cache = baseline
    return baseline


def _serializar_valor(val: Any) -> Any:
    if pd.isna(val):
        return None
    if isinstance(val, float):
        return round(val, 2)
    return val


def explicar_riesgo_matricula(
    conn: sqlite3.Connection,
    modelo: Any,
    id_matricula: int,
    top_n: int = 8,
) -> dict[str, Any]:
    """
    Calcula el impacto individual de cada variable sobre la probabilidad predicha
    reemplazando una variable a la vez por su valor de referencia de la cohorte.
    """
    X = extraer_features_matricula(conn, id_matricula)
    baseline = _obtener_baseline(conn)
    prob_actual = float(modelo.predict_proba(X)[0][1])

    # Construir matriz counterfactual en batch (una fila por feature) para evaluar en una sola llamada
    rows = []
    for col in FEATURES:
        row = X.iloc[0].copy()
        row[col] = baseline[col]
        rows.append(row)

    df_cf = pd.DataFrame(rows, columns=FEATURES)
    for col in FEATURES:
        if col in CAT_COLS:
            df_cf[col] = df_cf[col].where(df_cf[col].notna(), other=pd.NA).astype("object")
            df_cf[col] = df_cf[col].apply(lambda v: None if pd.isna(v) else str(v))
        else:
            df_cf[col] = pd.to_numeric(df_cf[col], errors="coerce").astype("float64")

    probs_cf = modelo.predict_proba(df_cf)[:, 1]

    aumentan = []
    reducen = []
    for idx, col in enumerate(FEATURES):
        delta = prob_actual - float(probs_cf[idx])
        if abs(delta) < 0.001:
            continue

        item = {
            "variable": col,
            "etiqueta": FEATURE_LABELS.get(col, col),
            "valor_estudiante": _serializar_valor(X.iloc[0][col]),
            "valor_referencia": _serializar_valor(baseline[col]),
            "impacto_probabilidad": round(delta, 4),
            "efecto": "Aumenta el riesgo" if delta > 0 else "Reduce el riesgo",
        }
        if delta > 0:
            aumentan.append(item)
        else:
            reducen.append(item)

    aumentan.sort(key=lambda x: x["impacto_probabilidad"], reverse=True)
    reducen.sort(key=lambda x: x["impacto_probabilidad"])

    row_riesgo = conn.execute(
        "SELECT factores FROM riesgo WHERE id_matricula = ? ORDER BY id_periodo DESC LIMIT 1",
        (id_matricula,),
    ).fetchone()

    return {
        "id_matricula": id_matricula,
        "probabilidad": round(prob_actual, 4),
        "nivel": clasificar_nivel(prob_actual),
        "factores_aumentan_riesgo": aumentan[:top_n],
        "factores_reducen_riesgo": reducen[:top_n],
        "resumen_bd": row_riesgo["factores"] if row_riesgo else None,
    }


def normalizar_df_features(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura el orden de las 59 columnas y los tipos exactos esperados por el modelo."""
    out = df[FEATURES].copy()
    for col in FEATURES:
        if col in CAT_COLS:
            out[col] = out[col].where(out[col].notna(), other=pd.NA).astype("object")
            out[col] = out[col].apply(lambda v: None if pd.isna(v) else str(v))
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def construir_df_desde_dict(conn: sqlite3.Connection, variables: dict[str, Any], base_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Construye un DataFrame de 1 fila a partir de `base_df` (o la mediana/moda de la cohorte)
    aplicando los valores enviados en `variables`. Devuelve (df, lista_campos_aplicados).
    """
    desconocidas = [k for k in variables if k not in FEATURES]
    if desconocidas:
        raise ValueError(f"Variables no reconocidas por el modelo: {', '.join(desconocidas)}")

    if base_df is not None:
        fila = base_df.iloc[0].to_dict()
    else:
        fila = dict(_obtener_baseline(conn))

    for k, v in variables.items():
        fila[k] = v

    df = normalizar_df_features(pd.DataFrame([fila], columns=FEATURES))
    return df, list(variables.keys())


def obtener_importancia_global(modelo: Any, top_n: int = 15) -> list[dict[str, Any]]:
    """Extrae la importancia global de variables del RandomForest agregada por feature original."""
    if not hasattr(modelo, "named_steps"):
        return []

    prep = modelo.named_steps.get("prep")
    clf = modelo.named_steps.get("modelo") or list(modelo.named_steps.values())[-1]
    if prep is None or not hasattr(clf, "feature_importances_"):
        return []

    nombres_trans = prep.get_feature_names_out()
    importancias = clf.feature_importances_

    agregado: dict[str, float] = {col: 0.0 for col in FEATURES}
    for nombre, imp in zip(nombres_trans, importancias):
        raw = nombre.split("__", 1)[-1]
        # Para OneHotEncoder (ej. zona_Urbana), empatar por prefijo más largo de FEATURES
        match = next((col for col in sorted(FEATURES, key=len, reverse=True) if raw == col or raw.startswith(f"{col}_")), None)
        if match:
            agregado[match] += float(imp)

    ranking = [
        {
            "variable": col,
            "etiqueta": FEATURE_LABELS.get(col, col),
            "importancia": round(val, 4),
        }
        for col, val in agregado.items()
        if val > 0
    ]
    ranking.sort(key=lambda x: x["importancia"], reverse=True)
    return ranking[:top_n]
