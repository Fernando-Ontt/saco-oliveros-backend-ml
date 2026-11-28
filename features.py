import sqlite3
import pandas as pd

# Configuración única del año escolar (cambiar BIMESTRE_ACTUAL a 4 cuando inicie el 4.º bimestre)
TOTAL_BIMESTRES = 4
BIMESTRE_ACTUAL = 3

PESOS_EVALUACION = {
    "Practica": 0.30,
    "Simulacro": 0.20,
    "Examen": 0.50,
}

UMBRAL_RIESGO = 0.4064030005224275

FEATURES = [
    "grado", "edad", "trabaja", "repitio_grado", "cursos_desaprobados_previos", "beca",
    "tipo_ingreso", "colegio_procedencia", "nota_diagnostico_ingreso", "zona",
    "distancia_km", "tiempo_traslado_min", "convive_con", "educacion_apoderado",
    "apoyo_familiar", "internet_casa", "horas_estudio_sem", "aspira_universidad",
    "relacion_familiar", "salud", "actividades_extra", "nro_hermanos",
    "prom_b1", "prom_b2", "prom_b12", "tendencia_notas", "min_curso_b2",
    "cursos_bajo_11_b2", "prom_matematica", "prom_comunicacion", "varianza_notas",
    "prom_simulacros", "pct_asistencia_b12", "pct_asistencia_b1", "pct_asistencia_b2",
    "tendencia_asistencia", "pct_tardanzas", "pct_faltas_injustificadas", "total_faltas",
    "pct_faltas_lun_vie", "faltas_consecutivas_max", "n_obs_positivas", "n_obs_leves",
    "n_obs_graves", "n_obs_socioemocional", "n_derivaciones_docente", "n_tutorias",
    "n_derivaciones_psicologia", "n_contactos_familia", "n_citaciones",
    "tasa_respuesta_familia", "meses_mora_b12", "meses_moroso_b12",
    "max_dias_atraso_b12", "bienestar_motivacion", "bienestar_estres",
    "bienestar_pares", "bienestar_familia", "bienestar_seguridad",
]

QUERY_FEATURES = """
WITH
corte AS (
    SELECT m.id_matricula,
           (SELECT fecha_fin FROM periodo WHERE anio = m.anio AND bimestre = 1) AS fin_b1,
           (SELECT fecha_fin FROM periodo WHERE anio = m.anio AND bimestre = 2) AS fin_b2
    FROM matricula m
    WHERE m.id_matricula = :id_matricula
),
nt AS (
    SELECT c.id_matricula, c.id_curso, p.bimestre, c.tipo_evaluacion, AVG(c.nota) AS nota
    FROM calificacion c JOIN periodo p ON p.id_periodo = c.id_periodo
    WHERE c.id_matricula = :id_matricula AND p.bimestre <= :bimestre_actual AND c.nota IS NOT NULL
    GROUP BY 1, 2, 3, 4
),
pcb AS (
    SELECT id_matricula, id_curso, bimestre,
           SUM(nota * CASE tipo_evaluacion WHEN 'Practica' THEN 0.3 WHEN 'Simulacro' THEN 0.2 ELSE 0.5 END)
         / SUM(CASE tipo_evaluacion WHEN 'Practica' THEN 0.3 WHEN 'Simulacro' THEN 0.2 ELSE 0.5 END) AS prom
    FROM nt GROUP BY 1, 2, 3
),
acad AS (
    SELECT p.id_matricula,
           AVG(CASE WHEN p.bimestre = 1 THEN p.prom END)                   AS prom_b1,
           AVG(CASE WHEN p.bimestre = 2 THEN p.prom END)                   AS prom_b2,
           AVG(CASE WHEN p.bimestre = :bimestre_actual THEN p.prom END)    AS prom_actual,
           AVG(p.prom)                                                     AS prom_b12,
           COALESCE(
               MIN(CASE WHEN p.bimestre = :bimestre_actual THEN p.prom END),
               MIN(CASE WHEN p.bimestre = 2 THEN p.prom END)
           )                                                               AS min_curso_b2,
           CASE
               WHEN SUM(CASE WHEN p.bimestre = :bimestre_actual THEN 1 ELSE 0 END) > 0
               THEN SUM(CASE WHEN p.bimestre = :bimestre_actual AND p.prom < 11 THEN 1 ELSE 0 END)
               ELSE SUM(CASE WHEN p.bimestre = 2 AND p.prom < 11 THEN 1 ELSE 0 END)
           END                                                             AS cursos_bajo_11_b2,
           AVG(CASE WHEN cu.nombre = 'Matemática'   THEN p.prom END)       AS prom_matematica,
           AVG(CASE WHEN cu.nombre = 'Comunicación' THEN p.prom END)       AS prom_comunicacion
    FROM pcb p JOIN curso cu ON cu.id_curso = p.id_curso
    GROUP BY p.id_matricula
),
dist AS (
    SELECT c.id_matricula,
           AVG(c.nota * c.nota) - AVG(c.nota) * AVG(c.nota)               AS varianza_notas,
           AVG(CASE WHEN c.tipo_evaluacion = 'Simulacro' THEN c.nota END) AS prom_simulacros
    FROM calificacion c JOIN periodo p ON p.id_periodo = c.id_periodo
    WHERE c.id_matricula = :id_matricula AND p.bimestre <= :bimestre_actual AND c.nota IS NOT NULL
    GROUP BY c.id_matricula
),
asi AS (
    SELECT a.id_matricula, a.fecha, a.estado, k.fin_b1,
           CASE WHEN a.estado LIKE 'Falta%' THEN 1 ELSE 0 END AS falta,
           ROW_NUMBER() OVER (PARTITION BY a.id_matricula ORDER BY a.fecha) AS rn
    FROM asistencia a JOIN corte k ON k.id_matricula = a.id_matricula
    WHERE a.fecha <= k.fin_b2
),
rachas AS (
    SELECT id_matricula, MAX(largo) AS faltas_consecutivas_max
    FROM (SELECT id_matricula, COUNT(*) AS largo
          FROM (SELECT id_matricula,
                       rn - ROW_NUMBER() OVER (PARTITION BY id_matricula ORDER BY fecha) AS grupo
                FROM asi WHERE falta = 1)
          GROUP BY id_matricula, grupo)
    GROUP BY id_matricula
),
asis AS (
    SELECT id_matricula,
           100.0 * (1 - AVG(falta))                                                  AS pct_asistencia_b12,
           100.0 * (1 - AVG(CASE WHEN fecha <= fin_b1 THEN falta END))               AS pct_asistencia_b1,
           100.0 * (1 - AVG(CASE WHEN fecha >  fin_b1 THEN falta END))               AS pct_asistencia_b2,
           100.0 * AVG(CASE WHEN estado = 'Tardanza' THEN 1.0 ELSE 0 END)            AS pct_tardanzas,
           100.0 * AVG(CASE WHEN estado = 'Falta injustificada' THEN 1.0 ELSE 0 END) AS pct_faltas_injustificadas,
           SUM(falta)                                                                AS total_faltas,
           100.0 * SUM(CASE WHEN falta = 1 AND strftime('%w', fecha) IN ('1','5') THEN 1 ELSE 0 END)
                 / NULLIF(SUM(falta), 0)                                             AS pct_faltas_lun_vie
    FROM asi GROUP BY id_matricula
),
obs AS (
    SELECT o.id_matricula,
           SUM(o.tipo = 'Positiva') AS n_obs_positivas, SUM(o.tipo = 'Leve') AS n_obs_leves,
           SUM(o.tipo = 'Grave') AS n_obs_graves, SUM(o.tipo = 'Socioemocional') AS n_obs_socioemocional,
           SUM(o.deriva_a_tutor) AS n_derivaciones_docente
    FROM observacion_conducta o JOIN corte k ON k.id_matricula = o.id_matricula
    WHERE o.fecha <= k.fin_b2 GROUP BY o.id_matricula
),
tut AS (
    SELECT t.id_matricula, COUNT(*) AS n_tutorias, SUM(t.derivacion = 'Psicologia') AS n_derivaciones_psicologia
    FROM tutoria t JOIN corte k ON k.id_matricula = t.id_matricula
    WHERE t.fecha <= k.fin_b2 GROUP BY t.id_matricula
),
fam AS (
    SELECT f.id_matricula, COUNT(*) AS n_contactos_familia, SUM(f.tipo = 'Citacion') AS n_citaciones,
           AVG(f.respondio) AS tasa_respuesta_familia
    FROM contacto_familia f JOIN corte k ON k.id_matricula = f.id_matricula
    WHERE f.fecha <= k.fin_b2 GROUP BY f.id_matricula
),
pag AS (
    SELECT id_matricula, SUM(estado <> 'Pagado') AS meses_mora_b12, SUM(estado = 'Moroso') AS meses_moroso_b12,
           MAX(dias_atraso) AS max_dias_atraso_b12
    FROM pago_pension WHERE id_matricula = :id_matricula AND mes IN (3, 4, 5, 6, 7) GROUP BY id_matricula
),
bien AS (
    SELECT e.id_matricula, AVG(e.motivacion) AS bienestar_motivacion, AVG(e.estres) AS bienestar_estres,
           AVG(e.relacion_pares) AS bienestar_pares, AVG(e.relacion_familia) AS bienestar_familia,
           AVG(e.seguridad) AS bienestar_seguridad
    FROM encuesta_bienestar e JOIN periodo p ON p.id_periodo = e.id_periodo
    WHERE e.id_matricula = :id_matricula AND p.bimestre <= 2 GROUP BY e.id_matricula
)
SELECT
    m.id_matricula, m.id_estudiante, m.anio, s.grado,
    m.edad, m.trabaja, m.repitio_grado, m.cursos_desaprobados_previos, m.beca,
    e.tipo_ingreso, e.colegio_procedencia, e.nota_diagnostico_ingreso, e.zona, e.distancia_km, e.tiempo_traslado_min,
    e.convive_con, e.educacion_apoderado, e.apoyo_familiar, e.internet_casa, e.horas_estudio_sem,
    e.aspira_universidad, e.relacion_familiar, e.salud, e.actividades_extra, e.nro_hermanos,
    a.prom_b1, a.prom_b2, a.prom_b12, COALESCE(a.prom_actual, a.prom_b2) - a.prom_b1 AS tendencia_notas,
    a.min_curso_b2, a.cursos_bajo_11_b2, a.prom_matematica, a.prom_comunicacion,
    d.varianza_notas, d.prom_simulacros,
    s2.pct_asistencia_b12, s2.pct_asistencia_b1, s2.pct_asistencia_b2,
    s2.pct_asistencia_b2 - s2.pct_asistencia_b1 AS tendencia_asistencia,
    s2.pct_tardanzas, s2.pct_faltas_injustificadas, s2.total_faltas, s2.pct_faltas_lun_vie,
    COALESCE(r.faltas_consecutivas_max, 0) AS faltas_consecutivas_max,
    COALESCE(o.n_obs_positivas, 0) AS n_obs_positivas, COALESCE(o.n_obs_leves, 0) AS n_obs_leves,
    COALESCE(o.n_obs_graves, 0) AS n_obs_graves, COALESCE(o.n_obs_socioemocional, 0) AS n_obs_socioemocional,
    COALESCE(o.n_derivaciones_docente, 0) AS n_derivaciones_docente,
    COALESCE(t.n_tutorias, 0) AS n_tutorias, COALESCE(t.n_derivaciones_psicologia, 0) AS n_derivaciones_psicologia,
    COALESCE(f.n_contactos_familia, 0) AS n_contactos_familia, COALESCE(f.n_citaciones, 0) AS n_citaciones,
    f.tasa_respuesta_familia,
    COALESCE(p.meses_mora_b12, 0) AS meses_mora_b12, COALESCE(p.meses_moroso_b12, 0) AS meses_moroso_b12,
    COALESCE(p.max_dias_atraso_b12, 0) AS max_dias_atraso_b12,
    b.bienestar_motivacion, b.bienestar_estres, b.bienestar_pares, b.bienestar_familia, b.bienestar_seguridad
FROM matricula m
JOIN estudiante e ON e.id_estudiante = m.id_estudiante
JOIN seccion s    ON s.id_seccion    = m.id_seccion
LEFT JOIN acad a   ON a.id_matricula  = m.id_matricula
LEFT JOIN dist d   ON d.id_matricula  = m.id_matricula
LEFT JOIN asis s2  ON s2.id_matricula = m.id_matricula
LEFT JOIN rachas r ON r.id_matricula  = m.id_matricula
LEFT JOIN obs o    ON o.id_matricula  = m.id_matricula
LEFT JOIN tut t    ON t.id_matricula  = m.id_matricula
LEFT JOIN fam f    ON f.id_matricula  = m.id_matricula
LEFT JOIN pag p    ON p.id_matricula  = m.id_matricula
LEFT JOIN bien b   ON b.id_matricula  = m.id_matricula
WHERE m.id_matricula = :id_matricula
"""


CAT_COLS = {"tipo_ingreso", "colegio_procedencia", "zona", "convive_con", "educacion_apoderado"}


def extraer_features_matricula(conn: sqlite3.Connection, id_matricula: int) -> pd.DataFrame:
    """Devuelve un DataFrame de 1 fila con las 59 columnas exactas del modelo y tipos idénticos a Colab."""
    df = pd.read_sql_query(
        QUERY_FEATURES,
        conn,
        params={"id_matricula": id_matricula, "bimestre_actual": BIMESTRE_ACTUAL},
    )
    if df.empty:
        raise ValueError(f"No se encontró la matrícula con id_matricula={id_matricula}")
    df = df[FEATURES].copy()
    for col in FEATURES:
        if col in CAT_COLS:
            df[col] = df[col].where(df[col].notna(), other=pd.NA).astype("object")
            df[col] = df[col].apply(lambda v: None if pd.isna(v) else str(v))
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def clasificar_nivel(probabilidad: float) -> str:
    """Clasifica la probabilidad en Alto, Medio o Bajo usando el umbral calibrado."""
    if probabilidad >= UMBRAL_RIESGO:
        return "Alto"
    elif probabilidad >= UMBRAL_RIESGO * 0.5:
        return "Medio"
    return "Bajo"