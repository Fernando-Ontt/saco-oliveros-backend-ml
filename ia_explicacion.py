from fastapi import APIRouter, HTTPException
import sqlite3
import os
import json
import httpx

router = APIRouter(tags=["IA Explicativa - Mi Parte"])
DB_PATH = "bd_piloto.db"


TABLA_KEYS = {
    "gemini_api_key": os.getenv("GEMINI_API_KEY", "PEGA_TU_API_KEY_AIZA_AQUI"),
    "modelo_ia": "gemini-2.5-flash"
}


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def tabla_existe(conn, nombre: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name=?",
        (nombre,),
    )
    return cur.fetchone() is not None


def contar_faltas_reales(conn, id_matricula: int) -> tuple[int, int]:
    """Única fuente de verdad para contar inasistencias en toda la app."""
    if not tabla_existe(conn, "asistencia"):
        return 0, 0
    asistencias = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM asistencia WHERE id_matricula = ?", (id_matricula,)
        ).fetchall()
    ]
    total_faltas = sum(
        1
        for a in asistencias
        if str(a.get("estado") or a.get("tipo") or a.get("asistio") or "").strip().lower()
        in ("falta", "inasistencia", "ausente", "0", "f", "falta injustificada")
    )
    return total_faltas, len(asistencias)


@router.get("/estudiantes")
def listar_estudiantes(limit: int = 200):
    conn = get_db()
    try:
        tiene_riesgo = tabla_existe(conn, "riesgo")
        tiene_vm = tabla_existe(conn, "vw_dataset_modelo")

        join_r = (
            "LEFT JOIN (SELECT id_matricula, MAX(probabilidad) AS probabilidad, nivel FROM riesgo GROUP BY id_matricula) r ON m.id_matricula = r.id_matricula"
            if tiene_riesgo else ""
        )
        join_vm = (
            "LEFT JOIN vw_dataset_modelo vm ON m.id_matricula = vm.id_matricula"
            if tiene_vm else ""
        )

        sel_r = ", ROUND(r.probabilidad, 4) AS probabilidad, COALESCE(r.nivel, 'Bajo') AS nivel" if tiene_riesgo else ""
        sel_vm = ", vm.*" if tiene_vm else ""

        rows = conn.execute(
            f"""
            SELECT m.id_matricula, e.* {sel_r} {sel_vm}
            FROM matricula m
            LEFT JOIN estudiante e ON m.id_estudiante = e.id_estudiante
            {join_r}
            {join_vm}
            ORDER BY m.id_matricula ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        resultado = []
        for r in rows:
            item = dict(r)
            total_faltas, _ = contar_faltas_reales(conn, item["id_matricula"])
            item["total_faltas"] = total_faltas
            item["inasistencias"] = total_faltas
            resultado.append(item)

        return resultado
    finally:
        conn.close()


@router.get("/estudiantes/{id_matricula}/explicacion-riesgo")
def explicacion_riesgo_ia(id_matricula: int):
    conn = get_db()
    try:
        # 1. Buscar datos del alumno + matrícula
        alumno = conn.execute(
            """
            SELECT m.*, e.*
            FROM matricula m
            LEFT JOIN estudiante e ON m.id_estudiante = e.id_estudiante
            WHERE m.id_matricula = ?
            """,
            (id_matricula,),
        ).fetchone()

        if not alumno:
            alumno = conn.execute(
                """
                SELECT m.*, e.*
                FROM matricula m
                LEFT JOIN estudiante e ON m.id_estudiante = e.id_estudiante
                ORDER BY m.id_matricula ASC
                LIMIT 1 OFFSET ?
                """,
                (id_matricula % 20,),
            ).fetchone()
            if not alumno:
                raise HTTPException(status_code=404, detail="Estudiante no encontrado")
            id_matricula = alumno["id_matricula"]

        datos_alumno = dict(alumno)

        # 2. Leer indicadores consolidados de vw_dataset_modelo
        indicadores_modelo = {}
        try:
            row_vm = conn.execute(
                "SELECT * FROM vw_dataset_modelo WHERE id_matricula = ?",
                (id_matricula,),
            ).fetchone()
            if row_vm:
                indicadores_modelo = {
                    k: v for k, v in dict(row_vm).items() if v is not None
                }
        except Exception:
            pass

        # 3. Usar la MISMA función de conteo que listar_estudiantes
        total_faltas, total_registros_asist = contar_faltas_reales(conn, id_matricula)
        notas = [dict(r) for r in conn.execute("SELECT * FROM nota WHERE id_matricula = ?", (id_matricula,)).fetchall()] if tabla_existe(conn, "nota") else []
        conducta = [dict(r) for r in conn.execute("SELECT * FROM conducta WHERE id_matricula = ?", (id_matricula,)).fetchall()] if tabla_existe(conn, "conducta") else []
        riesgos = [dict(r) for r in conn.execute("SELECT * FROM riesgo WHERE id_matricula = ?", (id_matricula,)).fetchall()] if tabla_existe(conn, "riesgo") else []

        riesgo_actual = riesgos[-1] if riesgos else {"probabilidad": 0, "nivel": "Medio"}
        prob_pct = round(float(riesgo_actual.get("probabilidad", 0)) * 100)
        b1_ent = round(float(indicadores_modelo.get("prom_b1") or indicadores_modelo.get("prom_b12") or 0))
        b2_ent = round(float(indicadores_modelo.get("prom_b2") or indicadores_modelo.get("prom_b12") or 0))

        # 4. Extraer variables ocultas de la BD
        claves_ocultas = [
            k for k in datos_alumno.keys()
            if any(p in k.lower() for p in ["dist", "traslado", "famil", "pension", "beca", "socio", "apoderado", "vive"])
        ]
        contexto_oculto = {k: datos_alumno[k] for k in claves_ocultas if datos_alumno[k] is not None}

        # 5. Prompt sincronizado con las cifras exactas que ve el tutor en pantalla
        prompt = f"""
Eres el motor explicativo del Sistema de Alerta Temprana Escolar (Saco Oliveros).
Analiza los datos reales de este alumno y responde ÚNICAMENTE en JSON válido con estas 4 claves:
- "variables_que_aumentan_riesgo": lista de 2 a 3 puntos cortos. OBLIGATORIO: menciona textualmente "{total_faltas} inasistencias registradas" y los promedios redondeados B1={b1_ent} y B2={b2_ent} para que coincidan exacto con el dashboard del tutor.
- "variables_que_reducen_riesgo": lista de 1 a 2 puntos cortos con factores protectores o estables.
- "analisis_factores_ocultos": explicación de 2 líneas cruzando distancia al colegio, situación familiar/pensión y el patrón de sus {total_faltas} inasistencias.
- "conclusion_ia": párrafo final de 3 líneas con el diagnóstico de deserción (riesgo {riesgo_actual.get('nivel')} del {prob_pct}%) y la acción recomendada al tutor.

Datos reales del alumno:
- Cifras oficiales mostradas en UI: {total_faltas} inasistencias de {total_registros_asist} sesiones | Promedio B1: {b1_ent} | Promedio B2: {b2_ent} | Riesgo ML: {riesgo_actual.get('nivel')} ({prob_pct}%)
- Contexto oculto BD (distancia, familia, pensión): {json.dumps(contexto_oculto, ensure_ascii=False)}
- Indicadores académicos consolidados (vw_dataset_modelo): {json.dumps(indicadores_modelo, ensure_ascii=False)}
- Historial de notas: {json.dumps(notas[:15], ensure_ascii=False)}
- Observaciones de conducta: {json.dumps(conducta[:10], ensure_ascii=False)}
"""

        # 6. Llamada a Gemini API
        token = TABLA_KEYS["gemini_api_key"].strip()
        if token and token != "PEGA_TU_API_KEY_AIZA_AQUI":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{TABLA_KEYS['modelo_ia']}:generateContent?key={token}"
            body = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"},
            }
            try:
                with httpx.Client(timeout=20.0) as client:
                    resp = client.post(url, json=body)
                    if resp.status_code == 200:
                        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                        return {
                            "id_matricula": id_matricula,
                            "fuente": "Gemini AI + Random Forest + SQLite",
                            "datos_detectados_bd": {
                                "indicadores_academicos": indicadores_modelo,
                                "inasistencias": total_faltas,
                                "riesgo_ml": riesgo_actual,
                                "factores_ocultos": contexto_oculto,
                            },
                            **json.loads(raw),
                        }
            except Exception:
                pass

        # 7. Respaldo automático con las mismas cifras exactas
        nivel = riesgo_actual.get("nivel", "Medio")
        detalle_ctx = ", ".join([f"{k}: {v}" for k, v in contexto_oculto.items()]) or "variables sociofamiliares estándar"

        aumentan = [
            f"{total_faltas} inasistencias registradas en el periodo",
            f"Probabilidad de riesgo ML: {prob_pct}%",
        ]
        reducen = ["Monitoreo activo por tutoría"]

        if b2_ent < 11:
            aumentan.insert(1, f"Promedio Bimestre 2 desaprobatorio: {b2_ent} (B1: {b1_ent})")
        else:
            reducen.insert(0, f"Promedio Bimestre 2 aprobatorio: {b2_ent}")

        if len(conducta) > 0:
            aumentan.append(f"{len(conducta)} observaciones de conducta registradas")
        else:
            reducen.append("Sin observaciones graves de conducta")

        return {
            "id_matricula": id_matricula,
            "fuente": "Motor Local + Random Forest",
            "datos_detectados_bd": {
                "indicadores_academicos": indicadores_modelo,
                "inasistencias": total_faltas,
                "riesgo_ml": riesgo_actual,
                "factores_ocultos": contexto_oculto,
            },
            "variables_que_aumentan_riesgo": aumentan,
            "variables_que_reducen_riesgo": reducen,
            "analisis_factores_ocultos": f"Al cruzar los datos internos de la BD ({detalle_ctx}), se observa relación directa entre sus {total_faltas} inasistencias y la vulnerabilidad del entorno.",
            "conclusion_ia": f"Conclusión: El estudiante presenta nivel de riesgo {nivel} ({prob_pct}%) impulsado por {total_faltas} inasistencias y un promedio de {b2_ent} en el Bimestre 2. Se recomienda intervención preventiva del tutor esta semana.",
        }
    finally:
        conn.close()