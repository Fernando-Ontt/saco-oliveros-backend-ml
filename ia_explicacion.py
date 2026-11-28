from fastapi import APIRouter, HTTPException
import sqlite3
import os
import json
import httpx

from riesgo_solucion_service import RiesgoSolucionService

router = APIRouter(tags=["IA Explicativa - Tutor"])
DB_PATH = "bd_piloto.db"

TABLA_KEYS = {
    "gemini_api_key": "",
    "modelo_ia": "gemini-2.5-flash",
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


def redactar_entorno_para_tutor(datos: dict, total_faltas: int) -> str:
    """Convierte las columnas técnicas de la BD en una lectura humana para el tutor."""
    partes = []

    dist = datos.get("distancia_km")
    tiempo = datos.get("tiempo_traslado_min")
    if dist is not None and tiempo is not None:
        partes.append(f"vive a {dist} km del colegio ({tiempo} min de traslado)")
    elif dist is not None:
        partes.append(f"vive a {dist} km del colegio")

    convive = datos.get("convive_con")
    if convive:
        partes.append(f"convive con {str(convive).lower()}")

    apoyo = datos.get("apoyo_familiar")
    if apoyo is not None:
        nivel_apoyo = "bajo" if int(apoyo) <= 2 else ("moderado" if int(apoyo) == 3 else "alto")
        partes.append(f"cuenta con apoyo familiar {nivel_apoyo}")

    beca = datos.get("beca")
    if beca is not None and int(beca) == 0:
        partes.append("no cuenta con beca")

    if not partes:
        return (
            f"El entorno sociofamiliar se mantiene dentro de parámetros habituales, "
            f"aunque conviene conversar con el apoderado sobre las {total_faltas} inasistencias registradas."
        )

    resumen = ", ".join(partes)
    resumen = resumen[0].upper() + resumen[1:]
    if total_faltas >= 10:
        return (
            f"{resumen}. Esta combinación de traslado y dinámica en casa puede estar "
            f"dificultando su continuidad diaria ({total_faltas} faltas acumuladas en el periodo)."
        )
    return f"{resumen}. Este contexto permite orientar mejor la entrevista con el apoderado y el acompañamiento en tutoría."


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

        # 2. Consultar RiesgoSolucionService (motivo principal, factores y soluciones históricas)
        servicio_soluciones = RiesgoSolucionService(conn)
        diag_soluciones = servicio_soluciones.obtener_diagnostico_y_soluciones(id_matricula)
        soluciones_ctx = diag_soluciones.model_dump() if diag_soluciones else {}

        # 3. Leer indicadores consolidados de vw_dataset_modelo
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

        # 4. Conteo oficial de faltas y riesgo actual
        total_faltas, total_registros_asist = contar_faltas_reales(conn, id_matricula)
        conducta = [dict(r) for r in conn.execute("SELECT * FROM observacion_conducta WHERE id_matricula = ?", (id_matricula,)).fetchall()] if tabla_existe(conn, "observacion_conducta") else []
        riesgos = [dict(r) for r in conn.execute("SELECT * FROM riesgo WHERE id_matricula = ?", (id_matricula,)).fetchall()] if tabla_existe(conn, "riesgo") else []

        riesgo_actual = riesgos[-1] if riesgos else {"probabilidad": 0, "nivel": "Medio"}
        prob_pct = round(float(riesgo_actual.get("probabilidad", 0)) * 100)
        b1_ent = round(float(indicadores_modelo.get("prom_b1") or indicadores_modelo.get("prom_b12") or 0))
        b2_ent = round(float(indicadores_modelo.get("prom_b2") or indicadores_modelo.get("prom_b12") or 0))

        datos_completos = {**datos_alumno, **indicadores_modelo}
        texto_entorno = redactar_entorno_para_tutor(datos_completos, total_faltas)
        nombre_alumno = f"{datos_alumno.get('nombres', 'El estudiante')} {datos_alumno.get('apellidos', '')}".strip()

        # 5. Prompt enriquecido para una conclusión completa, pedagógica y accionable
        prompt = f"""
Eres un coordinador de tutoría y orientación educativa que redacta informes claros, empáticos y bien fundamentados para un TUTOR ESCOLAR de secundaria (sin jerga técnica, sin siglas de sistemas y sin nombres de columnas de BD).
Analiza la situación de {nombre_alumno} y responde ÚNICAMENTE en JSON válido con estas 4 claves:
- "variables_que_aumentan_riesgo": lista de 3 puntos concretos y detallados sobre lo que afecta al alumno (inasistencias exactas: {total_faltas}, notas B1={b1_ent} y B2={b2_ent}, cursos desaprobados o atrasos en pensiones si existen).
- "variables_que_reducen_riesgo": lista de 2 puntos concretos con fortalezas o factores protectores del estudiante.
- "analisis_factores_ocultos": párrafo de 2 a 3 líneas explicando en lenguaje natural cómo su traslado, convivencia y apoyo en casa se relacionan con su asistencia y rendimiento.
- "conclusion_ia": texto detallado y estructurado en 3 párrafos breves separados por salto de línea (\\n\\n):
  Párrafo 1 (Diagnóstico integral): Explica por qué {nombre_alumno} está en riesgo {riesgo_actual.get('nivel')} ({prob_pct}%), conectando el motivo principal ({soluciones_ctx.get('motivo_principal') or 'rendimiento y asistencia'}) con la evolución entre Bimestre 1 y 2.
  Párrafo 2 (Historial de seguimiento): Indica si ya se aplicaron tutorías previas y cuál fue su resultado, o si requiere primera intervención inmediata.
  Párrafo 3 (Plan de acción recomendado): Recomienda la solución más efectiva según los casos del colegio (citando su porcentaje de mejora y casos similares) y qué compromiso concreto debe acordar el tutor con el estudiante y su apoderado esta semana.

Datos del estudiante:
- Nombre: {nombre_alumno}
- Resumen académico: {total_faltas} inasistencias de {total_registros_asist} sesiones | Promedio B1: {b1_ent} | Promedio B2: {b2_ent} | Cursos bajo 11 en B2: {indicadores_modelo.get('cursos_bajo_11_b2', 0)} | Meses mora: {indicadores_modelo.get('meses_mora_b12', 0)} | Nivel de riesgo: {riesgo_actual.get('nivel')} ({prob_pct}%)
- Contexto familiar y traslado: {texto_entorno}
- Factores y soluciones históricas del colegio: {json.dumps(soluciones_ctx, ensure_ascii=False)}
"""

        # 6. Llamada a Gemini API
        token = (TABLA_KEYS.get("gemini_api_key") or "").strip()
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
                            "fuente": "Análisis IA + Historial del colegio",
                            "datos_detectados_bd": {
                                "indicadores_academicos": indicadores_modelo,
                                "inasistencias": total_faltas,
                                "riesgo_ml": riesgo_actual,
                                "soluciones_historicas": soluciones_ctx,
                            },
                            **json.loads(raw),
                        }
            except Exception:
                pass

        # 7. Respaldo inteligente detallado y estructurado para el tutor
        nivel = riesgo_actual.get("nivel", "Medio")
        motivo_ppal = soluciones_ctx.get("motivo_principal") or "Rendimiento y asistencia"
        soluciones = soluciones_ctx.get("soluciones", [])
        ya_aplicadas = soluciones_ctx.get("ya_aplicadas", [])

        cursos_desap = int(indicadores_modelo.get("cursos_bajo_11_b2") or 0)
        meses_mora = int(indicadores_modelo.get("meses_mora_b12") or 0)
        pct_asist = round(float(indicadores_modelo.get("pct_asistencia_b12") or 0), 1)

        aumentan: list[str] = []
        if total_faltas > 0:
            detalle_asist = f" ({pct_asist}% de asistencia acumulada)" if pct_asist > 0 else ""
            aumentan.append(f"{total_faltas} inasistencias registradas en el periodo{detalle_asist}.")

        if b2_ent < 11:
            detalle_cursos = f", con {cursos_desap} curso(s) desaprobado(s)" if cursos_desap > 0 else ""
            aumentan.append(
                f"Promedio desaprobatorio en el Bimestre 2 ({b2_ent} de 20; venía de {b1_ent} en el Bimestre 1){detalle_cursos}."
            )
        elif b2_ent < b1_ent:
            aumentan.append(f"Descenso en el promedio bimestral (bajó de {b1_ent} a {b2_ent} puntos).")

        if meses_mora > 0:
            aumentan.append(f"Registra {meses_mora} mes(es) de atraso en pensiones, factor que suele asociarse a riesgo de retiro.")

        if len(conducta) > 0:
            aumentan.append(f"Presenta {len(conducta)} observación(es) de conducta registradas en aula.")

        if not aumentan:
            aumentan.append(f"Indicadores generales compatibles con alerta preventiva ({prob_pct}%).")

        reducen = ["Seguimiento activo y acompañamiento disponible desde tutoría."]
        if b2_ent >= 11:
            reducen.insert(0, f"Mantiene promedio general aprobatorio ({b2_ent} de 20) en el Bimestre 2.")
        if int(datos_completos.get("apoyo_familiar") or 0) >= 3:
            reducen.append("Cuenta con respaldo y disposición de diálogo desde su entorno familiar.")
        if len(conducta) == 0:
            reducen.append("Mantiene buen comportamiento en clase, sin reportes disciplinarios en el periodo.")

        # Redacción estructurada en 3 párrafos para el tutor
        diff_notas = b2_ent - b1_ent
        evolucion_txt = (
            f"una caída de {abs(diff_notas)} punto(s) respecto al Bimestre 1 (pasando de {b1_ent} a {b2_ent})"
            if diff_notas < 0
            else (
                f"un avance de {diff_notas} punto(s) respecto al Bimestre 1 (de {b1_ent} a {b2_ent})"
                if diff_notas > 0
                else f"un promedio estable de {b2_ent} puntos entre ambos bimestres"
            )
        )

        parrafo_1 = (
            f"1. Diagnóstico integral: {nombre_alumno} presenta un nivel de riesgo {nivel.lower()} ({prob_pct}%), "
            f"impulsado principalmente por {motivo_ppal.lower()}. En el último corte muestra {evolucion_txt} "
            f"y acumula {total_faltas} inasistencias, lo que exige atención prioritaria para evitar que el rezago académico se profundice."
        )

        if ya_aplicadas:
            historial_items = [
                f"«{a['accion']}» ({a['fecha']}, {'con mejora' if a.get('mejoro_sim') else 'sin mejora suficiente'})"
                for a in ya_aplicadas[-2:]
            ]
            parrafo_2 = (
                f"2. Antecedentes de acompañamiento: El estudiante ya recibió seguimiento previo mediante "
                f"{' y '.join(historial_items)}. Dado que los indicadores de riesgo persisten, es necesario ajustar la estrategia de intervención."
            )
        else:
            parrafo_2 = (
                "2. Antecedentes de acompañamiento: Aún no registra intervenciones formales de tutoría para este motivo, "
                "por lo que una acción temprana esta semana tiene alta probabilidad de revertir la tendencia."
            )

        mejor_sol = next((s for s in soluciones if s.get("recomendada")), soluciones[0] if soluciones else None)
        if mejor_sol:
            alcance_txt = "en 3.º de Secundaria" if mejor_sol.get("alcance") == "3.º" else "en todo el colegio"
            parrafo_3 = (
                f"3. Plan de acción sugerido: Se recomienda aplicar «{mejor_sol['nombre']}», estrategia que logró mejoras "
                f"en el {mejor_sol['pct_mejoro']}% de {mejor_sol['n_casos']} casos similares {alcance_txt} "
                f"(con una recuperación promedio de +{mejor_sol['variacion_media']} puntos). Se sugiere convocar al apoderado "
                f"para justificar las inasistencias, revisar las causas del traslado/entorno y firmar metas quincenales de asistencia y entregas."
            )
        else:
            parrafo_3 = (
                "3. Plan de acción sugerido: Se recomienda programar una entrevista individual con el estudiante y citar "
                "al apoderado esta semana para establecer un compromiso de asistencia regular y refuerzo en los cursos con menor promedio."
            )

        conclusion_detallada = f"{parrafo_1}\n\n{parrafo_2}\n\n{parrafo_3}"

        return {
            "id_matricula": id_matricula,
            "fuente": "Orientación automatizada + Historial del colegio",
            "datos_detectados_bd": {
                "indicadores_academicos": indicadores_modelo,
                "inasistencias": total_faltas,
                "riesgo_ml": riesgo_actual,
                "soluciones_historicas": soluciones_ctx,
            },
            "variables_que_aumentan_riesgo": aumentan[:3],
            "variables_que_reducen_riesgo": reducen[:3],
            "analisis_factores_ocultos": texto_entorno,
            "conclusion_ia": conclusion_detallada,
        }
    finally:
        conn.close()