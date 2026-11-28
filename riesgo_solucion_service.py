import re
import sqlite3
from datetime import date
from typing import Optional

from pydantic import BaseModel


# --- Modelos Pydantic ---

class RiesgoResumen(BaseModel):
    nivel: str
    probabilidad: float
    periodo: int


class FactorDetalle(BaseModel):
    factor: str
    motivo: str
    peso: float


class SolucionRecomendada(BaseModel):
    nombre: str
    n_casos: int
    pct_mejoro: int
    variacion_media: float
    alcance: str
    recomendada: bool


class AccionAplicada(BaseModel):
    accion: str
    fecha: str
    mejoro_sim: Optional[bool] = None


class RiesgoSolucionResponse(BaseModel):
    id_matricula: int
    riesgo: Optional[RiesgoResumen] = None
    motivo_principal: Optional[str] = None
    factores: list[FactorDetalle] = []
    soluciones: list[SolucionRecomendada] = []
    ya_aplicadas: list[AccionAplicada] = []


# --- Clase de Servicio ---

class RiesgoSolucionService:
    FACTOR_A_MOTIVO: dict[str, str] = {
        "Rendimiento en B1 y B2": "Bajo rendimiento",
        "Asistencia en B1 y B2": "Inasistencias",
        "Contexto del alumno y la familia": "Situación familiar",
        "Conducta, tutoría y familia": "Conducta",
        "Bienestar": "Socioemocional",
        "Pensiones (marzo a julio)": "Pensiones",
    }

    REGEX_FACTOR = re.compile(r"\s*(.*?)\s*\((-?[\d.]+) puntos\)")

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.row_factory = sqlite3.Row

    def existe_matricula(self, id_matricula: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM matricula WHERE id_matricula = ?",
            (id_matricula,),
        ).fetchone()
        return row is not None

    def _clasificar_motivo_texto(self, texto: str) -> str:
        t = texto.lower()
        if any(k in t for k in ("promedio", "curso", "desaprobado", "rendimiento", "nota")):
            return "Bajo rendimiento"
        if any(k in t for k in ("asistencia", "falta", "inasistencia", "tardanza")):
            return "Inasistencias"
        if any(k in t for k in ("socioemocional", "bienestar", "estres", "estrés", "motivacion", "motivación")):
            return "Socioemocional"
        if any(k in t for k in ("conducta", "observacion", "observación", "disciplina")):
            return "Conducta"
        if any(k in t for k in ("mora", "pension", "pensión", "atraso")):
            return "Pensiones"
        if any(k in t for k in ("familia", "apoderado", "distancia", "traslado", "contexto")):
            return "Situación familiar"
        return texto

    def parsear_factores(
        self, texto_factores: Optional[str]
    ) -> tuple[Optional[str], list[FactorDetalle]]:
        if not texto_factores or not texto_factores.strip():
            return None, []

        factores_parseados: list[FactorDetalle] = []
        segmentos = [s.strip() for s in texto_factores.split(";") if s.strip()]

        for idx, segmento in enumerate(segmentos):
            match = self.REGEX_FACTOR.fullmatch(segmento)
            if match:
                nombre_factor = match.group(1).strip()
                peso = float(match.group(2))
                motivo = self.FACTOR_A_MOTIVO.get(
                    nombre_factor, self._clasificar_motivo_texto(nombre_factor)
                )
            else:
                # Formato descriptivo de bd_piloto.db (ej. "Promedio B1-B2 bajo (11.94)")
                nombre_factor = segmento
                motivo = self._clasificar_motivo_texto(segmento)
                # Peso decreciente según orden de prioridad en la cadena de factores
                peso = round(max(0.10, 0.50 - idx * 0.08), 2)

            factores_parseados.append(
                FactorDetalle(factor=nombre_factor, motivo=motivo, peso=peso)
            )

        if not factores_parseados:
            return None, []

        principal = max(factores_parseados, key=lambda f: abs(f.peso))
        return principal.motivo, factores_parseados

    def _obtener_ultimo_riesgo(self, id_matricula: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT id_periodo, probabilidad, nivel, factores
            FROM riesgo
            WHERE id_matricula = ?
            ORDER BY id_periodo DESC, fecha_prediccion DESC
            LIMIT 1
            """,
            (id_matricula,),
        ).fetchone()

    def _obtener_acciones_aplicadas(
        self, id_matricula: int, motivo: str, fecha_corte: str
    ) -> list[AccionAplicada]:
        rows = self.conn.execute(
            """
            SELECT
                t.accion,
                t.fecha,
                sa.mejoro_sim
            FROM tutoria t
            LEFT JOIN seguimiento_accion sa ON sa.id_tutoria = t.id_tutoria
            WHERE t.id_matricula = ?
              AND t.motivo = ?
              AND t.fecha <= ?
            ORDER BY t.fecha ASC, t.id_tutoria ASC
            """,
            (id_matricula, motivo, fecha_corte),
        ).fetchall()

        return [
            AccionAplicada(
                accion=row["accion"],
                fecha=row["fecha"],
                mejoro_sim=bool(row["mejoro_sim"]) if row["mejoro_sim"] is not None else None,
            )
            for row in rows
        ]

    def _calcular_soluciones(
        self, motivo: str, acciones_previas: set[str]
    ) -> list[SolucionRecomendada]:
        if motivo == "Pensiones":
            return []

        rows = self.conn.execute(
            """
            SELECT
                t.accion AS nombre,
                CASE
                    WHEN SUM(CASE WHEN s.grado = 3 THEN 1 ELSE 0 END) >= 15
                    THEN SUM(CASE WHEN s.grado = 3 THEN 1 ELSE 0 END)
                    ELSE COUNT(*)
                END AS n_casos,
                CASE
                    WHEN SUM(CASE WHEN s.grado = 3 THEN 1 ELSE 0 END) >= 15
                    THEN CAST(ROUND(AVG(CASE WHEN s.grado = 3 THEN sa.mejoro_sim END) * 100) AS INTEGER)
                    ELSE CAST(ROUND(AVG(sa.mejoro_sim) * 100) AS INTEGER)
                END AS pct_mejoro,
                CASE
                    WHEN SUM(CASE WHEN s.grado = 3 THEN 1 ELSE 0 END) >= 15
                    THEN ROUND(AVG(CASE WHEN s.grado = 3 THEN sa.variacion_prom_sim END), 2)
                    ELSE ROUND(AVG(sa.variacion_prom_sim), 2)
                END AS variacion_media,
                CASE
                    WHEN SUM(CASE WHEN s.grado = 3 THEN 1 ELSE 0 END) >= 15
                    THEN '3.º'
                    ELSE 'colegio'
                END AS alcance
            FROM tutoria t
            INNER JOIN seguimiento_accion sa ON sa.id_tutoria = t.id_tutoria
            INNER JOIN matricula m ON m.id_matricula = t.id_matricula
            INNER JOIN seccion s ON s.id_seccion = m.id_seccion
            WHERE t.motivo = ?
              AND m.anio IN (2024, 2025, 2026)
              AND t.accion IS NOT NULL
              AND TRIM(t.accion) != ''
            GROUP BY t.accion
            ORDER BY pct_mejoro DESC, variacion_media DESC
            """,
            (motivo,),
        ).fetchall()

        if not rows:
            return []

        idx_recomendada = 0
        for idx, row in enumerate(rows):
            if row["nombre"] not in acciones_previas:
                idx_recomendada = idx
                break

        return [
            SolucionRecomendada(
                nombre=row["nombre"],
                n_casos=int(row["n_casos"]),
                pct_mejoro=int(row["pct_mejoro"]),
                variacion_media=float(row["variacion_media"]),
                alcance=row["alcance"],
                recomendada=(idx == idx_recomendada),
            )
            for idx, row in enumerate(rows)
        ]

    def obtener_diagnostico_y_soluciones(
        self, id_matricula: int
    ) -> Optional[RiesgoSolucionResponse]:
        """Retorna None si la matrícula no existe; en caso contrario arma la respuesta completa."""
        if not self.existe_matricula(id_matricula):
            return None

        riesgo_row = self._obtener_ultimo_riesgo(id_matricula)
        if not riesgo_row:
            return RiesgoSolucionResponse(id_matricula=id_matricula)

        riesgo_info = RiesgoResumen(
            nivel=riesgo_row["nivel"],
            probabilidad=round(float(riesgo_row["probabilidad"]), 3),
            periodo=int(riesgo_row["id_periodo"]),
        )

        motivo_principal, factores = self.parsear_factores(riesgo_row["factores"])
        if not motivo_principal:
            return RiesgoSolucionResponse(
                id_matricula=id_matricula,
                riesgo=riesgo_info,
                motivo_principal=None,
                factores=[],
                soluciones=[],
                ya_aplicadas=[],
            )

        hoy_iso = date.today().isoformat()
        ya_aplicadas = self._obtener_acciones_aplicadas(
            id_matricula, motivo_principal, hoy_iso
        )
        acciones_previas = {item.accion for item in ya_aplicadas}
        soluciones = self._calcular_soluciones(motivo_principal, acciones_previas)

        return RiesgoSolucionResponse(
            id_matricula=id_matricula,
            riesgo=riesgo_info,
            motivo_principal=motivo_principal,
            factores=factores,
            soluciones=soluciones,
            ya_aplicadas=ya_aplicadas,
        )