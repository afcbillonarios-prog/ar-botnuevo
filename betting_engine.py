"""
Modulo de apuestas deportivas para la final Junior vs Nacional.
Soporta: 1X2, Over/Under, Ambos Anotan, Marcador Exacto, Handicap Asiatico
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

@dataclass
class Apuesta:
    id: str
    nombre: str
    tipo: str          # "1X2", "over_under", "btts", "exacto", "handicap"
    seleccion: str     # ej: "1", "X", "2", "over_2.5", "exacto_2-1"
    cuota: float
    probabilidad: float
    valor_esperado: float
    stake_sugerido: float  # % del bankroll

class BettingEngine:
    """
    Motor de apuestas que calcula cuotas justas, valor esperado,
    y recomendaciones basadas en las probabilidades del modelo.
    """

    def __init__(self, bankroll: float = 100.0):
        self.bankroll = bankroll
        self.apuestas: List[Apuesta] = []
        self.seleccionadas: Dict[str, float] = {}  # apuesta_id -> stake

    @staticmethod
    def cuota_justa(prob: float) -> float:
        """Cuota justa basada en probabilidad (sin margen)"""
        return 1.0 / prob if prob > 0 else 999.0

    @staticmethod
    def cuota_con_margen(prob: float, margen: float = 0.06) -> float:
        """Cuota con margen de la casa (tipico 6%)"""
        return (1.0 - margen) / prob if prob > 0 else 999.0

    @staticmethod
    def valor_esperado(cuota: float, prob_real: float) -> float:
        """Valor esperado de una apuesta. > 0 significa value."""
        return (cuota * prob_real) - 1.0

    @staticmethod
    def stake_kelly(cuota: float, prob_real: float, bankroll: float) -> float:
        """Criterio de Kelly para tamaño de apuesta"""
        ve = BettingEngine.valor_esperado(cuota, prob_real)
        if ve <= 0:
            return 0.0
        b = cuota - 1.0
        p = prob_real
        q = 1 - p
        kelly = (b * p - q) / b
        return max(0.0, min(kelly * bankroll * 0.25, bankroll * 0.05))

    def generar_apuestas_1x2(self, prob_local: float, prob_emp: float, prob_visita: float):
        """Genera apuestas tipo 1X2 con cuotas de mercado realistas"""
        margen = 0.06
        cuotas = {
            "1": self.cuota_con_margen(prob_local, margen),
            "X": self.cuota_con_margen(prob_emp, margen),
            "2": self.cuota_con_margen(prob_visita, margen),
        }
        # Simular cuota de mercado (ligera variacion de la justa)
        cuota_mercado = {
            "1": round(cuotas["1"] * (1 + np.random.uniform(-0.03, 0.03)), 2),
            "X": round(cuotas["X"] * (1 + np.random.uniform(-0.03, 0.03)), 2),
            "2": round(cuotas["2"] * (1 + np.random.uniform(-0.03, 0.03)), 2),
        }

        probs = {"1": prob_local, "X": prob_emp, "2": prob_visita}
        nombres = {"1": "Junior gana", "X": "Empate", "2": "Nacional gana"}

        for key in ["1", "X", "2"]:
            ve = self.valor_esperado(cuota_mercado[key], probs[key])
            stake_k = self.stake_kelly(cuota_mercado[key], probs[key], self.bankroll)
            pct = round(stake_k / self.bankroll * 100, 1) if self.bankroll > 0 else 0
            self.apuestas.append(Apuesta(
                id=f"1x2_{key}",
                nombre=nombres[key],
                tipo="1X2",
                seleccion=key,
                cuota=cuota_mercado[key],
                probabilidad=round(probs[key], 3),
                valor_esperado=round(ve, 3),
                stake_sugerido=pct,
            ))

    def generar_apuestas_over_under(self, media_goles_local: float, media_goles_visita: float,
                                     linea: float = 2.5):
        """Genera apuestas Over/Under basado en Poisson"""
        lam = media_goles_local + media_goles_visita
        prob_over = 1.0 - sum(np.exp(-lam) * (lam ** k) / np.math.factorial(k) for k in range(int(linea) + 1))
        prob_under = 1.0 - prob_over

        margen = 0.06
        cuota_over = round(self.cuota_con_margen(prob_over, margen), 2)
        cuota_under = round(self.cuota_con_margen(prob_under, margen), 2)
        cuota_over_merc = round(cuota_over * (1 + np.random.uniform(-0.03, 0.03)), 2)
        cuota_under_merc = round(cuota_under * (1 + np.random.uniform(-0.03, 0.03)), 2)

        for key, prob, cuota_merc in [("over", prob_over, cuota_over_merc), ("under", prob_under, cuota_under_merc)]:
            ve = self.valor_esperado(cuota_merc, prob)
            stake_k = self.stake_kelly(cuota_merc, prob, self.bankroll)
            pct = round(stake_k / self.bankroll * 100, 1) if self.bankroll > 0 else 0
            self.apuestas.append(Apuesta(
                id=f"ou_{linea}_{key}",
                nombre=f"Over/Under {linea} - {'Over' if key == 'over' else 'Under'}",
                tipo="over_under",
                seleccion=f"{key}_{linea}",
                cuota=cuota_merc,
                probabilidad=round(prob, 3),
                valor_esperado=round(ve, 3),
                stake_sugerido=pct,
            ))

    def generar_apuestas_btts(self, prob_ambos_anotan: float):
        """Genera apuesta de Ambos Anotan (BTTS)"""
        prob_no = 1.0 - prob_ambos_anotan
        margen = 0.06
        cuota_si = round(self.cuota_con_margen(prob_ambos_anotan, margen), 2)
        cuota_no = round(self.cuota_con_margen(prob_no, margen), 2)
        cuota_si_merc = round(cuota_si * (1 + np.random.uniform(-0.03, 0.03)), 2)
        cuota_no_merc = round(cuota_no * (1 + np.random.uniform(-0.03, 0.03)), 2)

        for key, prob, cuota_merc in [("si", prob_ambos_anotan, cuota_si_merc), ("no", prob_no, cuota_no_merc)]:
            ve = self.valor_esperado(cuota_merc, prob)
            stake_k = self.stake_kelly(cuota_merc, prob, self.bankroll)
            pct = round(stake_k / self.bankroll * 100, 1) if self.bankroll > 0 else 0
            self.apuestas.append(Apuesta(
                id=f"btts_{key}",
                nombre=f"Ambos anotan - {'Si' if key == 'si' else 'No'}",
                tipo="btts",
                seleccion=key,
                cuota=cuota_merc,
                probabilidad=round(prob, 3),
                valor_esperado=round(ve, 3),
                stake_sugerido=pct,
            ))

    def generar_apuestas_exactas(self, prob_por_marcador: Dict[Tuple[int, int], float]):
        """Genera apuestas de marcador exacto"""
        margen = 0.10  # margen mayor en exactas
        for (goles_l, goles_v), prob in prob_por_marcador.items():
            if prob < 0.01:
                continue
            cuota_justa = self.cuota_justa(prob)
            cuota_merc = round(cuota_justa * (1 - margen), 2)
            ve = self.valor_esperado(cuota_merc, prob)
            self.apuestas.append(Apuesta(
                id=f"exacto_{goles_l}-{goles_v}",
                nombre=f"Exacto {goles_l}-{goles_v}",
                tipo="exacto",
                seleccion=f"{goles_l}-{goles_v}",
                cuota=cuota_merc,
                probabilidad=round(prob, 4),
                valor_esperado=round(ve, 3),
                stake_sugerido=0.0,  # Kelly no recomienda exactas normalmente
            ))

    def get_mejores_apuestas(self, top_n: int = 10) -> List[Apuesta]:
        """Retorna las apuestas con mejor valor esperado"""
        con_valor = [a for a in self.apuestas if a.valor_esperado > 0]
        con_valor.sort(key=lambda x: x.valor_esperado, reverse=True)
        return con_valor[:top_n]

    def calcular_combinada(self, ids_seleccionados: List[str], stakes: List[float]) -> Dict:
        """Calcula el payout de una apuesta combinada"""
        seleccionadas = [a for a in self.apuestas if a.id in ids_seleccionados]
        if not seleccionadas:
            return {"cuota_total": 0, "prob_total": 0, "payout": 0}
        cuota_total = 1.0
        prob_total = 1.0
        for a in seleccionadas:
            cuota_total *= a.cuota
            prob_total *= a.probabilidad
        stake_total = sum(stakes)
        return {
            "cuota_total": round(cuota_total, 2),
            "prob_total": round(prob_total, 4),
            "payout": round(stake_total * cuota_total, 2),
            "stake_total": round(stake_total, 2),
        }
