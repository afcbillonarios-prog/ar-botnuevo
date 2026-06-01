"""
Simulador Monte Carlo en vivo para la Final Junior vs Nacional.
Calcula distribuciones de probabilidad en tiempo real.
"""
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter

class LiveMonteCarlo:
    """
    Simulador que corre N iteraciones y calcula:
    - Probabilidades de resultado (1X2)
    - Distribucion de goles
    - Probabilidad de marcadores exactos
    - Over/Under
    - Ambos anotan
    """

    def __init__(self, n_simulaciones: int = 50000):
        self.n = n_simulaciones
        self.resultados: List[Tuple[int, int]] = []

    def simular(
        self,
        xg_local: float,
        xg_visita: float,
        ajuste_local: float = 1.0,
        ajuste_visita: float = 1.0
    ) -> Dict:
        """
        Corre la simulacion Poisson y retorna estadisticas completas.
        """
        lam_local = xg_local * ajuste_local
        lam_visita = xg_visita * ajuste_visita

        goles_local = np.random.poisson(lam_local, self.n)
        goles_visita = np.random.poisson(lam_visita, self.n)

        self.resultados = list(zip(goles_local, goles_visita))

        # Probabilidades basicas
        wins_l = sum(1 for gl, gv in self.resultados if gl > gv) / self.n
        empates = sum(1 for gl, gv in self.resultados if gl == gv) / self.n
        wins_v = sum(1 for gl, gv in self.resultados if gl < gv) / self.n

        # Marcadores exactos
        counter = Counter(self.resultados)
        marcadores = {}
        for (gl, gv), count in counter.most_common(15):
            marcadores[f"{gl}-{gv}"] = count / self.n

        # Over / Under
        ou = {}
        for linea in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
            totales = [gl + gv for gl, gv in self.resultados]
            ou[f"over_{linea}"] = sum(1 for t in totales if t > linea) / self.n
            ou[f"under_{linea}"] = sum(1 for t in totales if t <= linea) / self.n

        # Ambos anotan
        btts_si = sum(1 for gl, gv in self.resultados if gl > 0 and gv > 0) / self.n
        btts_no = 1.0 - btts_si

        # Goles totales esperados
        media_goles_local = float(np.mean(goles_local))
        media_goles_visita = float(np.mean(goles_visita))
        media_total = media_goles_local + media_goles_visita

        # Handicap -1.5, -1, 0, +1, +1.5
        handicaps = {}
        for h in [-2.5, -1.5, -1, 0, 1, 1.5, 2.5]:
            ajustados = [(gl - gv) for gl, gv in self.resultados]
            handicaps[f"local_{h}"] = sum(1 for x in ajustados if x > h) / self.n
            handicaps[f"visita_{h}"] = sum(1 for x in ajustados if x < -h) / self.n

        # Resultado al medio tiempo (estimacion simple)
        # Aproximacion: mitad de goles en cada tiempo
        mt_local = np.random.poisson(lam_local * 0.4, self.n)
        mt_visita = np.random.poisson(lam_visita * 0.4, self.n)
        empates_mt = sum(1 for gl, gv in zip(mt_local, mt_visita) if gl == gv) / self.n
        wins_l_mt = sum(1 for gl, gv in zip(mt_local, mt_visita) if gl > gv) / self.n
        wins_v_mt = sum(1 for gl, gv in zip(mt_local, mt_visita) if gl < gv) / self.n

        return {
            "prob_local": round(wins_l, 4),
            "prob_empate": round(empates, 4),
            "prob_visita": round(wins_v, 4),
            "xg_local": round(lam_local, 2),
            "xg_visita": round(lam_visita, 2),
            "media_goles_local": round(media_goles_local, 2),
            "media_goles_visita": round(media_goles_visita, 2),
            "media_goles_total": round(media_total, 2),
            "marcadores": marcadores,
            "over_under": ou,
            "btts_si": round(btts_si, 4),
            "btts_no": round(btts_no, 4),
            "handicaps": handicaps,
            "mt_prob_local": round(wins_l_mt, 4),
            "mt_prob_empate": round(empates_mt, 4),
            "mt_prob_visita": round(wins_v_mt, 4),
            "n": self.n,
        }

    def get_score_distribution(self, top_n: int = 10) -> Dict:
        """Distribucion de marcadores para grafica"""
        if not self.resultados:
            return {}
        counter = Counter(self.resultados)
        total = len(self.resultados)
        dist = {}
        for (gl, gv), count in counter.most_common(top_n):
            dist[f"{gl}-{gv}"] = count / total
        return dist

    def get_goal_distribution(self) -> Dict[str, List]:
        """Distribucion de goles totales para histograma"""
        if not self.resultados:
            return {"goles": [], "prob": []}
        totales = [gl + gv for gl, gv in self.resultados]
        counter = Counter(totales)
        max_goles = max(counter.keys()) if counter else 10
        goles = list(range(0, min(max_goles + 1, 13)))
        probs = [counter.get(g, 0) / self.n for g in goles]
        return {"goles": goles, "prob": probs}
