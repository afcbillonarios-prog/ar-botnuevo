"""
Decision Engine - Motor de decision para apuestas deportivas.
Evalua todas las posibilidades y recomienda la mejor accion basado en:
- Valor esperado (EV)
- Criterio de Kelly
- Confianza del modelo
- Risk/Reward ratio
- Correlacion entre mercados
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Recomendacion:
    """Recomendacion de apuesta con toda la informacion necesaria."""
    mercado: str                # "1X2", "Over/Under", "BTTS", "Exacto", "Handicap"
    seleccion: str              # "1", "X", "2", "over_2.5", etc.
    descripcion: str            # Legible para humanos
    probabilidad: float         # Probabilidad real del modelo
    cuota_justa: float          # Cuota sin margen
    cuota_mercado: float        # Cuota estimada en casas de apuestas
    valor_esperado: float       # EV > 0 significa que tiene valor
    kelly_stake_pct: float      # % del bankroll recomendado (Kelly fraccionado al 25%)
    confianza: str              # "ALTA", "MEDIA", "BAJA"
    rating_estrella: int        # 1-5 estrellas
    reasoning: str              # Explicacion


class DecisionEngine:
    """
    Evalua todas las opciones de apuesta para un partido
    y recomienda las mejores basado en valor esperado y confianza.
    """

    def __init__(self, bankroll: float = 100.0, aversión_riesgo: float = 0.25):
        """
        Args:
            bankroll: Capital total disponible
            aversión_riesgo: Factor de Kelly (0.25 = conservador, 0.5 = agresivo)
        """
        self.bankroll = bankroll
        self.kelly_factor = aversión_riesgo

    @staticmethod
    def cuota_con_margen(prob: float, margen: float = 0.03) -> float:
        """Cuota de mercado simulada que permite +EV ocasional."""
        cuota_base = 1.0 / prob if prob > 0 else 99.0
        # Las casas de apuestas tienen modelos diferentes al nuestro.
        # A veces sobrevaloran o infravaloran equipos.
        # Simulamos eso con ruido + margen variable.
        error_mercado = np.random.uniform(-0.08, 0.06)
        cuota_final = cuota_base * (1.0 - margen + error_mercado)
        return max(round(cuota_final, 2), 1.01)

    @staticmethod
    def valor_esperado(cuota: float, prob_real: float) -> float:
        """Valor esperado de una apuesta."""
        return (cuota * prob_real) - 1.0

    @staticmethod
    def kelly_fraction(cuota: float, prob_real: float, factor: float = 0.25) -> float:
        """Fraccion de Kelly (fraccionada para ser conservador)."""
        ev = (cuota * prob_real) - 1.0
        if ev <= 0:
            return 0.0
        b = cuota - 1.0
        p = prob_real
        q = 1.0 - p
        kelly = (b * p - q) / b
        return max(0.0, kelly * factor)

    def evaluar_1x2(self, prob_local: float, prob_emp: float, prob_vis: float,
                    local: str, visita: str) -> List[Recomendacion]:
        """Evalua mercado 1X2."""
        recs = []
        probs = [("1", prob_local, local), ("X", prob_emp, "Empate"), ("2", prob_vis, visita)]
        
        for key, prob, nombre in probs:
            cuota_merc = self.cuota_con_margen(prob)
            # Pequena variacion aleatoria para simular mercado real
            cuota_merc = round(cuota_merc * (1 + np.random.uniform(-0.02, 0.02)), 2)
            ev = self.valor_esperado(cuota_merc, prob)
            kelly = self.kelly_fraction(cuota_merc, prob, self.kelly_factor)
            kelly_pct = round(kelly * 100, 2)
            
            # Nivel de confianza
            if prob >= 0.55:
                conf = "ALTA"
            elif prob >= 0.40:
                conf = "MEDIA"
            else:
                conf = "BAJA"
            
            # Rating estrellas
            if ev > 0.05 and conf == "ALTA":
                stars = 5
            elif ev > 0.03 and conf in ("ALTA", "MEDIA"):
                stars = 4
            elif ev > 0:
                stars = 3
            elif ev > -0.05:
                stars = 2
            else:
                stars = 1
            
            # Reasoning
            if ev > 0:
                razon = f"VALOR DETECTADO: cuota ({cuota_merc}) > probabilidad real ({prob:.0%}). EV={ev:+.1%}."
            else:
                razon = f"Sin valor: cuota ({cuota_merc}) por debajo de la probabilidad real ({prob:.0%})."
            
            recs.append(Recomendacion(
                mercado="1X2",
                seleccion=key,
                descripcion=f"{nombre} gana",
                probabilidad=prob,
                cuota_justa=round(1.0/prob, 2) if prob > 0 else 99.0,
                cuota_mercado=cuota_merc,
                valor_esperado=round(ev, 4),
                kelly_stake_pct=kelly_pct,
                confianza=conf,
                rating_estrella=stars,
                reasoning=razon,
            ))
        
        return recs

    def evaluar_over_under(self, prob_over: Dict[float, float], prob_under: Dict[float, float]) -> List[Recomendacion]:
        """Evalua mercados Over/Under."""
        recs = []
        for linea in [2.5, 3.5]:
            key_over = f"over_{linea}"
            key_under = f"under_{linea}"
            
            for key, prob, nombre in [(key_over, prob_over.get(linea, 0.5), f"Over {linea}"),
                                       (key_under, prob_under.get(linea, 0.5), f"Under {linea}")]:
                cuota_merc = self.cuota_con_margen(prob, margen=0.04)
                ev = self.valor_esperado(cuota_merc, prob)
                kelly = self.kelly_fraction(cuota_merc, prob, self.kelly_factor)
                kelly_pct = round(kelly * 100, 2)
                
                conf = "ALTA" if prob >= 0.65 else "MEDIA" if prob >= 0.50 else "BAJA"
                stars = 5 if (ev > 0.05 and conf == "ALTA") else 4 if (ev > 0.03) else 3 if ev > 0 else 2
                
                recs.append(Recomendacion(
                    mercado="Over/Under",
                    seleccion=key,
                    descripcion=nombre,
                    probabilidad=prob,
                    cuota_justa=round(1.0/prob, 2) if prob > 0 else 99.0,
                    cuota_mercado=cuota_merc,
                    valor_esperado=round(ev, 4),
                    kelly_stake_pct=kelly_pct,
                    confianza=conf,
                    rating_estrella=stars,
                    reasoning=f"Probabilidad real: {prob:.0%}. {'VALOR' if ev > 0 else 'Sin valor'} (EV: {ev:+.1%})",
                ))
        
        return recs

    def evaluar_btts(self, btts_si: float, btts_no: float) -> List[Recomendacion]:
        """Evalua mercado de Ambos Anotan."""
        recs = []
        for key, prob, nombre in [("si", btts_si, "Ambos anotan: Si"),
                                   ("no", btts_no, "Ambos anotan: No")]:
            cuota_merc = self.cuota_con_margen(prob, margen=0.04)
            ev = self.valor_esperado(cuota_merc, prob)
            kelly = self.kelly_fraction(cuota_merc, prob, self.kelly_factor)
            kelly_pct = round(kelly * 100, 2)
            
            conf = "ALTA" if prob >= 0.65 else "MEDIA" if prob >= 0.50 else "BAJA"
            stars = 5 if (ev > 0.05 and conf == "ALTA") else 4 if (ev > 0.03) else 3 if ev > 0 else 2
            
            recs.append(Recomendacion(
                mercado="BTTS",
                seleccion=key,
                descripcion=nombre,
                probabilidad=prob,
                cuota_justa=round(1.0/prob, 2) if prob > 0 else 99.0,
                cuota_mercado=cuota_merc,
                valor_esperado=round(ev, 4),
                kelly_stake_pct=kelly_pct,
                confianza=conf,
                rating_estrella=stars,
                reasoning=f"Probabilidad real: {prob:.0%}. {'VALOR' if ev > 0 else 'Sin valor'} (EV: {ev:+.1%})",
            ))
        
        return recs

    def evaluar_marcadores_exactos(self, marcadores: Dict[str, float], top_n: int = 5) -> List[Recomendacion]:
        """Evalua mercados de marcador exacto."""
        recs = []
        margen_exacto = 0.12  # margen mayor para exactas
        
        for marc_str, prob in sorted(marcadores.items(), key=lambda x: -x[1])[:top_n]:
            if prob < 0.01:
                continue
            cuota_merc = self.cuota_con_margen(prob, margen_exacto)
            cuota_merc = round(cuota_merc * (1 + np.random.uniform(-0.03, 0.03)), 2)
            ev = self.valor_esperado(cuota_merc, prob)
            
            recs.append(Recomendacion(
                mercado="Marcador Exacto",
                seleccion=marc_str,
                descripcion=f"Exacto {marc_str}",
                probabilidad=prob,
                cuota_justa=round(1.0/prob, 2) if prob > 0 else 99.0,
                cuota_mercado=cuota_merc,
                valor_esperado=round(ev, 4),
                kelly_stake_pct=0.0,  # Kelly no recomienda exactas
                confianza="BAJA",
                rating_estrella=1 if ev > 0 else 0,
                reasoning=f"Baja probabilidad ({prob:.2%}), alto riesgo. Solo para combinadas.",
            ))
        
        return recs

    def evaluar_todo(self, pred) -> Dict:
        """
        Evaluacion completa de todos los mercados para un partido.
        Retorna las mejores recomendaciones organizadas.
        """
        todas = []
        todas.extend(self.evaluar_1x2(pred.prob_local, pred.prob_empate, pred.prob_visita,
                                       pred.local, pred.visitante))
        
        # Over/Under
        ou_over = {2.5: pred.over_under.get("over_2.5", 0.5), 3.5: pred.over_under.get("over_3.5", 0.3)}
        ou_under = {2.5: pred.over_under.get("under_2.5", 0.5), 3.5: pred.over_under.get("under_3.5", 0.7)}
        todas.extend(self.evaluar_over_under(ou_over, ou_under))
        
        todas.extend(self.evaluar_btts(pred.btts_si, pred.btts_no))
        todas.extend(self.evaluar_marcadores_exactos(pred.marcadores))
        
        # Ordenar por rating y EV
        mejores = sorted([r for r in todas if r.valor_esperado > 0], key=lambda x: -x.rating_estrella)
        sin_valor = sorted([r for r in todas if r.valor_esperado <= 0], key=lambda x: -x.probabilidad)
        
        return {
            "todas": todas,
            "mejores_con_valor": mejores,
            "sin_valor": sin_valor,
            "total_analizadas": len(todas),
            "total_con_valor": len(mejores),
            "bankroll_sugerido": self.bankroll,
        }

    def recomendar_para_hoy(self, predicciones: List) -> Dict:
        """
        Recomendacion global para todos los partidos del dia.
        Incluye: mejor apuesta del dia, riesgo total, stake total sugerido.
        """
        todas_recomendaciones = []
        for pred in predicciones:
            res = self.evaluar_todo(pred)
            todas_recomendaciones.extend(res["mejores_con_valor"])
        
        # Mejores del dia (top 5 por rating)
        mejores_del_dia = sorted(todas_recomendaciones, key=lambda x: (-x.rating_estrella, -x.valor_esperado))[:5]
        
        # Riesgo total si se apuestan todas
        stake_total = sum(r.kelly_stake_pct for r in mejores_del_dia)
        
        return {
            "mejores_del_dia": mejores_del_dia,
            "total_apuestas_recomendadas": len(mejores_del_dia),
            "stake_total_sugerido_pct": round(stake_total, 1),
            "riesgo_global": "BAJO" if stake_total < 5 else "MODERADO" if stake_total < 15 else "ALTO",
        }


if __name__ == "__main__":
    from real_predictor import RealPredictor
    from data_pipeline import get_proximos_partidos
    
    predictor = RealPredictor(10000)
    fixtures = get_proximos_partidos("colombia", 3)
    
    engine = DecisionEngine(bankroll=100.0)
    predicciones = []
    
    for f in fixtures:
        pred = predictor.predecir(f["local"], f["visitante"])
        predicciones.append(pred)
    
    resumen = engine.recomendar_para_hoy(predicciones)
    print(f"\n=== MEJORES APUESTAS DEL DIA ({resumen['total_apuestas_recomendadas']}) ===")
    print(f"Stake total sugerido: {resumen['stake_total_sugerido_pct']}% del bankroll")
    print(f"Riesgo: {resumen['riesgo_global']}")
    for r in resumen["mejores_del_dia"]:
        stars = "".join(chr(9733) for _ in range(r.rating_estrella))
        print(f"\n  {stars} {r.descripcion}")
        print(f"     Cuota: {r.cuota_mercado:.2f} | Prob: {r.probabilidad:.0%} | EV: {r.valor_esperado:+.1%}")
        print(f"     Stake: {r.kelly_stake_pct:.1f}% | Confianza: {r.confianza}")
        print(f"     {r.reasoning}")
