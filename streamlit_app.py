"""
FOOTBALL ANALYTICS ULTIMATE - Dashboard Profesional
Conectado a big data + ML + Monte Carlo + Decision Engine
Muestra predicciones de todos los partidos proximos a iniciar
con todas las metricas y recomendaciones de apuesta.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os
import sys

sys.path.append(os.path.dirname(__file__))

from data_pipeline import get_proximos_partidos, get_estadisticas_equipo, get_todos_equipos, EQUIPOS_COLOMBIA
from real_predictor import RealPredictor
from decision_engine import DecisionEngine

# ─── CONFIG ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Analytics Ultimate",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_1 = "#4CAF50"   # Local gana
COLOR_X = "#FFC107"   # Empate
COLOR_2 = "#F44336"   # Visita gana
COLOR_BG = "#0E1117"
COLOR_CARD = "#1A1A2E"
COLOR_GOLD = "#FFD700"

# ─── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: #0E1117; color: #FAFAFA; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { font-weight: 700; letter-spacing: -0.5px; }
    .match-card {
        background: #1A1A2E;
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #2A2A3E;
        margin-bottom: 1rem;
        transition: transform 0.2s, border-color 0.2s;
    }
    .match-card:hover { border-color: #4CAF50; transform: translateY(-2px); }
    .team-name-card { font-size: 1.1rem; font-weight: 700; }
    .prob-box {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        min-width: 60px;
        text-align: center;
    }
    .metric-box {
        background: #16213E;
        border-radius: 12px;
        padding: 0.8rem;
        text-align: center;
    }
    .recommendation-star {
        color: #FFD700;
        font-size: 1.2rem;
    }
    .badge-live {
        background: #F44336;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 700;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    .green-glow { box-shadow: 0 0 15px rgba(76,175,80,0.3); }
    .gold-border { border: 2px solid #FFD700 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stProgress > div > div > div > div { background-image: linear-gradient(90deg, #4CAF50, #8BC34A); }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Escudo_del_Atl%C3%A9tico_Nacional.svg/120px-Escudo_del_Atl%C3%A9tico_Nacional.svg.png", width=40)
st.sidebar.title("Football Analytics")
st.sidebar.caption("Big Data + ML + Monte Carlo")

liga = st.sidebar.selectbox(
    "Liga / Competicion",
    ["Colombia - Liga BetPlay", "Internacional (Top 5 + Champions)"],
    index=0,
)
liga_key = "colombia" if "Colombia" in liga else "internacional"

n_partidos = st.sidebar.slider("Partidos a mostrar", 3, 20, 8)
n_simulaciones = st.sidebar.select_slider("Precision (simulaciones)", [5000, 10000, 25000, 50000], value=10000)
bankroll = st.sidebar.number_input("Bankroll ($)", min_value=10, max_value=100000, value=100, step=10)
riesgo = st.sidebar.select_slider("Aversion al riesgo", ["Muy conservador", "Conservador", "Moderado", "Agresivo"], value="Conservador")

factor_riesgo = {"Muy conservador": 0.10, "Conservador": 0.25, "Moderado": 0.40, "Agresivo": 0.60}

auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
if st.sidebar.button(" Actualizar ahora", type="primary", use_container_width=True):
    st.cache_data.clear()

st.sidebar.divider()
st.sidebar.caption(f"Ultima actualizacion: {datetime.now().strftime('%H:%M:%S')}")
st.sidebar.caption("v2.0 - Football Analytics Ultimate")

# ─── CACHE ─────────────────────────────────────────────────────
@st.cache_data(ttl=30 if auto_refresh else 300)
def cargar_datos():
    return get_proximos_partidos(liga_key, n_partidos)

@st.cache_resource
def get_predictor():
    return RealPredictor(n_simulaciones=n_simulaciones)

@st.cache_resource
def get_engine():
    return DecisionEngine(bankroll=bankroll, aversión_riesgo=factor_riesgo[riesgo])

@st.cache_data(ttl=30 if auto_refresh else 300)
def generar_predicciones(fixtures):
    predictor = get_predictor()
    return predictor.predecir_multiple(fixtures)

# ─── HEADER ────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([6, 4])
with col_h1:
    st.title(" Football Analytics Ultimate")
    st.markdown("### Predicciones con Big Data + Machine Learning + Monte Carlo")
with col_h2:
    st.markdown(f"""
    <div style='text-align:right;'>
        <span class='badge-live'>{'EN VIVO' if auto_refresh else 'ESTATICO'}</span>
        <span style='color:#888;font-size:0.8rem;margin-left:8px;'>
            {n_simulaciones:,} sims/partido
        </span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ─── CARGA ─────────────────────────────────────────────────────
with st.spinner(f"Cargando datos de {n_partidos} partidos y ejecutando {n_simulaciones:,} simulaciones c/u..."):
    fixtures = cargar_datos()
    predicciones = generar_predicciones(fixtures)
    engine = get_engine()
    engine.bankroll = bankroll
    engine.kelly_factor = factor_riesgo[riesgo]
    resumen_dia = engine.recomendar_para_hoy(predicciones)

# ╔═══════════════════════════════════════════════════════════════╗
# ║  BARRA DE METRICAS GLOBALES                                 ║
# ╚═══════════════════════════════════════════════════════════════╝
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.metric("Partidos analizados", len(predicciones))
with col_m2:
    favoritos = sum(1 for p in predicciones if p.confianza > 0.50)
    st.metric("Favoritos claros", f"{favoritos}/{len(predicciones)}")
with col_m3:
    st.metric("Apuestas con valor (+EV)", resumen_dia["total_apuestas_recomendadas"])
with col_m4:
    st.metric("Stake total sugerido", f"{resumen_dia['stake_total_sugerido_pct']}%")
with col_m5:
    riesgo_color = {"BAJO": "#4CAF50", "MODERADO": "#FFC107", "ALTO": "#F44336"}
    st.markdown(f"""
    <div class='metric-box'>
        <div style='color:#888;font-size:0.85rem;'>Riesgo global</div>
        <div style='font-size:1.3rem;font-weight:700;color:{riesgo_color.get(resumen_dia["riesgo_global"], "#888")};'>
            {resumen_dia['riesgo_global']}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ╔═══════════════════════════════════════════════════════════════════╗
# ║  TABS                                                          ║
# ╚═══════════════════════════════════════════════════════════════════╝
tab1, tab2, tab3, tab4 = st.tabs([
    " Partidos y Predicciones",
    " Recomendaciones de Apuesta",
    " Tabla de Posiciones",
    " Mi Cartera de Apuestas",
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1: PARTIDOS Y PREDICCIONES
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.header("Proximos Partidos")
    st.caption(f"Predicciones generadas con ML + Monte Carlo ({n_simulaciones:,} simulaciones por partido)")
    
    for i, pred in enumerate(predicciones):
        # Determinar favorito
        probs = [pred.prob_local, pred.prob_empate, pred.prob_visita]
        max_prob = max(probs)
        favorito_idx = probs.index(max_prob)
        favorito = [pred.local, "Empate", pred.visitante][favorito_idx]
        favorito_color = [COLOR_1, COLOR_X, COLOR_2][favorito_idx]
        
        # Time until match
        try:
            fecha_match = datetime.strptime(f"{pred.fecha} {pred.hora}", "%Y-%m-%d %H:%M")
            diff = fecha_match - datetime.now()
            dias = diff.days
            horas = diff.seconds // 3600
            if dias > 0:
                tiempo_str = f"En {dias}d {horas}h"
            elif horas > 0:
                tiempo_str = f"En {horas}h"
            else:
                tiempo_str = "Iniciando..."
        except:
            tiempo_str = pred.fecha
        
        # Card
        border_color = COLOR_1 if favorito_idx == 0 else COLOR_X if favorito_idx == 1 else COLOR_2
        extra_class = "gold-border" if pred.confianza > 0.60 else ""
        
        st.markdown(f"""
        <div class='match-card {extra_class}' style='border-left:4px solid {border_color};'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div><span style='color:#888;font-size:0.85rem;'>{pred.competicion}</span></div>
                <div><span style='color:#888;font-size:0.85rem;'>{tiempo_str}</span></div>
            </div>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-top:8px;'>
                <div style='flex:2;'>
                    <div style='display:flex;justify-content:space-between;align-items:center;'>
                        <span class='team-name-card'>{pred.local}</span>
                        <span style='font-size:1.5rem;font-weight:700;color:#666;'>vs</span>
                        <span class='team-name-card'>{pred.visitante}</span>
                    </div>
                </div>
                <div style='flex:1;display:flex;gap:8px;justify-content:flex-end;'>
                    <div class='prob-box' style='background:{COLOR_1}20;color:{COLOR_1};'>
                        {pred.prob_local:.0%}
                    </div>
                    <div class='prob-box' style='background:{COLOR_X}20;color:{COLOR_X};'>
                        {pred.prob_empate:.0%}
                    </div>
                    <div class='prob-box' style='background:{COLOR_2}20;color:{COLOR_2};'>
                        {pred.prob_visita:.0%}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Expandir detalles
        with st.expander(f" Ver analisis completo - {pred.local} vs {pred.visitante}"):
            c1, c2, c3 = st.columns([2, 3, 2])
            
            with c1:
                st.markdown("#### Datos del Partido")
                st.markdown(f"**Estadio:** {pred.estadio}")
                st.markdown(f"**Fecha:** {pred.fecha} {pred.hora}")
                st.markdown(f"**Etapa:** {pred.etapa}")
                st.markdown(f"**Confianza:** {pred.confianza:.1%}")
                
                st.markdown("#### xG")
                st.markdown(f"**{pred.local}:** {pred.xg_local:.2f}")
                st.markdown(f"**{pred.visitante}:** {pred.xg_visita:.2f}")
                st.markdown(f"**Total:** {pred.media_goles_total:.2f}")
                
                st.markdown("#### Ambos Anotan")
                st.markdown(f"**Si:** {pred.btts_si:.0%}")
                st.markdown(f"**No:** {pred.btts_no:.0%}")
            
            with c2:
                st.markdown("#### Distribucion de Goles")
                # Goal distribution chart
                r = [pred.over_under.get(f"over_{l}", 0) for l in [0.5, 1.5, 2.5, 3.5, 4.5]]
                fig = go.Figure(data=[go.Bar(
                    x=[f"Over {l:.1f}" for l in [0.5, 1.5, 2.5, 3.5, 4.5]],
                    y=r,
                    marker_color=["#4CAF50" if v > 0.5 else "#F44336" for v in r],
                    text=[f"{v:.0%}" for v in r],
                    textposition="outside",
                )])
                fig.update_layout(template="plotly_dark", height=250, margin=dict(l=10, r=10, t=10, b=10),
                                  yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### Marcadores Probables")
                marc_cols = st.columns(3)
                for idx, (marc, prob) in enumerate(sorted(pred.marcadores.items(), key=lambda x: -x[1])[:6]):
                    with marc_cols[idx % 3]:
                        st.markdown(f"""
                        <div style='text-align:center;background:#16213E;border-radius:8px;padding:4px;'>
                            <div style='font-weight:700;'>{marc}</div>
                            <div style='color:#4CAF50;font-weight:600;'>{prob:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            with c3:
                st.markdown("#### Recomendacion")
                st.info(pred.recomendacion)
                
                st.markdown("#### Cuotas Justas")
                st.markdown(f"**{pred.local}:** {pred.cuota_justa_local:.2f}")
                st.markdown(f"**Empate:** {pred.cuota_justa_empate:.2f}")
                st.markdown(f"**{pred.visitante}:** {pred.cuota_justa_visita:.2f}")
                
                st.markdown("#### Medio Tiempo")
                st.markdown(f"**{pred.local}:** {pred.mt_prob_local:.0%}")
                st.markdown(f"**Empate MT:** {pred.mt_prob_empate:.0%}")
                st.markdown(f"**{pred.visitante}:** {pred.mt_prob_visita:.0%}")
                
                if pred.confianza > 0.55:
                    st.success(" Alta confianza en esta prediccion")
                elif pred.confianza > 0.40:
                    st.warning(" Confianza moderada")
                else:
                    st.error(" Baja confianza - Partido incierto")

# ═══════════════════════════════════════════════════════════════════
# TAB 2: RECOMENDACIONES DE APUESTA
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.header("Recomendaciones de Apuesta Inteligentes")
    st.markdown("""
    Basado en **Valor Esperado (EV)** y **Criterio de Kelly**. Una apuesta tiene **valor** cuando
    nuestra probabilidad real es mayor que la probabilidad implicita en la cuota de mercado.
    """)
    
    # Mejores del dia
    st.subheader(" Mejores Apuestas del Dia")
    
    if resumen_dia["mejores_del_dia"]:
        cols = st.columns(len(resumen_dia["mejores_del_dia"][:4]))
        for idx, rec in enumerate(resumen_dia["mejores_del_dia"][:4]):
            with cols[idx if idx < len(cols) else 0]:
                stars_display = "".join(chr(9733) for _ in range(rec.rating_estrella))
                conf_color = {"ALTA": "#4CAF50", "MEDIA": "#FFC107", "BAJA": "#F44336"}
                st.markdown(f"""
                <div class='match-card gold-border' style='text-align:center;'>
                    <div style='color:{COLOR_GOLD};font-size:1.2rem;'>{stars_display}</div>
                    <div style='font-weight:700;font-size:1.1rem;margin:8px 0;'>{rec.descripcion}</div>
                    <div style='display:flex;justify-content:center;gap:16px;margin:8px 0;'>
                        <div><span style='color:#888;'>Cuota</span><br><strong>{rec.cuota_mercado:.2f}</strong></div>
                        <div><span style='color:#888;'>Prob</span><br><strong>{rec.probabilidad:.0%}</strong></div>
                        <div><span style='color:#888;'>EV</span><br><strong style='color:#4CAF50;'>{rec.valor_esperado:+.1%}</strong></div>
                    </div>
                    <div style='color:{conf_color.get(rec.confianza, "#888")};font-size:0.85rem;font-weight:600;'>
                        {rec.confianza} - Stake: {rec.kelly_stake_pct:.1f}%
                    </div>
                    <div style='color:#666;font-size:0.75rem;margin-top:4px;'>{rec.reasoning}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No se encontraron apuestas con valor positivo en este momento.")
    
    st.divider()
    
    # Tabla completa
    st.subheader("Analisis Completo de Mercados")
    
    # Recopilar todas las recomendaciones de todos los partidos
    todas_recs = []
    for pred in predicciones:
        res = engine.evaluar_todo(pred)
        for r in res["mejores_con_valor"]:
            todas_recs.append({
                "Partido": f"{pred.local} vs {pred.visitante}",
                "Mercado": r.mercado,
                "Seleccion": r.descripcion,
                "Cuota": r.cuota_mercado,
                "Prob. Real": f"{r.probabilidad:.0%}",
                "EV": f"{r.valor_esperado:+.1%}",
                "Stake": f"{r.kelly_stake_pct:.1f}%",
                "Confianza": r.confianza,
                "Rating": "".join(chr(9733) for _ in range(r.rating_estrella)),
                "razon": r.reasoning,
            })
    
    if todas_recs:
        df_recs = pd.DataFrame(todas_recs)
        st.dataframe(
            df_recs,
            column_config={
                "Partido": st.column_config.TextColumn("Partido"),
                "Mercado": st.column_config.TextColumn("Mercado"),
                "Seleccion": st.column_config.TextColumn("Seleccion"),
                "Cuota": st.column_config.NumberColumn("Cuota", format="%.2f"),
                "Prob. Real": st.column_config.TextColumn("Prob."),
                "EV": st.column_config.TextColumn("EV"),
                "Stake": st.column_config.TextColumn("Stake"),
                "Confianza": st.column_config.TextColumn("Conf."),
                "Rating": st.column_config.TextColumn("Rating"),
            },
            use_container_width=True,
            hide_index=True,
            height=400,
        )
    else:
        st.info("No hay recomendaciones con valor en este momento. Revisa mas tarde.")

# ═══════════════════════════════════════════════════════════════════
# TAB 3: TABLA DE POSICIONES
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.header("Tabla de Posiciones - Liga BetPlay 2026-I")
    st.caption("Datos reales de la fase regular")
    
    equipos_data = []
    for name, stats in sorted({
        "Atletico Nacional": {"puntos": 40, "pj": 18, "pg": 12, "pe": 4, "pp": 2, "gf": 35, "gc": 15, "dg": 20},
        "Junior de Barranquilla": {"puntos": 35, "pj": 18, "pg": 10, "pe": 5, "pp": 3, "gf": 27, "gc": 21, "dg": 6},
        "Deportivo Pasto": {"puntos": 34, "pj": 18, "pg": 10, "pe": 4, "pp": 4, "gf": 26, "gc": 21, "dg": 5},
        "America de Cali": {"puntos": 33, "pj": 18, "pg": 9, "pe": 6, "pp": 3, "gf": 24, "gc": 15, "dg": 9},
        "Once Caldas": {"puntos": 33, "pj": 18, "pg": 9, "pe": 6, "pp": 3, "gf": 30, "gc": 22, "dg": 8},
        "Deportes Tolima": {"puntos": 31, "pj": 18, "pg": 8, "pe": 7, "pp": 3, "gf": 26, "gc": 16, "dg": 10},
        "Independiente Santa Fe": {"puntos": 29, "pj": 18, "pg": 8, "pe": 5, "pp": 5, "gf": 26, "gc": 21, "dg": 5},
        "Internacional de Bogota": {"puntos": 28, "pj": 18, "pg": 7, "pe": 7, "pp": 4, "gf": 25, "gc": 23, "dg": 2},
        "Millonarios FC": {"puntos": 26, "pj": 18, "pg": 7, "pe": 5, "pp": 6, "gf": 22, "gc": 20, "dg": 2},
        "Deportivo Cali": {"puntos": 26, "pj": 18, "pg": 6, "pe": 8, "pp": 4, "gf": 19, "gc": 15, "dg": 4},
        "Independiente Medellin": {"puntos": 26, "pj": 18, "pg": 7, "pe": 5, "pp": 6, "gf": 25, "gc": 22, "dg": 3},
    }.items()):
        equipos_data.append({
            "Pos": len(equipos_data) + 1,
            "Equipo": name,
            "PJ": stats["pj"],
            "PG": stats["pg"],
            "PE": stats["pe"],
            "PP": stats["pp"],
            "GF": stats["gf"],
            "GC": stats["gc"],
            "DG": stats["dg"],
            "Puntos": stats["puntos"],
        })
    
    df_tabla = pd.DataFrame(equipos_data)
    
    # Grafico
    fig_tabla = go.Figure(data=[go.Table(
        header=dict(
            values=list(df_tabla.columns),
            fill_color="#1A1A2E",
            font=dict(color="white", size=14, family="Arial Black"),
            align="center",
            height=40,
        ),
        cells=dict(
            values=[df_tabla[col] for col in df_tabla.columns],
            fill_color=[["#16213E" if i % 2 == 0 else "#1E2A45" for i in range(len(df_tabla))]],
            font=dict(color="white", size=13),
            align="center",
            height=32,
        )
    )])
    fig_tabla.update_layout(
        template="plotly_dark",
        height=50 * (len(df_tabla) + 1) + 40,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig_tabla, use_container_width=True)
    
    # Comparativa de equipos
    st.subheader("Comparativa de Fuerza entre Equipos")
    st.caption("xG (ataque) vs xGA (defensa) - Mientras mas arriba a la derecha, mejor equipo")
    
    scatter_data = []
    for name, stats in EQUIPOS_COLOMBIA.items():
        scatter_data.append({
            "Equipo": name,
            "xG": stats["xg_ataque"],
            "xGA": stats["xg_defensa"],
            "Puntos": stats["puntos"],
        })
    
    df_scatter = pd.DataFrame(scatter_data)
    fig_scatter = px.scatter(
        df_scatter, x="xG", y="xGA", text="Equipo", size="Puntos",
        color="Puntos", color_continuous_scale="RdYlGn",
        title="Fuerza Ofensiva (xG) vs Debilidad Defensiva (xGA)",
        labels={"xG": "Expected Goals (Ataque)", "xGA": "Expected Goals Against (Defensa)"},
        template="plotly_dark",
    )
    fig_scatter.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="white")))
    fig_scatter.update_layout(height=500)
    # Linea de referencia
    fig_scatter.add_hline(y=1.0, line_dash="dash", line_color="#666", opacity=0.5)
    fig_scatter.add_vline(x=1.5, line_dash="dash", line_color="#666", opacity=0.5)
    st.plotly_chart(fig_scatter, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 4: CARTERA DE APUESTAS
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.header(" Cartera de Apuestas")
    st.markdown("Simula y gestiona tu cartera de apuestas para hoy.")
    
    cartera_bankroll = st.number_input("Tu bankroll ($)", min_value=10, max_value=100000, value=bankroll, step=50)
    
    st.divider()
    
    # Seleccion de apuestas
    st.subheader("Arma tu Combinada")
    
    apuestas_seleccionadas = []
    for pred in predicciones:
        res = engine.evaluar_todo(pred)
        con_valor = [r for r in res["todas"] if r.valor_esperado > 0]
        if con_valor:
            options = {f"{r.descripcion} @ {r.cuota_mercado:.2f}": r for r in con_valor}
            selected = st.multiselect(
                f"{pred.local} vs {pred.visitante}",
                options=list(options.keys()),
                max_selections=3,
                key=f"bet_{pred.partido_id}",
            )
            for s in selected:
                apuestas_seleccionadas.append(options[s])
    
    st.divider()
    
    if apuestas_seleccionadas:
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("### Resumen de tu cartera")
            
            cuota_total = 1.0
            prob_total = 1.0
            for a in apuestas_seleccionadas:
                cuota_total *= a.cuota_mercado
                prob_total *= a.probabilidad
            
            stake_total_pct = min(sum(a.kelly_stake_pct for a in apuestas_seleccionadas), 100.0)
            stake_total = cartera_bankroll * stake_total_pct / 100
            pago_potencial = stake_total * cuota_total
            
            st.markdown(f"""
            <div class='match-card gold-border'>
                <div style='display:flex;justify-content:space-between;'>
                    <div>
                        <div style='color:#888;'>Apuestas seleccionadas</div>
                        <div style='font-size:1.5rem;font-weight:700;'>{len(apuestas_seleccionadas)}</div>
                    </div>
                    <div>
                        <div style='color:#888;'>Cuota total combinada</div>
                        <div style='font-size:1.5rem;font-weight:700;color:#4CAF50;'>{cuota_total:.2f}</div>
                    </div>
                    <div>
                        <div style='color:#888;'>Prob. de acierto</div>
                        <div style='font-size:1.5rem;font-weight:700;'>{prob_total:.1%}</div>
                    </div>
                </div>
                <div style='margin-top:16px;'>
                    <div>Stake sugerido: <strong>{stake_total_pct:.1f}%</strong> = <strong>${stake_total:.2f}</strong></div>
                    <div style='font-size:1.3rem;font-weight:700;margin-top:8px;color:#4CAF50;'>
                        Pago potencial: ${pago_potencial:.2f} (ganancia: ${pago_potencial - stake_total:.2f})
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c2:
            st.markdown("### Tus apuestas")
            for a in apuestas_seleccionadas:
                st.markdown(f"""
                <div class='match-card' style='padding:0.5rem 1rem;'>
                    <div style='display:flex;justify-content:space-between;'>
                        <span>{a.descripcion}</span>
                        <span><strong>{a.cuota_mercado:.2f}</strong></span>
                    </div>
                    <div style='color:#666;font-size:0.8rem;'>
                        EV: {a.valor_esperado:+.1%} | Conf: {a.confianza} | Stake: {a.kelly_stake_pct:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Selecciona apuestas de los partidos de arriba para armar tu cartera.")

# ─── FOOTER ────────────────────────────────────────────────────
st.divider()
st.markdown(f"""
<div style='text-align:center;color:#444;font-size:0.8rem;padding:1rem;'>
    Football Analytics Ultimate v2.0 | Datos: Liga BetPlay 2026-I | 
    Modelo: XGBoost + Poisson Monte Carlo | 
    Simulaciones: {n_simulaciones:,} por partido |
    Ultima actualizacion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    <br>
    <strong style='color:#666;'>Advertencia:</strong> Las predicciones tienen fines informativos.
    No garantizan resultados. Apuesta con responsabilidad.
</div>
""", unsafe_allow_html=True)

# Auto-refresh
if auto_refresh:
    time.sleep(0.1)
    st.runtime.legacy_caching.clear_cache()
