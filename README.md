# futbol_AR — Predicciones de Fútbol con IA

Dashboard interactivo de predicciones para la Liga BetPlay Colombia 2026-I, usando Monte Carlo, Machine Learning (XGBoost), y un motor de decisión con criterio de Kelly para recomendar apuestas con valor esperado positivo (+EV).

## Datos reales incluidos

Estadísticas reales de los 11 equipos de la Liga BetPlay 2026-I: puntos, xG, xGA, posesión, forma reciente, localía.

## Stack

- **Python 3.12+**
- **Streamlit** — frontend interactivo
- **Plotly** — gráficos profesionales
- **XGBoost + scikit-learn** — modelo ML híbrido
- **NumPy / Pandas** — procesamiento de datos
- **Joblib** — serialización del modelo

## Instalación local

```bash
git clone https://github.com/TU_USUARIO/futbol_AR.git
cd futbol_AR
python -m venv venv
.\venv\Scripts\Activate    # Windows
pip install -r requirements.txt
streamlit run streamlit_app.py --server.port 8510
```

## Despliegue en la nube

### Render
1. Conectá tu repositorio de GitHub
2. Creá un **Web Service** con:
   - **Start Command**: `streamlit run streamlit_app.py --server.port $PORT`
3. Listo.

### Streamlit Community Cloud
1. Subí el repo a GitHub
2. Entrá a https://streamlit.io/cloud
3. Conectá el repo y seleccioná `streamlit_app.py`

## Estructura del proyecto

```
futbol_AR/
├── streamlit_app.py      # Dashboard principal (4 tabs)
├── data_pipeline.py      # Datos reales de equipos y fixtures
├── real_predictor.py     # Predictor híbrido XGBoost + Monte Carlo
├── decision_engine.py    # Motor de decisión con Kelly
├── betting_engine.py     # Tipos de apuesta y pagos
├── live_simulator.py     # Simulación Monte Carlo
├── models/
│   ├── match_predictor.py
│   ├── xg_calculator.py
│   └── colombian_scouting.py
├── data/fixtures/        # Fixtures generados
├── requirements.txt
└── .gitignore
```

## Funcionalidades

- Predicción 1X2, xG, marcadores exactos, Over/Under, BTTS
- Recomendaciones con valor esperado y stake según Kelly
- Tabla de posiciones con gráficos comparativos
- Simulador de cartera de apuestas
- 50,000 iteraciones Monte Carlo por partido
