# ⚽ Premier League Analytics — Dashboard en Streamlit

Dashboard interactivo con datos reales de la Premier League (temporadas 2011-12 a 2025-26),
incluyendo resultados, goles, tiros, tarjetas, córners y árbitro por partido.

**Fuente de datos:** [football-data.co.uk](https://www.football-data.co.uk/), vía el dataset
público [`datasets/football-datasets`](https://github.com/datasets/football-datasets) en GitHub
(licencia Public Domain Dedication and License v1.0).



## Cómo ejecutarlo

1. Instala el proyecto (lee dependencias desde `pyproject.toml`):
   ```bash
   pip install -e .
   ```
   Para incluir además las herramientas de desarrollo (pytest, ruff), usa
   `pip install -e ".[dev]"` — ver [CONTRIBUTING.md](CONTRIBUTING.md).

2. Corre la app:
   ```bash
   streamlit run app.py
   ```

3. Se abrirá automáticamente en tu navegador en `http://localhost:8501`.

## Qué incluye

- **Tabla de posiciones** de la temporada seleccionada, con zonas de Champions/Europa/descenso resaltadas.
- **Goles y eficiencia**: goles a favor/en contra, diferencia de gol, precisión y conversión de tiros.
- **Local vs Visitante**: puntos promedio como local/visitante, distribución de resultados de la liga.
- **Disciplina**: tarjetas amarillas/rojas y faltas por equipo.
- **Cara a cara**: compara el historial entre dos equipos cualquiera a través de todas las temporadas.
- **Tendencias históricas**: evolución de goles por partido, ventaja de local y tarjetas a lo largo de los años.
- **Predicción**: estima la probabilidad de victoria local/empate/victoria visitante y los marcadores más probables para un partido hipotético entre dos equipos, usando un modelo de Poisson ataque/defensa (ver detalle abajo).

## Modelo de predicción

La pestaña **Predicción** usa un modelo de Poisson ataque/defensa (`src/predictive_model.py`),
inspirado en el modelo de Maher (1982), habitual como punto de partida en analítica deportiva:

1. Para cada equipo se calcula una fuerza de ataque y una de defensa, como local y como
   visitante, relativas al promedio de goles de la liga en la temporada seleccionada.
2. Los goles esperados (xG) de cada equipo en un partido combinan su fuerza de ataque con
   la fuerza de defensa del rival.
3. Los goles de cada equipo se modelan como variables Poisson independientes, lo que permite
   construir una matriz de probabilidad sobre todos los marcadores posibles y, a partir de
   ella, las probabilidades de resultado y los marcadores más probables.

**Limitaciones conocidas** (dejadas explícitas a propósito, no ocultas):
- No incluye la corrección de Dixon-Coles para marcadores bajos (0-0, 1-0, 0-1, 1-1).
- No pondera partidos recientes más que antiguos (sin decaimiento temporal / forma reciente).
- No considera lesiones, sanciones, ni contexto de calendario (descansos, competiciones paralelas).

Estas son extensiones naturales para una versión futura del modelo.

## Pruebas

El proyecto incluye pruebas unitarias para el modelo predictivo en `tests/test_predictive_model.py`,
cubriendo cálculo de fuerzas de equipo, la función de Poisson, la matriz de marcadores, y el
flujo completo de `predict_match`. Para correrlas:

```bash
pip install pytest
pytest tests/ -v
```

## Estructura del proyecto

```
pl-dashboard/
├── app.py                      # Presentación: layout de Streamlit, tabs, gráficos
├── src/
│   ├── __init__.py
│   ├── data.py                 # Carga del CSV de partidos
│   ├── metrics.py               # Tabla de posiciones, local/visitante, disciplina, tendencias
│   └── predictive_model.py     # Modelo de predicción Poisson ataque/defensa
├── tests/
│   ├── test_metrics.py
│   └── test_predictive_model.py
├── data/
│   └── pl_matches.csv          # Dataset (incluido en el repo)
├── .github/
│   └── workflows/
│       └── tests.yml           # CI: ruff + pytest en cada push/PR
├── pyproject.toml              # Metadatos del proyecto, dependencias, config de pytest
└── .gitignore
```

`app.py` solo se encarga de la interfaz; toda la lógica de datos y cálculos vive en `src/`, sin
depender de Streamlit, lo que permite probarla de forma aislada con pytest.

Con `pyproject.toml` el proyecto queda instalable en modo editable:

```bash
pip install -e ".[dev]"
pytest
```

## Integración continua (CI)

Cada vez que subes cambios a cualquier rama, o abres un Pull Request, GitHub Actions
(`.github/workflows/tests.yml`) corre automáticamente:

1. `ruff check .` — revisa el estilo del código.
2. `pytest` — corre toda la suite de pruebas.

Si algo falla, se ve directamente en la pestaña "Actions" del repo en GitHub, sin que
tengas que correrlo manualmente antes de cada push.

## Decisiones de diseño

Un resumen de por qué el proyecto está armado así, no solo qué hace:

- **¿Por qué Streamlit y no Flask/Django?** Streamlit permite pasar de datos a un dashboard
  interactivo sin escribir HTML/JS/CSS por separado. Para un proyecto de análisis de datos
  (más que una aplicación web de propósito general), reduce mucho el tiempo de desarrollo.

- **¿Por qué este dataset y no la API oficial de la Premier League?** football-data.co.uk es de
  dominio público y no requiere autenticación ni llaves de API, lo que hace el proyecto
  reproducible por cualquiera que clone el repo, sin pasos adicionales de configuración.

- **¿Por qué separar `app.py` de `src/`?** `app.py` mezclar la lógica de cálculo con la interfaz
  dificulta probar el código (no se puede probar una tabla de posiciones sin levantar Streamlit).
  Separarlos permite escribir pruebas unitarias rápidas sobre `src/metrics.py` y
  `src/predictive_model.py`, sin depender de la UI.

- **¿Por qué un modelo de Poisson simple y no algo como XGBoost?** El objetivo de la pestaña de
  predicción es que el resultado sea explicable: se puede justificar cada número (fuerza de
  ataque, fuerza de defensa, goles esperados) con un cálculo directo. Un modelo de caja negra
  daría predicciones sin poder explicar el porqué, que es menos valioso en un portafolio.

- **¿Por qué pytest + ruff en CI y no algo más?** Es el combo mínimo que cubre las dos preguntas
  básicas de calidad: "¿el código funciona?" (pytest) y "¿el código está limpio?" (ruff), sin
  agregar herramientas que compliquen el flujo de trabajo para un proyecto de este tamaño.

## Personalizarlo

- Para agregar temporadas más antiguas o de otras ligas (La Liga, Serie A, Bundesliga, Ligue 1),
  puedes clonar el mismo repositorio de datos y añadir esos CSV a `data/`, ajustando `app.py`.
- Los colores del tema (`ACCENT`, `ACCENT_2`, `ACCENT_3` al inicio de `app.py`) están basados en
  la identidad visual de la Premier League — cámbialos si prefieres otra paleta.
