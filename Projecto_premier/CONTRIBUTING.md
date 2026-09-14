# Contribuir a Premier League Analytics

Gracias por tu interés en contribuir. Esta guía explica cómo configurar el entorno de
desarrollo y qué se espera antes de abrir un Pull Request.

## Configurar el entorno

1. Clona el repositorio y entra a la carpeta del proyecto.
2. Instala el proyecto en modo editable, con las dependencias de desarrollo incluidas:
   ```bash
   pip install -e ".[dev]"
   ```
   Esto instala Streamlit, pandas, plotly, numpy, pytest y ruff de una vez.

## Antes de abrir un Pull Request

Corre estos dos comandos y asegúrate de que ambos pasen — es exactamente lo que revisa el CI:

```bash
ruff check .
pytest
```

Si `ruff` marca algo que se puede arreglar automáticamente, prueba primero:

```bash
ruff check . --fix
```

## Convenciones del proyecto

- **Separación de capas:** la lógica de datos y cálculos va en `src/` (sin depender de
  Streamlit); `app.py` solo se encarga de la interfaz. Si agregas una métrica nueva, debería
  vivir como una función en `src/metrics.py` (o un módulo nuevo si no encaja ahí), no directo
  en `app.py`.
- **Pruebas:** toda función nueva en `src/` debería tener al menos un par de pruebas en
  `tests/`, con un DataFrame pequeño de ejemplo donde el resultado esperado se pueda calcular
  a mano (así se puede verificar que la prueba misma esté bien planteada).
- **Estilo:** el proyecto usa `ruff` con una longitud de línea de 110 caracteres
  (ver `[tool.ruff]` en `pyproject.toml`). No hace falta memorizar las reglas — con correr
  `ruff check . --fix` antes de cada commit alcanza.
- **Commits:** mensajes cortos y en modo imperativo ("Agrega pestaña de predicción", no
  "Agregada pestaña de predicción"), en español o inglés, pero consistente dentro del mismo PR.

## Reportar un problema

Si encuentras un bug o tienes una idea de mejora, abre un Issue describiendo:
- Qué esperabas que pasara.
- Qué pasó en realidad.
- Pasos para reproducirlo (si aplica).
