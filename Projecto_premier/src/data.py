"""
Carga de datos para el dashboard de Premier League Analytics.

Este módulo no depende de Streamlit a propósito: así se puede probar con
pytest y reutilizar (por ejemplo, en un notebook o script) sin necesitar
una app corriendo.
"""

from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pl_matches.csv"


def load_data(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Carga el CSV de partidos de la Premier League y añade una columna
    `Matchweek_order` con el número de partido dentro de cada temporada,
    en el orden en que aparecen en el archivo.

    Parameters
    ----------
    path : ruta al CSV de partidos. Por defecto, data/pl_matches.csv en la
        raíz del proyecto.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Matchweek_order"] = df.groupby("Season").cumcount() + 1
    return df
