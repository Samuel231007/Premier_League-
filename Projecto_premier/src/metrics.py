"""
Métricas agregadas para el dashboard de Premier League Analytics.

Igual que `data.py`, este módulo no depende de Streamlit a propósito: se
puede probar con pytest y reutilizar fuera de la app.

Todas las funciones esperan un DataFrame de partidos con, como mínimo, las
columnas usadas por football-data.co.uk: HomeTeam, AwayTeam, FTHG, FTAG,
FTR, y (para disciplina) HY, AY, HR, AR, HF, AF. `historical_trends`
además espera una columna `Season`.
"""

from __future__ import annotations

import pandas as pd

_HOME_PTS = {"H": 3, "D": 1, "A": 0}
_AWAY_PTS = {"H": 0, "D": 1, "A": 3}


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la tabla de posiciones a partir de los partidos de una
    temporada (o cualquier subconjunto de `df`).

    Devuelve un DataFrame con columnas Equipo, PJ, Pts, GF, GC, DG,
    ordenado por Pts, luego DG, luego GF (de mayor a menor), con el índice
    empezando en 1 (posición en la tabla).
    """
    home = df[["HomeTeam", "FTHG", "FTAG", "FTR"]].rename(
        columns={"HomeTeam": "Equipo", "FTHG": "GF", "FTAG": "GC"}
    )
    home["Pts"] = home["FTR"].map(_HOME_PTS)

    away = df[["AwayTeam", "FTAG", "FTHG", "FTR"]].rename(
        columns={"AwayTeam": "Equipo", "FTAG": "GF", "FTHG": "GC"}
    )
    away["Pts"] = away["FTR"].map(_AWAY_PTS)

    combined = pd.concat(
        [home[["Equipo", "GF", "GC", "Pts"]], away[["Equipo", "GF", "GC", "Pts"]]],
        ignore_index=True,
    )

    table = (
        combined.groupby("Equipo")
        .agg(PJ=("Pts", "size"), GF=("GF", "sum"), GC=("GC", "sum"), Pts=("Pts", "sum"))
        .reset_index()
    )
    table["DG"] = table["GF"] - table["GC"]
    table = table.sort_values(["Pts", "DG", "GF"], ascending=False).reset_index(drop=True)
    table.index = table.index + 1

    return table[["Equipo", "PJ", "Pts", "GF", "GC", "DG"]]


def home_away_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resume el rendimiento de cada equipo como local y como visitante:
    partidos jugados, puntos totales y puntos promedio por partido en cada
    condición.

    Devuelve columnas: Equipo, PJ_local, Pts como local, Prom. local,
    PJ_visitante, Pts como visitante, Prom. visitante.
    """
    home = df.copy()
    home["Pts_local"] = home["FTR"].map(_HOME_PTS)
    home_grp = (
        home.groupby("HomeTeam")
        .agg(PJ_local=("Pts_local", "size"), Pts_como_local=("Pts_local", "sum"))
        .reset_index()
        .rename(columns={"HomeTeam": "Equipo"})
    )

    away = df.copy()
    away["Pts_visitante"] = away["FTR"].map(_AWAY_PTS)
    away_grp = (
        away.groupby("AwayTeam")
        .agg(PJ_visitante=("Pts_visitante", "size"), Pts_como_visitante=("Pts_visitante", "sum"))
        .reset_index()
        .rename(columns={"AwayTeam": "Equipo"})
    )

    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    result = pd.DataFrame({"Equipo": teams})
    result = result.merge(home_grp, on="Equipo", how="left").merge(away_grp, on="Equipo", how="left")

    fill_cols = ["PJ_local", "Pts_como_local", "PJ_visitante", "Pts_como_visitante"]
    result[fill_cols] = result[fill_cols].fillna(0)

    result["Prom. local"] = result.apply(
        lambda r: r["Pts_como_local"] / r["PJ_local"] if r["PJ_local"] > 0 else 0.0, axis=1
    )
    result["Prom. visitante"] = result.apply(
        lambda r: r["Pts_como_visitante"] / r["PJ_visitante"] if r["PJ_visitante"] > 0 else 0.0, axis=1
    )

    result = result.rename(
        columns={"Pts_como_local": "Pts como local", "Pts_como_visitante": "Pts como visitante"}
    )

    return result[
        ["Equipo", "PJ_local", "Pts como local", "Prom. local", "PJ_visitante", "Pts como visitante", "Prom. visitante"]
    ]


def discipline_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Suma tarjetas amarillas, rojas y faltas cometidas por cada equipo,
    combinando sus partidos como local y como visitante.

    Devuelve columnas Equipo, Amarillas, Rojas, Faltas, ordenado por
    Amarillas de forma descendente.
    """
    home = df[["HomeTeam", "HY", "HR", "HF"]].rename(
        columns={"HomeTeam": "Equipo", "HY": "Amarillas", "HR": "Rojas", "HF": "Faltas"}
    )
    away = df[["AwayTeam", "AY", "AR", "AF"]].rename(
        columns={"AwayTeam": "Equipo", "AY": "Amarillas", "AR": "Rojas", "AF": "Faltas"}
    )
    combined = pd.concat([home, away], ignore_index=True)

    disc = (
        combined.groupby("Equipo")
        .agg(Amarillas=("Amarillas", "sum"), Rojas=("Rojas", "sum"), Faltas=("Faltas", "sum"))
        .reset_index()
    )
    return disc.sort_values("Amarillas", ascending=False).reset_index(drop=True)


def historical_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega, por temporada, el promedio de goles por partido, el porcentaje
    de victorias locales y el promedio de tarjetas por partido.

    Devuelve columnas Season, Goles/partido, % victorias local,
    Tarjetas/partido, ordenado por Season.
    """

    def _agg(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        goals = group["FTHG"].sum() + group["FTAG"].sum()
        cards = group["HY"].sum() + group["AY"].sum() + group["HR"].sum() + group["AR"].sum()
        return pd.Series(
            {
                "Goles/partido": goals / n,
                "% victorias local": (group["FTR"] == "H").mean() * 100,
                "Tarjetas/partido": cards / n,
            }
        )

    trend = df.groupby("Season").apply(_agg).reset_index()
    return trend.sort_values("Season").reset_index(drop=True)
