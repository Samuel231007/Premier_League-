import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import (
    build_table,
    discipline_summary,
    historical_trends,
    home_away_summary,
)


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    """
    Liga ficticia de 3 equipos (A, B, C), un partido de ida entre cada par,
    con estadísticas de goles, tarjetas y faltas fijadas a mano:

    A 3-1 B   (local A gana)
    A 1-1 C   (empate)
    B 0-2 C   (visitante C gana)
    """
    return pd.DataFrame([
        {"Season": "2024-25", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 3, "FTAG": 1, "FTR": "H",
         "HY": 1, "AY": 2, "HR": 0, "AR": 0, "HF": 10, "AF": 12},
        {"Season": "2024-25", "HomeTeam": "A", "AwayTeam": "C", "FTHG": 1, "FTAG": 1, "FTR": "D",
         "HY": 0, "AY": 1, "HR": 0, "AR": 0, "HF": 8, "AF": 9},
        {"Season": "2024-25", "HomeTeam": "B", "AwayTeam": "C", "FTHG": 0, "FTAG": 2, "FTR": "A",
         "HY": 2, "AY": 0, "HR": 1, "AR": 0, "HF": 14, "AF": 7},
    ])


def test_build_table_points_and_ranking(sample_matches):
    table = build_table(sample_matches)

    # A: 1 victoria (3 pts) + 1 empate (1 pt) = 4 pts, GF=4, GC=2
    row_a = table[table["Equipo"] == "A"].iloc[0]
    assert row_a["Pts"] == 4
    assert row_a["GF"] == 4
    assert row_a["GC"] == 2
    assert row_a["DG"] == 2

    # C: 1 victoria (3 pts) + 1 empate (1 pt) = 4 pts, GF=3, GC=1 -> mejor DG que A
    row_c = table[table["Equipo"] == "C"].iloc[0]
    assert row_c["Pts"] == 4
    assert row_c["DG"] == 2

    # B: 1 derrota + 1 derrota = 0 pts
    row_b = table[table["Equipo"] == "B"].iloc[0]
    assert row_b["Pts"] == 0

    # El índice de la tabla debe empezar en 1 (posición 1, no 0)
    assert table.index[0] == 1


def test_build_table_matches_played(sample_matches):
    table = build_table(sample_matches)
    # Cada equipo jugó exactamente 2 partidos en este dataset
    assert (table["PJ"] == 2).all()


def test_home_away_summary_points(sample_matches):
    hv = home_away_summary(sample_matches)

    row_a = hv[hv["Equipo"] == "A"].iloc[0]
    # A jugó 2 partidos como local: ganó uno (3 pts) y empató otro (1 pt) = 4 pts
    assert row_a["Pts como local"] == 4
    assert row_a["Pts como visitante"] == 0  # A nunca jugó como visitante

    row_c = hv[hv["Equipo"] == "C"].iloc[0]
    # C jugó 2 partidos como visitante: empató uno (1 pt) y ganó otro (3 pts) = 4 pts
    assert row_c["Pts como visitante"] == 4


def test_discipline_summary_totals(sample_matches):
    disc = discipline_summary(sample_matches)

    row_b = disc[disc["Equipo"] == "B"].iloc[0]
    # B: amarillas como visitante vs A (2) + amarillas como local vs C (2) = 4
    assert row_b["Amarillas"] == 4
    # B: rojas como visitante vs A (0) + rojas como local vs C (1) = 1
    assert row_b["Rojas"] == 1

    # Ordenado por amarillas descendente
    assert disc.iloc[0]["Amarillas"] >= disc.iloc[-1]["Amarillas"]


def test_historical_trends_single_season(sample_matches):
    trend = historical_trends(sample_matches)
    assert len(trend) == 1
    row = trend.iloc[0]
    assert row["Season"] == "2024-25"
    # Total de goles: (3+1)+(1+1)+(0+2) = 8, entre 3 partidos
    assert row["Goles/partido"] == pytest.approx(8 / 3)
    # 1 de 3 partidos con victoria local
    assert row["% victorias local"] == pytest.approx(100 / 3)


def test_historical_trends_multiple_seasons(sample_matches):
    other_season = sample_matches.copy()
    other_season["Season"] = "2023-24"
    combined = pd.concat([sample_matches, other_season], ignore_index=True)

    trend = historical_trends(combined)
    assert set(trend["Season"]) == {"2024-25", "2023-24"}
    assert len(trend) == 2
