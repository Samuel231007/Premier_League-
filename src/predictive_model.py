"""
Modelo predictivo de resultados de la Premier League.

Implementa un modelo de Poisson independiente por equipo (ataque/defensa),
inspirado en el modelo de Maher (1982), para estimar la probabilidad de
victoria local, empate y victoria visitante de un partido hipotético entre
dos equipos, a partir de su historial de goles anotados/recibidos como
local y como visitante.

Idea del modelo:
- Cada equipo tiene una fuerza de ataque y una fuerza de defensa, tanto en
  condición de local como de visitante, relativas al promedio de la liga.
- Los goles esperados de un equipo en un partido combinan su fuerza de
  ataque con la fuerza de defensa del rival.
- Los goles de cada equipo se modelan como variables Poisson independientes,
  lo que permite construir una matriz de probabilidad sobre todos los
  marcadores posibles y, a partir de ella, las probabilidades de resultado.

Limitaciones conocidas (dejadas explícitas a propósito):
- No incluye la corrección de Dixon-Coles para marcadores bajos
  (0-0, 1-0, 0-1, 1-1), que en la literatura mejora el ajuste en partidos
  de pocos goles.
- No pondera partidos recientes más que antiguos (sin decaimiento temporal).
- No considera lesiones, sanciones, calendario ni contexto de la jornada.
Estas son extensiones naturales para una futura versión.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

MAX_GOALS = 8  # rango de marcadores considerado al construir la matriz de probabilidad


@dataclass
class TeamStrengths:
    attack_home: dict
    defense_home: dict
    attack_away: dict
    defense_away: dict
    avg_home_goals: float
    avg_away_goals: float


def compute_team_strengths(df: pd.DataFrame) -> TeamStrengths:
    """
    Calcula la fuerza de ataque y defensa de cada equipo, como local y como
    visitante, relativas al promedio de la liga.

    Parameters
    ----------
    df : DataFrame con columnas HomeTeam, AwayTeam, FTHG, FTAG (una fila
        por partido). Normalmente será el subconjunto de una temporada.

    Raises
    ------
    ValueError si `df` está vacío.
    """
    if df.empty:
        raise ValueError("No hay partidos para calcular las fuerzas de los equipos.")

    avg_home_goals = df["FTHG"].mean()
    avg_away_goals = df["FTAG"].mean()

    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))

    attack_home, defense_home = {}, {}
    attack_away, defense_away = {}, {}

    for team in teams:
        home_games = df[df["HomeTeam"] == team]
        away_games = df[df["AwayTeam"] == team]

        if len(home_games) > 0 and avg_home_goals > 0 and avg_away_goals > 0:
            attack_home[team] = home_games["FTHG"].mean() / avg_home_goals
            defense_home[team] = home_games["FTAG"].mean() / avg_away_goals
        else:
            attack_home[team] = 1.0
            defense_home[team] = 1.0

        if len(away_games) > 0 and avg_home_goals > 0 and avg_away_goals > 0:
            attack_away[team] = away_games["FTAG"].mean() / avg_away_goals
            defense_away[team] = away_games["FTHG"].mean() / avg_home_goals
        else:
            attack_away[team] = 1.0
            defense_away[team] = 1.0

    return TeamStrengths(
        attack_home=attack_home,
        defense_home=defense_home,
        attack_away=attack_away,
        defense_away=defense_away,
        avg_home_goals=avg_home_goals,
        avg_away_goals=avg_away_goals,
    )


def _poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) para X ~ Poisson(lam). Implementado a mano para no depender de scipy."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam**k / math.factorial(k)


def expected_goals(home_team: str, away_team: str, strengths: TeamStrengths) -> tuple[float, float]:
    """
    Goles esperados (xG) del equipo local y del visitante en un partido
    hipotético, combinando el ataque de un equipo con la defensa del rival.
    Si algún equipo no está en `strengths` (p. ej. no jugó esa temporada),
    se le asigna fuerza neutra (1.0).
    """
    home_attack = strengths.attack_home.get(home_team, 1.0)
    away_defense = strengths.defense_away.get(away_team, 1.0)
    away_attack = strengths.attack_away.get(away_team, 1.0)
    home_defense = strengths.defense_home.get(home_team, 1.0)

    home_xg = strengths.avg_home_goals * home_attack * away_defense
    away_xg = strengths.avg_away_goals * away_attack * home_defense
    return home_xg, away_xg


def score_probability_matrix(home_xg: float, away_xg: float, max_goals: int = MAX_GOALS) -> list[list[float]]:
    """
    Matriz de probabilidad P(marcador local = i, marcador visitante = j)
    para i, j en [0, max_goals], asumiendo goles de cada equipo como
    variables Poisson independientes.
    """
    home_probs = [_poisson_pmf(i, home_xg) for i in range(max_goals + 1)]
    away_probs = [_poisson_pmf(j, away_xg) for j in range(max_goals + 1)]
    return [[hp * ap for ap in away_probs] for hp in home_probs]


def outcome_probabilities(matrix: list[list[float]]) -> dict:
    """A partir de la matriz de marcadores, agrega P(local gana), P(empate), P(visitante gana)."""
    home_win = draw = away_win = 0.0
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p
    return {"home_win": home_win, "draw": draw, "away_win": away_win}


def most_likely_scorelines(matrix: list[list[float]], top_n: int = 5) -> list[tuple[tuple[int, int], float]]:
    """Devuelve los `top_n` marcadores más probables como [((local, visitante), prob), ...]."""
    scored = [
        ((i, j), matrix[i][j])
        for i in range(len(matrix))
        for j in range(len(matrix[0]))
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def predict_match(home_team: str, away_team: str, df: pd.DataFrame, max_goals: int = MAX_GOALS) -> dict:
    """
    Función de conveniencia de extremo a extremo: calcula las fuerzas de los
    equipos a partir de `df` y devuelve goles esperados, probabilidades de
    resultado y los marcadores más probables para home_team vs away_team.
    """
    strengths = compute_team_strengths(df)
    home_xg, away_xg = expected_goals(home_team, away_team, strengths)
    matrix = score_probability_matrix(home_xg, away_xg, max_goals)
    outcomes = outcome_probabilities(matrix)
    top_scores = most_likely_scorelines(matrix, top_n=5)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "matrix": matrix,
        "outcomes": outcomes,
        "top_scorelines": top_scores,
    }
