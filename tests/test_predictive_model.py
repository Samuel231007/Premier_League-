import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predictive_model import (
    _poisson_pmf,
    compute_team_strengths,
    expected_goals,
    most_likely_scorelines,
    outcome_probabilities,
    predict_match,
    score_probability_matrix,
)


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    """
    Liga ficticia de 3 equipos (A, B, C) con 4 partidos y goles fijados a
    mano, para poder verificar los cálculos de fuerzas de equipo.
    """
    data = [
        {"HomeTeam": "A", "AwayTeam": "B", "FTHG": 3, "FTAG": 1},
        {"HomeTeam": "B", "AwayTeam": "A", "FTHG": 0, "FTAG": 2},
        {"HomeTeam": "A", "AwayTeam": "C", "FTHG": 1, "FTAG": 1},
        {"HomeTeam": "C", "AwayTeam": "B", "FTHG": 2, "FTAG": 0},
    ]
    return pd.DataFrame(data)


def test_poisson_pmf_sums_to_one():
    lam = 1.7
    total = sum(_poisson_pmf(k, lam) for k in range(50))
    assert math.isclose(total, 1.0, abs_tol=1e-6)


def test_poisson_pmf_zero_lambda():
    assert _poisson_pmf(0, 0.0) == 1.0
    assert _poisson_pmf(1, 0.0) == 0.0


def test_compute_team_strengths_known_values(sample_matches):
    strengths = compute_team_strengths(sample_matches)

    # Promedios de liga: FTHG = (3+0+1+2)/4 = 1.5 ; FTAG = (1+2+1+0)/4 = 1.0
    assert math.isclose(strengths.avg_home_goals, 1.5)
    assert math.isclose(strengths.avg_away_goals, 1.0)

    # Equipo A como local jugó 2 partidos, anotó (3+1)/2 = 2.0 goles en promedio
    assert math.isclose(strengths.attack_home["A"], 2.0 / 1.5)


def test_compute_team_strengths_raises_on_empty_df():
    empty = pd.DataFrame(columns=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    with pytest.raises(ValueError):
        compute_team_strengths(empty)


def test_expected_goals_positive(sample_matches):
    strengths = compute_team_strengths(sample_matches)
    home_xg, away_xg = expected_goals("A", "B", strengths)
    assert home_xg > 0
    assert away_xg > 0


def test_expected_goals_unknown_team_uses_neutral_strength(sample_matches):
    strengths = compute_team_strengths(sample_matches)
    # "Z" nunca jugó: debe usar fuerza neutra (1.0) sin lanzar error
    home_xg, away_xg = expected_goals("Z", "B", strengths)
    assert home_xg > 0
    assert away_xg > 0


def test_score_probability_matrix_sums_to_one():
    matrix = score_probability_matrix(home_xg=1.4, away_xg=1.1, max_goals=10)
    total = sum(sum(row) for row in matrix)
    assert math.isclose(total, 1.0, abs_tol=1e-4)


def test_outcome_probabilities_sum_to_one():
    matrix = score_probability_matrix(home_xg=1.4, away_xg=1.1, max_goals=10)
    outcomes = outcome_probabilities(matrix)
    total = outcomes["home_win"] + outcomes["draw"] + outcomes["away_win"]
    assert math.isclose(total, 1.0, abs_tol=1e-4)


def test_outcome_probabilities_favor_stronger_home_side():
    # Local con xG mucho mayor que el visitante -> debe favorecer claramente al local
    matrix = score_probability_matrix(home_xg=3.0, away_xg=0.5, max_goals=10)
    outcomes = outcome_probabilities(matrix)
    assert outcomes["home_win"] > outcomes["draw"]
    assert outcomes["home_win"] > outcomes["away_win"]


def test_most_likely_scorelines_sorted_desc():
    matrix = score_probability_matrix(home_xg=1.8, away_xg=0.9, max_goals=6)
    top = most_likely_scorelines(matrix, top_n=5)
    probs = [p for _, p in top]
    assert probs == sorted(probs, reverse=True)
    assert len(top) == 5


def test_predict_match_end_to_end(sample_matches):
    # Este dataset de juguete produce un xG local alto (~3.3) por el tamaño de
    # muestra tan pequeño, así que se usa un max_goals más amplio para que la
    # matriz capture prácticamente toda la masa de probabilidad.
    result = predict_match("A", "B", sample_matches, max_goals=20)
    assert result["home_team"] == "A"
    assert result["away_team"] == "B"
    assert result["home_xg"] > 0
    assert result["away_xg"] > 0
    total_prob = sum(result["outcomes"].values())
    assert math.isclose(total_prob, 1.0, abs_tol=1e-4)
    assert len(result["top_scorelines"]) == 5
