import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import load_data
from src.metrics import build_table, discipline_summary, historical_trends, home_away_summary
from src.predictive_model import predict_match

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Premier League Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#37003C"      # PL purple
ACCENT_2 = "#00FF87"    # PL green
ACCENT_3 = "#04F5FF"    # PL cyan
BG = "#0E1017"
CARD_BG = "#171A26"
TEXT_MUTED = "#9AA0B4"

PLOTLY_TEMPLATE = "plotly_dark"

# --------------------------------------------------------------------------------------
# STYLES
# --------------------------------------------------------------------------------------
st.markdown(f"""
<style>
    .stApp {{
        background: radial-gradient(circle at top left, #1a0f2e 0%, {BG} 45%);
    }}
    #MainMenu, footer {{visibility: hidden;}}

    .hero {{
        padding: 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #37003C 0%, #6A00A8 55%, #00A8B5 120%);
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 30px rgba(55,0,60,0.45);
    }}
    .hero h1 {{
        color: white;
        font-size: 2.1rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.5px;
    }}
    .hero p {{
        color: #E4D9F5;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }}

    div[data-testid="stMetric"] {{
        background: {CARD_BG};
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 0.9rem 1rem 0.6rem 1rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }}
    div[data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED};
    }}
    div[data-testid="stMetricValue"] {{
        color: white;
    }}

    section[data-testid="stSidebar"] {{
        background: #12141f;
        border-right: 1px solid rgba(255,255,255,0.06);
    }}

    .block-container {{
        padding-top: 1.4rem;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {CARD_BG};
        border-radius: 10px 10px 0 0;
        padding: 8px 16px;
        color: {TEXT_MUTED};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #37003C;
        color: white !important;
    }}

    .section-note {{
        color: {TEXT_MUTED};
        font-size: 0.85rem;
        margin-top: -0.4rem;
        margin-bottom: 0.8rem;
    }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------------------
@st.cache_data
def get_data():
    return load_data()


df_all = get_data()
seasons = sorted(df_all["Season"].unique())
teams_all = sorted(set(df_all["HomeTeam"]) | set(df_all["AwayTeam"]))


# --------------------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Filtros")
    season_sel = st.selectbox("Temporada", options=seasons[::-1], index=0)
    st.caption("La mayoría de las secciones usan la temporada seleccionada.")
    st.divider()
    st.markdown("### 📊 Datos")
    st.caption(
        "Resultados y estadísticas por partido de la Premier League "
        f"({seasons[0]} – {seasons[-1]}), incluyendo tiros, tarjetas, córners y árbitro."
    )
    st.caption("Fuente: football-data.co.uk vía dataset público en GitHub.")

df_season = df_all[df_all["Season"] == season_sel].copy()
table = build_table(df_season)
teams_all_season = sorted(set(df_season["HomeTeam"]) | set(df_season["AwayTeam"]))

# --------------------------------------------------------------------------------------
# HERO + KPIs
# --------------------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
    <h1>⚽ Premier League Analytics</h1>
    <p>Panel interactivo de resultados, goles, disciplina y forma — temporada {season_sel}</p>
</div>
""", unsafe_allow_html=True)

total_matches = len(df_season)
total_goals = int(df_season["FTHG"].sum() + df_season["FTAG"].sum())
avg_goals = total_goals / total_matches if total_matches else 0
home_win_pct = (df_season["FTR"] == "H").mean() * 100 if total_matches else 0
avg_cards = (df_season["HY"].sum() + df_season["AY"].sum() + df_season["HR"].sum() + df_season["AR"].sum()) / total_matches if total_matches else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Partidos jugados", f"{total_matches}")
k2.metric("Goles totales", f"{total_goals}")
k3.metric("Goles / partido", f"{avg_goals:.2f}")
k4.metric("Victorias locales", f"{home_win_pct:.1f}%")
k5.metric("Tarjetas / partido", f"{avg_cards:.2f}")

st.write("")

# --------------------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏆 Tabla de posiciones",
    "🎯 Goles y eficiencia",
    "🏠 Local vs Visitante",
    "🟨 Disciplina",
    "🆚 Cara a cara",
    "📈 Tendencias históricas",
    "🔮 Predicción",
])

# ---- TAB 1: LEAGUE TABLE ----
with tab1:
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader(f"Tabla — {season_sel}")

        def highlight_zone(row):
            pos = row.name
            if pos <= 4:
                color = "rgba(0,255,135,0.14)"
            elif pos <= 6:
                color = "rgba(4,245,255,0.10)"
            elif pos >= len(table) - 2:
                color = "rgba(255,60,60,0.14)"
            else:
                color = "transparent"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            table.style.apply(highlight_zone, axis=1).format(precision=0),
            use_container_width=True,
            height=560,
        )
        st.caption("🟩 Champions League · 🟦 Europa/Conference · 🟥 Descenso (zonas aproximadas)")

    with right:
        st.subheader("Puntos por equipo")
        top_n = st.slider("Mostrar top N equipos", 5, len(table), min(10, len(table)))
        fig = px.bar(
            table.head(top_n).iloc[::-1],
            x="Pts", y="Equipo", orientation="h",
            color="Pts", color_continuous_scale=["#37003C", "#00FF87"],
            template=PLOTLY_TEMPLATE, text="Pts",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=560,
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---- TAB 2: GOALS & EFFICIENCY ----
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Goles a favor vs en contra")
        gdf = table.sort_values("GF", ascending=False)
        fig = go.Figure()
        fig.add_bar(x=gdf["Equipo"], y=gdf["GF"], name="A favor", marker_color=ACCENT_2)
        fig.add_bar(x=gdf["Equipo"], y=gdf["GC"], name="En contra", marker_color="#FF3C64")
        fig.update_layout(barmode="group", template=PLOTLY_TEMPLATE, height=430,
                           xaxis_tickangle=-45, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Eficiencia de disparo (equipo local)")
        shots = df_season.groupby("HomeTeam").agg(
            Tiros=("HS", "sum"), Al_arco=("HST", "sum"), Goles=("FTHG", "sum")
        ).reset_index().rename(columns={"HomeTeam": "Equipo"})
        shots["Precision_%"] = (shots["Al_arco"] / shots["Tiros"].replace(0, np.nan) * 100).round(1)
        shots["Conversion_%"] = (shots["Goles"] / shots["Al_arco"].replace(0, np.nan) * 100).round(1)
        fig = px.scatter(
            shots, x="Precision_%", y="Conversion_%", size="Goles", color="Goles",
            hover_name="Equipo", template=PLOTLY_TEMPLATE,
            color_continuous_scale=["#37003C", "#04F5FF", "#00FF87"],
            labels={"Precision_%": "% tiros al arco", "Conversion_%": "% conversión a gol"},
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<p class="section-note">Basado en partidos como local. Burbuja = goles anotados.</p>', unsafe_allow_html=True)

    st.subheader("Diferencia de goles por equipo")
    gd_sorted = table.sort_values("DG", ascending=True)
    fig = px.bar(
        gd_sorted, x="DG", y="Equipo", orientation="h",
        color="DG", color_continuous_scale=["#FF3C64", "#2b2b3d", "#00FF87"],
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=650, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---- TAB 3: HOME VS AWAY ----
with tab3:
    st.subheader("Rendimiento como local vs visitante")

    hv = home_away_summary(df_season)

    fig = go.Figure()
    fig.add_bar(x=hv["Equipo"], y=hv["Prom. local"], name="Puntos/partido (local)", marker_color=ACCENT_2)
    fig.add_bar(x=hv["Equipo"], y=hv["Prom. visitante"], name="Puntos/partido (visitante)", marker_color=ACCENT_3)
    fig.update_layout(barmode="group", template=PLOTLY_TEMPLATE, height=460,
                       xaxis_tickangle=-45, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        res_counts = df_season["FTR"].value_counts().rename({"H": "Local", "D": "Empate", "A": "Visitante"})
        fig = px.pie(
            values=res_counts.values, names=res_counts.index,
            color=res_counts.index,
            color_discrete_map={"Local": ACCENT_2, "Empate": "#8888aa", "Visitante": ACCENT_3},
            template=PLOTLY_TEMPLATE, hole=0.55,
        )
        fig.update_layout(height=380, title="Distribución de resultados de la liga")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### Ventaja de local")
        st.markdown(
            f"En la temporada **{season_sel}**, el equipo local ganó el "
            f"**{(df_season['FTR']=='H').mean()*100:.1f}%** de los partidos, "
            f"empató el **{(df_season['FTR']=='D').mean()*100:.1f}%** "
            f"y perdió el **{(df_season['FTR']=='A').mean()*100:.1f}%**."
        )
        st.markdown(
            "Compara esto con la pestaña **Tendencias históricas** para ver cómo "
            "ha evolucionado la ventaja de jugar en casa a lo largo de los años."
        )

# ---- TAB 4: DISCIPLINE ----
with tab4:
    st.subheader("Tarjetas por equipo")
    disc = discipline_summary(df_season)

    fig = go.Figure()
    fig.add_bar(x=disc["Equipo"], y=disc["Amarillas"], name="Amarillas", marker_color="#F5C518")
    fig.add_bar(x=disc["Equipo"], y=disc["Rojas"] * 5, name="Rojas (x5 visual)", marker_color="#FF3C64")
    fig.update_layout(barmode="stack", template=PLOTLY_TEMPLATE, height=460,
                       xaxis_tickangle=-45, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(
        disc, x="Faltas", y="Amarillas", size=(disc["Rojas"] + 1), color="Amarillas",
        hover_name="Equipo", template=PLOTLY_TEMPLATE,
        color_continuous_scale=["#00FF87", "#F5C518", "#FF3C64"],
        labels={"Faltas": "Faltas cometidas", "Amarillas": "Tarjetas amarillas"},
    )
    fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('<p class="section-note">Tamaño de burbuja = tarjetas rojas.</p>', unsafe_allow_html=True)

# ---- TAB 5: HEAD TO HEAD ----
with tab5:
    st.subheader("Comparación cara a cara (histórico multi-temporada)")
    c1, c2 = st.columns(2)
    with c1:
        team_a = st.selectbox("Equipo A", teams_all, index=teams_all.index("Arsenal") if "Arsenal" in teams_all else 0)
    with c2:
        default_b = "Chelsea" if "Chelsea" in teams_all else teams_all[1]
        team_b = st.selectbox("Equipo B", teams_all, index=teams_all.index(default_b))

    h2h = df_all[
        ((df_all["HomeTeam"] == team_a) & (df_all["AwayTeam"] == team_b)) |
        ((df_all["HomeTeam"] == team_b) & (df_all["AwayTeam"] == team_a))
    ].sort_values("Date")

    if h2h.empty:
        st.info("No hay enfrentamientos registrados entre estos equipos en el rango de datos disponible.")
    else:
        wins_a = ((h2h["HomeTeam"] == team_a) & (h2h["FTR"] == "H")).sum() + \
                 ((h2h["AwayTeam"] == team_a) & (h2h["FTR"] == "A")).sum()
        wins_b = ((h2h["HomeTeam"] == team_b) & (h2h["FTR"] == "H")).sum() + \
                 ((h2h["AwayTeam"] == team_b) & (h2h["FTR"] == "A")).sum()
        draws = (h2h["FTR"] == "D").sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Victorias {team_a}", wins_a)
        m2.metric("Empates", draws)
        m3.metric(f"Victorias {team_b}", wins_b)
        m4.metric("Enfrentamientos totales", len(h2h))

        fig = px.pie(
            values=[wins_a, draws, wins_b],
            names=[team_a, "Empate", team_b],
            color_discrete_sequence=[ACCENT_2, "#8888aa", ACCENT_3],
            template=PLOTLY_TEMPLATE, hole=0.55,
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Últimos enfrentamientos")
        show = h2h[["Date", "HomeTeam", "FTHG", "FTAG", "AwayTeam", "Season"]].tail(10).iloc[::-1]
        show.columns = ["Fecha", "Local", "GL", "GV", "Visitante", "Temporada"]
        st.dataframe(show, use_container_width=True, hide_index=True)

# ---- TAB 6: HISTORICAL TRENDS ----
with tab6:
    st.subheader("Evolución histórica de la liga")

    trend = historical_trends(df_all)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(
            trend, x="Season", y="Goles/partido", markers=True,
            template=PLOTLY_TEMPLATE, color_discrete_sequence=[ACCENT_2],
        )
        fig.update_layout(height=380, title="Goles por partido a través de las temporadas")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(
            trend, x="Season", y="% victorias local", markers=True,
            template=PLOTLY_TEMPLATE, color_discrete_sequence=[ACCENT_3],
        )
        fig.update_layout(height=380, title="Ventaja de local a través del tiempo")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        trend, x="Season", y="Tarjetas/partido",
        template=PLOTLY_TEMPLATE, color="Tarjetas/partido",
        color_continuous_scale=["#37003C", "#F5C518"],
    )
    fig.update_layout(height=380, title="Tarjetas por partido a través de las temporadas", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-note">Datos agregados de todas las temporadas disponibles en el dataset cargado.</p>', unsafe_allow_html=True)

# ---- TAB 7: PREDICTION ----
with tab7:
    st.subheader("Predicción de resultado (modelo Poisson ataque/defensa)")
    st.markdown(
        "Estima la probabilidad de victoria local, empate y victoria visitante para un "
        "partido hipotético, a partir de los goles anotados y recibidos por cada equipo "
        "como local y como visitante en la temporada seleccionada. Asume que los goles de "
        "cada equipo siguen una distribución de Poisson independiente (modelo de Maher, 1982)."
    )

    if len(teams_all_season) < 2:
        st.info("Se necesitan al menos dos equipos con partidos en esta temporada para predecir.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            pred_home = st.selectbox("Equipo local", teams_all_season, key="pred_home", index=0)
        with c2:
            default_away_idx = 1 if teams_all_season[0] == pred_home else 0
            pred_away = st.selectbox("Equipo visitante", teams_all_season, key="pred_away", index=default_away_idx)

        if pred_home == pred_away:
            st.warning("Selecciona dos equipos diferentes para generar una predicción.")
        else:
            result = predict_match(pred_home, pred_away, df_season)

            m1, m2 = st.columns(2)
            m1.metric(f"Goles esperados — {pred_home}", f"{result['home_xg']:.2f}")
            m2.metric(f"Goles esperados — {pred_away}", f"{result['away_xg']:.2f}")

            outcomes = result["outcomes"]
            fig = go.Figure(go.Bar(
                x=["Local", "Empate", "Visitante"],
                y=[outcomes["home_win"] * 100, outcomes["draw"] * 100, outcomes["away_win"] * 100],
                marker_color=[ACCENT_2, "#8888aa", ACCENT_3],
                text=[f"{outcomes['home_win']*100:.1f}%", f"{outcomes['draw']*100:.1f}%", f"{outcomes['away_win']*100:.1f}%"],
                textposition="outside",
            ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE, height=380,
                title=f"Probabilidad de resultado: {pred_home} vs {pred_away}",
                yaxis_title="Probabilidad (%)", margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Marcadores más probables")
            top_df = pd.DataFrame([
                {"Marcador": f"{i} - {j}", "Probabilidad": f"{p * 100:.1f}%"}
                for (i, j), p in result["top_scorelines"]
            ])
            st.dataframe(top_df, use_container_width=True, hide_index=True)

            st.markdown(
                '<p class="section-note">Modelo simplificado: no incluye corrección de Dixon-Coles para '
                'marcadores bajos, ni forma reciente, lesiones o calendario. Pensado como estimación '
                'orientativa a partir de goles históricos, no como predicción definitiva.</p>',
                unsafe_allow_html=True,
            )

st.write("")
st.caption("Datos: football-data.co.uk (dataset público). Panel construido con Streamlit + Plotly.")
