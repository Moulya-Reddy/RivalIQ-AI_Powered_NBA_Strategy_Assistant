"""
RivalIQ Streamlit app: pick two NBA teams, get a RAG-grounded scouting
report, an RL-derived clutch-strategy recommendation with a learning
curve vs baselines, and a visual team comparison.
"""

import os
import tempfile
import streamlit as st

from data.client import BallDontLieClient
from data.analytics import build_team_profile
from rag.notes import build_corpus
from rag.retriever import ScoutingRetriever
from rag.generator import ScoutingReportGenerator
from eval.faithfulness import check_faithfulness
from rl.team_params import derive_team_skill, derive_opponent_skill
from rl.environment import ClutchPossessionEnv, ACTION_NAMES, TWO_POINTER, THREE_POINTER
from rl.agent import QLearningAgent, train, evaluate_random_baseline, evaluate_fixed_baseline
from genai.visuals import render_team_comparison_radar, render_learning_curve

st.set_page_config(page_title="RivalIQ - NBA Strategy Assistant", page_icon="🏀", layout="wide")

st.title("🏀 RivalIQ - NBA Strategy Assistant")
st.caption(
    "RAG-grounded scouting reports + an RL agent that learns clutch-time strategy, "
    "calibrated to real team data. Extends multi-agent RL research into a live product."
)

with st.sidebar:
    st.header("API keys")
    provider_label = st.radio(
        "Report-generation provider",
        ["Groq (free)", "Anthropic (paid)"],
        help=(
            "Groq's API has a free tier (no credit card) and runs Llama 3.3 70B. "
            "Anthropic's Claude API is pay-as-you-go, no ongoing free tier."
        ),
    )
    provider = "groq" if provider_label.startswith("Groq") else "anthropic"

    if provider == "groq":
        llm_key_env = "GROQ_API_KEY"
        llm_key_label = "Groq API key"
        llm_key_help = "Free at console.groq.com — no credit card required."
    else:
        llm_key_env = "ANTHROPIC_API_KEY"
        llm_key_label = "Anthropic API key"
        llm_key_help = "Pay-as-you-go at console.anthropic.com."

    llm_key = st.text_input(llm_key_label, type="password",
                             value=os.environ.get(llm_key_env, ""),
                             help=llm_key_help)
    bdl_key = st.text_input("balldontlie API key", type="password",
                             value=os.environ.get("BALLDONTLIE_API_KEY", ""),
                             help="Free key (no cost) at app.balldontlie.io")
    st.divider()
    season = st.number_input("Season", min_value=2015, max_value=2025, value=2024)
    num_possessions = st.slider("Clutch possessions to simulate", 4, 10, 6)
    training_episodes = st.select_slider("RL training episodes", [2000, 4000, 8000, 15000], value=8000)


@st.cache_data(show_spinner=False)
def load_teams(_client_key):
    client = BallDontLieClient(api_key=_client_key)
    teams = client.get_teams()
    return {t["abbreviation"]: t for t in teams}


col1, col2 = st.columns(2)
with col1:
    team_a_abbr = st.text_input("Team A abbreviation", value="BOS")
with col2:
    team_b_abbr = st.text_input("Team B abbreviation", value="LAL")

run = st.button("Analyze matchup", type="primary")

if run:
    if not llm_key or not bdl_key:
        st.error(f"Please enter both your {llm_key_label} and your balldontlie API key in the sidebar.")
        st.stop()

    with st.spinner("Fetching team data..."):
        client = BallDontLieClient(api_key=bdl_key)
        teams_by_abbr = load_teams(bdl_key)

        if team_a_abbr.upper() not in teams_by_abbr or team_b_abbr.upper() not in teams_by_abbr:
            st.error("Team abbreviation not recognized. Try e.g. BOS, LAL, GSW, MIL, NYK.")
            st.stop()

        team_a = teams_by_abbr[team_a_abbr.upper()]
        team_b = teams_by_abbr[team_b_abbr.upper()]

        games_a = client.get_games(team_ids=[team_a["id"]], seasons=[season], max_pages=3)
        games_b = client.get_games(team_ids=[team_b["id"]], seasons=[season], max_pages=3)

        profile_a = build_team_profile(team_a, games_a)
        profile_b = build_team_profile(team_b, games_b)

    if not profile_a.games or not profile_b.games:
        st.warning("Not enough finished games found for this season for one of these teams.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📋 Scouting Report (RAG)", "📊 Team Comparison", "🎯 Clutch Strategy (RL)"])

    with tab1:
        st.subheader("RAG-grounded scouting report")
        corpus = build_corpus([profile_a, profile_b])
        retriever = ScoutingRetriever(corpus)
        retrieved = retriever.retrieve_for_matchup(team_a_abbr, team_b_abbr)

        with st.expander("Retrieved scouting notes (ground truth)", expanded=False):
            for note in retrieved:
                st.markdown(f"**{note.team_abbreviation}**: {note.text}")

        with st.spinner(f"Generating report via {provider_label}..."):
            generator = ScoutingReportGenerator(api_key=llm_key, provider=provider)
            report = generator.generate(retrieved)

        st.markdown(report)

        faith = check_faithfulness(report, retrieved)
        badge = "🟢" if faith.faithfulness_score >= 0.9 else "🟡" if faith.faithfulness_score >= 0.6 else "🔴"
        st.metric("Faithfulness score", f"{badge} {faith.faithfulness_score:.0%}",
                   help="Fraction of numeric claims in the report traceable to retrieved stats")
        if faith.ungrounded_values:
            st.warning(f"Ungrounded values detected: {faith.ungrounded_values}")

    with tab2:
        st.subheader("Team comparison")
        col_a, col_b = st.columns(2)
        col_a.json(profile_a.to_stats_dict())
        col_b.json(profile_b.to_stats_dict())

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            radar_path = render_team_comparison_radar(
                profile_a.to_stats_dict(), profile_b.to_stats_dict(), tmp.name
            )
            st.image(radar_path, caption=f"{team_a_abbr} vs {team_b_abbr}")

    with tab3:
        st.subheader("RL-derived clutch strategy")
        st.caption(
            "A Q-learning agent trained on a simulated end-game possession exchange, "
            "with success rates calibrated from each team's real scoring efficiency."
        )

        skill_a = derive_team_skill(profile_a)
        skill_b = derive_opponent_skill(profile_b)

        with st.spinner(f"Training RL agent for {training_episodes} episodes..."):
            env = ClutchPossessionEnv(team_skill=skill_a, opponent_skill=skill_b,
                                       num_possessions=num_possessions)
            agent = QLearningAgent(epsilon_decay_episodes=max(2000, training_episodes // 2))
            curve = train(env, agent, num_episodes=training_episodes,
                           eval_every=max(200, training_episodes // 10), eval_episodes=400)

            baselines = {
                "Random baseline": evaluate_random_baseline(env, 1500),
                "Always 2PT baseline": evaluate_fixed_baseline(env, 1500, TWO_POINTER),
                "Always 3PT baseline": evaluate_fixed_baseline(env, 1500, THREE_POINTER),
            }

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trained agent win rate", f"{curve[-1]['win_rate']:.1%}")
        m2.metric("Random baseline", f"{baselines['Random baseline']:.1%}")
        m3.metric("Always 2PT baseline", f"{baselines['Always 2PT baseline']:.1%}")
        m4.metric("Always 3PT baseline", f"{baselines['Always 3PT baseline']:.1%}")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            curve_path = render_learning_curve(curve, baselines, tmp.name)
            st.image(curve_path, caption="Learning curve: agent win rate vs training episodes")

        st.markdown("**Learned policy for a few key states** (score bucket, possessions left):")
        example_states = [(0, 1, "Trailing big, last possession"),
                           (3, 1, "Leading close, last possession"),
                           (2, num_possessions, "Tied, full clock left")]
        for score_bucket, poss_left, label in example_states:
            q_values = {ACTION_NAMES[a]: round(agent.q[((score_bucket, poss_left), a)], 3) for a in range(4)}
            best = max(q_values, key=q_values.get)
            st.markdown(f"- **{label}**: recommends `{best}` — {q_values}")
