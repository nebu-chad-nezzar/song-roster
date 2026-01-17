import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass

st.set_page_config(page_title="Weekly Song Roster", layout="wide")


@dataclass
class Config:
    total_weeks: int
    songs_per_week: int
    min_repeat_gap: int
    style_targets: dict
    rotation_weights: dict
    seed: int


REQUIRED_COLS = ["Song name", "Style", "Rotation"]


def normalise_style_targets(style_targets: dict) -> dict:
    # Remove blanks/non-positive, then normalise to sum=1
    cleaned = {k: float(v) for k, v in style_targets.items() if k and float(v) > 0}
    s = sum(cleaned.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in cleaned.items()}


def generate_roster(df: pd.DataFrame, cfg: Config) -> list[list[str]]:
    rng = np.random.default_rng(cfg.seed)

    # Basic sanitisation
    df = df.copy()
    df["Song name"] = df["Song name"].astype(str).str.strip()
    df["Style"] = df["Style"].astype(str).str.strip()
    df["Rotation"] = df["Rotation"].astype(str).str.strip().str.lower()

    # Cooldown tracker: song -> earliest week index (0-based) it can appear again
    cooldown_until = {name: 0 for name in df["Song name"].tolist()}

    roster = []
    for w in range(cfg.total_weeks):
        week_songs = []
        used_this_week = set()

        for _ in range(cfg.songs_per_week):
            # Eligible songs: not in cooldown and not already used this week
            eligible = df[
                df["Song name"].map(lambda s: cooldown_until.get(s, 0) <= w)
                & (~df["Song name"].isin(used_this_week))
            ].copy()

            if eligible.empty:
                # If constraints are too tight, break early (or you could relax rules here)
                break

            # Compute weights: style_target * rotation_weight
            eligible["style_w"] = eligible["Style"].map(lambda s: cfg.style_targets.get(s, 0.0))
            eligible["rot_w"] = eligible["Rotation"].map(lambda r: cfg.rotation_weights.get(r, 1.0))

            # If a style wasn't given a target, give it a small baseline chance
            # so the algorithm still works with incomplete style targets.
            eligible.loc[eligible["style_w"] <= 0, "style_w"] = 0.05

            eligible["w"] = eligible["style_w"] * eligible["rot_w"]

            # If all weights end up zero, fallback to uniform
            weights = eligible["w"].to_numpy(dtype=float)
            if weights.sum() <= 0:
                weights = np.ones(len(eligible), dtype=float)

            probs = weights / weights.sum()
            picked_idx = rng.choice(len(eligible), p=probs)
            picked_song = eligible.iloc[picked_idx]["Song name"]

            week_songs.append(picked_song)
            used_this_week.add(picked_song)
            cooldown_until[picked_song] = w + cfg.min_repeat_gap + 1  # next eligible week

        roster.append(week_songs)

    return roster


def roster_to_text(roster: list[list[str]]) -> str:
    lines = []
    for i, songs in enumerate(roster, start=1):
        lines.append(f"Week {i}")
        for j, s in enumerate(songs, start=1):
            lines.append(f"Song {j}: {s}")
        lines.append("")  # blank line between weeks
    return "\n".join(lines).strip()


st.title("Weekly Song Roster Generator")

uploaded = st.file_uploader("Upload master song list (CSV)", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV with columns: Song name, Style, Rotation")
    st.stop()

df = pd.read_csv(uploaded)

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

# ---- Config UI
st.subheader("Configuration")

col1, col2, col3, col4 = st.columns(4)
with col1:
    total_weeks = st.number_input("Total weeks to roster", min_value=1, max_value=260, value=12, step=1)
with col2:
    songs_per_week = st.number_input("Songs per week", min_value=1, max_value=50, value=4, step=1)
with col3:
    min_repeat_gap = st.number_input("Min repeat frequency (weeks)", min_value=0, max_value=260, value=4, step=1)
with col4:
    seed = st.number_input("Random seed (for repeatable output)", min_value=0, max_value=10_000_000, value=42, step=1)

# Rotation weights (non-configurable field per your spec, but mapping is useful to define once)
st.markdown("**Rotation weights** (used to bias selection frequency)")
rotation_weights = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3,
}

rot_present = sorted(df["Rotation"].astype(str).str.strip().str.lower().unique().tolist())
unknown_rots = [r for r in rot_present if r not in rotation_weights]
if unknown_rots:
    st.warning(f"Unknown Rotation values found: {unknown_rots}. They will default to weight=1.0.")

# Style targets table
styles = sorted(df["Style"].astype(str).str.strip().unique().tolist())
st.markdown("**Target selection probability for each style** (will be normalised to sum to 1)")

default = pd.DataFrame({"Style": styles, "Target": [0.0] * len(styles)})
# Give a simple default: if only 1 style, set it to 1
if len(styles) == 1:
    default["Target"] = [1.0]

style_df = st.data_editor(
    default,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "Target": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
    },
)

style_targets_raw = dict(zip(style_df["Style"], style_df["Target"]))
style_targets = normalise_style_targets(style_targets_raw)

if not style_targets:
    st.warning("No positive style targets set. Styles will be treated nearly uniformly with a small baseline weight.")

cfg = Config(
    total_weeks=int(total_weeks),
    songs_per_week=int(songs_per_week),
    min_repeat_gap=int(min_repeat_gap),
    style_targets=style_targets,
    rotation_weights=rotation_weights,
    seed=int(seed),
)

# ---- Generate
if st.button("Generate roster", type="primary"):
    roster = generate_roster(df, cfg)

    st.subheader("Roster output")
    out_text = roster_to_text(roster)
    st.text(out_text)

    st.download_button(
        "Download as .txt",
        data=out_text.encode("utf-8"),
        file_name="weekly_roster.txt",
        mime="text/plain",
    )

    # Optional: diagnostics
    st.subheader("Diagnostics (optional)")
    flat = [s for week in roster for s in week]
    if flat:
        counts = pd.Series(flat).value_counts().reset_index()
        counts.columns = ["Song name", "Times selected"]
        st.dataframe(counts, use_container_width=True)

        # Style distribution achieved
        name_to_style = dict(zip(df["Song name"], df["Style"]))
        style_counts = pd.Series([name_to_style.get(s, "Unknown") for s in flat]).value_counts()
        achieved = (style_counts / style_counts.sum()).reset_index()
        achieved.columns = ["Style", "Achieved proportion"]
        st.dataframe(achieved, use_container_width=True)