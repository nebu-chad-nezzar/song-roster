import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Weekly Song Roster", layout="wide")


# -----------------------------
# Models / Config
# -----------------------------
@dataclass
class Config:
    total_weeks: int
    songs_per_week: int
    min_repeat_gap: int
    style_targets: dict
    rotation_weights: dict
    seed: int

    hymn_style_name: str
    max_hymns_per_week: int

    must_include_styles: set  # styles that must appear at least once every week


REQUIRED_COLS = ["Title", "Style", "Rotation"]


# -----------------------------
# Helpers
# -----------------------------
def normalise_style_targets(style_targets: dict) -> dict:
    cleaned = {k: float(v) for k, v in style_targets.items() if k and float(v) > 0}
    s = sum(cleaned.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in cleaned.items()}


def is_style(value: str, style_name: str) -> bool:
    return str(value).strip().lower() == str(style_name).strip().lower()


def is_hymn(style_value: str, hymn_style_name: str) -> bool:
    return is_style(style_value, hymn_style_name)


def max_uses_per_song(total_weeks: int, min_repeat_gap: int) -> int:
    return math.ceil(total_weeks / (min_repeat_gap + 1))


def weighted_pick_one(rng, eligible: pd.DataFrame, cfg: Config) -> str:
    eligible = eligible.copy()
    eligible["style_w"] = eligible["Style"].map(lambda s: cfg.style_targets.get(s, 0.0))
    eligible["rot_w"] = eligible["Rotation"].map(lambda r: cfg.rotation_weights.get(r, 1.0))

    # Baseline chance for styles not explicitly targeted
    eligible.loc[eligible["style_w"] <= 0, "style_w"] = 0.05

    eligible["w"] = eligible["style_w"] * eligible["rot_w"]
    weights = eligible["w"].to_numpy(dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(eligible), dtype=float)

    probs = weights / weights.sum()
    idx = rng.choice(len(eligible), p=probs)
    return eligible.iloc[idx]["Title"]


def feasibility_checks(df: pd.DataFrame, cfg: Config):
    """
    Lightweight checks + warnings (doesn't try to prove feasibility with must-include constraints).
    """
    n_songs = df["Title"].astype(str).str.strip().nunique()
    demand = cfg.total_weeks * cfg.songs_per_week

    weekly_ok = cfg.songs_per_week <= n_songs

    per_song_cap = max_uses_per_song(cfg.total_weeks, cfg.min_repeat_gap)
    supply = n_songs * per_song_cap
    global_ok = supply >= demand

    messages = []
    severity = "ok"

    if not weekly_ok:
        severity = "error"
        messages.append(
            f"Weekly uniqueness impossible: songs_per_week={cfg.songs_per_week} > unique_songs={n_songs}."
        )

    if not global_ok:
        if severity != "error":
            severity = "warning"
        messages.append(
            "Repeat-gap may make the plan incomplete.\n"
            f"- Demand (total picks): {demand}\n"
            f"- Supply upper bound: {supply} (= {n_songs} songs × {per_song_cap} max uses each)\n"
            f"- Consider reducing min_repeat_gap or songs_per_week, or increasing master list size."
        )

    # Must-include sanity: you cannot require more styles than slots in a week
    if len(cfg.must_include_styles) > cfg.songs_per_week:
        severity = "error"
        messages.append(
            f"Must-include styles count ({len(cfg.must_include_styles)}) exceeds songs_per_week ({cfg.songs_per_week})."
        )

    hymn_songs = int((df["Style"].map(lambda s: is_hymn(s, cfg.hymn_style_name))).sum())
    stats = {
        "unique_songs": int(n_songs),
        "demand": int(demand),
        "supply": int(supply),
        "per_song_cap": int(per_song_cap),
        "hymn_songs": hymn_songs,
    }
    return severity, messages, stats


# -----------------------------
# Core generator
# -----------------------------
def generate_roster(df: pd.DataFrame, cfg: Config) -> list[list[str]]:
    """
    Rules:
    - No duplicate songs within a week
    - Min repeat gap across weeks (cooldown), EXCEPT:
        - If Hymn is required (must-include) and no hymn is eligible due to gap,
          relax the repeat-gap FOR HYMNS ONLY to satisfy requirement ("C")
    - Must include at least 1 song from each style in cfg.must_include_styles each week (when possible)
    - Hard rule: max 1 hymn per week
    """
    rng = np.random.default_rng(cfg.seed)

    df = df.copy()
    df["Title"] = df["Title"].astype(str).str.strip()
    df["Style"] = df["Style"].astype(str).str.strip()
    df["Rotation"] = df["Rotation"].astype(str).str.strip().str.lower()

    cooldown_until = {name: 0 for name in df["Title"].tolist()}
    roster: list[list[str]] = []

    for w in range(cfg.total_weeks):
        week_songs: list[str] = []
        used_this_week = set()
        hymns_this_week = 0

        def base_eligible(local_df: pd.DataFrame) -> pd.Series:
            return (
                local_df["Title"].map(lambda s: cooldown_until.get(s, 0) <= w)
                & (~local_df["Title"].isin(used_this_week))
            )

        def eligible_with_hymn_cap(local_df: pd.DataFrame, mask: pd.Series) -> pd.Series:
            if cfg.max_hymns_per_week is not None and hymns_this_week >= cfg.max_hymns_per_week:
                mask = mask & (~local_df["Style"].map(lambda s: is_hymn(s, cfg.hymn_style_name)))
            return mask

        def pick_and_commit(song_name: str):
            nonlocal hymns_this_week
            week_songs.append(song_name)
            used_this_week.add(song_name)

            style_val = df.loc[df["Title"] == song_name, "Style"].iloc[0]
            if is_hymn(style_val, cfg.hymn_style_name):
                hymns_this_week += 1

            cooldown_until[song_name] = w + cfg.min_repeat_gap + 1

        # 1) Satisfy must-include styles first (one per required style)
        #    Order matters slightly; do hymn first so the max-1 constraint is respected early.
        required_styles = list(cfg.must_include_styles)
        required_styles.sort(key=lambda s: 0 if is_hymn(s, cfg.hymn_style_name) else 1)

        for style_name in required_styles:
            if len(week_songs) >= cfg.songs_per_week:
                break

            # If hymn cap already reached, we cannot satisfy hymn requirement
            if is_hymn(style_name, cfg.hymn_style_name) and hymns_this_week >= cfg.max_hymns_per_week:
                continue

            # Eligible (strict, respecting cooldown)
            strict_mask = base_eligible(df)
            strict_mask = eligible_with_hymn_cap(df, strict_mask)
            strict_style = strict_mask & df["Style"].map(lambda s: is_style(s, style_name))
            eligible_strict = df[strict_style].copy()

            if not eligible_strict.empty:
                picked = weighted_pick_one(rng, eligible_strict, cfg)
                pick_and_commit(picked)
                continue

            # Behaviour "C": if the required style is HYMN and cooldown blocks it, relax cooldown for hymns only
            if is_hymn(style_name, cfg.hymn_style_name):
                relaxed_mask = (~df["Title"].isin(used_this_week))  # ignore cooldown
                relaxed_mask = eligible_with_hymn_cap(df, relaxed_mask)
                relaxed_style = relaxed_mask & df["Style"].map(lambda s: is_hymn(s, cfg.hymn_style_name))
                eligible_relaxed = df[relaxed_style].copy()

                if not eligible_relaxed.empty:
                    picked = weighted_pick_one(rng, eligible_relaxed, cfg)
                    pick_and_commit(picked)
                # else: no hymns at all (or only duplicates), can’t satisfy

        # 2) Fill remaining slots with weighted picks under normal eligibility rules
        while len(week_songs) < cfg.songs_per_week:
            mask = base_eligible(df)
            mask = eligible_with_hymn_cap(df, mask)
            eligible = df[mask].copy()

            if eligible.empty:
                break

            picked = weighted_pick_one(rng, eligible, cfg)
            pick_and_commit(picked)

        roster.append(week_songs)

    return roster


def roster_to_text(roster: list[list[str]]) -> str:
    lines = []
    for i, songs in enumerate(roster, start=1):
        lines.append(f"Week {i}")
        for j, s in enumerate(songs, start=1):
            lines.append(f"Song {j}: {s}")
        lines.append("")
    return "\n".join(lines).strip()


# -----------------------------
# UI
# -----------------------------
st.title("Weekly Song Roster Generator")

uploaded = st.file_uploader("Upload master song list (CSV)", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV with columns: Title, Style, Rotation")
    st.stop()

df = pd.read_csv(uploaded)
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Clean key columns early
df["Title"] = df["Title"].astype(str).str.strip()
df["Style"] = df["Style"].astype(str).str.strip()
df["Rotation"] = df["Rotation"].astype(str).str.strip().str.lower()

st.subheader("Preview")
st.dataframe(df.head(25), use_container_width=True)

# Main configuration
st.subheader("Configuration")
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_weeks = st.number_input("Total weeks to roster", min_value=1, max_value=260, value=12, step=1)
with col2:
    songs_per_week = st.number_input("Songs per week", min_value=1, max_value=50, value=4, step=1)
with col3:
    min_repeat_gap = st.number_input("Min repeat frequency (weeks)", min_value=0, max_value=260, value=4, step=1)
with col4:
    seed = st.number_input("Random seed (repeatable output)", min_value=0, max_value=10_000_000, value=42, step=1)

# Rotation weights (fixed mapping)
st.markdown("**Rotation weights** (fixed mapping used to bias selection frequency)")
rotation_weights = {"high": 1.0, "medium": 0.6, "low": 0.3}
unknown_rots = [r for r in sorted(df["Rotation"].unique().tolist()) if r not in rotation_weights]
if unknown_rots:
    st.warning(f"Unknown Rotation values found: {unknown_rots}. They will default to weight=1.0.")

# Style targets
styles = sorted(df["Style"].unique().tolist())
st.markdown("**Target selection probability for each style** (values will be normalised to sum to 1)")
default_targets = pd.DataFrame({"Style": styles, "Target": [0.0] * len(styles)})
if len(styles) == 1:
    default_targets["Target"] = [1.0]

style_targets_df = st.data_editor(
    default_targets,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "Target": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
    },
)
style_targets_raw = dict(zip(style_targets_df["Style"], style_targets_df["Target"]))
style_targets = normalise_style_targets(style_targets_raw)
if not style_targets:
    st.warning("No positive style targets set. Styles will be treated near-uniformly (small baseline weight).")

# Must-include styles screen (your request)
st.subheader("Must-include styles (at least 1 per week)")
must_df = pd.DataFrame({"Style": styles, "Must include (>=1 per week)": [False] * len(styles)})

must_df = st.data_editor(
    must_df,
    use_container_width=True,
    num_rows="fixed",
)
must_include_styles = set(must_df.loc[must_df["Must include (>=1 per week)"] == True, "Style"].tolist())

# Hymn rules (hard constraint remains)
st.subheader("Hymn rules")
colh1, colh2 = st.columns(2)
with colh1:
    hymn_style_name = st.text_input("Style name that counts as a hymn", value="Hymn")
with colh2:
    max_hymns_per_week = st.number_input("Max hymns per week (hard rule)", min_value=0, max_value=10, value=1, step=1)

cfg = Config(
    total_weeks=int(total_weeks),
    songs_per_week=int(songs_per_week),
    min_repeat_gap=int(min_repeat_gap),
    style_targets=style_targets,
    rotation_weights=rotation_weights,
    seed=int(seed),
    hymn_style_name=str(hymn_style_name),
    max_hymns_per_week=int(max_hymns_per_week),
    must_include_styles=must_include_styles,
)

# Feasibility check
severity, messages, stats = feasibility_checks(df=df, cfg=cfg)

st.markdown("### Feasibility check")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Unique songs", stats["unique_songs"])
m2.metric("Total picks (weeks × songs/week)", stats["demand"])
m3.metric("Max uses per song (upper bound)", stats["per_song_cap"])
m4.metric("Supply upper bound", stats["supply"])
m5.metric("Hymn songs", stats["hymn_songs"])

if cfg.must_include_styles:
    st.info(f"Must include each week: {sorted(cfg.must_include_styles)}")

if severity == "ok":
    st.success("Feasibility check passed: settings look achievable with the current master list.")
elif severity == "warning":
    st.warning("Feasibility warning:\n\n" + "\n\n".join(messages))
else:
    st.error("Feasibility error:\n\n" + "\n\n".join(messages))

disable_generate = (severity == "error")

# Generate
if st.button("Generate roster", type="primary", disabled=disable_generate):
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

    # Diagnostics
    st.subheader("Diagnostics (optional)")
    flat = [s for week in roster for s in week]
    if flat:
        counts = pd.Series(flat).value_counts().reset_index()
        counts.columns = ["Title", "Times selected"]
        st.dataframe(counts, use_container_width=True)

        name_to_style = dict(zip(df["Title"], df["Style"]))
        style_counts = pd.Series([name_to_style.get(s, "Unknown") for s in flat]).value_counts()
        achieved = (style_counts / style_counts.sum()).reset_index()
        achieved.columns = ["Style", "Achieved proportion"]
        st.dataframe(achieved, use_container_width=True)