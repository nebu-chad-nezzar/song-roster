import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Weekly Song Roster", layout="wide")


# -----------------------------
# HARD-CODED HYMN RULES
# -----------------------------
HYMN_STYLE_NAME = "Hymn"
MAX_HYMNS_PER_WEEK = 1


# -----------------------------
# Models / Config
# -----------------------------
@dataclass
class Config:
    total_weeks: int
    songs_per_week: int
    min_repeat_gap: int

    style_targets: dict
    feel_targets: dict

    seed: int
    rotation_alpha: float  # controls how strongly RotationScore biases selection

    must_include_styles: set  # styles that must appear at least once every week
    must_exclude_styles: set  # styles that must NOT appear (0 per week)


REQUIRED_COLS = ["Song name", "Style", "Feel", "RotationScore"]


# -----------------------------
# Helpers
# -----------------------------
def normalise_targets(targets: dict) -> dict:
    """
    Normalise positive targets so they sum to 1.
    Ignores missing/blank keys and non-positive values.
    """
    cleaned = {k: float(v) for k, v in targets.items() if k and float(v) > 0}
    s = sum(cleaned.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in cleaned.items()}


def is_style(value: str, style_name: str) -> bool:
    return str(value).strip().lower() == str(style_name).strip().lower()


def is_hymn(style_value: str) -> bool:
    return is_style(style_value, HYMN_STYLE_NAME)


def max_uses_per_song(total_weeks: int, min_repeat_gap: int) -> int:
    return math.ceil(total_weeks / (min_repeat_gap + 1))


def weighted_pick_one(rng, eligible: pd.DataFrame, cfg: Config) -> str:
    """
    Weight = style_weight * feel_weight * rotation_weight
    rotation_weight = RotationScore ** alpha

    RotationScore is allowed to be 0..5:
      - 0 => weight 0 => never selected (hard exclusion)
    """
    eligible = eligible.copy()

    # Style weight
    eligible["style_w"] = eligible["Style"].map(lambda s: cfg.style_targets.get(s, 0.0))
    eligible.loc[eligible["style_w"] <= 0, "style_w"] = 0.05  # baseline for non-targeted

    # Feel weight
    eligible["feel_w"] = eligible["Feel"].map(lambda f: cfg.feel_targets.get(f, 0.0))
    eligible.loc[eligible["feel_w"] <= 0, "feel_w"] = 0.05  # baseline for non-targeted

    # Rotation weight (0..5)
    rot = pd.to_numeric(eligible["RotationScore"], errors="coerce").fillna(0.0).clip(lower=0.0)
    eligible["rot_w"] = rot.astype(float) ** float(cfg.rotation_alpha)

    # Combined weight
    eligible["w"] = eligible["style_w"] * eligible["feel_w"] * eligible["rot_w"]

    weights = eligible["w"].to_numpy(dtype=float)
    if weights.sum() <= 0:
        # If all weights are 0 (e.g., all RotationScore=0), no valid pick
        return None

    probs = weights / weights.sum()
    idx = rng.choice(len(eligible), p=probs)
    return eligible.iloc[idx]["Song name"]


def feasibility_checks(df: pd.DataFrame, cfg: Config):
    """
    Checks:
    - weekly uniqueness possible
    - global upper bound capacity (rough)
    - must-include count <= songs per week
    - must-include and must-exclude conflict
    - hymn must-include sanity
    - warn if many songs have RotationScore=0
    """
    n_songs = df["Song name"].astype(str).str.strip().nunique()
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

    if len(cfg.must_include_styles) > cfg.songs_per_week:
        severity = "error"
        messages.append(
            f"Must-include styles count ({len(cfg.must_include_styles)}) exceeds songs_per_week ({cfg.songs_per_week})."
        )

    conflicts = cfg.must_include_styles.intersection(cfg.must_exclude_styles)
    if conflicts:
        severity = "error"
        messages.append(
            f"Conflicting rules: these styles are both must-include and must-exclude: {sorted(conflicts)}"
        )

    hymn_songs = int((df["Style"].map(is_hymn)).sum())
    if hymn_songs == 0 and any(is_style(s, HYMN_STYLE_NAME) for s in cfg.must_include_styles):
        if severity != "error":
            severity = "warning"
        messages.append(
            f"'{HYMN_STYLE_NAME}' is marked must-include, but there are 0 songs with Style='{HYMN_STYLE_NAME}'."
        )

    # RotationScore=0 count (informational)
    rot0 = int((df["RotationScore"] == 0).sum())
    if rot0 > 0:
        messages.append(f"Info: {rot0} songs have RotationScore=0 (they will never be selected).")

    stats = {
        "unique_songs": int(n_songs),
        "demand": int(demand),
        "supply": int(supply),
        "per_song_cap": int(per_song_cap),
        "hymn_songs": int(hymn_songs),
        "rotation_zero": int(rot0),
    }
    return severity, messages, stats


# -----------------------------
# Core generator
# -----------------------------
def generate_roster(df: pd.DataFrame, cfg: Config) -> list[list[str]]:
    """
    Rules:
    - No duplicate songs within a week
    - Global min repeat gap across weeks (cooldown), EXCEPT:
        - If Hymn is required (must-include) and no hymn is eligible due to gap,
          relax cooldown FOR HYMNS ONLY to satisfy requirement ("C")
    - Must include >=1 song per required style each week (when possible)
    - Must exclude styles always
    - Hard rule: max 1 hymn per week
    - RotationScore biases selection probabilities; RotationScore=0 => never selected
    """
    rng = np.random.default_rng(cfg.seed)

    df = df.copy()
    df["Song name"] = df["Song name"].astype(str).str.strip()
    df["Style"] = df["Style"].astype(str).str.strip()
    df["Feel"] = df["Feel"].astype(str).str.strip()
    df["RotationScore"] = pd.to_numeric(df["RotationScore"], errors="coerce").fillna(0).clip(lower=0)

    cooldown_until = {name: 0 for name in df["Song name"].tolist()}
    roster: list[list[str]] = []

    def is_excluded_style(style_val: str) -> bool:
        return any(is_style(style_val, x) for x in cfg.must_exclude_styles)

    for w in range(cfg.total_weeks):
        week_songs: list[str] = []
        used_this_week = set()
        hymns_this_week = 0

        def not_excluded(local_df: pd.DataFrame) -> pd.Series:
            if not cfg.must_exclude_styles:
                return pd.Series([True] * len(local_df), index=local_df.index)
            return ~local_df["Style"].map(lambda s: is_excluded_style(s))

        def not_rotation_zero(local_df: pd.DataFrame) -> pd.Series:
            # Hard exclusion: RotationScore=0 means never pick
            return local_df["RotationScore"].astype(float) > 0

        def base_eligible(local_df: pd.DataFrame) -> pd.Series:
            return (
                not_excluded(local_df)
                & not_rotation_zero(local_df)
                & local_df["Song name"].map(lambda s: cooldown_until.get(s, 0) <= w)
                & (~local_df["Song name"].isin(used_this_week))
            )

        def eligible_with_hymn_cap(local_df: pd.DataFrame, mask: pd.Series) -> pd.Series:
            if hymns_this_week >= MAX_HYMNS_PER_WEEK:
                mask = mask & (~local_df["Style"].map(is_hymn))
            return mask

        def pick_and_commit(song_name: str):
            nonlocal hymns_this_week
            week_songs.append(song_name)
            used_this_week.add(song_name)

            style_val = df.loc[df["Song name"] == song_name, "Style"].iloc[0]
            if is_hymn(style_val):
                hymns_this_week += 1

            cooldown_until[song_name] = w + cfg.min_repeat_gap + 1

        # 1) Satisfy must-include styles first
        required_styles = list(cfg.must_include_styles)
        required_styles.sort(key=lambda s: 0 if is_style(s, HYMN_STYLE_NAME) else 1)

        for style_name in required_styles:
            if len(week_songs) >= cfg.songs_per_week:
                break

            if any(is_style(style_name, x) for x in cfg.must_exclude_styles):
                continue

            if is_style(style_name, HYMN_STYLE_NAME) and hymns_this_week >= MAX_HYMNS_PER_WEEK:
                continue

            strict_mask = base_eligible(df)
            strict_mask = eligible_with_hymn_cap(df, strict_mask)
            strict_style = strict_mask & df["Style"].map(lambda s: is_style(s, style_name))
            eligible_strict = df[strict_style].copy()

            if not eligible_strict.empty:
                picked = weighted_pick_one(rng, eligible_strict, cfg)
                if picked is not None:
                    pick_and_commit(picked)
                continue

            # Behaviour C: relax cooldown for hymns only (still respects RotationScore>0, must-exclude, no dup, max 1 hymn)
            if is_style(style_name, HYMN_STYLE_NAME):
                relaxed_mask = (
                    not_excluded(df)
                    & not_rotation_zero(df)
                    & (~df["Song name"].isin(used_this_week))  # ignore cooldown
                )
                relaxed_mask = eligible_with_hymn_cap(df, relaxed_mask)
                relaxed_style = relaxed_mask & df["Style"].map(is_hymn)
                eligible_relaxed = df[relaxed_style].copy()

                if not eligible_relaxed.empty:
                    picked = weighted_pick_one(rng, eligible_relaxed, cfg)
                    if picked is not None:
                        pick_and_commit(picked)

        # 2) Fill remaining slots
        while len(week_songs) < cfg.songs_per_week:
            mask = base_eligible(df)
            mask = eligible_with_hymn_cap(df, mask)
            eligible = df[mask].copy()
            if eligible.empty:
                break

            picked = weighted_pick_one(rng, eligible, cfg)
            if picked is None:
                break
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
    st.info("Upload a CSV with columns: Song name, Style, Feel, RotationScore")
    st.stop()

df = pd.read_csv(uploaded)
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Clean key columns early
df["Song name"] = df["Song name"].astype(str).str.strip()
df["Style"] = df["Style"].astype(str).str.strip()
df["Feel"] = df["Feel"].astype(str).str.strip()
df["RotationScore"] = pd.to_numeric(df["RotationScore"], errors="coerce").fillna(0).clip(lower=0)

st.subheader("Preview")
st.dataframe(df.head(25), use_container_width=True)

st.subheader("Configuration")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    total_weeks = st.number_input("Total weeks to roster", min_value=1, max_value=260, value=12, step=1)
with col2:
    songs_per_week = st.number_input("Songs per week", min_value=1, max_value=50, value=4, step=1)
with col3:
    min_repeat_gap = st.number_input("Min repeat frequency (weeks)", min_value=0, max_value=260, value=4, step=1)
with col4:
    seed = st.number_input("Random seed (repeatable output)", min_value=0, max_value=10_000_000, value=42, step=1)
with col5:
    rotation_alpha = st.slider("Rotation strength (alpha)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)

# Style targets
styles = sorted(df["Style"].unique().tolist())
st.markdown("**Target selection probability for each Style** (values will be normalised to sum to 1)")
default_style_targets = pd.DataFrame({"Style": styles, "Target": [0.0] * len(styles)})
if len(styles) == 1:
    default_style_targets["Target"] = [1.0]

style_targets_df = st.data_editor(
    default_style_targets,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "Target": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
    },
)
style_targets_raw = dict(zip(style_targets_df["Style"], style_targets_df["Target"]))
style_targets = normalise_targets(style_targets_raw)
if not style_targets:
    st.warning("No positive Style targets set. Styles will be treated near-uniformly (small baseline weight).")

# Feel targets
feels = sorted(df["Feel"].unique().tolist())
st.markdown("**Target selection probability for each Feel** (values will be normalised to sum to 1)")
default_feel_targets = pd.DataFrame({"Feel": feels, "Target": [0.0] * len(feels)})
if len(feels) == 1:
    default_feel_targets["Target"] = [1.0]

feel_targets_df = st.data_editor(
    default_feel_targets,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "Target": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
    },
)
feel_targets_raw = dict(zip(feel_targets_df["Feel"], feel_targets_df["Target"]))
feel_targets = normalise_targets(feel_targets_raw)
if not feel_targets:
    st.warning("No positive Feel targets set. Feels will be treated near-uniformly (small baseline weight).")

# Must include / exclude per style (unchanged)
st.subheader("Per-style rules")
rules_df = pd.DataFrame(
    {
        "Style": styles,
        "Must include (>=1 per week)": [False] * len(styles),
        "Must exclude (0 per week)": [False] * len(styles),
    }
)
rules_df = st.data_editor(rules_df, use_container_width=True, num_rows="fixed")

must_include_styles = set(rules_df.loc[rules_df["Must include (>=1 per week)"] == True, "Style"].tolist())
must_exclude_styles = set(rules_df.loc[rules_df["Must exclude (0 per week)"] == True, "Style"].tolist())

cfg = Config(
    total_weeks=int(total_weeks),
    songs_per_week=int(songs_per_week),
    min_repeat_gap=int(min_repeat_gap),
    style_targets=style_targets,
    feel_targets=feel_targets,
    seed=int(seed),
    rotation_alpha=float(rotation_alpha),
    must_include_styles=must_include_styles,
    must_exclude_styles=must_exclude_styles,
)

# Feasibility check
severity, messages, stats = feasibility_checks(df=df, cfg=cfg)

st.markdown("### Feasibility check")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Unique songs", stats["unique_songs"])
m2.metric("Total picks (weeks × songs/week)", stats["demand"])
m3.metric("Max uses per song (upper bound)", stats["per_song_cap"])
m4.metric("Supply upper bound", stats["supply"])
m5.metric("Hymn songs", stats["hymn_songs"])
m6.metric("RotationScore=0 songs", stats["rotation_zero"])

st.caption(f"Hard-coded hymn rules: Style='{HYMN_STYLE_NAME}', max {MAX_HYMNS_PER_WEEK} hymn per week.")

if cfg.must_include_styles:
    st.info(f"Must include each week: {sorted(cfg.must_include_styles)}")
if cfg.must_exclude_styles:
    st.info(f"Must exclude (never pick): {sorted(cfg.must_exclude_styles)}")

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
        counts.columns = ["Song name", "Times selected"]
        st.dataframe(counts, use_container_width=True)

        name_to_style = dict(zip(df["Song name"], df["Style"]))
        style_counts = pd.Series([name_to_style.get(s, "Unknown") for s in flat]).value_counts()
        achieved_style = (style_counts / style_counts.sum()).reset_index()
        achieved_style.columns = ["Style", "Achieved proportion"]
        st.dataframe(achieved_style, use_container_width=True)

        name_to_feel = dict(zip(df["Song name"], df["Feel"]))
        feel_counts = pd.Series([name_to_feel.get(s, "Unknown") for s in flat]).value_counts()
        achieved_feel = (feel_counts / feel_counts.sum()).reset_index()
        achieved_feel.columns = ["Feel", "Achieved proportion"]
        st.dataframe(achieved_feel, use_container_width=True)