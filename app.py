import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Weekly Song Roster", layout="wide")


# -----------------------------
# HARD-CODED HYMN CAP ONLY
# Hymn is identified by Type == "Hymn"
# -----------------------------
HYMN_TYPE_NAME = "Hymn"
MAX_HYMNS_PER_WEEK = 1


# -----------------------------
# Soft preference vocabulary (no "Never" - use Must exclude instead)
# -----------------------------
PREFERENCE_LEVELS = ["Rarely", "Sometimes", "About half", "Mostly"]

# Multipliers used as soft weights in selection
PREFERENCE_TO_WEIGHT = {
    "Rarely": 0.6,
    "Sometimes": 1.0,
    "About half": 1.8,
    "Mostly": 2.6,
}

# Tooltip / help text shown in the UI
PREFERENCE_HELP = (
    "Soft preference only (not a guarantee). Higher preference increases selection likelihood.\n"
    "- Rarely: discourage\n"
    "- Sometimes: neutral\n"
    "- About half: prefer strongly\n"
    "- Mostly: prefer very strongly\n"
    "Use 'Must exclude' to ban a Type completely."
)


# -----------------------------
# Models / Config
# -----------------------------
@dataclass
class Config:
    total_weeks: int
    songs_per_week: int
    min_repeat_gap: int

    seed: int
    rotation_alpha: float  # controls how strongly RotationScore biases selection

    # Soft preferences -> weights
    type_pref_weight: dict   # Type -> multiplier
    style_pref_weight: dict  # Style -> multiplier

    # Hard rules (per Type)
    must_include_types: set
    must_exclude_types: set


REQUIRED_COLS = ["Title", "Artist", "Type", "Style", "RotationScore"]


# -----------------------------
# Helpers
# -----------------------------
def is_value(value: str, name: str) -> bool:
    return str(value).strip().lower() == str(name).strip().lower()


def is_hymn(song_type: str) -> bool:
    return is_value(song_type, HYMN_TYPE_NAME)


def max_uses_per_song(total_weeks: int, min_repeat_gap: int) -> int:
    return math.ceil(total_weeks / (min_repeat_gap + 1))


def build_song_key(df: pd.DataFrame) -> pd.Series:
    """
    Unique key per song row used for cooldown + selection uniqueness:
      SongKey = "Title — Artist"
    """
    title = df["Title"].astype(str).str.strip().fillna("")
    artist = df["Artist"].astype(str).str.strip().fillna("")
    title = title.replace("nan", "")
    artist = artist.replace("nan", "")
    return (title + " — " + artist).str.strip(" —")


def preference_table_to_weight_map(values: list[str], preference_df: pd.DataFrame, value_col: str) -> dict:
    """
    Convert a dataframe like [Type, Preference] into a map: Type -> weight multiplier.
    """
    mapping = {}
    for _, row in preference_df.iterrows():
        key = str(row[value_col]).strip()
        pref = str(row["Preference"]).strip()
        if key == "" or key.lower() == "nan":
            continue
        mapping[key] = float(PREFERENCE_TO_WEIGHT.get(pref, 1.0))
    # Ensure any missing keys get neutral weight
    for v in values:
        mapping.setdefault(v, 1.0)
    return mapping


def weighted_pick_one(rng, eligible: pd.DataFrame, cfg: Config) -> str | None:
    """
    Weight = type_pref_weight * style_pref_weight * (RotationScore ** alpha)

    RotationScore is allowed to be 0..N:
      - 0 => hard exclusion (weight 0; never selected)
    Returns the chosen SongKey.
    """
    eligible = eligible.copy()

    eligible["type_w"] = eligible["Type"].map(lambda t: cfg.type_pref_weight.get(t, 1.0))
    eligible["style_w"] = eligible["Style"].map(lambda s: cfg.style_pref_weight.get(s, 1.0))

    rot = pd.to_numeric(eligible["RotationScore"], errors="coerce").fillna(0.0).clip(lower=0.0)
    eligible["rot_w"] = rot.astype(float) ** float(cfg.rotation_alpha)

    eligible["w"] = eligible["type_w"] * eligible["style_w"] * eligible["rot_w"]

    weights = eligible["w"].to_numpy(dtype=float)
    if weights.sum() <= 0:
        return None

    probs = weights / weights.sum()
    idx = rng.choice(len(eligible), p=probs)
    return eligible.iloc[idx]["SongKey"]


def feasibility_checks(df: pd.DataFrame, cfg: Config):
    """
    Checks:
    - weekly uniqueness possible
    - global upper bound capacity (rough)
    - must-include Type count <= songs per week
    - must-include and must-exclude Type conflict
    - info about RotationScore=0 disabled songs
    - hymn row count uses Type
    """
    unique_song_keys = df["SongKey"].nunique()
    demand = cfg.total_weeks * cfg.songs_per_week

    weekly_ok = cfg.songs_per_week <= unique_song_keys

    per_song_cap = max_uses_per_song(cfg.total_weeks, cfg.min_repeat_gap)
    supply = unique_song_keys * per_song_cap
    global_ok = supply >= demand

    messages = []
    severity = "ok"

    if not weekly_ok:
        severity = "error"
        messages.append(
            f"Weekly uniqueness impossible: songs_per_week={cfg.songs_per_week} > unique_songs={unique_song_keys}."
        )

    if not global_ok:
        if severity != "error":
            severity = "warning"
        messages.append(
            "Repeat-gap may make the plan incomplete.\n"
            f"- Demand (total picks): {demand}\n"
            f"- Supply upper bound: {supply} (= {unique_song_keys} songs × {per_song_cap} max uses each)\n"
            f"- Consider reducing min_repeat_gap or songs_per_week, or increasing master list size."
        )

    if len(cfg.must_include_types) > cfg.songs_per_week:
        severity = "error"
        messages.append(
            f"Must-include Types count ({len(cfg.must_include_types)}) exceeds songs_per_week ({cfg.songs_per_week})."
        )

    conflicts = cfg.must_include_types.intersection(cfg.must_exclude_types)
    if conflicts:
        severity = "error"
        messages.append(
            f"Conflicting rules: these Types are both must-include and must-exclude: {sorted(conflicts)}"
        )

    rot0 = int((df["RotationScore"] == 0).sum())
    if rot0 > 0:
        messages.append(f"Info: {rot0} songs have RotationScore=0 (they will never be selected).")

    hymn_rows = int((df["Type"].map(is_hymn)).sum())

    stats = {
        "unique_songs": int(unique_song_keys),
        "demand": int(demand),
        "supply": int(supply),
        "per_song_cap": int(per_song_cap),
        "hymn_rows": int(hymn_rows),
        "rotation_zero": int(rot0),
    }
    return severity, messages, stats


# -----------------------------
# Core generator
# -----------------------------
def generate_roster(df: pd.DataFrame, cfg: Config) -> list[list[str]]:
    """
    Returns roster as list of weeks; each week is list of SongKey.

    Rules:
    - No duplicate SongKey within a week
    - Global min repeat gap across weeks (cooldown)
    - Must include >=1 song per required Type each week (when possible)
    - Must exclude Types always
    - Hard rule: max 1 hymn per week (Type == "Hymn")
    - RotationScore biases selection probabilities; RotationScore=0 => never selected
    - Soft preferences bias selection via multipliers
    """
    rng = np.random.default_rng(cfg.seed)

    df = df.copy()
    df["Title"] = df["Title"].astype(str).str.strip()
    df["Artist"] = df["Artist"].astype(str).str.strip()
    df["Type"] = df["Type"].astype(str).str.strip()
    df["Style"] = df["Style"].astype(str).str.strip()
    df["RotationScore"] = pd.to_numeric(df["RotationScore"], errors="coerce").fillna(0).clip(lower=0)
    df["SongKey"] = build_song_key(df)

    cooldown_until = {k: 0 for k in df["SongKey"].tolist()}
    roster: list[list[str]] = []

    def is_excluded_type(type_val: str) -> bool:
        return any(is_value(type_val, x) for x in cfg.must_exclude_types)

    for w in range(cfg.total_weeks):
        week_keys: list[str] = []
        used_this_week = set()
        hymns_this_week = 0

        def not_excluded(local_df: pd.DataFrame) -> pd.Series:
            if not cfg.must_exclude_types:
                return pd.Series([True] * len(local_df), index=local_df.index)
            return ~local_df["Type"].map(lambda t: is_excluded_type(t))

        def not_rotation_zero(local_df: pd.DataFrame) -> pd.Series:
            return local_df["RotationScore"].astype(float) > 0

        def base_eligible(local_df: pd.DataFrame) -> pd.Series:
            return (
                not_excluded(local_df)
                & not_rotation_zero(local_df)
                & local_df["SongKey"].map(lambda k: cooldown_until.get(k, 0) <= w)
                & (~local_df["SongKey"].isin(used_this_week))
            )

        def eligible_with_hymn_cap(local_df: pd.DataFrame, mask: pd.Series) -> pd.Series:
            if hymns_this_week >= MAX_HYMNS_PER_WEEK:
                mask = mask & (~local_df["Type"].map(is_hymn))
            return mask

        def pick_and_commit(song_key: str):
            nonlocal hymns_this_week
            week_keys.append(song_key)
            used_this_week.add(song_key)

            type_val = df.loc[df["SongKey"] == song_key, "Type"].iloc[0]
            if is_hymn(type_val):
                hymns_this_week += 1

            cooldown_until[song_key] = w + cfg.min_repeat_gap + 1

        # 1) Must-include Types first (one per required Type)
        required_types = list(cfg.must_include_types)
        # If Hymn is must-include, do it first to respect max-1 rule early
        required_types.sort(key=lambda t: 0 if is_value(t, HYMN_TYPE_NAME) else 1)

        for type_name in required_types:
            if len(week_keys) >= cfg.songs_per_week:
                break

            if any(is_value(type_name, x) for x in cfg.must_exclude_types):
                continue

            if is_value(type_name, HYMN_TYPE_NAME) and hymns_this_week >= MAX_HYMNS_PER_WEEK:
                continue

            mask = base_eligible(df)
            mask = eligible_with_hymn_cap(df, mask)
            mask = mask & df["Type"].map(lambda t: is_value(t, type_name))
            eligible = df[mask].copy()

            if eligible.empty:
                continue

            picked = weighted_pick_one(rng, eligible, cfg)
            if picked is not None:
                pick_and_commit(picked)

        # 2) Fill remaining slots
        while len(week_keys) < cfg.songs_per_week:
            mask = base_eligible(df)
            mask = eligible_with_hymn_cap(df, mask)
            eligible = df[mask].copy()
            if eligible.empty:
                break

            picked = weighted_pick_one(rng, eligible, cfg)
            if picked is None:
                break

            pick_and_commit(picked)

        roster.append(week_keys)

    return roster


def roster_to_text(roster: list[list[str]], df: pd.DataFrame) -> str:
    """Output includes Title + Artist via SongKey."""
    songkey_to_display = dict(zip(df["SongKey"], df["SongKey"]))

    lines = []
    for i, keys in enumerate(roster, start=1):
        lines.append(f"Week {i}")
        for j, k in enumerate(keys, start=1):
            lines.append(f"Song {j}: {songkey_to_display.get(k, k)}")
        lines.append("")
    return "\n".join(lines).strip()


# -----------------------------
# UI
# -----------------------------
st.title("Weekly Song Roster Generator")

uploaded = st.file_uploader("Upload master song list (CSV)", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV with columns: Title, Artist, Type, Style, RotationScore")
    st.stop()

df = pd.read_csv(uploaded)
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Clean and add SongKey early
df = df.copy()
df["Title"] = df["Title"].astype(str).str.strip()
df["Artist"] = df["Artist"].astype(str).str.strip()
df["Type"] = df["Type"].astype(str).str.strip()
df["Style"] = df["Style"].astype(str).str.strip()
df["RotationScore"] = pd.to_numeric(df["RotationScore"], errors="coerce").fillna(0).clip(lower=0)
df["SongKey"] = build_song_key(df)

# Drop blank keys (defensive)
df = df[df["SongKey"].astype(str).str.strip() != ""].copy()

st.subheader("Preview")
st.dataframe(df[["Title", "Artist", "Type", "Style", "RotationScore"]].head(25), use_container_width=True)

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

st.caption(
    "Soft preferences bias selection but do not guarantee exact weekly proportions. "
    "Use 'Must include' / 'Must exclude' for hard rules."
)

# --- Type preferences (dropdown)
types = sorted(df["Type"].unique().tolist())
st.subheader("Type preferences (soft)")
type_pref_df = pd.DataFrame({"Type": types, "Preference": ["Sometimes"] * len(types)})
type_pref_df = st.data_editor(
    type_pref_df,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "Preference": st.column_config.SelectboxColumn(
            "Preference",
            options=PREFERENCE_LEVELS,
            help=PREFERENCE_HELP,
        )
    },
)
type_pref_weight = preference_table_to_weight_map(types, type_pref_df, "Type")

# --- Style preferences (dropdown)
styles = sorted(df["Style"].unique().tolist())
st.subheader("Style preferences (soft)")
style_pref_df = pd.DataFrame({"Style": styles, "Preference": ["Sometimes"] * len(styles)})
style_pref_df = st.data_editor(
    style_pref_df,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "Preference": st.column_config.SelectboxColumn(
            "Preference",
            options=PREFERENCE_LEVELS,
            help=PREFERENCE_HELP,
        )
    },
)
style_pref_weight = preference_table_to_weight_map(styles, style_pref_df, "Style")

# --- Must include / exclude per Type (hard)
st.subheader("Per-Type rules (hard)")
rules_df = pd.DataFrame(
    {
        "Type": types,
        "Must include (>=1 per week)": [False] * len(types),
        "Must exclude (0 per week)": [False] * len(types),
    }
)
rules_df = st.data_editor(rules_df, use_container_width=True, num_rows="fixed")

must_include_types = set(rules_df.loc[rules_df["Must include (>=1 per week)"] == True, "Type"].tolist())
must_exclude_types = set(rules_df.loc[rules_df["Must exclude (0 per week)"] == True, "Type"].tolist())

cfg = Config(
    total_weeks=int(total_weeks),
    songs_per_week=int(songs_per_week),
    min_repeat_gap=int(min_repeat_gap),
    seed=int(seed),
    rotation_alpha=float(rotation_alpha),
    type_pref_weight=type_pref_weight,
    style_pref_weight=style_pref_weight,
    must_include_types=must_include_types,
    must_exclude_types=must_exclude_types,
)

# Feasibility check
severity, messages, stats = feasibility_checks(df=df, cfg=cfg)

st.markdown("### Feasibility check")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Unique songs (Title — Artist)", stats["unique_songs"])
m2.metric("Total picks (weeks × songs/week)", stats["demand"])
m3.metric("Max uses per song (upper bound)", stats["per_song_cap"])
m4.metric("Supply upper bound", stats["supply"])
m5.metric("Hymn rows (Type='Hymn')", stats["hymn_rows"])
m6.metric("RotationScore=0 rows", stats["rotation_zero"])

st.caption(f"Hard-coded hymn cap: Type='{HYMN_TYPE_NAME}', max {MAX_HYMNS_PER_WEEK} hymn per week.")

if cfg.must_include_types:
    st.info(f"Must include each week (Types): {sorted(cfg.must_include_types)}")
if cfg.must_exclude_types:
    st.info(f"Must exclude (never pick) (Types): {sorted(cfg.must_exclude_types)}")

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
    out_text = roster_to_text(roster, df)
    st.text(out_text)

    st.download_button(
        "Download as .txt",
        data=out_text.encode("utf-8"),
        file_name="weekly_roster.txt",
        mime="text/plain",
    )

    # Diagnostics
    st.subheader("Diagnostics (optional)")
    flat = [k for week in roster for k in week]
    if flat:
        counts = pd.Series(flat).value_counts().reset_index()
        counts.columns = ["Song (Title — Artist)", "Times selected"]
        st.dataframe(counts, use_container_width=True)

        songkey_to_type = dict(zip(df["SongKey"], df["Type"]))
        type_counts = pd.Series([songkey_to_type.get(k, "Unknown") for k in flat]).value_counts()
        achieved_type = (type_counts / type_counts.sum()).reset_index()
        achieved_type.columns = ["Type", "Achieved proportion"]
        st.dataframe(achieved_type, use_container_width=True)

        songkey_to_style = dict(zip(df["SongKey"], df["Style"]))
        style_counts = pd.Series([songkey_to_style.get(k, "Unknown") for k in flat]).value_counts()
        achieved_style = (style_counts / style_counts.sum()).reset_index()
        achieved_style.columns = ["Style", "Achieved proportion"]
        st.dataframe(achieved_style, use_container_width=True)