"""
data_processor.py
Loads raw 15-sec telemetry CSVs, aggregates them to workflow segments,
and engineers features (including power_vs_baseline) for the hybrid engine.
"""

import glob
import io
import os

import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

PHASE_BASELINES_W = {
    "idle": 135.0,
    "processing": 200.0,
    "live_view_monitoring": 180.0,
    "tile_scan_acquisition": 170.0,
}

OPTIMAL_TILE_OVERLAP = {
    "low": 10.0,
    "medium": 12.0,
    "high": 18.0,
}

# Columns that indicate a file is already aggregated (has segment-level rows)
_AGGREGATED_MARKER_COL = "phase_segment_id"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mode(series: pd.Series):
    """Return the most frequent value, or NaN if empty."""
    m = series.mode()
    return m.iloc[0] if len(m) > 0 else None


def _phase_baseline(phase: str) -> float:
    return PHASE_BASELINES_W.get(str(phase).strip().lower(), 170.0)


# ── Core API ──────────────────────────────────────────────────────────────────

def load_training_data(training_dir: str) -> pd.DataFrame:
    """
    Load all S*_v4.csv files from training_dir, concatenate them.
    Returns a raw 15-sec telemetry DataFrame with `recommended_action` column.
    """
    pattern = os.path.join(training_dir, "S*_v4.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No training files found at {pattern}")
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df["_source_file"] = os.path.basename(f)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def aggregate_to_segments(df_raw: pd.DataFrame, has_label: bool = True) -> pd.DataFrame:
    """
    Group raw 15-sec rows by (session_id, workflow_block_id) and compute
    segment-level features.

    Parameters
    ----------
    df_raw   : raw telemetry DataFrame (columns as per data dictionary)
    has_label: True when training data (recommended_action column present)

    Returns
    -------
    DataFrame with one row per segment, ready for rule engine + MLP.
    """
    # Normalise column names to lowercase with underscores
    df_raw = df_raw.copy()
    df_raw.columns = [c.strip().lower().replace(" ", "_") for c in df_raw.columns]

    # ── Column alias normalisation ────────────────────────────────────────────
    # Raw telemetry CSVs use 'quality_constraint' but the aggregation code
    # expects 'quality_constraint_mode'.  Create an alias so both schemas work.
    if "quality_constraint_mode" not in df_raw.columns and "quality_constraint" in df_raw.columns:
        df_raw["quality_constraint_mode"] = df_raw["quality_constraint"]

    # Boolean feature columns exist with a '_flag' suffix in all raw CSVs.
    # Create bare-name aliases so the share() helper finds them correctly.
    for _bare, _flagged in (
        ("live_view_enabled",               "live_view_enabled_flag"),
        ("monitoring_required",             "monitoring_required_flag"),
        ("user_interacting",                "user_interacting_flag"),
        ("continuous_acquisition_display",  "continuous_acquisition_display_flag"),
        ("tile_scan_enabled",               "tile_scan_enabled_flag"),
        ("experiment_running",              "experiment_running_flag"),
    ):
        if _bare not in df_raw.columns and _flagged in df_raw.columns:
            df_raw[_bare] = df_raw[_flagged]
    # ─────────────────────────────────────────────────────────────────────────

    group_keys = ["session_id", "workflow_block_id"]
    # Some files use slightly different casing — be defensive
    available = set(df_raw.columns)
    group_keys = [k for k in group_keys if k in available]
    if not group_keys:
        # Fallback: treat entire file as single segment
        df_raw["session_id"] = "unknown"
        df_raw["workflow_block_id"] = 0
        group_keys = ["session_id", "workflow_block_id"]

    records = []
    for keys, grp in df_raw.groupby(group_keys, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        session_id = keys[0] if len(keys) > 0 else "unknown"
        block_id = keys[1] if len(keys) > 1 else 0

        n = len(grp)
        phase = _mode(grp.get("workflow_phase", pd.Series(["idle"] * n)))
        if phase is None:
            phase = "idle"
        phase = str(phase).strip().lower()

        # Power metrics column names (defined here, used throughout)
        power_col = "estimated_system_power_w"
        gpu_col = "perf_gpu_power_w"
        energy_col = "estimated_energy_wh_interval"
        overlap_col = "tile_overlap_pct"

        quality = _mode(grp.get("quality_constraint_mode", pd.Series(["medium"] * n)))
        if quality is None or str(quality).strip().lower() in ("", "nan", "unknown", "?"):
            # Infer quality from overlap pattern when not recorded in telemetry:
            #   overlap >= 15%  → high   (high-quality scan needs more coverage)
            #   overlap >= 11%  → medium (default scanning)
            #   overlap <  11%  → low    (fast screening, minimal overlap)
            _q_overlap = grp[overlap_col].mean() if overlap_col in grp else 12.0
            if _q_overlap >= 15.0:
                quality = "high"
            elif _q_overlap >= 11.0:
                quality = "medium"
            else:
                quality = "low"
        quality = str(quality).strip().lower()
        inactivity_col = "seconds_since_last_ui_interaction"
        inflight_col = "processing_items_in_flight"

        avg_power = grp[power_col].mean() if power_col in grp else 0.0
        gpu_power = grp[gpu_col].mean() if gpu_col in grp else 0.0
        total_energy = grp[energy_col].sum() if energy_col in grp else 0.0
        overlap_mean = grp[overlap_col].mean() if overlap_col in grp else 0.0
        inflight_mean = grp[inflight_col].mean() if inflight_col in grp else 0.0

        # Inactivity
        if inactivity_col in grp:
            median_inactivity = grp[inactivity_col].median()
        else:
            median_inactivity = 0.0

        # Share features (boolean columns → float mean = share)
        def share(col):
            if col in grp:
                return grp[col].astype(float).mean()
            return 0.0

        live_view_share = share("live_view_enabled")
        user_interacting_share = share("user_interacting")
        monitoring_share = share("monitoring_required")
        cont_acq_share = share("continuous_acquisition_display")
        tile_scan_share = share("tile_scan_enabled")
        exp_running_share = share("experiment_running")

        # Additional numeric metrics
        def col_mean(col):
            return float(grp[col].mean()) if col in grp else 0.0

        camera_light_mean = col_mean("camera_light_usage_index_pct")
        cpu_pct_mean      = col_mean("perf_cpu_pct")
        gpu_usage_mean    = col_mean("perf_gpu_usage_pct")
        disk_write_mean   = col_mean("perf_disk_write_mb_s")
        incoming_data_mean = col_mean("perf_incoming_data_mb_s")
        planned_area_mean = col_mean("planned_scan_area_mm2")
        preview_res_mean  = col_mean("preview_resolution_pct")
        tile_x_mean       = col_mean("tile_count_x")
        tile_y_mean       = col_mean("tile_count_y")
        total_tiles_mean  = tile_x_mean * tile_y_mean

        # Engineered feature: deviation from per-phase power baseline
        power_vs_baseline = avg_power - _phase_baseline(phase)

        rec = {
            "session_id": session_id,
            "workflow_block_id": block_id,
            "phase_name": phase,
            "quality_constraint_mode": quality,
            "experiment_type_mode": _mode(grp.get("experiment_type", pd.Series(["standard"]))),
            "duration_sec": n * 15,
            "live_view_enabled_share": round(live_view_share, 4),
            "user_interacting_share": round(user_interacting_share, 4),
            "monitoring_required_share": round(monitoring_share, 4),
            "median_seconds_since_last_ui_interaction": round(float(median_inactivity), 1),
            "tile_overlap_pct_mean": round(float(overlap_mean), 2),
            "perf_gpu_power_w_mean": round(float(gpu_power), 2),
            "processing_items_in_flight_mean": round(float(inflight_mean), 2),
            "estimated_system_power_w_mean": round(float(avg_power), 2),
            "estimated_energy_wh_interval_sum": round(float(total_energy), 4),
            "power_vs_baseline": round(float(power_vs_baseline), 2),
            # Extended feature set
            "camera_light_usage_index_pct_mean": round(camera_light_mean, 2),
            "perf_cpu_pct_mean": round(cpu_pct_mean, 2),
            "perf_gpu_usage_pct_mean": round(gpu_usage_mean, 2),
            "perf_disk_write_mb_s_mean": round(disk_write_mean, 3),
            "perf_incoming_data_mb_s_mean": round(incoming_data_mean, 3),
            "planned_scan_area_mm2_mean": round(planned_area_mean, 2),
            "preview_resolution_pct_mean": round(preview_res_mean, 2),
            "continuous_acquisition_display_share": round(cont_acq_share, 4),
            "tile_scan_enabled_share": round(tile_scan_share, 4),
            "experiment_running_share": round(exp_running_share, 4),
            "total_tiles_mean": round(total_tiles_mean, 1),
        }

        if has_label and "recommended_action" in grp.columns:
            rec["label"] = _mode(grp["recommended_action"])

        records.append(rec)

    return pd.DataFrame(records)


def _normalise_aggregated(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns from pre-aggregated test_segments.csv schema to match
    the schema produced by aggregate_to_segments().
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rename_map = {
        # phase_segment_id → used as segment_id later
        "phase_name": "phase_name",
        "workflow_phase": "phase_name",
        "quality_constraint_mode": "quality_constraint_mode",
        "total_duration_sec": "duration_sec",
        "live_view_enabled_share": "live_view_enabled_share",
        "user_interacting_share": "user_interacting_share",
        "monitoring_required_share": "monitoring_required_share",
        "median_seconds_since_last_ui_interaction": "median_seconds_since_last_ui_interaction",
        "tile_overlap_pct_mean": "tile_overlap_pct_mean",
        "perf_gpu_power_w_mean": "perf_gpu_power_w_mean",
        "processing_items_in_flight_mean": "processing_items_in_flight_mean",
        "estimated_system_power_w_mean": "estimated_system_power_w_mean",
        "estimated_energy_wh_interval_sum": "estimated_energy_wh_interval_sum",
    }
    # Apply any renames that actually exist in the df
    actual_rename = {k: v for k, v in rename_map.items() if k in df.columns and k != v}
    df = df.rename(columns=actual_rename)

    # Normalise phase_name to lowercase
    if "phase_name" in df.columns:
        df["phase_name"] = df["phase_name"].astype(str).str.strip().str.lower()

    # Compute power_vs_baseline if not present
    if "power_vs_baseline" not in df.columns and "estimated_system_power_w_mean" in df.columns:
        df["power_vs_baseline"] = df.apply(
            lambda r: r["estimated_system_power_w_mean"] - _phase_baseline(r.get("phase_name", "idle")),
            axis=1,
        )

    # Fill missing numeric columns with sensible defaults
    numeric_defaults = {
        "live_view_enabled_share": 0.0,
        "user_interacting_share": 0.0,
        "monitoring_required_share": 0.0,
        "median_seconds_since_last_ui_interaction": 0.0,
        "tile_overlap_pct_mean": 0.0,
        "perf_gpu_power_w_mean": 0.0,
        "processing_items_in_flight_mean": 0.0,
        "estimated_system_power_w_mean": 0.0,
        "estimated_energy_wh_interval_sum": 0.0,
        "duration_sec": 0.0,
        "power_vs_baseline": 0.0,
    }
    for col, default in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    if "quality_constraint_mode" not in df.columns:
        df["quality_constraint_mode"] = "medium"

    return df


def detect_and_load(file_bytes: bytes) -> tuple[pd.DataFrame, bool]:
    """
    Detect whether uploaded CSV is raw telemetry or pre-aggregated.
    Returns (dataframe, is_raw).
    Raw → aggregate_to_segments()
    Pre-aggregated → normalise column names.
    """
    df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    is_aggregated = _AGGREGATED_MARKER_COL in df.columns
    if is_aggregated:
        return _normalise_aggregated(df), False
    else:
        return aggregate_to_segments(df, has_label=False), True
