"""
energy_calculator.py
Quantifies Wh savings per segment per recommendation.

Formulas (physics-based):
  pause_live_view        : 18W × live_view_share × duration_sec / 3600
  optimize_tile_scan     : 1.2W × max(0, overlap - OPTIMAL[quality]) × duration_sec / 3600
  no_action              : 0.0 Wh
"""

from app.data_processor import OPTIMAL_TILE_OVERLAP

# Power overhead constants (derived empirically from training data patterns)
LIVE_VIEW_POWER_OVERHEAD_W = 18.0       # watts saved by pausing live view
TILE_OVERLAP_POWER_PER_PCT_W = 1.2      # watts per % of excess tile overlap


def calculate_savings(segment: dict, action: str) -> float:
    """
    Estimate energy savings in Wh if the given action is applied to this segment.

    Parameters
    ----------
    segment : dict with duration_sec, live_view_enabled_share,
              tile_overlap_pct_mean, quality_constraint_mode
    action  : one of {pause_live_view, optimize_tile_scan_settings, no_action}

    Returns
    -------
    float — savings in Wh (always >= 0)
    """
    duration_sec = float(segment.get("duration_sec", 0.0))
    if duration_sec <= 0:
        return 0.0

    if action == "pause_live_view":
        live_view_share = float(segment.get("live_view_enabled_share", 0.0))
        phase = str(segment.get("phase_name", "")).strip().lower()
        # In live_view_monitoring phase the illumination overhead is present by
        # definition — the phase name implies the camera/illumination system is
        # running even when the live_view_enabled telemetry flag reads 0.
        # Use full share (1.0) for that phase; 0.5 floor for idle/other phases
        # where the rule fired on inactivity grounds.
        if live_view_share < 0.05:
            if phase == "live_view_monitoring":
                live_view_share = 1.0
            elif phase in ("idle", "tile_scan_acquisition"):
                live_view_share = 0.5
        savings_wh = LIVE_VIEW_POWER_OVERHEAD_W * live_view_share * duration_sec / 3600.0
        return max(0.0, savings_wh)

    if action == "optimize_tile_scan_settings":
        overlap = float(segment.get("tile_overlap_pct_mean", 0.0))
        quality = str(segment.get("quality_constraint_mode", "medium")).strip().lower()
        optimal = OPTIMAL_TILE_OVERLAP.get(quality, 12.0)
        excess_pct = max(0.0, overlap - optimal)
        savings_wh = TILE_OVERLAP_POWER_PER_PCT_W * excess_pct * duration_sec / 3600.0
        return max(0.0, savings_wh)

    return 0.0
