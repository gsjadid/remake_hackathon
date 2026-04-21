"""
rule_engine.py
70% component of the hybrid decision engine.
Rules are empirically validated from raw S1/S2/S4/S9 telemetry.

Key finding: seconds_since_last_ui_interaction >= 120 is the PRIMARY
driver for pause_live_view — validated directly from raw row sequences.
"""

from dataclasses import dataclass
from typing import Optional

from app.data_processor import OPTIMAL_TILE_OVERLAP

# Inactivity threshold (seconds) that triggers live-view pause recommendations
INACTIVITY_THRESHOLD_S = 120.0


@dataclass
class RuleResult:
    action: str
    confidence: float
    issue_description: str
    rule_id: Optional[str] = None


def evaluate(segment: dict) -> RuleResult:
    """
    Evaluate the 5 empirically validated rules in priority order.

    Parameters
    ----------
    segment : dict with keys matching aggregate_to_segments() output columns.

    Returns
    -------
    RuleResult with action, confidence [0..1], and human-readable issue text.
    """
    phase = str(segment.get("phase_name", "")).strip().lower()
    quality = str(segment.get("quality_constraint_mode", "medium")).strip().lower()

    live_view_share: float = float(segment.get("live_view_enabled_share", 0.0))
    user_interacting: float = float(segment.get("user_interacting_share", 0.0))
    monitoring_share: float = float(segment.get("monitoring_required_share", 0.0))
    inactivity: float = float(segment.get("median_seconds_since_last_ui_interaction", 0.0))
    overlap: float = float(segment.get("tile_overlap_pct_mean", 0.0))
    duration: float = float(segment.get("duration_sec", 0.0))

    optimal_overlap = OPTIMAL_TILE_OVERLAP.get(quality, 12.0)
    overlap_excess = overlap - optimal_overlap

    tile_scan_share: float = float(segment.get("tile_scan_enabled_share", 0.0))

    # ── R1: Inactivity-driven live-view pause (primary signal) ───────────────
    # Validated: S1 rows >=120s inactivity always → pause_live_view
    # Guard: skip during active tile scans — the correct action there is R4
    # (overlap optimisation).  Pausing live-view mid-acquisition is wrong.
    if (
        live_view_share > 0.3
        and inactivity >= INACTIVITY_THRESHOLD_S
        and monitoring_share < 0.5
        and tile_scan_share < 0.5          # not in an active tile scan
    ):
        idle_min = round(inactivity / 60, 1)
        lv_pct = round(live_view_share * 100)
        return RuleResult(
            action="pause_live_view",
            confidence=0.95,
            issue_description=(
                f"Live view active {lv_pct}% of segment with no user interaction "
                f"for {idle_min} min — unnecessary power consumption"
            ),
            rule_id="R1",
        )

    # ── R2: Idle phase with live view on ─────────────────────────────────────
    # Validated: S9 idle + live_view=True → pause_live_view
    if phase == "idle" and live_view_share > 0.3:
        lv_pct = round(live_view_share * 100)
        return RuleResult(
            action="pause_live_view",
            confidence=0.90,
            issue_description=(
                f"System idle but live view is active {lv_pct}% of the time "
                f"({round(duration/60, 1)} min) — wasting illumination power"
            ),
            rule_id="R2",
        )

    # ── R3: Live-view monitoring with no actual monitoring happening ──────────
    # Real-world condition: only flag when the user has been genuinely absent
    # for >=90 s AND true monitoring activity is near-zero.
    # A user who just stepped away briefly may return; a user absent for 2+ min
    # with zero monitoring traffic is a clear waste pattern.
    # Confidence scales with absence duration so the ML can more easily override
    # brief-absence cases (conf=0.80) vs long-absence cases (conf=0.92).
    if (
        phase == "live_view_monitoring"
        and monitoring_share < 0.15       # tightened: truly no monitoring traffic
        and user_interacting < 0.10       # tightened: truly no interaction
        and inactivity >= 90.0            # must be genuinely absent (not just a pause)
    ):
        r3_conf = 0.92 if inactivity >= INACTIVITY_THRESHOLD_S else 0.80
        return RuleResult(
            action="pause_live_view",
            confidence=r3_conf,
            issue_description=(
                f"Live-view monitoring phase with only {round(monitoring_share*100)}% "
                f"actual monitoring and user absent for {round(inactivity/60, 1)} min "
                f"— microscope illumination running without purpose"
            ),
            rule_id="R3",
        )

    # ── R4: Tile scan with excess overlap ───────────────────────────────────────
    # Threshold: 2.0% excess — at 1.2W/% this is already measurable over long scans.
    # Excess overlap wastes energy regardless of user presence.
    # Higher confidence when user has walked away (inactivity ≥ threshold) because
    # the scan is running unattended with no opportunity for manual correction.
    if (
        phase == "tile_scan_acquisition"
        and overlap_excess > 2.0
    ):
        # Physics-based certainty: excess overlap always wastes energy.
        # 0.93 (mandatory) when inactivity >= 90s (user genuinely absent).
        # 0.82 (soft) when user is still active — recommend but don't force.
        R4_INACTIVITY_THRESHOLD = 90.0
        conf = 0.93 if inactivity >= R4_INACTIVITY_THRESHOLD else 0.82
        context = (
            f"unattended ({round(inactivity/60, 1)} min idle)"
            if inactivity >= INACTIVITY_THRESHOLD_S
            else "user active"
        )
        return RuleResult(
            action="optimize_tile_scan_settings",
            confidence=conf,
            issue_description=(
                f"Tile overlap {round(overlap, 1)}% is {round(overlap_excess, 1)}% "
                f"above the {quality}-quality optimal {round(optimal_overlap, 0)}% "
                f"({context}) — redundant scan area wastes energy"
            ),
            rule_id="R4",
        )

    # ── R5: Processing phase with unused live view ────────────────────────────
    if (
        phase == "processing"
        and live_view_share > 0.5
        and inactivity >= INACTIVITY_THRESHOLD_S
    ):
        lv_pct = round(live_view_share * 100)
        return RuleResult(
            action="pause_live_view",
            confidence=0.75,
            issue_description=(
                f"Post-acquisition processing with live view on {lv_pct}% of the time "
                f"and no user interaction — camera illumination is unnecessary"
            ),
            rule_id="R5",
        )

    # ── Default: segment is efficient ────────────────────────────────────────
    return RuleResult(
        action="no_action",
        confidence=0.50,
        issue_description="",
        rule_id=None,
    )
