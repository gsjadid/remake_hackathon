"""
hybrid_engine.py
ML-primary hybrid decision engine.

Architecture:
  ML is the primary decision maker (MLP_WEIGHT=0.60).
  Rules act as evidence boosters — they raise the score for their matched action
  but cannot override a confident ML disagreement for soft rules.

  Tiered rule weights:
    Hard rules (conf ≥ 0.90):  RULE_WEIGHT_HARD = 0.45 — empirically certain patterns
    Soft rules (conf < 0.90):   RULE_WEIGHT_SOFT = 0.28 — suggestive, ML can override

  No-rule path:
    When rules find nothing wrong (returns no_action), we run PURE ML.
    Rules must not suppress the ML's ability to surface subtle inefficiencies.

Math example:
  R3 soft (conf=0.80): rule contributes 0.80×0.28=0.224
    ML needs ~0.37+ confidence in a different action to override — achievable.
  R1 hard (conf=0.95): rule contributes 0.95×0.45=0.428
    ML (0.60 weight) needs >0.71 confidence to override — very hard, as intended.
"""

from typing import Any

from app import rule_engine, ml_model
from app.energy_calculator import calculate_savings

RULE_WEIGHT_MANDATORY = 0.70  # conf ≥ 0.93 — physics-based certainty, takes precedence over ML
RULE_WEIGHT_HARD = 0.45   # conf ≥ 0.90 — empirically certain, hard to override
RULE_WEIGHT_SOFT = 0.28   # conf < 0.90 — suggestive, ML can override with moderate confidence
MLP_WEIGHT       = 0.60   # ML is the primary classifier

_ALL_ACTIONS = ["no_action", "optimize_tile_scan_settings", "pause_live_view"]


def analyze_segment(segment: dict) -> dict[str, Any]:
    """
    Run the hybrid engine on a single segment dict.

    Returns
    -------
    dict with all fields needed by the API response:
      segment_id, phase, duration_sec, avg_power_w, total_energy_wh,
      overlap_pct, live_view_share, user_interaction_share,
      is_energy_spike, spike_magnitude_w, issue,
      recommended_action, action_reason, estimated_savings_wh,
      confidence, rule_id, mlp_probabilities, power_vs_baseline
    """
    # ── Rule engine ───────────────────────────────────────────────────────────
    rule_result = rule_engine.evaluate(segment)

    # ── MLP inference ─────────────────────────────────────────────────────────
    ml_result = ml_model.model.predict(segment)

    # ── Score fusion (ML-primary, tiered rule evidence) ───────────────────────────
    ml_proba = ml_result["probabilities"]
    scores: dict[str, float] = {}

    if rule_result.action == "no_action":
        # Rules found nothing wrong — let ML decide entirely.
        # Do NOT give the rule's no_action a score boost; that would suppress
        # the ML from surfacing subtle inefficiencies the rules don't cover.
        scores = {a: ml_proba.get(a, 0.0) for a in _ALL_ACTIONS}
    else:
        # A rule fired — blend ML + rule evidence.
        # Mandatory rules (conf ≥ 0.93) encode physics-based certainty; they
        # override ML even when ML disagrees (e.g. excess tile overlap is always waste).
        # Hard rules (conf ≥ 0.90) get strong weight; soft rules yield to ML.
        if rule_result.confidence >= 0.93:
            rule_weight = RULE_WEIGHT_MANDATORY
        elif rule_result.confidence >= 0.90:
            rule_weight = RULE_WEIGHT_HARD
        else:
            rule_weight = RULE_WEIGHT_SOFT
        for action in _ALL_ACTIONS:
            ml_score   = ml_proba.get(action, 0.0) * MLP_WEIGHT
            rule_score = rule_result.confidence * rule_weight if rule_result.action == action else 0.0
            scores[action] = ml_score + rule_score

    final_action = max(scores, key=lambda a: scores[a])
    combined_confidence = scores[final_action]

    # ── Energy savings ────────────────────────────────────────────────────────
    savings_wh = calculate_savings(segment, final_action)

    # ── Compose reason string ─────────────────────────────────────────────────
    if final_action == rule_result.action and rule_result.issue_description:
        action_reason = rule_result.issue_description
    elif ml_result["explanation"]:
        action_reason = ml_result["explanation"]
    else:
        action_reason = f"Pattern matches {final_action.replace('_', ' ')} class"

    # ── Build output record ───────────────────────────────────────────────────
    phase = str(segment.get("phase_name", "unknown"))
    segment_id = str(
        segment.get("phase_segment_id",
        segment.get("workflow_block_id",
        segment.get("segment_id", "SEG")))
    )

    return {
        "segment_id": segment_id,
        "phase": phase,
        "duration_sec": int(segment.get("duration_sec", 0)),
        "avg_power_w": round(float(segment.get("estimated_system_power_w_mean", 0.0)), 2),
        "total_energy_wh": round(float(segment.get("estimated_energy_wh_interval_sum", 0.0)), 4),
        "overlap_pct": round(float(segment.get("tile_overlap_pct_mean", 0.0)), 2),
        "live_view_share": round(float(segment.get("live_view_enabled_share", 0.0)), 4),
        "user_interaction_share": round(float(segment.get("user_interacting_share", 0.0)), 4),
        "monitoring_required_share": round(float(segment.get("monitoring_required_share", 0.0)), 4),
        "quality": str(segment.get("quality_constraint_mode", "medium")),
        "power_vs_baseline": round(float(segment.get("power_vs_baseline", 0.0)), 2),
        "is_energy_spike": ml_result["is_energy_spike"],
        "spike_magnitude_w": ml_result["spike_magnitude_w"],
        "issue": rule_result.issue_description if final_action != "no_action" else "",
        "recommended_action": final_action,
        "action_reason": action_reason,
        "estimated_savings_wh": round(savings_wh, 4),
        "confidence": round(combined_confidence, 4),
        "rule_id": rule_result.rule_id,
        "mlp_probabilities": {k: round(v, 4) for k, v in ml_result["probabilities"].items()},
    }
