"""
test_random_dataset.py
Generates a synthetic random microscopy workflow CSV and runs it through
the full ZEISS hybrid engine pipeline to validate end-to-end correctness.
"""
import sys, random, io
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from app.data_processor import aggregate_to_segments, load_training_data
from app import rule_engine, ml_model
from app.energy_calculator import calculate_savings

random.seed(0)
np.random.seed(0)

# ── Train model ───────────────────────────────────────────────────────────────
print("Training model on real S1-S13 data...")
train_raw = load_training_data('data/training')
train_segs = aggregate_to_segments(train_raw, has_label=True)
stats = ml_model.model.train(train_segs)
print(f"  MLP acc={stats['accuracy']:.4f}  f1={stats['f1_macro']:.4f}\n")

ALL_ACTIONS = ['no_action', 'optimize_tile_scan_settings', 'pause_live_view']

def hybrid_score(seg):
    rr = rule_engine.evaluate(seg)
    ml_res = ml_model.model.predict(seg)
    ml_p = ml_res['probabilities']
    if rr.action == 'no_action':
        scores = {a: ml_p.get(a, 0.0) for a in ALL_ACTIONS}
    else:
        rw = 0.70 if rr.confidence >= 0.93 else (0.45 if rr.confidence >= 0.90 else 0.28)
        scores = {a: ml_p.get(a, 0.0) * 0.60 + (rr.confidence * rw if rr.action == a else 0.0)
                  for a in ALL_ACTIONS}
    final = max(scores, key=lambda a: scores[a])
    return final, round(scores[final], 3), rr, ml_p, calculate_savings(seg, final)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario builder — each scenario has known EXPECTED output
# ─────────────────────────────────────────────────────────────────────────────

def make_rows(session, block, phase, n_rows,
              live_view=0, user_interacting=0, monitoring=0,
              inactivity_start=0, overlap=0.0, quality='medium',
              system_power=None, cont_acq=0, tile_scan_en=0,
              camera_light=0.0, gpu_power=50.0, cpu=20.0,
              disk_write=2.0, incoming=2.0, gpu_usage=40.0,
              exp_running=1, inflight=1.0, planned_area=200.0):
    rows = []
    baselines = {'idle': 135, 'processing': 200,
                 'live_view_monitoring': 180, 'tile_scan_acquisition': 170}
    base_pwr = system_power or baselines.get(phase, 170)
    for i in range(n_rows):
        inact = inactivity_start + i * 15
        pwr = base_pwr + np.random.normal(0, 2)
        energy = pwr * 15 / 3600
        rows.append({
            'session_id': session,
            'workflow_block_id': block,
            'workflow_phase': phase,
            'quality_constraint': quality,
            'live_view_enabled_flag': live_view,
            'user_interacting_flag': user_interacting,
            'monitoring_required_flag': monitoring,
            'continuous_acquisition_display_flag': cont_acq,
            'tile_scan_enabled_flag': tile_scan_en,
            'experiment_running_flag': exp_running,
            'seconds_since_last_ui_interaction': inact,
            'tile_overlap_pct': overlap + np.random.uniform(-0.3, 0.3),
            'perf_gpu_power_w': gpu_power + np.random.normal(0, 1),
            'perf_gpu_usage_pct': gpu_usage,
            'perf_cpu_pct': cpu,
            'perf_disk_write_mb_s': disk_write,
            'perf_incoming_data_mb_s': incoming,
            'processing_items_in_flight': inflight,
            'estimated_system_power_w': pwr,
            'estimated_energy_wh_interval': energy,
            'camera_light_usage_index_pct': camera_light + np.random.normal(0, 1),
            'preview_resolution_pct': 80.0 if live_view else 0.0,
            'planned_scan_area_mm2': planned_area,
            'tile_count_x': max(1, int(np.sqrt(planned_area / 30))),
            'tile_count_y': max(1, int(np.sqrt(planned_area / 30))),
            'recommended_action': None,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Define test scenarios with EXPECTED predictions
# ─────────────────────────────────────────────────────────────────────────────
SCENARIOS = [
    # name, expected_action, description, rows
    (
        "IDLE_CLEAN",
        "no_action",
        "Idle phase, no live view, low power — should be efficient",
        make_rows('RND001', 1, 'idle', 10,
                  live_view=0, user_interacting=0, monitoring=0,
                  inactivity_start=300, overlap=0, system_power=136,
                  camera_light=0, gpu_power=15, cpu=5, disk_write=0.1,
                  incoming=0.0, inflight=0, planned_area=0)
    ),
    (
        "IDLE_LIVE_VIEW_ON_UNATTENDED",
        "pause_live_view",
        "Idle phase with live view on, user absent 4+ minutes — R2 should fire",
        make_rows('RND001', 2, 'idle', 20,
                  live_view=1, user_interacting=0, monitoring=0,
                  inactivity_start=240, overlap=0, system_power=180,
                  camera_light=65, gpu_power=50, cpu=12, disk_write=0.5,
                  incoming=0.5, inflight=0, planned_area=0)
    ),
    (
        "LIVE_VIEW_LONG_INACTIVITY",
        "pause_live_view",
        "Live view monitoring, user absent 8+ min, no monitoring traffic — R3/R1",
        make_rows('RND001', 3, 'live_view_monitoring', 24,
                  live_view=1, user_interacting=0, monitoring=0,
                  inactivity_start=480, overlap=0, system_power=182,
                  camera_light=70, gpu_power=52, cpu=14, disk_write=0.3,
                  incoming=0.4, inflight=0, planned_area=0)
    ),
    (
        "LIVE_VIEW_ATTENDED_MONITORING",
        "no_action",
        "Live view on, user actively interacting — no recommendation expected",
        make_rows('RND001', 4, 'live_view_monitoring', 16,
                  live_view=1, user_interacting=1, monitoring=1,
                  inactivity_start=0, overlap=0, system_power=183,
                  camera_light=68, gpu_power=53, cpu=18, disk_write=0.4,
                  incoming=0.6, inflight=0, planned_area=0)
    ),
    (
        "TILE_SCAN_EXCESS_OVERLAP_LOW_QUALITY",
        "optimize_tile_scan_settings",
        "Tile scan quality=low, overlap=16% (optimal=10%) — R4 should fire",
        make_rows('RND001', 5, 'tile_scan_acquisition', 32,
                  live_view=1, user_interacting=0, monitoring=0,
                  inactivity_start=90, overlap=16.0, quality='low',
                  system_power=210, camera_light=62, gpu_power=55,
                  cpu=22, disk_write=18.0, incoming=16.0,
                  inflight=3, planned_area=350, tile_scan_en=1, cont_acq=1)
    ),
    (
        "TILE_SCAN_OPTIMAL_OVERLAP_HIGH_QUALITY",
        "no_action",
        "Tile scan quality=high, overlap=18% (optimal=18%) — exactly optimal, no action",
        # User started scan and stepped away; inactivity grows but tile_scan is active
        # so R1 must NOT fire — R4 only fires if overlap_excess > 2%, which it isn't.
        make_rows('RND001', 6, 'tile_scan_acquisition', 24,
                  live_view=1, user_interacting=0, monitoring=0,
                  inactivity_start=120, overlap=18.0, quality='high',
                  system_power=175, camera_light=66, gpu_power=50,
                  cpu=20, disk_write=19.0, incoming=18.0,
                  inflight=2, planned_area=400, tile_scan_en=1, cont_acq=1)
    ),
    (
        "TILE_SCAN_MEDIUM_QUALITY_SMALL_EXCESS",
        "optimize_tile_scan_settings",
        "Tile scan quality=medium, overlap=15% (optimal=12%, excess=3%) — R4",
        make_rows('RND001', 7, 'tile_scan_acquisition', 30,
                  live_view=1, user_interacting=0, monitoring=0,
                  inactivity_start=120, overlap=15.0, quality='medium',
                  system_power=208, camera_light=60, gpu_power=54,
                  cpu=21, disk_write=17.0, incoming=16.5,
                  inflight=2, planned_area=300, tile_scan_en=1, cont_acq=1)
    ),
    (
        "PROCESSING_EFFICIENT",
        "no_action",
        "Processing phase, no live view, active compute queue — should be efficient",
        make_rows('RND001', 8, 'processing', 16,
                  live_view=0, user_interacting=0, monitoring=0,
                  inactivity_start=180, overlap=0, system_power=202,
                  camera_light=0, gpu_power=80, cpu=75, disk_write=25.0,
                  incoming=0.5, inflight=8, planned_area=0)
    ),
    (
        "PROCESSING_LIVE_VIEW_IDLE_WASTE",
        "pause_live_view",
        "Processing phase with live view still on, user absent 3+ min — R5",
        make_rows('RND001', 9, 'processing', 20,
                  live_view=1, user_interacting=0, monitoring=0,
                  inactivity_start=180, overlap=0, system_power=215,
                  camera_light=60, gpu_power=55, cpu=30, disk_write=5.0,
                  incoming=0.2, inflight=2, planned_area=0)
    ),
    (
        "IDLE_SHORT_INACTIVITY_NO_LIVE_VIEW",
        "no_action",
        "Idle, 30s inactivity, no live view — should be efficient (brief pause)",
        make_rows('RND001', 10, 'idle', 8,
                  live_view=0, user_interacting=0, monitoring=0,
                  inactivity_start=30, overlap=0, system_power=137,
                  camera_light=0, gpu_power=18, cpu=6, disk_write=0.1,
                  incoming=0.0, inflight=0, planned_area=0)
    ),
    (
        "EXTREME_OVERLAP_WASTE",
        "optimize_tile_scan_settings",
        "Tile scan quality=low, extreme overlap=25% (optimal=10%, excess=15%)",
        make_rows('RND001', 11, 'tile_scan_acquisition', 40,
                  live_view=1, user_interacting=0, monitoring=0,
                  inactivity_start=200, overlap=25.0, quality='low',
                  system_power=230, camera_light=65, gpu_power=60,
                  cpu=25, disk_write=20.0, incoming=19.0,
                  inflight=4, planned_area=500, tile_scan_en=1, cont_acq=1)
    ),
    (
        "LIVE_VIEW_ACTIVE_BARELY_INACTIVITY",
        "no_action",
        "Live view, only 45s inactivity with active monitoring — below R1/R3 threshold",
        make_rows('RND001', 12, 'live_view_monitoring', 12,
                  live_view=1, user_interacting=0, monitoring=1,
                  inactivity_start=45, overlap=0, system_power=181,
                  camera_light=67, gpu_power=51, cpu=14, disk_write=0.3,
                  incoming=0.5, inflight=0, planned_area=0)
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Run each scenario
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("RANDOM SYNTHETIC DATASET TEST")
print("=" * 80)

passed = 0
failed = 0
results_table = []

for scenario_name, expected, description, rows in SCENARIOS:
    df_raw = pd.DataFrame(rows)
    segs = aggregate_to_segments(df_raw, has_label=False)

    if segs.empty:
        print(f"  [SKIP] {scenario_name}: no segments produced")
        continue

    seg = segs.iloc[0].to_dict()
    final, conf, rr, ml_p, savings_wh = hybrid_score(seg)

    status = "PASS" if final == expected else "FAIL"
    if final == expected:
        passed += 1
    else:
        failed += 1

    results_table.append({
        'scenario': scenario_name,
        'expected': expected,
        'predicted': final,
        'rule': rr.rule_id or '-',
        'rule_action': rr.action[:12],
        'rule_conf': rr.confidence,
        'ml_na': round(ml_p.get('no_action', 0), 3),
        'ml_plv': round(ml_p.get('pause_live_view', 0), 3),
        'ml_tile': round(ml_p.get('optimize_tile_scan_settings', 0), 3),
        'conf': conf,
        'savings_wh': round(savings_wh, 3),
        'status': status,
    })

print(f"\n{'Scenario':<42} {'Expected':<30} {'Predicted':<30} {'Rule':<5} {'Conf':>5} {'Save':>6} Status")
print("-" * 130)
for r in results_table:
    mark = "✓" if r['status'] == 'PASS' else "✗"
    print(f"  {r['scenario']:<40} {r['expected']:<30} {r['predicted']:<30} {r['rule']:<5} {r['conf']:>5.3f} {r['savings_wh']:>6.3f}  {mark} {r['status']}")

print()
print(f"RESULT: {passed}/{passed+failed} PASSED  ({passed/(passed+failed)*100:.1f}%)")
print()

# ── Detailed breakdown of FAILED scenarios ────────────────────────────────────
failures = [r for r in results_table if r['status'] == 'FAIL']
if failures:
    print("=" * 80)
    print("FAILED SCENARIOS — DETAIL")
    print("=" * 80)
    for r in failures:
        # Find the scenario
        for sname, exp, desc, rows in SCENARIOS:
            if sname == r['scenario']:
                df_raw = pd.DataFrame(rows)
                segs = aggregate_to_segments(df_raw, has_label=False)
                seg = segs.iloc[0].to_dict()
                print(f"\n  {r['scenario']}")
                print(f"  Description: {desc}")
                print(f"  Expected: {exp}  |  Got: {r['predicted']}")
                print(f"  Rule: {r['rule']}/{r['rule_action']} (conf={r['rule_conf']:.2f})")
                print(f"  ML probs: no_action={r['ml_na']}  pause_lv={r['ml_plv']}  tile={r['ml_tile']}")
                print(f"  Key features:")
                print(f"    live_view_share={seg.get('live_view_enabled_share', 0):.3f}")
                print(f"    camera_light={seg.get('camera_light_usage_index_pct_mean', 0):.1f}")
                print(f"    user_interacting={seg.get('user_interacting_share', 0):.3f}")
                print(f"    monitoring_share={seg.get('monitoring_required_share', 0):.3f}")
                print(f"    inactivity={seg.get('median_seconds_since_last_ui_interaction', 0):.0f}s")
                print(f"    overlap={seg.get('tile_overlap_pct_mean', 0):.2f}%")
                print(f"    quality={seg.get('quality_constraint_mode')}")
                print(f"    tile_scan_share={seg.get('tile_scan_enabled_share', 0):.3f}")
                break
else:
    print("All scenarios passed!")

# ── Savings sanity check ─────────────────────────────────────────────────────
print()
print("=" * 80)
print("SAVINGS SANITY CHECK")
print("=" * 80)
print(f"{'Scenario':<42} {'Action':<30} {'Duration':>8} {'Savings Wh':>10}  Formula check")
print("-" * 110)
for r in results_table:
    for sname, exp, desc, rows in SCENARIOS:
        if sname == r['scenario']:
            df_raw = pd.DataFrame(rows)
            segs = aggregate_to_segments(df_raw, has_label=False)
            seg = segs.iloc[0].to_dict()
            dur = seg.get('duration_sec', 0)
            overlap = seg.get('tile_overlap_pct_mean', 0)
            quality = seg.get('quality_constraint_mode', 'medium')
            lv_share = seg.get('live_view_enabled_share', 0)

            from app.data_processor import OPTIMAL_TILE_OVERLAP
            if r['predicted'] == 'pause_live_view':
                effective_lv = max(lv_share, 1.0 if seg.get('phase_name') == 'live_view_monitoring' else 0.5 if lv_share < 0.05 else lv_share)
                expected_save = round(18.0 * effective_lv * dur / 3600, 3)
                formula = f"18W × {effective_lv:.2f} × {dur}s / 3600"
            elif r['predicted'] == 'optimize_tile_scan_settings':
                opt = OPTIMAL_TILE_OVERLAP.get(quality, 12.0)
                excess = max(0, overlap - opt)
                expected_save = round(1.2 * excess * dur / 3600, 3)
                formula = f"1.2W/% × {excess:.2f}% × {dur}s / 3600"
            else:
                expected_save = 0.0
                formula = "n/a (no_action)"

            match = "✓" if abs(r['savings_wh'] - expected_save) < 0.01 else f"✗ expected {expected_save}"
            print(f"  {r['scenario']:<40} {r['predicted']:<30} {dur:>8}s {r['savings_wh']:>10.3f}  {match}")
            break
