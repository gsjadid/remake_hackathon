import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from app.data_processor import aggregate_to_segments, load_training_data, OPTIMAL_TILE_OVERLAP
from app import rule_engine, ml_model
from app.energy_calculator import calculate_savings

# -- Train MLP first ----------------------------------------------------
train_raw = load_training_data('data/training')
train_segs = aggregate_to_segments(train_raw, has_label=True)
stats = ml_model.model.train(train_segs)
print(f"MLP trained: acc={stats['accuracy']:.4f}  f1={stats['f1_macro']:.4f}  n_train={stats['n_train']}")
print()

ALL_ACTIONS = ['no_action','optimize_tile_scan_settings','pause_live_view']

def hybrid_score(seg):
    rr = rule_engine.evaluate(seg)
    ml_res = ml_model.model.predict(seg)
    ml_p = ml_res['probabilities']
    if rr.action == 'no_action':
        scores = {a: ml_p.get(a, 0.0) for a in ALL_ACTIONS}
    else:
        if rr.confidence >= 0.93:
            rw = 0.70   # mandatory: physics-based rule overrides ML
        elif rr.confidence >= 0.90:
            rw = 0.45   # hard rule
        else:
            rw = 0.28   # soft rule, ML can override
        scores = {a: ml_p.get(a, 0.0)*0.60 + (rr.confidence*rw if rr.action==a else 0.0) for a in ALL_ACTIONS}
    final = max(scores, key=lambda a: scores[a])
    savings = calculate_savings(seg, final)
    return final, scores[final], rr, ml_p, savings

# -- S13 detailed analysis --------------------------------------------
print("=" * 80)
print("S13 RAW TELEMETRY TRUTH CHECK")
print("=" * 80)
s13_raw = pd.read_csv('data/training/S13_low_priority_screening_mode_v4.csv', low_memory=False)
s13_raw.columns = [c.strip().lower().replace(' ','_') for c in s13_raw.columns]

# Ground truth labels per phase
if 'recommended_action' in s13_raw.columns:
    print("Ground truth recommended_action per phase:")
    for phase, grp in s13_raw.groupby('workflow_phase'):
        print(f"  {phase}: {grp['recommended_action'].value_counts().to_dict()}")
else:
    print("NOTE: S13 has NO recommended_action column (it is a test/eval file, not labelled training data)")
print()

# Actual feature values per phase  
print("Key feature values per phase (raw rows):")
for phase, grp in s13_raw.groupby('workflow_phase'):
    lv = grp['live_view_enabled'].astype(float).mean() if 'live_view_enabled' in grp.columns else 0
    ui = grp['user_interacting'].astype(float).mean() if 'user_interacting' in grp.columns else 0
    mon = grp['monitoring_required'].astype(float).mean() if 'monitoring_required' in grp.columns else 0
    inact_med = grp['seconds_since_last_ui_interaction'].median() if 'seconds_since_last_ui_interaction' in grp.columns else 0
    inact_min = grp['seconds_since_last_ui_interaction'].min() if 'seconds_since_last_ui_interaction' in grp.columns else 0
    inact_max = grp['seconds_since_last_ui_interaction'].max() if 'seconds_since_last_ui_interaction' in grp.columns else 0
    overlap = grp['tile_overlap_pct'].mean() if 'tile_overlap_pct' in grp.columns else 0
    quality = grp['quality_constraint_mode'].mode()[0] if 'quality_constraint_mode' in grp.columns else '?'
    opt_overlap = OPTIMAL_TILE_OVERLAP.get(quality, 12.0)
    power = grp['estimated_system_power_w'].mean() if 'estimated_system_power_w' in grp.columns else 0
    n_blocks = grp['workflow_block_id'].nunique() if 'workflow_block_id' in grp.columns else 0
    print(f"  {phase} (blocks={n_blocks}, quality={quality}, optOverlap={opt_overlap}):")
    print(f"    live_view={lv:.3f}  user={ui:.3f}  monitoring={mon:.3f}")
    print(f"    inactivity: min={inact_min:.0f}s  median={inact_med:.0f}s  max={inact_max:.0f}s")
    print(f"    tile_overlap={overlap:.2f}%  excess={overlap-opt_overlap:.2f}%  power={power:.1f}W")
print()

# -- Engine analysis per segment -------------------------------------
print("=" * 80)
print("S13 SEGMENT ENGINE PREDICTIONS")
print("=" * 80)
s13_segs = aggregate_to_segments(s13_raw, has_label=True)

by_phase = {}
total_savings = 0.0
for _, row in s13_segs.iterrows():
    seg = row.to_dict()
    phase = seg['phase_name']
    label = seg.get('label', '?')
    final, conf, rr, ml_p, savings = hybrid_score(seg)
    total_savings += savings

    if phase not in by_phase:
        by_phase[phase] = {'segments': [], 'total_savings': 0.0}
    by_phase[phase]['segments'].append({
        'block': seg.get('workflow_block_id'),
        'inact': seg.get('median_seconds_since_last_ui_interaction', 0),
        'lv': seg.get('live_view_enabled_share', 0),
        'mon': seg.get('monitoring_required_share', 0),
        'ui': seg.get('user_interacting_share', 0),
        'overlap': seg.get('tile_overlap_pct_mean', 0),
        'power': seg.get('estimated_system_power_w_mean', 0),
        'duration': seg.get('duration_sec', 0),
        'energy': seg.get('estimated_energy_wh_interval_sum', 0),
        'label': label,
        'rule': rr.rule_id,
        'rule_action': rr.action,
        'rule_conf': rr.confidence,
        'ml_no_action': round(ml_p.get('no_action', 0), 3),
        'ml_pause': round(ml_p.get('pause_live_view', 0), 3),
        'ml_tile': round(ml_p.get('optimize_tile_scan_settings', 0), 3),
        'final': final,
        'conf': round(conf, 3),
        'savings': round(savings, 3),
    })
    by_phase[phase]['total_savings'] += savings

for phase, info in sorted(by_phase.items()):
    segs_list = info['segments']
    print(f"\n-- {phase.upper()} ({len(segs_list)} segments, total savings={info['total_savings']:.2f} Wh) --")
    # Count predictions
    preds = pd.Series([s['final'] for s in segs_list]).value_counts().to_dict()
    rules_fired = pd.Series([s['rule'] for s in segs_list]).value_counts().to_dict()
    gt_dist = pd.Series([s['label'] for s in segs_list]).value_counts().to_dict()
    print(f"  Predictions: {preds}")
    print(f"  Rules fired: {rules_fired}")
    print(f"  Ground truth: {gt_dist}")
    # Show first 3 + last 3 segments detail
    to_show = segs_list[:3] + (segs_list[-3:] if len(segs_list) > 6 else [])
    print(f"  Sample segments (first 3 / last 3):")
    for s in to_show:
        print(f"    blk={s['block']:>4}  inact={s['inact']:>5.0f}s  lv={s['lv']:.2f}  mon={s['mon']:.2f}  "
              f"overlap={s['overlap']:>5.1f}%  dur={s['duration']:>4}s  energy={s['energy']:>5.2f}Wh  "
              f"rule={s['rule']}/{s['rule_action'][:12]}({s['rule_conf']:.2f})  "
              f"ml=[na={s['ml_no_action']} plv={s['ml_pause']} tile={s['ml_tile']}]  "
              f"=> {s['final']} ({s['conf']:.3f})  save={s['savings']:.3f}Wh  GT={s['label']}")

print()
print(f"TOTAL S13 ESTIMATED SAVINGS: {total_savings:.2f} Wh")

# -- Cross-scenario comparison ----------------------------------------
print()
print("=" * 80)
print("CROSS-SCENARIO COMPARISON (all S files)")
print("=" * 80)
files = sorted(__import__('glob').glob('data/training/S*_v4.csv'))
for fpath in files:
    sname = __import__('os').path.basename(fpath).replace('_v4.csv','')
    raw = pd.read_csv(fpath, low_memory=False)
    raw.columns = [c.strip().lower().replace(' ','_') for c in raw.columns]
    segs = aggregate_to_segments(raw, has_label=True)
    results = {'no_action': 0, 'pause_live_view': 0, 'optimize_tile_scan_settings': 0}
    total_e = 0.0
    total_s = 0.0
    for _, row in segs.iterrows():
        seg = row.to_dict()
        final, conf, rr, ml_p, savings = hybrid_score(seg)
        results[final] = results.get(final, 0) + 1
        total_e += seg.get('estimated_energy_wh_interval_sum', 0)
        total_s += savings
    n = len(segs)
    eff_pct = round(results['no_action']/n*100)
    plv_pct = round(results.get('pause_live_view',0)/n*100)
    tile_pct = round(results.get('optimize_tile_scan_settings',0)/n*100)
    gt_dist = segs['label'].value_counts().to_dict() if 'label' in segs.columns else {}
    print(f"{sname:<45} segs={n:>4}  energy={total_e:>8.1f}Wh  savings={total_s:>7.2f}Wh  "
          f"eff={eff_pct:>3}%  plv={plv_pct:>3}%  tile={tile_pct:>3}%  GT={gt_dist}")
