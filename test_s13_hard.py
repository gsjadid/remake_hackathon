import sys, pandas as pd, numpy as np, glob
sys.path.insert(0, '.')
from app.data_processor import aggregate_to_segments, load_training_data, OPTIMAL_TILE_OVERLAP
from app import rule_engine, ml_model
from app.energy_calculator import calculate_savings

# Train model
train_raw = load_training_data('data/training')
train_segs = aggregate_to_segments(train_raw, has_label=True)
stats = ml_model.model.train(train_segs)
print(f"MLP: acc={stats['accuracy']:.4f}  f1={stats['f1_macro']:.4f}  classes={stats['classes']}")
print()

ALL_ACTIONS = ['no_action','optimize_tile_scan_settings','pause_live_view']

def hybrid_score(seg):
    rr = rule_engine.evaluate(seg)
    ml_res = ml_model.model.predict(seg)
    ml_p = ml_res['probabilities']
    if rr.action == 'no_action':
        scores = {a: ml_p.get(a, 0.0) for a in ALL_ACTIONS}
    else:
        rw = 0.70 if rr.confidence >= 0.93 else (0.45 if rr.confidence >= 0.90 else 0.28)
        scores = {a: ml_p.get(a,0.0)*0.60 + (rr.confidence*rw if rr.action==a else 0.0) for a in ALL_ACTIONS}
    return max(scores, key=lambda a: scores[a])

# ── Per-scenario accuracy on LABELLED files only ──────────────────────────────
print("=" * 70)
print("PER-SCENARIO ACCURACY  (labelled training files only)")
print("=" * 70)
tp_total, n_total = 0, 0
for sname in ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10']:
    files = glob.glob(f'data/training/{sname}_*.csv')
    if not files:
        continue
    raw = pd.read_csv(files[0], low_memory=False)
    raw.columns = [c.strip().lower().replace(' ','_') for c in raw.columns]
    segs = aggregate_to_segments(raw, has_label=True)
    if 'label' not in segs.columns:
        print(f"{sname}: no label column")
        continue
    labeled = segs[segs['label'].notna() & (segs['label'] != '?')]
    if labeled.empty:
        print(f"{sname}: no valid labels")
        continue
    correct = 0
    errors = []
    for _, row in labeled.iterrows():
        seg = row.to_dict()
        pred = hybrid_score(seg)
        gt = seg['label']
        if pred == gt:
            correct += 1
        else:
            errors.append((seg['phase_name'], gt, pred))
    acc = correct / len(labeled) * 100
    tp_total += correct
    n_total += len(labeled)
    print(f"  {sname}: {correct:>3}/{len(labeled):>3} correct  acc={acc:>6.1f}%  errors={len(errors)}")
    for ph, g, p in errors[:5]:
        print(f"      phase={ph}  GT={g}  PRED={p}")

print()
if n_total > 0:
    print(f"  OVERALL HYBRID ACCURACY: {tp_total}/{n_total} = {tp_total/n_total*100:.2f}%")

# ── S13 column mapping investigation ─────────────────────────────────────────
print()
print("=" * 70)
print("S13 COLUMN NAME vs EXPECTED FEATURE CHECK")
print("=" * 70)
s13_raw = pd.read_csv('data/training/S13_low_priority_screening_mode_v4.csv', low_memory=False)
s13_raw.columns = [c.strip().lower().replace(' ','_') for c in s13_raw.columns]

segs = aggregate_to_segments(s13_raw, has_label=False)
print("Quality distribution in S13 aggregated segments:")
print(" ", segs['quality_constraint_mode'].value_counts().to_dict())
print()
print("Actual 'quality_constraint' values in raw S13 rows:")
if 'quality_constraint' in s13_raw.columns:
    print(" ", s13_raw['quality_constraint'].value_counts().to_dict())
else:
    print("  column 'quality_constraint' NOT FOUND")
print()
print("Code expects: 'quality_constraint_mode'")
print("CSV has:      'quality_constraint'   <- MISMATCH!")
print()
print("Live-view related column names in S13 raw CSV:")
lv_cols = [c for c in s13_raw.columns if any(k in c for k in ['quality','view','monitor','interact','flag'])]
print(" ", lv_cols)
print()
print("Code expects: live_view_enabled, monitoring_required, user_interacting")
print("CSV has:      live_view_enabled_flag, monitoring_required_flag, user_interacting_flag  <- MISMATCH (_flag suffix)")

# Check if same mismatch is in S1
s1_raw = pd.read_csv('data/training/S1_high_energy_continuous_supervision_v4.csv', nrows=1, low_memory=False)
s1_raw.columns = [c.strip().lower().replace(' ','_') for c in s1_raw.columns]
s1_lv = [c for c in s1_raw.columns if any(k in c for k in ['quality','view','monitor','interact'])]
print()
print("S1 same columns (for comparison):")
print(" ", s1_lv)

# ── What WOULD quality be if column was read correctly for S13 tile scans ─────
print()
print("=" * 70)
print("S13 TILE SCAN QUALITY & SAVINGS IMPACT")
print("=" * 70)
tile_segs = segs[segs['phase_name'] == 'tile_scan_acquisition']
print(f"Tile scan segments: {len(tile_segs)}")
print(f"Overlap range: {tile_segs['tile_overlap_pct_mean'].min():.2f}% - {tile_segs['tile_overlap_pct_mean'].max():.2f}%")
print(f"Quality inferred by code: {tile_segs['quality_constraint_mode'].unique()}")
print(f"Optimal overlap used (medium): {OPTIMAL_TILE_OVERLAP['medium']}")
print()

actual_quality_raw = s13_raw['quality_constraint'].mode()[0] if 'quality_constraint' in s13_raw.columns else 'unknown'
print(f"Actual quality in raw data: {actual_quality_raw}")
print(f"Optimal overlap SHOULD BE (low): {OPTIMAL_TILE_OVERLAP['low']}")
print()

# Calculate savings delta
savings_current = 0.0
savings_corrected = 0.0
for _, row in tile_segs.iterrows():
    seg = row.to_dict()
    overlap = seg.get('tile_overlap_pct_mean', 0)
    dur = seg.get('duration_sec', 0)
    # Current (medium quality, optimal=12)
    excess_current = max(0, overlap - OPTIMAL_TILE_OVERLAP['medium'])
    s_current = 1.2 * excess_current * dur / 3600
    # Corrected (low quality, optimal=10)
    excess_correct = max(0, overlap - OPTIMAL_TILE_OVERLAP['low'])
    s_correct = 1.2 * excess_correct * dur / 3600
    savings_current += s_current
    savings_corrected += s_correct

print(f"Tile savings WITH CURRENT (wrong medium quality): {savings_current:.2f} Wh")
print(f"Tile savings IF quality='low' (correct):          {savings_corrected:.2f} Wh")
print(f"Underestimated savings due to quality bug:        {savings_corrected - savings_current:.2f} Wh")

# ── Confusion matrix for all labelled segments ────────────────────────────────
print()
print("=" * 70)
print("CONFUSION MATRIX (all labelled segments combined)")
print("=" * 70)
from collections import defaultdict
conf_matrix = defaultdict(lambda: defaultdict(int))
for sname in ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10']:
    files = glob.glob(f'data/training/{sname}_*.csv')
    if not files: continue
    raw = pd.read_csv(files[0], low_memory=False)
    raw.columns = [c.strip().lower().replace(' ','_') for c in raw.columns]
    segs = aggregate_to_segments(raw, has_label=True)
    if 'label' not in segs.columns: continue
    labeled = segs[segs['label'].notna() & (segs['label'] != '?')]
    for _, row in labeled.iterrows():
        seg = row.to_dict()
        pred = hybrid_score(seg)
        gt = seg['label']
        conf_matrix[gt][pred] += 1

print(f"{'':>35}  {'PREDICTED':^42}")
header = f"{'TRUE':>35}  {'no_action':>12}  {'pause_lv':>12}  {'opt_tile':>12}"
print(header)
print("-" * len(header))
for gt_label in ['no_action','pause_live_view','optimize_tile_scan_settings']:
    row_vals = conf_matrix[gt_label]
    total_gt = sum(row_vals.values())
    no_act = row_vals.get('no_action', 0)
    plv    = row_vals.get('pause_live_view', 0)
    tile   = row_vals.get('optimize_tile_scan_settings', 0)
    print(f"  {gt_label:>33}  {no_act:>12}  {plv:>12}  {tile:>12}   (n={total_gt})")
