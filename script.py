import csv
import os
from collections import defaultdict

base = r"c:\Users\dxdel\Downloads\zeis\data\training"

files = {
    "S1":  "S1_high_energy_continuous_supervision_v4.csv",
    "S2":  "S2_low_energy_deferred_batch_v4.csv",
    "S3":  "S3_frequent_short_scans_v4.csv",
    "S4":  "S4_live_view_left_on_monitoring_v4.csv",
    "S5":  "S5_large_area_batch_reconstruction_v4.csv",
    "S6":  "S6_critical_live_view_supervision_v4.csv",
    "S7":  "S7_high_overlap_required_quality_v4.csv",
    "S8":  "S8_post_acquisition_reconstruction_only_v4.csv",
    "S9":  "S9_post_experiment_idle_waste_v4.csv",
    "S10": "S10_confounded_background_load_v4.csv",
    "S13": "S13_low_priority_screening_mode_v4.csv",
}

def read_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

# Load all datasets
data = {}
for k, fname in files.items():
    fpath = os.path.join(base, fname)
    if os.path.exists(fpath):
        data[k] = read_csv(fpath)
        print(f"{k}: {len(data[k])} rows loaded from {fname}")
    else:
        print(f"Warning: File not found: {fpath}")

if "S13" not in data:
    print("Error: S13 data not loaded. Exiting.")
    exit()

print("\n" + "="*70)
print("=== PER-SCENARIO ENERGY SUMMARY (S1-S10) ===")
print("="*70)

scenario_stats = {}
for k in ["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10"]:
    if k not in data: continue
    rows = data[k]
    energies = [float(r['estimated_energy_wh_interval']) for r in rows]
    powers   = [float(r['estimated_system_power_w']) for r in rows]
    total_e  = sum(energies)
    avg_p    = sum(powers)/len(powers) if powers else 0
    max_p    = max(powers) if powers else 0
    min_p    = min(powers) if powers else 0
    
    # per-phase breakdown
    phase_e  = defaultdict(float)
    phase_cnt= defaultdict(int)
    action_cnt = defaultdict(int)
    for r in rows:
        phase_e[r['workflow_phase']] += float(r['estimated_energy_wh_interval'])
        phase_cnt[r['workflow_phase']] += 1
        if 'recommended_action' in r:
            action_cnt[r['recommended_action']] += 1
    
    # overlap stats
    overlaps = [float(r['tile_overlap_pct']) for r in rows if r.get('tile_overlap_pct')]
    avg_overlap = sum(overlaps)/len(overlaps) if overlaps else 0
    
    # live_view waste
    lv_rows = [r for r in rows if r['live_view_enabled_flag']=='True' and r['user_interacting_flag']=='False']
    lv_waste_e = sum(float(r['estimated_energy_wh_interval']) for r in lv_rows)
    
    scenario_stats[k] = {
        'total_e': total_e,
        'avg_p':   avg_p,
        'max_p':   max_p,
        'min_p':   min_p,
        'phase_e': dict(phase_e),
        'phase_cnt': dict(phase_cnt),
        'action_cnt': dict(action_cnt),
        'avg_overlap': avg_overlap,
        'lv_waste_e': lv_waste_e,
        'n_rows': len(rows)
    }
    
    print(f"\n--- {k} ---")
    print(f"  Rows: {len(rows)}  |  Total Energy: {total_e:.3f} Wh  |  Avg Power: {avg_p:.2f} W  |  Range: [{min_p:.1f}–{max_p:.1f}] W")
    print(f"  Avg tile overlap: {avg_overlap:.2f}%")
    print(f"  Live-view unattended waste: {lv_waste_e:.3f} Wh")
    print(f"  Phase energy breakdown:")
    for ph, e in sorted(phase_e.items()):
        pct = e/total_e*100 if total_e else 0
        print(f"    {ph:<35s}: {e:8.3f} Wh  ({pct:5.1f}%)  [{phase_cnt[ph]} rows]")
    print(f"  Recommended actions:")
    for a, c in sorted(action_cnt.items()):
        print(f"    {a:<40s}: {c} rows ({c/len(rows)*100:.1f}%)")

print("\n" + "="*70)
print("=== S13 DETAILED ANALYSIS ===")
print("="*70)

rows13 = data["S13"]
energies13 = [float(r['estimated_energy_wh_interval']) for r in rows13]
powers13   = [float(r['estimated_system_power_w'])     for r in rows13]
total_e13  = sum(energies13)
avg_p13    = sum(powers13)/len(powers13)
max_p13    = max(powers13)
min_p13    = min(powers13)

phase_e13   = defaultdict(float)
phase_cnt13 = defaultdict(int)
phase_power13 = defaultdict(list)
action_cnt13  = defaultdict(int)
for r in rows13:
    phase_e13[r['workflow_phase']] += float(r['estimated_energy_wh_interval'])
    phase_cnt13[r['workflow_phase']] += 1
    phase_power13[r['workflow_phase']].append(float(r['estimated_system_power_w']))
    if 'recommended_action' in r:
        action_cnt13[r['recommended_action']] += 1

overlaps13 = [float(r['tile_overlap_pct']) for r in rows13 if r.get('tile_overlap_pct')]
avg_overlap13 = sum(overlaps13)/len(overlaps13) if overlaps13 else 0

lv_rows13 = [r for r in rows13 if r['live_view_enabled_flag']=='True' and r['user_interacting_flag']=='False']
lv_waste_e13 = sum(float(r['estimated_energy_wh_interval']) for r in lv_rows13)

idle_rows13 = [r for r in rows13 if r['workflow_phase']=='idle']
idle_e13 = sum(float(r['estimated_energy_wh_interval']) for r in idle_rows13)

high_overlap_rows13 = [r for r in rows13 if float(r.get('tile_overlap_pct',0)) > 20]
print(f"S13 Rows: {len(rows13)}")
print(f"S13 Total Energy: {total_e13:.3f} Wh")
print(f"S13 Avg Power: {avg_p13:.2f} W  |  Range: [{min_p13:.1f}–{max_p13:.1f}] W")
print(f"S13 Avg tile overlap: {avg_overlap13:.2f}%")
print(f"S13 Live-view unattended waste: {lv_waste_e13:.3f} Wh ({lv_waste_e13/total_e13*100:.1f}%)")
print(f"S13 Idle energy: {idle_e13:.3f} Wh ({idle_e13/total_e13*100:.1f}% of total)")
print(f"S13 High-overlap rows (>20%): {len(high_overlap_rows13)}")
print("\nS13 Phase energy breakdown:")
for ph, e in sorted(phase_e13.items()):
    pct = e/total_e13*100
    avg_ph_power = sum(phase_power13[ph])/len(phase_power13[ph])
    print(f"  {ph:<35s}: {e:8.3f} Wh  ({pct:5.1f}%)  [rows={phase_cnt13[ph]}, avg_power={avg_ph_power:.1f}W]")
print("\nS13 Recommended actions distribution:")
for a, c in sorted(action_cnt13.items()):
    print(f"  {a:<40s}: {c} rows ({c/len(rows13)*100:.1f}%)")

# ----------------------------------------------------------------
# ENERGY SAVING CALCULATIONS
# ----------------------------------------------------------------
print("\n" + "="*70)
print("=== ENERGY SAVING OPPORTUNITIES FOR S13 ===")
print("="*70)

# Saving 1: Pause live-view when unattended
lv_saving = lv_waste_e13
print(f"\n[1] PAUSE LIVE VIEW WHEN UNATTENDED")
print(f"    S13 unattended live-view energy: {lv_waste_e13:.3f} Wh")
print(f"    If paused → save up to {lv_saving:.3f} Wh ({lv_saving/total_e13*100:.1f}% of S13 total)")

s13_lv_gpu_powers = [float(r['perf_gpu_power_w']) for r in lv_rows13]
s13_no_lv_gpu_powers = [float(r['perf_gpu_power_w']) for r in rows13 if r['live_view_enabled_flag']=='False']
avg_lv_gpu = sum(s13_lv_gpu_powers)/len(s13_lv_gpu_powers) if s13_lv_gpu_powers else 0
avg_no_lv_gpu = sum(s13_no_lv_gpu_powers)/len(s13_no_lv_gpu_powers) if s13_no_lv_gpu_powers else 0
print(f"    S13 avg GPU power WITH live-view: {avg_lv_gpu:.2f} W")
print(f"    S13 avg GPU power WITHOUT live-view: {avg_no_lv_gpu:.2f} W")
gpu_diff = max(0, avg_lv_gpu - avg_no_lv_gpu)
print(f"    GPU power reduction by pausing: {gpu_diff:.2f} W")
lv_duration_sec = len(lv_rows13) * 15
realistic_lv_saving_wh = gpu_diff * lv_duration_sec / 3600
print(f"    Unattended live-view duration: {lv_duration_sec/60:.1f} min")
print(f"    Realistic GPU-based saving: {realistic_lv_saving_wh:.3f} Wh")

# Saving 2: Optimize tile scan settings (reduce overlap)
print(f"\n[2] OPTIMIZE TILE SCAN SETTINGS (Reduce Overlap)")
tile_scan_rows13 = [r for r in rows13 if r['workflow_phase']=='tile_scan_acquisition']
ts_total_e13 = sum(float(r['estimated_energy_wh_interval']) for r in tile_scan_rows13)
ts_overlaps13 = [float(r['tile_overlap_pct']) for r in tile_scan_rows13]
avg_ts_overlap13 = sum(ts_overlaps13)/len(ts_overlaps13) if ts_overlaps13 else 0

if "S2" in data:
    tile_scan_rows_s2 = [r for r in data['S2'] if r['workflow_phase']=='tile_scan_acquisition']
    ts_overlaps_s2 = [float(r['tile_overlap_pct']) for r in tile_scan_rows_s2]
    avg_ts_overlap_s2 = sum(ts_overlaps_s2)/len(ts_overlaps_s2) if ts_overlaps_s2 else 0
    print(f"    S13 tile_scan_acquisition total energy: {ts_total_e13:.3f} Wh")
    print(f"    S13 avg tile overlap: {avg_ts_overlap13:.2f}%")
    print(f"    S2  avg tile overlap (baseline): {avg_ts_overlap_s2:.2f}%")
    if avg_ts_overlap13 > 0:
        overlap_saving_factor = 1 - ((1 + avg_ts_overlap_s2/100)**2) / ((1 + avg_ts_overlap13/100)**2)
        overlap_energy_saving = max(0, ts_total_e13 * overlap_saving_factor)
        print(f"    Estimated tile-scan energy saving: {overlap_energy_saving:.3f} Wh")
    else: overlap_energy_saving = 0
else:
    print("    S2 baseline not available for overlap comparison")
    overlap_energy_saving = 0

# Saving 3: Reduce idle energy
print(f"\n[3] REDUCE IDLE PHASE ENERGY")
idle_avg_powers = {}
for k, scenario in data.items():
    if k == "S13": continue
    irows = [r for r in scenario if r['workflow_phase']=='idle']
    if irows:
        idle_avg_powers[k] = sum(float(r['estimated_system_power_w']) for r in irows)/len(irows)

if idle_rows13 and idle_avg_powers:
    avg_idle_power13 = sum(float(r['estimated_system_power_w']) for r in idle_rows13)/len(idle_rows13)
    best_idle_k = min(idle_avg_powers, key=idle_avg_powers.get)
    best_idle_power = idle_avg_powers[best_idle_k]
    idle_power_saving = max(0, avg_idle_power13 - best_idle_power)
    idle_duration_sec = len(idle_rows13) * 15
    idle_saving_wh = idle_power_saving * idle_duration_sec / 3600
    print(f"    S13 avg idle power: {avg_idle_power13:.2f} W")
    print(f"    Best ref idle: {best_idle_k} at {best_idle_power:.2f} W")
    print(f"    Potential idle saving: {idle_saving_wh:.3f} Wh")
else:
    idle_saving_wh = 0
    print("    No idle rows or references available.")

# Final Summary
print("\n" + "="*70)
print("=== FINAL ENERGY SAVING SUMMARY FOR S13 ===")
print("="*70)
print(f"S13 total energy: {total_e13:.3f} Wh")
total_saving = lv_saving + overlap_energy_saving + idle_saving_wh
print(f"Total potential saving: {total_saving:.3f} Wh ({total_saving/total_e13*100:.1f}%)")
print(f"Optimized S13: {max(0, total_e13 - total_saving):.3f} Wh")
