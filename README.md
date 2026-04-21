# ZEISS Smart Assistant for Energy-Efficient Microscopy Workflows

A hackathon project built for ZEISS — an AI-powered energy intelligence dashboard that ingests microscopy workflow telemetry, detects inefficiencies, and recommends concrete power-saving actions with quantified Wh savings.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Getting Started](#3-getting-started)
4. [Architecture Overview](#4-architecture-overview)
5. [Backend — API Layer (`main.py`)](#5-backend--api-layer-mainpy)
6. [Data Pipeline (`app/data_processor.py`)](#6-data-pipeline-appdata_processorpy)
7. [Rule Engine (`app/rule_engine.py`)](#7-rule-engine-apprule_enginepy)
8. [Machine Learning Model (`app/ml_model.py`)](#8-machine-learning-model-appml_modelpy)
9. [Hybrid Engine (`app/hybrid_engine.py`)](#9-hybrid-engine-apphybrid_enginepy)
10. [Energy Calculator (`app/energy_calculator.py`)](#10-energy-calculator-appenergy_calculatorpy)
11. [Frontend Dashboard (`templates/index.html`)](#11-frontend-dashboard-templatesindexhtml)
12. [Data & Training Scenarios](#12-data--training-scenarios)
13. [API Reference](#13-api-reference)
14. [Design System](#14-design-system)
15. [Key Constants & Thresholds](#15-key-constants--thresholds)
16. [Decision Flow — End to End](#16-decision-flow--end-to-end)

---

## 1. Project Overview

Modern electron microscopes consume significant power (100–300 W) continuously. Many workflows leave the live view illumination on while the user is absent, run tile scans with unnecessarily high overlap, or leave the system fully powered during post-experiment idle time. None of this is flagged to the operator.

This assistant:

- Accepts a workflow CSV (raw 15-second telemetry rows **or** pre-aggregated segment files)
- Groups rows into workflow segments by phase
- Runs a **hybrid AI engine** (rule-based + MLP neural network) on each segment
- Returns an action recommendation for each segment (`pause_live_view`, `optimize_tile_scan_settings`, or `no_action`)
- Quantifies the energy saved (Wh) if each recommendation is followed
- Renders all of this in a real-time ZEISS-branded dashboard with a waveform timeline, per-segment cards, KPI chips, and phase labels

---

## 2. Repository Structure

```
zeis/
├── main.py                          # FastAPI app entry point, routes, lifespan trainer
├── requirements.txt                 # Python dependencies
├── diagnostic_s13.py                # Standalone diagnostic script for scenario S13
├── script.py                        # Utility / scratch script
│
├── app/
│   ├── __init__.py
│   ├── data_processor.py            # CSV loading, aggregation, feature engineering
│   ├── rule_engine.py               # 5 physics-validated deterministic rules
│   ├── ml_model.py                  # MLP classifier (128→64→32) + spike detector
│   ├── hybrid_engine.py             # Blends rule + ML scores; produces final decision
│   └── energy_calculator.py        # Wh savings formula per recommended action
│
├── templates/
│   └── index.html                   # Full ZEISS dashboard (Tailwind + vanilla JS)
│
└── data/
    ├── data_dictionary_energy.csv   # Column definitions for telemetry CSVs
    ├── optional/
    │   └── Energy for Workflow data.csv
    └── training/
        ├── S1_high_energy_continuous_supervision_v4.csv
        ├── S2_low_energy_deferred_batch_v4.csv
        ├── S3_frequent_short_scans_v4.csv
        ├── S4_live_view_left_on_monitoring_v4.csv
        ├── S5_large_area_batch_reconstruction_v4.csv
        ├── S6_critical_live_view_supervision_v4.csv
        ├── S7_high_overlap_required_quality_v4.csv
        ├── S8_post_acquisition_reconstruction_only_v4.csv
        ├── S9_post_experiment_idle_waste_v4.csv
        ├── S10_confounded_background_load_v4.csv
        ├── S13_low_priority_screening_mode_v4.csv
        ├── model_training.ipynb
        └── test_segments.csv        # Pre-aggregated segments for all scenarios
```

---

## 3. Getting Started

### Prerequisites

- Python 3.10 or later
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `fastapi`, `uvicorn[standard]`, `pandas`, `scikit-learn`, `python-multipart`, `jinja2`

### Run the server

```bash
python main.py
```

Server starts at **http://localhost:3000**

On startup the server trains the MLP automatically from all `S*_v4.csv` files in `data/training/`. You will see a log line like:

```
[ZEISS AI] MLP trained: accuracy=93.27%  F1=90.68%  (N train / M val segments)
```

### Port conflict

If port 3000 is already in use:

```powershell
Get-NetTCPConnection -LocalPort 3000 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

Then run `python main.py` again.

---

## 4. Architecture Overview

```
CSV Upload (raw telemetry OR pre-aggregated)
         │
         ▼
  data_processor.py
  ┌────────────────────────────────────┐
  │  detect_and_load()                 │
  │  → raw? aggregate_to_segments()    │
  │  → pre-agg? _normalise_aggregated()│
  └────────────────────────────────────┘
         │  one row per segment
         ▼
  hybrid_engine.analyze_segment()  ← called per segment
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  rule_engine.evaluate()          ml_model.predict() │
  │  → R1–R5 deterministic rules     → MLP(128,64,32)   │
  │  → RuleResult(action, conf)      → prob dict        │
  │                                                     │
  │         Score Fusion (tiered weights)               │
  │  mandatory (conf≥0.93): rule×0.70 + ml×0.60        │
  │  hard      (conf≥0.90): rule×0.45 + ml×0.60        │
  │  soft      (conf<0.90): rule×0.28 + ml×0.60        │
  │  no-rule: pure ML scores                            │
  │                                                     │
  │  final_action = argmax(blended scores)              │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  energy_calculator.calculate_savings()
  → Wh savings if action is applied
         │
         ▼
  JSON response → renderResults() in index.html
```

The system is **ML-primary**: the MLP is the default decision maker. Rules act as evidence boosters, escalating confidence for patterns they were specifically designed to catch. Physics-based mandatory rules (tile overlap waste, long-inactivity live view) can override the MLP entirely.

---

## 5. Backend — API Layer (`main.py`)

### Startup — `lifespan(app)`

An async context manager that runs once before the server accepts any request. It:

1. Calls `load_training_data(TRAINING_DIR)` to load all `S*_v4.csv` files
2. Calls `aggregate_to_segments(raw_df, has_label=True)` to build the segment DataFrame
3. Calls `mlp_model.train(segments_df)` to fit the MLP + scaler
4. Logs accuracy and F1 to stdout

If training fails the server still starts — the MLP will return `no_action` for all predictions and the rule engine still operates normally.

### Routes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the Jinja2 dashboard (`templates/index.html`) |
| `GET` | `/health` | Returns model readiness, accuracy, and F1 |
| `POST` | `/analyze` | Accepts a CSV file upload; returns full analysis JSON |
| `GET` | `/analyze-sample?name=S1` | Runs analysis on a bundled demo scenario from `test_segments.csv` |

### `_run_analysis(raw_bytes, scenario_name)`

The shared pipeline called by both `/analyze` and `/analyze-sample`:

1. `detect_and_load(raw_bytes)` — parse CSV, auto-detect format
2. Loop over every segment row → `analyze_segment(row.to_dict())`
3. Aggregate summary stats:
   - `total_energy_wh` — sum of all segment energies
   - `total_savings_wh` — sum of all estimated savings
   - `savings_pct` — `(savings / total) × 100`
   - `spike_count` — segments flagged as energy spikes
   - `issues_count` — segments where `recommended_action != no_action`
4. Returns a `JSONResponse` with all segment results + summary

### Allowed sample scenarios

`_SAMPLE_SCENARIOS = {"S1","S2","S3","S4","S5","S6","S7","S8","S9","S10","S13"}`

Any code outside this set returns HTTP 404. This is an allowlist — not derived from filesystem, so there is no directory traversal risk.

---

## 6. Data Pipeline (`app/data_processor.py`)

### Constants

```python
PHASE_BASELINES_W = {
    "idle": 135.0,
    "processing": 200.0,
    "live_view_monitoring": 180.0,
    "tile_scan_acquisition": 170.0,
}

OPTIMAL_TILE_OVERLAP = {
    "low": 10.0,    # fast screening
    "medium": 12.0, # default
    "high": 18.0,   # critical quality scans
}
```

These baseline power values represent the expected system draw for each workflow phase. The feature `power_vs_baseline = avg_power − baseline` captures anomalous power draw and is the primary signal for energy spike detection.

Optimal overlap values encode the minimum overlap needed for image stitching at each quality level. Anything above these values is wasted scan area.

### `load_training_data(training_dir)`

Glob-loads all `S*_v4.csv` files from the training directory, tags each row with its source filename, and concatenates them into a single raw DataFrame. Used only at startup for training.

### `aggregate_to_segments(df_raw, has_label)`

The core feature engineering function. Takes raw 15-second telemetry rows and produces one aggregated row per workflow segment.

**Grouping**: rows are grouped by `(session_id, workflow_block_id)`. Each group becomes one segment.

**Per-segment features computed:**

| Feature | Derivation |
|---------|-----------|
| `phase_name` | Mode of `workflow_phase` in the group |
| `quality_constraint_mode` | Mode of the quality column; inferred from overlap if missing |
| `duration_sec` | `n_rows × 15` seconds |
| `live_view_enabled_share` | Mean of boolean `live_view_enabled` column |
| `user_interacting_share` | Mean of boolean `user_interacting` column |
| `monitoring_required_share` | Mean of boolean `monitoring_required` column |
| `median_seconds_since_last_ui_interaction` | Median of inactivity column |
| `tile_overlap_pct_mean` | Mean of `tile_overlap_pct` |
| `perf_gpu_power_w_mean` | Mean GPU power draw |
| `processing_items_in_flight_mean` | Mean in-flight processing queue depth |
| `estimated_system_power_w_mean` | Mean system power |
| `estimated_energy_wh_interval_sum` | Sum of per-interval energy in Wh |
| `power_vs_baseline` | `avg_power − PHASE_BASELINES_W[phase]` |

**Quality inference** — when `quality_constraint_mode` is blank, `"?"`, or `NaN`:

```
overlap >= 15.0%  →  quality = "high"
overlap >= 11.0%  →  quality = "medium"
overlap <  11.0%  →  quality = "low"
```

This prevents S13-style scenarios (where quality is not recorded) from defaulting to `"medium"` and incorrectly suppressing tile-overlap recommendations.

### `detect_and_load(file_bytes)`

Auto-detects whether the uploaded CSV is raw telemetry or already aggregated.

- **Detection**: presence of column `phase_segment_id` → pre-aggregated
- **Raw path**: `aggregate_to_segments(df, has_label=False)`
- **Pre-aggregated path**: `_normalise_aggregated(df)` (renames columns, fills defaults, computes `power_vs_baseline` if missing)

Returns `(DataFrame, is_raw: bool)`.

### `_normalise_aggregated(df)`

Maps column names from the pre-aggregated test file schema to the schema produced by `aggregate_to_segments()`. Also fills all numeric columns with sensible defaults (0.0) when absent, and normalises phase names to lowercase.

---

## 7. Rule Engine (`app/rule_engine.py`)

The rule engine implements 5 physics-validated deterministic rules. Rules are evaluated in priority order — the first match wins. If no rule fires, `no_action` is returned.

### `RuleResult` dataclass

```python
@dataclass
class RuleResult:
    action: str             # recommended action string
    confidence: float       # 0.0–1.0; determines rule weight in fusion
    issue_description: str  # human-readable explanation shown in UI
    rule_id: Optional[str]  # "R1"–"R5" or None
```

### R1 — Inactivity-driven live-view pause

**Condition:**
```
live_view_share > 0.30
AND inactivity >= 120 seconds
AND monitoring_share < 0.50
```

**Action:** `pause_live_view`  
**Confidence:** `0.95` (HARD rule)  
**Basis:** Validated from raw S1 telemetry — rows with ≥120s inactivity always correspond to unattended illumination. The 0.5 monitoring guard ensures we don't fire on legitimate supervised monitoring sessions.

This is the highest-priority rule. An operator absent for 2+ minutes with live view running is the most common and most costly inefficiency pattern in the training data.

### R2 — Idle phase with live view on

**Condition:**
```
phase == "idle"
AND live_view_share > 0.30
```

**Action:** `pause_live_view`  
**Confidence:** `0.90` (HARD rule)  
**Basis:** Validated from S9. When the system is explicitly in idle phase, there is no scientific justification for illumination to be on. The 0.3 threshold tolerates brief transient activations.

### R3 — Live-view monitoring with no actual monitoring

**Condition:**
```
phase == "live_view_monitoring"
AND monitoring_share < 0.15
AND user_interacting < 0.10
AND inactivity >= 90 seconds
```

**Action:** `pause_live_view`  
**Confidence:** `0.92` if `inactivity >= 120s`, else `0.80` (SOFT)

**Basis:** The phase name says "monitoring" but the telemetry shows no monitoring traffic and the user has not interacted. The confidence is deliberately scaled — a user who stepped away for 90s may return, whereas 2+ minutes with zero interaction is a clear waste pattern. The lower confidence at 90–120s allows the ML to override if it has learned this specific pattern corresponds to normal operator behaviour.

### R4 — Tile scan with excess overlap

**Condition:**
```
phase == "tile_scan_acquisition"
AND (tile_overlap_pct_mean − OPTIMAL[quality]) > 2.0%
```

**Action:** `optimize_tile_scan_settings`  
**Confidence:** `0.93` (MANDATORY) if `inactivity >= 90s`, else `0.82` (SOFT)

**Basis:** Excess overlap wastes scan area proportionally. At 1.2 W/%, even a 2% excess over a 30-minute scan wastes ~1.2 Wh. The mandatory tier (conf = 0.93) applies when the user is absent — the scan is running unattended with no opportunity for manual correction, so the recommendation is physics-certain. The threshold was lowered from 3.0% to 2.0% because the savings are measurable even at small excess values.

### R5 — Processing phase with unused live view

**Condition:**
```
phase == "processing"
AND live_view_share > 0.50
AND inactivity >= 120 seconds
```

**Action:** `pause_live_view`  
**Confidence:** `0.75` (SOFT)  
**Basis:** During post-acquisition processing, the camera illumination serves no purpose. This is a softer rule because some reconstruction workflows do keep live view open for monitoring purposes — the ML can override with moderate confidence.

### Default

If no rule matches: `RuleResult(action="no_action", confidence=0.50, issue_description="", rule_id=None)`

---

## 8. Machine Learning Model (`app/ml_model.py`)

### Architecture

```
MLPClassifier(hidden_layer_sizes=(128, 64, 32))
  activation  : relu
  max_iter    : 500
  early_stopping: True
  validation_fraction: 0.15
  n_iter_no_change: 20
  random_state: 42
```

3 output classes: `no_action`, `optimize_tile_scan_settings`, `pause_live_view`

### Feature Set (11 features, order-sensitive)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `live_view_enabled_share` | Fraction of segment with live view on |
| 2 | `user_interacting_share` | Fraction of segment with active user input |
| 3 | `monitoring_required_share` | Fraction of segment where monitoring is flagged |
| 4 | `median_seconds_since_last_ui_interaction` | Inactivity metric |
| 5 | `tile_overlap_pct_mean` | Average tile overlap percentage |
| 6 | `perf_gpu_power_w_mean` | Average GPU power draw |
| 7 | `processing_items_in_flight_mean` | Average compute queue depth |
| 8 | `estimated_system_power_w_mean` | Average total system power |
| 9 | `power_vs_baseline` | Delta from per-phase expected power |
| 10 | `phase_encoded` | Integer encoding of workflow phase (0–3) |
| 11 | `quality_encoded` | Integer encoding of quality level (0–2) |

Phase encoding: `idle=0, processing=1, live_view_monitoring=2, tile_scan_acquisition=3`  
Quality encoding: `low=0, medium=1, high=2`

### Phase-context correction in `_build_feature_row()`

Some telemetry files (e.g. S13) report `live_view_enabled = 0` even when the microscope is in `live_view_monitoring` phase. Without correction the MLP would see 0% live view and default to `no_action`. The correction:

```python
if live_view_share < 0.05:
    if phase == "live_view_monitoring":
        live_view_share = 1.0      # illumination must be on by definition
    elif phase in ("idle", "tile_scan_acquisition"):
        live_view_share = 0.5      # likely on; use conservative mid-point
```

This same correction is applied in `energy_calculator.py` for consistent savings calculations.

### Training — `EnergyMLP.train(segments_df)`

1. Drops rows with missing labels
2. Encodes phase and quality columns
3. Fills missing numeric features with 0.0
4. Splits 80/20 train/val (stratified)
5. Fits `StandardScaler` on training set only
6. Trains MLP with early stopping
7. Evaluates accuracy + macro F1 on validation set
8. Sets `_trained = True`

Returns `{accuracy, f1_macro, n_train, n_val, classes}`.

Typical result on the full S1–S13 training corpus: **accuracy ≈ 93%, F1 ≈ 91%**

### Inference — `EnergyMLP.predict(segment)`

1. Calls `_build_feature_row(segment)` with phase-context correction
2. Scales with the fitted scaler
3. Runs `predict_proba` → probability distribution over 3 classes
4. Detects energy spike: `power_vs_baseline > 20 W` → `is_energy_spike = True`
5. Builds a human-readable explanation highlighting the highest-magnitude feature
6. Returns `{action, probabilities, is_energy_spike, spike_magnitude_w, explanation}`

The singleton instance `model = EnergyMLP()` is instantiated at module import. It is trained once in the lifespan hook and then used read-only for all inference requests.

### Energy Spike Detection

A segment is flagged as an energy spike when:

```
power_vs_baseline > SPIKE_THRESHOLD_W  (20 W)
```

`power_vs_baseline` is an engineered feature: the difference between the segment's average system power and the expected baseline for its phase. A spike means the microscope is drawing significantly more power than normal for that activity.

---

## 9. Hybrid Engine (`app/hybrid_engine.py`)

The hybrid engine fuses rule and ML evidence into a single final decision.

### Weight Constants

| Tier | Condition | Rule Weight | Description |
|------|-----------|-------------|-------------|
| Mandatory | `conf >= 0.93` | `0.70` | Physics-based certainty; intended to override ML |
| Hard | `conf >= 0.90` | `0.45` | Empirically certain; hard for ML to override |
| Soft | `conf < 0.90` | `0.28` | Suggestive; ML with moderate confidence can override |
| ML baseline | always | `0.60` | Primary classifier weight |

### Fusion Algorithm

**When rule returns `no_action`** — pure ML path:
```python
scores = {action: ml_proba[action] for action in ALL_ACTIONS}
```
Rules must never suppress the ML from surfacing inefficiencies they don't cover.

**When a rule fires** — blended scoring:
```python
for action in ALL_ACTIONS:
    ml_score   = ml_proba[action] * MLP_WEIGHT           # 0.60
    rule_score = rule_conf * rule_weight if rule_result.action == action else 0.0
    scores[action] = ml_score + rule_score
```

`final_action = argmax(scores)`  
`combined_confidence = scores[final_action]`

### Override math examples

**R1 (conf=0.95, HARD, weight=0.45):**
- Rule boost for `pause_live_view` = `0.95 × 0.45 = 0.428`
- ML would need `> 0.71` confidence in a different action to override: `ml_proba[other] × 0.60 > 0.428 + ml_proba[pause] × 0.60`
- In practice: very hard to override, as intended.

**R3 (conf=0.80, SOFT, weight=0.28):**
- Rule boost = `0.80 × 0.28 = 0.224`
- ML needs `> 0.37` confidence in a different action to override — achievable for scenarios where R3 fires but the pattern is actually benign (e.g. S6 supervised monitoring)

**R4 (conf=0.93, MANDATORY, weight=0.70):**
- Rule boost = `0.93 × 0.70 = 0.651`
- ML would need `> 1.085` confidence to override — impossible (probabilities max at 1.0)
- Tile overlap waste is physics-certain; the rule always wins.

### Output from `analyze_segment(segment)`

Returns a dict with all fields needed by the API and dashboard:

| Field | Type | Description |
|-------|------|-------------|
| `segment_id` | str | ID from source data |
| `phase` | str | Workflow phase name |
| `duration_sec` | int | Segment duration |
| `avg_power_w` | float | Average system power |
| `total_energy_wh` | float | Total energy consumed |
| `overlap_pct` | float | Mean tile overlap |
| `live_view_share` | float | Fraction with live view on |
| `user_interaction_share` | float | Fraction with user interacting |
| `monitoring_required_share` | float | Fraction flagged for monitoring |
| `quality` | str | Quality constraint mode |
| `power_vs_baseline` | float | Delta from phase baseline |
| `is_energy_spike` | bool | True if >20W above baseline |
| `spike_magnitude_w` | float | Watt excess above baseline |
| `issue` | str | Human-readable issue description |
| `recommended_action` | str | Final action recommendation |
| `action_reason` | str | Reasoning string for UI tooltip |
| `estimated_savings_wh` | float | Energy saved if action applied |
| `confidence` | float | Blended decision confidence |
| `rule_id` | str\|null | Which rule fired (R1–R5) |
| `mlp_probabilities` | dict | Raw MLP output {action: prob} |

---

## 10. Energy Calculator (`app/energy_calculator.py`)

### `calculate_savings(segment, action) → float (Wh)`

Physics-based savings estimation. Always returns `>= 0`.

#### `pause_live_view`

```
savings_Wh = 18 W × live_view_share × duration_sec / 3600
```

- `18 W` = empirically measured illumination overhead (from training data power analysis)
- `live_view_share` = fraction of segment with live view on
- Phase-context correction applied (same as `ml_model.py`): if `live_view_share < 0.05`, use `1.0` for `live_view_monitoring` phase or `0.5` for `idle`/`tile_scan_acquisition`

#### `optimize_tile_scan_settings`

```
excess_pct  = max(0, tile_overlap_pct_mean − OPTIMAL_TILE_OVERLAP[quality])
savings_Wh  = 1.2 W/% × excess_pct × duration_sec / 3600
```

- `1.2 W/%` = power per percentage point of excess overlap
- `OPTIMAL_TILE_OVERLAP` = `{"low": 10.0, "medium": 12.0, "high": 18.0}`

#### Example (S13 scenario)

A 45-minute tile scan with 18% overlap and quality "medium" (optimal = 12%):
```
excess_pct  = 18 − 12 = 6.0%
savings_Wh  = 1.2 × 6.0 × 2700 / 3600 = 5.4 Wh
```

Across the full S13 scenario (multiple segments), verified savings = **24.26 Wh total** (17.33 Wh from live view + 6.93 Wh from tile scan).

---

## 11. Frontend Dashboard (`templates/index.html`)

Single-file full-stack dashboard. No build step. Uses Tailwind CDN v3, Inter font (Google Fonts), and vanilla JS.

### Views / Sections

The dashboard has three logical view states controlled by adding CSS classes to `<body>`:

| Body class | Visible section | Hidden sections |
|------------|----------------|-----------------|
| _(default)_ | `#hero-view` + `#upload-section` | `#results-view` |
| `view-results` | `#results-view` | `#hero-view`, `#upload-section` |

### CSS Variables (Design Tokens)

```css
--bg         : page background
--surface    : card/panel background
--border     : border color
--text        : primary text
--muted       : secondary text
--faint       : tertiary/placeholder text
--zeiss       : #009FE3 (ZEISS brand blue)
--ok          : #10B981 (green — efficient)
--warn        : #F59E0B (amber — tile scan issue)
--spike       : #F43F5E (red — live view issue / energy spike)
--pill-bg     : pill/chip background
```

Dark mode is toggled by adding/removing the `dark` class on `<html>`.

### JavaScript Functions (A–Z)

#### `activateSegment(idx)`
Sets the currently active segment. Updates the waveform cursor position via `updateWavePosition(idx)`, scrolls the corresponding segment card into view, highlights it with a teal left border and `ring-2` outline, and populates the detail panel (Phase, Duration, Power, Savings, Confidence, Rule, MLP probabilities bar chart).

#### `actionBadge(action)`
Returns an HTML string for a colour-coded badge pill:
- `pause_live_view` → red `⏸ Pause Live View`
- `optimize_tile_scan_settings` → amber `⚙ Optimize Scan`
- `no_action` → green `✓ Efficient`

#### `actionColor(action)`
Returns the CSS variable string for an action's colour (`--spike`, `--warn`, or `--ok`).

#### `applyFilter()`
Filters the global `_allSegs` array to `_visibleIdx` based on the current `_filter` value and then calls `renderSegments()`. Hides/shows the `#segments-empty` message when no segments match.

#### `applyTheme(mode)`
Sets light/dark mode. Toggles `dark` on `<html>`, swaps the theme button icon, persists the choice to `localStorage` under key `'zeiss-theme'`.

On first call with no argument, reads the stored preference or falls back to `prefers-color-scheme`.

#### `buildImpact(wh)`
Converts a Wh savings number into human-scale equivalents shown in the summary card:
- LED bulb hours equivalent (`wh / 10 × 1000 ms → hours`)
- Phone charges (one smartphone charge ≈ 15 Wh)
- CO₂ offset estimate

#### `fmtDuration(sec)`
Formats a duration in seconds to a human-readable string: `"30s"`, `"2m 30s"`, `"1h 5m"`.

#### `handleDrop(e)` / `handleFileSelect(e)`
Drag-and-drop and file-input change handlers. Both call `selectFile(f)`.

#### `jumpToSegment(idx)`
Looks up `idx` in `_visibleIdx` (the filtered index), then calls `activateSegment()` with the corresponding position. Removes `collapsed` class from the target segment card if it was collapsed.

#### `pollHealth()`
Async function. Fetches `GET /health` every 3 seconds until the model reports `model_ready: true`. Updates the `#health-chip` badge:
- `"Initializing…"` (amber, pulsing) → while polling
- `"Model Ready — XX.X% acc"` (green) → when trained

#### `renderFilterChips(segs)`
Builds the phase-filter chip row from the unique phases present in the analysis results. Each chip calls `setFilter(phase)` when clicked. Also adds an "All" chip.

#### `renderResults(data)`
Main render function. Called once when analysis completes. Receives the full API JSON response. Orchestrates all other render functions:
1. Populates KPI chips (total energy, savings Wh, savings %, spike count)
2. Calls `verdict(data)` to set the headline verdict banner
3. Calls `renderFilterChips(segs)` for the phase filter row
4. Calls `renderWaveform(segs)` for the timeline
5. Calls `renderSegments(segs)` for the segment card list
6. Switches body to `view-results` class
7. Auto-activates the first segment (`activateSegment(0)`)

#### `renderSegments(segs)`
Renders the full segment card list into `#seg-list`. For each segment in `_visibleIdx`:
- Creates a `div.seg-card` with severity class (`seg-ok`, `seg-spike`, `seg-warn`)
- Card always starts open (no `collapsed` class)
- Shows segment number, duration, power, energy, phase label, action badge
- Shows `seg-details` block with issue text, savings, and action reason
- Attaches click listener → `jumpToSegment(i)`

#### `renderWaveform(segs)`
Renders the power-over-time waveform at the top of the results view. Each segment is a bar scaled proportionally to its duration. Bar colour reflects severity. Phase-change boundary labels are shown. Clicking a bar calls `jumpToSegment(i)`. A scrubber cursor tracks the active segment via `updateWavePosition()`.

#### `runAnalysis()`
Async. Reads `_selectedFile`, POSTs it to `POST /analyze` using `FormData`. Manages the upload state machine via `setUploadState()`. On success, calls `renderResults(data)`. On error, shows an error message.

#### `runScenario(name)`
Async. Fetches `GET /analyze-sample?name=<name>`. On success, calls `renderResults(data)`. This function remains in the JS even without the demo section in the DOM — it's available for programmatic use or future UI additions.

#### `selectFile(f)`
Validates that the file has a `.csv` extension, stores it in `_selectedFile`, updates the upload zone UI with the filename, and enables the Analyze button.

#### `setFilter(f)`
Sets `_filter` to `f` (a phase name or `"all"`), updates active chip styling, and calls `applyFilter()`.

#### `setTab(name)`
Switches the detail panel tab between `"detail"` and `"mlp"`. Hides/shows the corresponding pane.

#### `setUploadState(state, msg)`
Controls the upload flow state machine. States:
- `"idle"` — default, no file selected
- `"ready"` — file selected, button enabled
- `"loading"` — analysis running, spinner visible, button disabled
- `"done"` — analysis complete
- `"error"` — shows error message in red

#### `startPlay()` / `stopPlay()`
Auto-play mode. `startPlay()` sets a 1.2-second interval that calls `stepSegment(1)` to advance through segments. `stopPlay()` clears the interval. The play/pause button toggles between these two states.

#### `stepSegment(dir)`
Advances the active segment by `dir` (+1 forward, −1 backward) within the visible (filtered) segment list. Wraps around at boundaries.

#### `toast(msg)`
Shows a temporary toast notification at the bottom-right of the screen. Fades out after 2.5 seconds.

#### `toggleEfficient(show)`
Called by the "Show efficient" checkbox. When `show = false`, adds `collapsed` to all `.seg-efficient` cards, hiding their details. When `show = true`, removes `collapsed` to expand them.

#### `toggleHelp(force)`
Shows/hides the keyboard shortcuts help overlay (`#help-overlay`). Pass `true` to show, `false` to hide, or call without argument to toggle.

#### `updateWavePosition(idx)`
Moves the waveform scrubber cursor to align with segment at index `idx`. Calculates the cumulative width offset based on the duration-proportional widths of all preceding segments.

#### `verdict(data)`
Generates the summary headline and sub-text based on `savings_pct`:
- `>= 20%` — "Significant Waste Detected" (red)
- `>= 8%` — "Moderate Inefficiency" (amber)
- `< 8%` — "Workflow Looks Efficient" (green)

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `t` | Toggle dark/light theme |
| `←` | Previous segment |
| `→` | Next segment |
| `Space` | Toggle auto-play |
| `?` | Toggle help overlay |

### Global JS State Variables

```javascript
_allSegs     // full array of segment objects from API
_visibleIdx  // filtered array of indices into _allSegs
_activeIdx   // currently selected segment index (in _visibleIdx)
_filter      // current phase filter string or "all"
_selectedFile // File object from the input/drop
_playTimer   // setInterval handle for auto-play
```

### Phase Styles (`PHASE_STYLES`)

```javascript
const PHASE_STYLES = {
  idle:                   { col: '#94A3B8', label: 'Idle' },
  processing:             { col: '#A78BFA', label: 'Processing' },
  live_view_monitoring:   { col: '#38BDF8', label: 'Live View' },
  tile_scan_acquisition:  { col: '#34D399', label: 'Tile Scan' },
}
```

---

## 12. Data & Training Scenarios

### Training Scenarios (S1–S10, S13)

| Code | Name | Key inefficiency |
|------|------|-----------------|
| S1 | High-Energy Continuous Supervision | Live view on for 2+ min inactivity, high overlap |
| S2 | Low-Energy Deferred Batch | Efficient baseline — no_action expected |
| S3 | Frequent Short Scans | Rapid acquisitions, typically efficient |
| S4 | Live View Left On Monitoring | Live view running during unattended monitoring |
| S5 | Large-Area Batch Reconstruction | Multi-region tile scans, overlap waste |
| S6 | Critical Live View Supervision | User-attended — no recommendation expected |
| S7 | High Overlap Required Quality | High-quality scan, overlap justified |
| S8 | Post-Acquisition Reconstruction Only | Processing phase, typically efficient |
| S9 | Post-Experiment Idle Waste | System left powered after experiment ends |
| S10 | Confounded Background Load | Background processing during live monitoring |
| S13 | Low-Priority Screening Mode | Tile overlap waste + idle live view |

### Raw Telemetry Column Schema

Each `S*_v4.csv` file contains 15-second rows with (at minimum):

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | str | Unique session identifier |
| `workflow_block_id` | int | Segment identifier within session |
| `workflow_phase` | str | Phase name (idle, processing, live_view_monitoring, tile_scan_acquisition) |
| `quality_constraint_mode` | str | Scan quality level (low/medium/high) |
| `live_view_enabled` | bool | Whether live view is active |
| `user_interacting` | bool | Whether operator is interacting |
| `monitoring_required` | bool | Whether monitoring is flagged |
| `seconds_since_last_ui_interaction` | float | Inactivity counter in seconds |
| `tile_overlap_pct` | float | Current tile overlap percentage |
| `estimated_system_power_w` | float | Total system power in watts |
| `perf_gpu_power_w` | float | GPU sub-component power |
| `estimated_energy_wh_interval` | float | Energy for this 15s interval |
| `processing_items_in_flight` | int | Active compute jobs |
| `recommended_action` | str | Ground-truth label (training only) |

### `test_segments.csv`

Pre-aggregated segment file used by `/analyze-sample`. Contains one row per segment with `scenario_code` and `scenario_name` columns for filtering. Schema matches the output of `aggregate_to_segments()`.

---

## 13. API Reference

### `GET /health`

```json
{
  "status": "ok",
  "model_ready": true,
  "model_accuracy": 0.9327,
  "model_f1": 0.9068
}
```

### `POST /analyze`

**Request:** `multipart/form-data` with a `file` field containing a CSV.

**Response:**

```json
{
  "scenario_name": "my_workflow",
  "total_energy_wh": 145.72,
  "total_savings_wh": 24.26,
  "savings_pct": 16.65,
  "spike_count": 3,
  "issues_count": 7,
  "segments": [
    {
      "segment_id": "42",
      "phase": "live_view_monitoring",
      "duration_sec": 2700,
      "avg_power_w": 198.4,
      "total_energy_wh": 14.88,
      "overlap_pct": 18.0,
      "live_view_share": 0.0,
      "user_interaction_share": 0.0,
      "monitoring_required_share": 0.08,
      "quality": "medium",
      "power_vs_baseline": 18.4,
      "is_energy_spike": false,
      "spike_magnitude_w": 18.4,
      "issue": "Live-view monitoring phase with only 8% actual monitoring...",
      "recommended_action": "pause_live_view",
      "action_reason": "Live-view monitoring phase with only 8% actual monitoring...",
      "estimated_savings_wh": 13.5,
      "confidence": 0.852,
      "rule_id": "R3",
      "mlp_probabilities": {
        "no_action": 0.12,
        "pause_live_view": 0.71,
        "optimize_tile_scan_settings": 0.17
      }
    }
  ]
}
```

### `GET /analyze-sample?name=S1`

Same response structure as `/analyze`. Runs the full pipeline on the bundled `test_segments.csv` data for the specified scenario code.

**Error codes:**
- `400` — not a CSV file, or empty file
- `404` — unknown scenario code
- `422` — could not parse CSV, or no segments found
- `500` — internal server error (demo data missing)

---

## 14. Design System

### Brand

- **Primary colour:** `#009FE3` (ZEISS blue)
- **Font:** Inter (Google Fonts) — `font-display: swap`
- **Mode:** Light by default, dark toggle available

### Severity Colour Coding

| Severity | Colour | CSS var | Used for |
|----------|--------|---------|----------|
| Efficient | `#10B981` | `--ok` | `no_action` segments |
| Warning | `#F59E0B` | `--warn` | `optimize_tile_scan_settings` |
| Spike/Error | `#F43F5E` | `--spike` | `pause_live_view`, energy spikes |
| Brand | `#009FE3` | `--zeiss` | Header, links, accents |

### Typography Scale

- Labels: `text-[10.5px]` uppercase tracking-widest
- Body: `text-[13px]`
- Card headers: `text-[12.5px] font-600`
- KPI numbers: `text-[20px] font-700 t-mono`
- Verdict headline: `text-[20–22px] font-700`

### Component Patterns

- **Surface cards:** `class="surface shadow-soft p-5"` — uses `--surface` background, subtle shadow
- **Badges/pills:** `class="pill pill-ok|pill-spike|pill-warn"` — inline coloured labels
- **Buttons:** `class="btn btn-primary|btn-ghost|btn-icon"` — consistent sizing, hover states
- **Tags:** `class="tag"` — inline category chips with custom background + text colours
- **Segment cards:** `class="seg-card seg-ok|seg-spike|seg-warn"` — left-border coloured by severity

---

## 15. Key Constants & Thresholds

| Constant | Value | File | Purpose |
|----------|-------|------|---------|
| `INACTIVITY_THRESHOLD_S` | 120.0 s | rule_engine.py | Trigger threshold for R1/R5 |
| `R4_INACTIVITY_THRESHOLD` | 90.0 s | rule_engine.py | Trigger for R4 mandatory tier |
| `LIVE_VIEW_POWER_OVERHEAD_W` | 18.0 W | energy_calculator.py | Power saved by pausing live view |
| `TILE_OVERLAP_POWER_PER_PCT_W` | 1.2 W/% | energy_calculator.py | Power per % of excess overlap |
| `SPIKE_THRESHOLD_W` | 20.0 W | ml_model.py | Excess above baseline to flag spike |
| `RULE_WEIGHT_MANDATORY` | 0.70 | hybrid_engine.py | Weight for conf ≥ 0.93 rules |
| `RULE_WEIGHT_HARD` | 0.45 | hybrid_engine.py | Weight for conf ≥ 0.90 rules |
| `RULE_WEIGHT_SOFT` | 0.28 | hybrid_engine.py | Weight for conf < 0.90 rules |
| `MLP_WEIGHT` | 0.60 | hybrid_engine.py | ML classifier weight |
| Phase baseline idle | 135 W | data_processor.py | Expected power during idle |
| Phase baseline processing | 200 W | data_processor.py | Expected power during processing |
| Phase baseline live_view | 180 W | data_processor.py | Expected power during live view |
| Phase baseline tile_scan | 170 W | data_processor.py | Expected power during tile scan |
| Optimal overlap low | 10.0% | data_processor.py | Min overlap for low-quality scans |
| Optimal overlap medium | 12.0% | data_processor.py | Min overlap for standard scans |
| Optimal overlap high | 18.0% | data_processor.py | Min overlap for high-quality scans |

---

## 16. Decision Flow — End to End

Below is the complete path from a CSV upload to a rendered result card.

```
1. User drops CSV on upload zone
   └─ selectFile(f) validates .csv extension, stores in _selectedFile

2. User clicks "Analyze"
   └─ runAnalysis() → POST /analyze with FormData

3. FastAPI /analyze endpoint
   ├─ validate: is it a CSV? is it non-empty?
   └─ _run_analysis(raw_bytes, scenario_name)
       ├─ detect_and_load(raw_bytes)
       │   ├─ pd.read_csv(raw_bytes)
       │   ├─ has "phase_segment_id"?
       │   │   ├─ YES → _normalise_aggregated(df) → (df, is_raw=False)
       │   │   └─ NO  → aggregate_to_segments(df, has_label=False) → (df, is_raw=True)
       │   └─ returns (segments_df, was_raw)
       │
       └─ for each segment row:
           └─ analyze_segment(row.to_dict())
               ├─ rule_engine.evaluate(segment)
               │   ├─ Check R1: inactivity + live_view + monitoring?
               │   ├─ Check R2: idle + live_view?
               │   ├─ Check R3: lv_monitoring + no_monitoring + no_interaction?
               │   ├─ Check R4: tile_scan + excess_overlap > 2%?
               │   ├─ Check R5: processing + live_view + inactivity?
               │   └─ Default: no_action, conf=0.50
               │
               ├─ ml_model.model.predict(segment)
               │   ├─ _build_feature_row(segment)  [phase-context correction]
               │   ├─ scaler.transform(features)
               │   ├─ mlp.predict_proba(scaled)
               │   ├─ spike = power_vs_baseline > 20W
               │   └─ returns {action, probabilities, is_spike, ...}
               │
               ├─ Score fusion
               │   ├─ rule = no_action? → pure ML scores
               │   └─ rule fired? → tiered blend
               │       ├─ conf ≥ 0.93 → rule×0.70 + ml×0.60
               │       ├─ conf ≥ 0.90 → rule×0.45 + ml×0.60
               │       └─ conf < 0.90 → rule×0.28 + ml×0.60
               │
               ├─ final_action = argmax(blended scores)
               │
               └─ energy_calculator.calculate_savings(segment, final_action)
                   ├─ pause_live_view: 18W × share × dur / 3600
                   └─ optimize_tile_scan: 1.2W × excess% × dur / 3600

4. JSON response returned to browser

5. renderResults(data)
   ├─ Populate KPI chips (energy, savings, %, spikes)
   ├─ verdict() → headline banner
   ├─ renderFilterChips() → phase filter row
   ├─ renderWaveform() → power timeline
   ├─ renderSegments() → segment card list
   ├─ body.classList.add("view-results")
   └─ activateSegment(0) → highlight first segment
```
