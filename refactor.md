# ZeroCausal — Repo Refactor Plan for GitHub

**Purpose:** Instructions for an AI (or human) to clean up and restructure `work2/` into a proper, publishable GitHub repository. Read every section before executing. Each step lists exact commands or file-by-file actions.

**Last updated:** 2026-06-19  
**Sections added in this revision:** files-to-delete audit (§1.5), Docker overhaul (§8), .dockerignore (§8.1)

---

## 0. Current State (what exists and why it's messy)

```
work2/
├── 30+ Python scripts dumped at root (numbered 00–15 + misc)
├── 9 reference PDFs at root (arxiv + journal papers)
├── 2 ZIP archives at root (BETH.zip, streamspot.zip) — raw data backups
├── ZeroCausal_Paper.tex + LaTeX artifacts at root (.aux .bbl .blg .pdf)
├── refs.bib at root (duplicate of paper/refs.bib)
├── pdflatex_out.txt at root (stray log)
├── sections/ at root — DUPLICATE of thesis/sections/ and paper/sections/
├── optc_edges.csv at root — a raw data file that belongs in data/
├── zerocausal_architecture_v2.png at root — belongs in plots/
├── zerocausal_core.py at root — the main package file
├── venv/ directory (old, unused virtual environment)
├── Multiple .md files scattered at root
├── results/ — contains hundreds of .pkl cache files (should NOT be in git)
├── logs/ — log files (should NOT be in git)
└── paper/ and thesis/ both have their own sections/ (3 copies total)
```

**What to keep in git:** source code, configs, final results, plots, paper/thesis LaTeX source, docs  
**What NOT to keep in git:** data files, pkl caches, log files, old venvs, zip archives, LaTeX build artifacts

---

## 1. Target Directory Structure

```
work2/
├── .gitignore                     # NEW — critical first step
├── README.md                      # UPDATE — fix paths after restructure
├── requirements.txt               # KEEP AS-IS
├── Dockerfile                     # KEEP AS-IS
│
├── zerocausal/                    # NEW Python package
│   ├── __init__.py                # NEW (empty or minimal)
│   └── core.py                    # RENAMED from zerocausal_core.py
│
├── scripts/                       # ALL pipeline Python scripts MOVED here
│   ├── preprocess/
│   │   ├── 00_preprocess_optc.py
│   │   ├── 00b_preprocess_streamspot.py
│   │   └── 00c_preprocess_beth.py
│   ├── download/
│   │   ├── 01_download_data.py
│   │   ├── 01b_download_streamspot.py
│   │   └── 01c_download_beth.py
│   ├── evaluate/
│   │   ├── 05_evaluate_zerocausal.py
│   │   ├── 09_evaluate_additional_datasets.py
│   │   ├── 12_contamination_sweep.py
│   │   ├── 13_drift_fpr_comparison.py
│   │   ├── 14_novel_evaluation.py
│   │   └── 15_beat_papers_evaluation.py
│   ├── plot/
│   │   ├── 08_plot_results.py
│   │   ├── 10_plot_comparisons.py
│   │   ├── generate_architecture_diagram.py
│   │   ├── generate_latex.py
│   │   └── plot_v1_vs_v2.py
│   ├── analysis/
│   │   ├── 07_offline_tuning.py
│   │   ├── 11_sensitivity_analysis.py
│   │   └── run_all_seeds.py
│   └── utils/                     # Shared dependencies (actually imported by evals)
│       ├── baselines.py           # KEEP — imported by 5 evaluation scripts
│       └── compute_operational_metrics.py  # KEEP — standalone metric reporting
│
├── bin/                           # Shell scripts (ALL NEW or MOVED)
│   ├── setup.sh                   # NEW — create venv, install deps
│   ├── preprocess.sh              # NEW — run all preprocess scripts
│   ├── run_eval.sh                # MOVED/UPDATED from run_all_evaluations.sh
│   └── run_plots.sh               # NEW — generate all plots
│
├── configs/                       # KEEP AS-IS
│   ├── default.yaml
│   └── tuned.yaml
│
├── paper/                         # LaTeX paper (CLEANED UP)
│   ├── ZeroCausal_Paper.tex       # MOVED from root
│   ├── refs.bib                   # KEEP (remove duplicate from root)
│   ├── Makefile                   # KEEP AS-IS
│   └── sections/                  # KEEP AS-IS (already here)
│       ├── abstract.tex ... (11 section files)
│
├── thesis/                        # LaTeX thesis (CLEANED UP)
│   ├── main.tex                   # KEEP AS-IS
│   ├── refs.bib                   # KEEP AS-IS
│   ├── Makefile                   # KEEP AS-IS
│   ├── sections/                  # KEEP AS-IS
│   ├── chapters/                  # KEEP AS-IS
│   └── figures/                   # KEEP AS-IS
│
├── docs/                          # NEW — consolidate all documentation
│   ├── architecture.md            # MOVED from root
│   ├── Artifact_Appendix.md       # MOVED from root
│   ├── Cover_Letter.md            # MOVED from root
│   ├── simulation_protocols.md    # MOVED from root
│   └── references/                # NEW — reference PDFs
│       ├── 2501.06997v2.pdf
│       ├── 2502.08963v1.pdf
│       ├── 2510.15188v2.pdf
│       ├── 2603.07560v1.pdf
│       ├── 2604.17870v1.pdf
│       ├── 1-s2.0-S0167404825002779-main.pdf
│       ├── 1-s2.0-S1389128625005195-main.pdf
│       ├── CausalGraph_When_Causal_Reasoning_Meets_Large_Language_Models_for_Intrusion_Detection_Systems.pdf
│       ├── Causal-IDS_Detecting_Network_Intrusions_as_Causal_Mechanism_Violations.pdf
│       ├── LiNGAM-SF_Causal_Structural_Learning_Method_With_Linear_Non-Gaussian_Acyclic_Models_for_Streaming_Features.pdf
│       └── TraceCluster_A_Lightweight_and_Adaptive_Clustering-Based_Subgraph_Attention_Network_for_APT_Detection_in_Provenance_Graphs.pdf
│
├── data/                          # NOT in git (gitignored)
│   ├── raw/                       # KEEP AS-IS (beth/, streamspot/)
│   └── processed/                 # KEEP AS-IS (beth_edges.csv, streamspot_edges.csv, optc_edges.csv MOVED here)
│
├── results/                       # Partially in git
│   └── final/                     # KEEP in git (JSON results only, no pkl)
│   # All .pkl, .csv, .log files OUTSIDE final/ are gitignored
│
└── plots/                         # KEEP in git (PNG outputs)
    # zerocausal_architecture_v2.png MOVED here from root
```

---

## 2. Step-by-Step Execution Plan

Work inside `/DATA/shourya_2211mc14/Arp/work2/` for all commands.

### STEP 1 — Create .gitignore (do this first)

Create file `work2/.gitignore` with this exact content:

```gitignore
# Virtual environments
venv/
.venv/
.venv_zc/
*.egg-info/

# Python cache
__pycache__/
*.pyc
*.pyo

# Data files (too large for git)
data/raw/
data/processed/
*.zip
*.csv
optc_edges.csv

# Experiment caches and logs
logs/
results/pcmci_cache_*.pkl
results/*.pkl
results/*.log
results/*.csv
# Keep results/final/ JSON summaries
!results/final/
!results/final/*.json

# LaTeX build artifacts
*.aux
*.bbl
*.blg
*.log
*.out
*.toc
*.lof
*.lot
pdflatex_out.txt
paper/CausalML_Hybrid_Ensemble_Paper.*

# Old archives
BETH.zip
streamspot.zip
thesis-1.pdf

# Stray text extracts
*.txt
# except known docs
!2502.08963v1.txt
!2510.15188v2.txt
!2603.07560v1.txt
!2604.17870v1.txt
!CausalGraph_*.txt
!Causal-IDS_*.txt
!LiNGAM-SF_*.txt
!TraceCluster_*.txt

# IDE
.vscode/
*.code-workspace
```

> **Note to human:** Review the .gitignore carefully. The `*.csv` rule above will also ignore `optc_edges.csv`. If you want some CSVs in git (e.g., final benchmark tables), add `!results/final/*.csv`.

---

### STEP 1.5 — Files to DELETE (unused / superseded / junk)

Before moving anything, delete these first so they don't pollute the new structure.

#### Hard deletes — confirmed unused (no script imports or calls them)

```bash
# generate_latex.py — hardcoded LaTeX strings from v1 paper draft.
# The real paper source is paper/ZeroCausal_Paper.tex. This file does nothing.
rm generate_latex.py

# 02_build_provenance_graph.py — early prototype before zerocausal_core.py existed.
# Superseded entirely by zerocausal_core.py + 09_evaluate_additional_datasets.py.
rm 02_build_provenance_graph.py

# 03_zero_causal_detector.py — v1 detector prototype. Same logic lives in zerocausal_core.py.
rm 03_zero_causal_detector.py

# 04_verify_synthetic_anomaly.py — only prints a note saying to run 03_zero_causal_detector.py.
# Nothing calls it; no useful logic.
rm 04_verify_synthetic_anomaly.py

# check_split.py — one-off debug script. Reads streamspot_edges.csv and prints stats.
# Not referenced anywhere.
rm check_split.py

# diagnose_distributions.py — one-off debug/diagnostic script. Not referenced anywhere.
rm diagnose_distributions.py

# fix_streamspot.py — one-off data fix. StreamSpot data is already fixed in data/processed/.
# Not referenced anywhere.
rm fix_streamspot.py

# 08_plot_results.py — early plotting script for v1 results format.
# 10_plot_comparisons.py is the current replacement. Not referenced anywhere.
rm 08_plot_results.py

# plot_v1_vs_v2.py — comparison plot between old v1 and v2. One-off analysis.
# Not referenced anywhere. Results already captured in plots/.
rm plot_v1_vs_v2.py

# generate_architecture_diagram.py — generates zerocausal_architecture_v2.png.
# The PNG already exists in plots/ (move it there per Step 5). Once in git, the
# script is not needed for reproducibility — the image is the deliverable.
# If you want to keep it for re-generation ability, move to scripts/plot/ instead.
# Decision: DELETE (image is already in plots/) or MOVE to scripts/plot/ (see below).
# Default recommendation: MOVE (don't delete — useful if figure needs updating).
mv generate_architecture_diagram.py scripts/plot/

# pdflatex_out.txt — stray LaTeX stdout dump. Useless.
rm pdflatex_out.txt

# thesis-1.pdf — compiled PDF output. Not source. Compiled by editors, not scripts.
rm thesis-1.pdf

# venv/ — old virtual environment. Dependencies tracked via requirements.txt.
rm -rf venv/
```

#### Stale .md files at root — consolidate into docs/

These are planning/session notes. Move them, not delete — they provide context.

```bash
# Do NOT delete: architecture.md, imp.md, postwork.md, todo.md, work.md,
# simulation_protocols.md, Artifact_Appendix.md, Cover_Letter.md
# → These all move to docs/ in Step 8.
```

#### Files that look unused but ARE active dependencies — do NOT delete

```bash
# baselines.py — imported by: 05_evaluate_zerocausal.py, 09_evaluate_additional_datasets.py,
#                12_contamination_sweep.py, 13_drift_fpr_comparison.py, 15_beat_papers_evaluation.py
# KEEP. Move to scripts/utils/ in Step 3.

# run_all_seeds.py — used as CMD in Dockerfile. KEEP.
# Move to scripts/analysis/ in Step 3, then update Dockerfile.

# compute_operational_metrics.py — standalone but may still be useful for paper numbers.
# Move to scripts/utils/ (don't delete without checking paper tables).
```

---

### STEP 2 — Create package directory and move core file

```bash
mkdir -p zerocausal
touch zerocausal/__init__.py
mv zerocausal_core.py zerocausal/core.py
```

Then update the `__init__.py` to re-export the main classes:

```python
# zerocausal/__init__.py
from .core import (
    ZeroCausalDetector,
    CausalFeatureExtractor,
    HybridAnomalyScorer,
    WeightedConformalCalibrator,
    RobustParCorr,
    KalmanSCM,
)
```

> **IMPORTANT:** After moving `zerocausal_core.py`, every script that does `import zerocausal_core` or `from zerocausal_core import ...` must be updated to `from zerocausal.core import ...` or `import zerocausal.core as zerocausal_core`. Search with:
> ```bash
> grep -rn "zerocausal_core" scripts/ scripts/evaluate/ scripts/analysis/
> ```
> Fix all hits.

---

### STEP 3 — Create scripts/ directory and move Python files

```bash
mkdir -p scripts/preprocess scripts/download scripts/evaluate scripts/plot scripts/analysis scripts/utils

# Preprocess
mv 00_preprocess_optc.py scripts/preprocess/
mv 00b_preprocess_streamspot.py scripts/preprocess/
mv 00c_preprocess_beth.py scripts/preprocess/

# Download
mv 01_download_data.py scripts/download/
mv 01b_download_streamspot.py scripts/download/
mv 01c_download_beth.py scripts/download/

# Evaluate
mv 05_evaluate_zerocausal.py scripts/evaluate/
mv 09_evaluate_additional_datasets.py scripts/evaluate/
mv 12_contamination_sweep.py scripts/evaluate/
mv 13_drift_fpr_comparison.py scripts/evaluate/
mv 14_novel_evaluation.py scripts/evaluate/
mv 15_beat_papers_evaluation.py scripts/evaluate/

# Plot (only these two survive; others were deleted in Step 1.5)
mv 10_plot_comparisons.py scripts/plot/
mv generate_architecture_diagram.py scripts/plot/

# Analysis
mv 07_offline_tuning.py scripts/analysis/
mv 11_sensitivity_analysis.py scripts/analysis/
mv run_all_seeds.py scripts/analysis/

# Utilities — only keep files that are actually imported by evaluation scripts
mv baselines.py scripts/utils/
mv compute_operational_metrics.py scripts/utils/

# Files already deleted in Step 1.5 — do NOT mv these:
# check_split.py, diagnose_distributions.py, fix_streamspot.py
# generate_latex.py, plot_v1_vs_v2.py, 08_plot_results.py
# 02_build_provenance_graph.py, 03_zero_causal_detector.py, 04_verify_synthetic_anomaly.py
```

> **Note:** Scripts 02, 03, 04 are early prototypes. Check if any current script imports from them before deciding to move vs. delete.

---

### STEP 4 — Move data file from root

```bash
mv optc_edges.csv data/processed/
```

---

### STEP 5 — Move stray image from root

```bash
mv zerocausal_architecture_v2.png plots/
```

---

### STEP 6 — Move LaTeX paper files from root into paper/

```bash
mv ZeroCausal_Paper.tex paper/
mv ZeroCausal_Paper.pdf paper/     # compiled output — optional to keep
rm -f ZeroCausal_Paper.aux ZeroCausal_Paper.bbl ZeroCausal_Paper.blg
rm -f refs.bib                     # duplicate — canonical copy is paper/refs.bib
rm -f pdflatex_out.txt
```

---

### STEP 7 — Delete stray duplicate sections/ at root

> First verify it is identical to thesis/sections/:
> ```bash
> diff -r sections/ thesis/sections/
> ```
> If no differences (or only whitespace), delete:
> ```bash
> rm -rf sections/
> ```
> If there ARE differences, merge the unique content into thesis/sections/ first, then delete.

---

### STEP 8 — Create docs/ and move documentation

```bash
mkdir -p docs/references

# Move markdown docs
mv architecture.md docs/
mv Artifact_Appendix.md docs/
mv Cover_Letter.md docs/
mv simulation_protocols.md docs/

# Move reference PDFs into docs/references/
mv 2501.06997v2.pdf docs/references/
mv 2502.08963v1.pdf docs/references/
mv 2510.15188v2.pdf docs/references/
mv 2603.07560v1.pdf docs/references/
mv 2604.17870v1.pdf docs/references/
mv 1-s2.0-S0167404825002779-main.pdf docs/references/
mv 1-s2.0-S1389128625005195-main.pdf docs/references/
mv "CausalGraph_When_Causal_Reasoning_Meets_Large_Language_Models_for_Intrusion_Detection_Systems.pdf" docs/references/
mv "Causal-IDS_Detecting_Network_Intrusions_as_Causal_Mechanism_Violations.pdf" docs/references/
mv "LiNGAM-SF_Causal_Structural_Learning_Method_With_Linear_Non-Gaussian_Acyclic_Models_for_Streaming_Features.pdf" docs/references/
mv "TraceCluster_A_Lightweight_and_Adaptive_Clustering-Based_Subgraph_Attention_Network_for_APT_Detection_in_Provenance_Graphs.pdf" docs/references/
```

Also move the .txt extracts of these papers (they are the plain-text versions):
```bash
mv 2502.08963v1.txt docs/references/
mv 2510.15188v2.txt docs/references/
mv 2603.07560v1.txt docs/references/
mv 2604.17870v1.txt docs/references/
mv "CausalGraph_"*.txt docs/references/
mv "Causal-IDS_"*.txt docs/references/
mv "LiNGAM-SF_"*.txt docs/references/
mv "TraceCluster_"*.txt docs/references/
```

---

### STEP 9 — Clean up old/stale items at root

```bash
# Delete old venv (packages are tracked via requirements.txt)
rm -rf venv/

# Remove data zip archives (data is downloaded via scripts, not stored in git)
rm -f BETH.zip streamspot.zip

# Remove stale thesis PDF (compiled output, not source)
rm -f thesis-1.pdf

# Remove __pycache__ at root
rm -rf __pycache__/
```

> **Ask the human before deleting** `BETH.zip` and `streamspot.zip` if these are the ONLY copies of the raw data. Confirm that `data/raw/beth/` and `data/raw/streamspot/` are already extracted and intact.

---

### STEP 10 — Create bin/ shell scripts

Create the following 5 shell scripts. All should be `chmod +x`.

#### `bin/setup.sh` — Environment setup
```bash
#!/bin/bash
# Creates .venv_zc and installs all dependencies.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "[setup] Creating virtual environment .venv_zc ..."
python3 -m venv .venv_zc
echo "[setup] Installing dependencies..."
.venv_zc/bin/pip install --upgrade pip
.venv_zc/bin/pip install -r requirements.txt
echo "[setup] Done. Activate with: source .venv_zc/bin/activate"
```

#### `bin/preprocess.sh` — Dataset preprocessing
```bash
#!/bin/bash
# Preprocesses all datasets. Run once after downloading data.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
PYTHON=".venv_zc/bin/python3"

echo "[preprocess] OpTC..."
$PYTHON scripts/preprocess/00_preprocess_optc.py

echo "[preprocess] StreamSpot..."
$PYTHON scripts/preprocess/00b_preprocess_streamspot.py

echo "[preprocess] BETH..."
$PYTHON scripts/preprocess/00c_preprocess_beth.py

echo "[preprocess] All done."
```

#### `bin/run_eval.sh` — Full evaluation pipeline
```bash
#!/bin/bash
# Runs the full ZeroCausal evaluation on all 5 datasets.
# Usage: nohup bash bin/run_eval.sh > logs/master_eval.log 2>&1 &
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
PYTHON=".venv_zc/bin/python3"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

echo "=== ZeroCausal Full Evaluation — $(date) ==="
echo $$ > "$LOG_DIR/master_eval.pid"
FAILURES=""

echo "[1/5] TC3..."
$PYTHON scripts/evaluate/09_evaluate_additional_datasets.py \
    --dataset tc3 --baseline --simulate-drift --std-floor 1.0 \
    --run-name tc3_default --checkpoint-interval 500 \
    2>&1 | tee "$LOG_DIR/tc3_eval_${TIMESTAMP}.log" || FAILURES="$FAILURES tc3"

echo "[2/5] NODLINK..."
$PYTHON scripts/evaluate/09_evaluate_additional_datasets.py \
    --dataset nodlink --baseline --std-floor 1.0 \
    --run-name nodlink_default --checkpoint-interval 500 \
    2>&1 | tee "$LOG_DIR/nodlink_eval_${TIMESTAMP}.log" || FAILURES="$FAILURES nodlink"

echo "[3/5] StreamSpot..."
$PYTHON scripts/evaluate/09_evaluate_additional_datasets.py \
    --dataset streamspot --baseline --train-limit 5000 \
    --run-name streamspot_default --checkpoint-interval 500 \
    2>&1 | tee "$LOG_DIR/streamspot_eval_${TIMESTAMP}.log" || FAILURES="$FAILURES streamspot"

echo "[4/5] BETH..."
$PYTHON scripts/evaluate/09_evaluate_additional_datasets.py \
    --dataset beth --baseline \
    --run-name beth_default --checkpoint-interval 500 \
    2>&1 | tee "$LOG_DIR/beth_eval_${TIMESTAMP}.log" || FAILURES="$FAILURES beth"

echo "[5/5] OpTC (tuned)..."
$PYTHON scripts/evaluate/05_evaluate_zerocausal.py \
    --a-p 0.3632 --b-p 3.5284 --a-r 5.8843 --b-r 0.1495 \
    --target-fpr 0.0822 --conformal-lr 0.0204 --run-name optc_tuned \
    2>&1 | tee "$LOG_DIR/optc_eval_${TIMESTAMP}.log" || FAILURES="$FAILURES optc"

echo ""
echo "=== Results ==="
for ds in tc3 nodlink streamspot beth optc; do
    SUMMARY="results/${ds}_default_summary.json"
    [ "$ds" = "optc" ] && SUMMARY="results/optc_tuned_summary.json"
    if [ -f "$SUMMARY" ]; then
        AUC=$($PYTHON -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d['metrics']['auc']:.4f}\")" 2>/dev/null || echo "N/A")
        echo "  $ds: AUC = $AUC"
    fi
done

[ -n "$FAILURES" ] && echo "FAILURES: $FAILURES" || echo "All evaluations completed."
rm -f "$LOG_DIR/master_eval.pid"
```

#### `bin/run_plots.sh` — Generate all plots
```bash
#!/bin/bash
# Generates evaluation figures after run_eval.sh completes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
PYTHON=".venv_zc/bin/python3"

echo "[plots] Architecture diagram..."
$PYTHON scripts/plot/generate_architecture_diagram.py

echo "[plots] Comparison ROC curves..."
$PYTHON scripts/plot/10_plot_comparisons.py

echo "[plots] Sensitivity analysis..."
$PYTHON scripts/analysis/11_sensitivity_analysis.py

echo "[plots] Done. Check plots/ directory."
```

Make all scripts executable:
```bash
chmod +x bin/setup.sh bin/preprocess.sh bin/run_eval.sh bin/run_plots.sh
```

---

### STEP 11 — Fix import paths in scripts

After moving `zerocausal_core.py` → `zerocausal/core.py` and all scripts into `scripts/`, the imports will break. Fix systematically:

**Find all broken imports:**
```bash
grep -rn "import zerocausal_core\|from zerocausal_core" scripts/
```

**Replace pattern:** Change every occurrence of:
- `import zerocausal_core` → `from zerocausal import core as zerocausal_core`
- `from zerocausal_core import X` → `from zerocausal.core import X`

**Fix sys.path if needed:** Some scripts may do `sys.path.insert(0, '..')` — update to point to the repo root. Or add a `pyproject.toml` to make `zerocausal` installable (see STEP 13).

**Check for relative data path assumptions:** Scripts may assume `data/` and `results/` are relative to the script location. Since scripts moved from root → `scripts/evaluate/`, add this near the top of each affected script:
```python
import os, sys
# ensure repo root is on path and cwd is repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
```
Or run all scripts from the repo root (the `bin/*.sh` scripts already `cd "$SCRIPT_DIR"` to do this).

---

### STEP 12 — Update README.md

Update `README.md` to reflect the new structure:
- Change venv activation to `source .venv_zc/bin/activate` (already correct in README but double-check)
- Change all script paths from `python3 05_evaluate_zerocausal.py` to `python3 scripts/evaluate/05_evaluate_zerocausal.py`
- Add section: **Quick Start** — `bash bin/setup.sh && bash bin/preprocess.sh && bash bin/run_eval.sh`
- Update the directory tree in the README

---

### STEP 13 — (Optional but recommended) Add pyproject.toml

Create `work2/pyproject.toml` so that the `zerocausal` package is importable from scripts without path hacks:

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "zerocausal"
version = "0.1.0"
description = "Zero-label causal APT detection for provenance graphs"
requires-python = ">=3.9"
dependencies = []

[tool.setuptools.packages.find]
where = ["."]
include = ["zerocausal*"]
```

Then in `bin/setup.sh`, add after pip install:
```bash
.venv_zc/bin/pip install -e .
```

This makes `from zerocausal.core import ...` work from any script without sys.path manipulation.

---

## 3. Markdown/Doc files at root — what to do with each

| File | Action |
|------|--------|
| `README.md` | KEEP at root, UPDATE paths |
| `architecture.md` | MOVE to `docs/architecture.md` |
| `Artifact_Appendix.md` | MOVE to `docs/Artifact_Appendix.md` |
| `Cover_Letter.md` | MOVE to `docs/Cover_Letter.md` |
| `simulation_protocols.md` | MOVE to `docs/simulation_protocols.md` |
| `imp.md` | MOVE to `docs/imp.md` (RCF implementation plan — useful reference) |
| `postwork.md` | MOVE to `docs/postwork.md` (session log — useful for context) |
| `todo.md` | MOVE to `docs/todo.md` |
| `work.md` | MOVE to `docs/work.md` |

---

## 4. Results directory — what to keep in git

The `results/` directory is large. Keep only what reproducibility requires:

**Keep in git (`results/final/`):**
- `*.json` summary files (small, contain final AUC/FPR metrics)

**Gitignore (add to .gitignore):**
- `results/*.pkl` — PCMCI caches (large, regenerated by running evaluation)
- `results/*.csv` — per-step data (large, regenerated)
- `results/*.log` — run logs

**Check results/final/ exists and has the key JSON files** before gitignoring everything else.

---

## 5. Verification Checklist (run after all steps)

After completing the restructure, verify everything still works:

```bash
# 1. Check package imports work
source .venv_zc/bin/activate
python3 -c "from zerocausal.core import ZeroCausalDetector; print('OK')"

# 2. Check no broken imports in evaluate scripts
python3 -c "import scripts.evaluate.05_evaluate_zerocausal"  # or run directly

# 3. Run a quick single-dataset eval to confirm end-to-end works
python3 scripts/evaluate/09_evaluate_additional_datasets.py \
    --dataset tc3 --baseline --run-name test_refactor --checkpoint-interval 9999

# 4. Check root is clean
ls *.py 2>/dev/null && echo "ERROR: Python files still at root" || echo "Root clean"
ls *.tex 2>/dev/null && echo "ERROR: LaTeX files still at root" || echo "Root clean"

# 5. Check git status
git status --short
```

---

## 6. Things to Decide (Human Input Required)

These require a judgment call before execution:

1. **`BETH.zip` and `streamspot.zip`**: Delete only if `data/raw/beth/` and `data/raw/streamspot/` are complete and intact. If these zips are the only backup, keep them somewhere outside the git repo (e.g., the parent Arp directory).

2. **Scripts 02/03/04**: `02_build_provenance_graph.py`, `03_zero_causal_detector.py`, `04_verify_synthetic_anomaly.py` are early-prototype scripts. Confirm they are superseded by `zerocausal/core.py` + `09_evaluate_additional_datasets.py` before deleting.

3. **`paper/CausalML_Hybrid_Ensemble_Paper.*`**: There is a second paper (`CausalML_Hybrid_Ensemble_Paper.tex`) inside `paper/`. Is this a separate paper or an earlier draft of `ZeroCausal_Paper.tex`? If it's obsolete, delete it. If it's a different submission, keep it and organize under `paper/causalml/`.

4. **`thesis/` vs `paper/`**: Both have `sections/` with identical file names. Are these actually the same files symlinked, or is one a copy that has diverged? Run `diff -r paper/sections/ thesis/sections/` to check.

5. **`run_all_evaluations.sh` (old)**: After creating `bin/run_eval.sh`, delete the old `run_all_evaluations.sh` at root.

---

## 8. Docker Overhaul

The current `Dockerfile` is a thin wrapper — it copies everything and runs `run_all_seeds.py`. Problems:
- Copies `data/`, `results/`, `logs/` into the image (bloat, sensitive)
- No `.dockerignore`
- Single hardcoded CMD — can't switch between eval/preprocess/plots without rebuilding
- No volume mounts defined — results disappear when the container exits

### 8.1 Create `.dockerignore`

Create `work2/.dockerignore` — this is the most important change. Prevents the image from including data, caches, and git history:

```dockerignore
# Git
.git/
.gitignore

# Virtual envs
venv/
.venv/
.venv_zc/

# Data (mount as volume at runtime)
data/

# Results and logs (mount as volume at runtime)
results/
logs/

# ZIP archives
*.zip

# Python cache
__pycache__/
*.pyc
*.pyo

# LaTeX build artifacts
*.aux
*.bbl
*.blg
*.toc
*.out
*.lof
*.lot
pdflatex_out.txt

# IDE
.vscode/
```

### 8.2 Rewrite `Dockerfile`

Replace the current `Dockerfile` with this multi-target version:

```dockerfile
# ─────────────────────────────────────────────────────
#  Stage 1: dependency builder
# ─────────────────────────────────────────────────────
FROM python:3.9-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─────────────────────────────────────────────────────
#  Stage 2: runtime image
# ─────────────────────────────────────────────────────
FROM python:3.9-slim AS runtime

WORKDIR /app

# Bring in compiled packages from builder (keeps runtime image small)
COPY --from=builder /install /usr/local

# Install runtime system libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source code (data/ and results/ are excluded by .dockerignore)
COPY zerocausal/ ./zerocausal/
COPY scripts/    ./scripts/
COPY bin/        ./bin/
COPY configs/    ./configs/
COPY requirements.txt .

# Mount points — data and results must be mounted at runtime
VOLUME ["/app/data", "/app/results", "/app/logs"]

# Default: run full multi-seed eval on OpTC
CMD ["python", "scripts/analysis/run_all_seeds.py"]
```

### 8.3 How to use Docker for each task

After the refactor, run specific tasks by overriding CMD with `-e` or by passing a command:

```bash
# ── Build the image ──
docker build -t zerocausal:latest .

# ── Preprocess datasets (mount your data dir) ──
docker run --rm \
  -v /path/to/your/data:/app/data \
  zerocausal:latest \
  bash bin/preprocess.sh

# ── Run full evaluation ──
docker run --rm \
  -v /path/to/your/data:/app/data \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/logs:/app/logs \
  zerocausal:latest \
  bash bin/run_eval.sh

# ── Run a single dataset (e.g., BETH) ──
docker run --rm \
  -v /path/to/your/data:/app/data \
  -v $(pwd)/results:/app/results \
  zerocausal:latest \
  python scripts/evaluate/09_evaluate_additional_datasets.py \
    --dataset beth --baseline --run-name beth_default

# ── Generate plots (results must already exist) ──
docker run --rm \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/plots:/app/plots \
  zerocausal:latest \
  bash bin/run_plots.sh

# ── Interactive shell for debugging ──
docker run --rm -it \
  -v /path/to/your/data:/app/data \
  -v $(pwd)/results:/app/results \
  zerocausal:latest \
  bash
```

### 8.4 Create `docker-compose.yml` for convenience

Create `work2/docker-compose.yml` so you don't have to type long docker run commands:

```yaml
version: "3.9"

x-common: &common
  image: zerocausal:latest
  build: .
  volumes:
    - ./data:/app/data
    - ./results:/app/results
    - ./logs:/app/logs
    - ./plots:/app/plots

services:
  preprocess:
    <<: *common
    command: bash bin/preprocess.sh

  eval:
    <<: *common
    command: bash bin/run_eval.sh

  eval-beth:
    <<: *common
    command: >
      python scripts/evaluate/09_evaluate_additional_datasets.py
      --dataset beth --baseline --run-name beth_default

  eval-optc:
    <<: *common
    command: >
      python scripts/evaluate/05_evaluate_zerocausal.py
      --a-p 0.3632 --b-p 3.5284 --a-r 5.8843 --b-r 0.1495
      --target-fpr 0.0822 --conformal-lr 0.0204 --run-name optc_tuned

  plots:
    <<: *common
    command: bash bin/run_plots.sh

  seeds:
    <<: *common
    command: python scripts/analysis/run_all_seeds.py

  shell:
    <<: *common
    command: bash
    stdin_open: true
    tty: true
```

Usage after creating docker-compose.yml:
```bash
docker compose build            # build image once
docker compose run --rm eval    # run full evaluation
docker compose run --rm plots   # generate plots
docker compose run --rm shell   # debug shell
```

### 8.5 Fix Dockerfile CMD after moving run_all_seeds.py

The current `CMD ["python", "run_all_seeds.py"]` will break after the refactor because the file moves to `scripts/analysis/run_all_seeds.py`. The new Dockerfile above already uses the correct path. No further action needed beyond replacing the old Dockerfile with the new one.

---

## 7. Git Commit Strategy

Do the refactor in 3 commits to keep history clean:

```
Commit 1: "chore: add .gitignore and remove gitignored files from tracking"
Commit 2: "refactor: restructure work2 into package layout (scripts/, zerocausal/, bin/, docs/)"
Commit 3: "fix: update import paths after zerocausal_core.py → zerocausal/core.py rename"
```

Use `git mv` instead of `mv` to preserve file history:
```bash
git mv zerocausal_core.py zerocausal/core.py
git mv 05_evaluate_zerocausal.py scripts/evaluate/05_evaluate_zerocausal.py
# etc.
```
