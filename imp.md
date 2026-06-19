# ZeroCausal — Robust Calibration Filter (RCF) Implementation Plan

---

## The Core Problem (one paragraph)

ZeroCausal's causal discovery backbone (PCMCI) is stable under contaminated training data — the learned causal graph barely changes at 1–5% contamination. But the **H-CAS calibration CDF** — the empirical distribution of residual energies built from the training calibration partition — absorbs every window fed into it, including attack-like contaminated windows. When those contaminated windows inflate the calibration baseline, real test attack windows score no higher than the inflated calibration scores, destroying the ranking signal. The result: AUC drops from 0.982 (clean) to 0.674 at just 1% contamination, and operational recall at 5% FPR is only 24.21% on real OpTC data. The fix is a **filter layer that gates what enters the calibration queue** — using techniques already established in the literature.

---

## What Each Related Paper Contributes to the Filter

| Paper | Core Technical Idea | How We Borrow It |
|-------|-------------------|-----------------|
| **Causal-IDS** (ICOIN 2026) | SCM mechanism violations detected via causal p-values. Attacks = external interventions that break learned causal laws | Use the already-computed causal p-values to self-screen calibration candidates. If a window itself violates causal laws (low min p-value), it is likely contamination — exclude it from calibration |
| **OCR-APT** (CCS 2025) | One-class RGCN + OCSVM trained on structural behavior patterns to distinguish normal vs. anomalous nodes without node attributes | Adapt the one-class classification concept: maintain a streaming Tukey-fence one-class energy gate using running IQR of H-CAS scores. Windows whose energy lies outside the normal energy envelope are excluded |
| **TraceCluster** (IEEE TIFS 2026) | Clustering-based subgraph partitioning with adaptive category weighting to handle imbalanced normal/attack data | Use density clustering on the rolling calibration feature vectors. Only windows that fall inside the dominant normal cluster are admitted. Outlier-cluster windows (rare and anomalous) are excluded |
| **LiNGAM-SF** (IEEE TNNLS 2025) | Latent confounder detection by distinguishing uniform residual correlation (hidden common cause) from selective residual spikes (direct mechanism violation) | Use the residual structure: if residuals are uniformly elevated across all features, the window is a global confounder event (benign system load spike) → admit. If only specific features have high residuals → mechanism violation → exclude |
| **ModePlait** (arXiv 2502.08963) | Time-evolving causality: detects regime transitions (dynamical pattern switches) in streaming data to track when causal structure changes | Use the already-existing AdaptiveWindowDetector change-point signal. Exclude calibration candidates that fall within a transition buffer (N steps after a detected change-point) since the SCM baseline is in flux and these windows do not represent stable normal behavior |
| **CausalGraph-IDS** (ICCE 2026) | Conformal Risk Control (CRC) + influence-driven counterfactual testing to verify whether alerts persist under feasible interventions | Use the influence function concept: before permanently adding a score to the calibration queue, test its influence on the calibration distribution's key quantiles (Q95). High-influence windows are statistical outliers in the calibration set — roll them back |

---

## The Proposed Layer: Robust Calibration Filter (RCF)

The RCF sits **between the raw streaming window and the conformal calibration queue**. It is a four-stage sequential pipeline. A window must pass ALL active stages to be admitted to the calibration queue. Rejected windows are still scored and alarmed normally — they are simply not used to update the reference distribution.

```
Raw Window
    │
    ▼
[Stage 1] Causal Self-Gate          ← from Causal-IDS
    │  (uses existing PCMCI p-values, zero extra cost)
    ▼
[Stage 2] One-Class Energy Gate     ← from OCR-APT + TraceCluster
    │  (streaming Tukey fence on H-CAS energy)
    ▼
[Stage 3] Residual Structure Gate   ← from LiNGAM-SF
    │  (confounder vs. mechanism violation discrimination)
    ▼
[Stage 4] Transition Buffer         ← from ModePlait
    │  (cooldown after detected change-points)
    ▼
[ADMIT to Calibration Queue]
```

---

## Stage-by-Stage Technical Specification

### Stage 1 — Causal Self-Gate (Causal-IDS)

**What it does:** Any window that itself shows a significant causal mechanism violation should not be used to teach the system what "normal" looks like.

**How:** The min causal p-value `p_min = min_j p_val^j` is already computed for every window during scoring. Before adding a window to calibration:
```
if p_min < alpha_gate:        # default alpha_gate = 0.10
    REJECT (likely mechanism violation in training data)
else:
    PASS to Stage 2
```

**Why 0.10 not 0.01:** The calibration gate should be lenient (exclude only clear violations). Using the same alpha as PCMCI (0.01) would be too strict and starve the calibration queue.

**Cost:** Zero — `p_min` is already computed.

---

### Stage 2 — One-Class Energy Gate (OCR-APT + TraceCluster)

**What it does:** OCR-APT uses one-class SVM to classify nodes as normal or anomalous based purely on behavioral patterns (not node attributes). TraceCluster uses adaptive category weighting to handle imbalanced normal/attack data. We adapt this as a streaming one-class energy test.

**How:** Maintain a rolling buffer of the last `W_iqr` (default 500) admitted calibration H-CAS energy scores. Compute Q1 and Q3. Apply Tukey fence:
```
IQR = Q3 - Q1
upper_fence = Q3 + k * IQR        # default k = 1.5
lower_fence = Q1 - k * IQR

if h_cas_energy > upper_fence:    # anomalously high energy
    REJECT (outlier in energy space)
else:
    PASS to Stage 3
```

**Why this maps to OCR-APT/TraceCluster:** OCR-APT's OCSVM draws a boundary around normal behavioral patterns and excludes everything outside it. Tukey fence does the same for the scalar H-CAS energy distribution, which is the summary of all behavioral features. TraceCluster's adaptive weighting handles imbalance — by using Q3+1.5*IQR rather than a fixed threshold, the fence automatically adapts to whatever normal energy level the current data exhibits.

**Cold start:** For the first `W_iqr` windows, use the initial calibration scores from `train_calib` to seed the Q1/Q3 estimates.

---

### Stage 3 — Residual Structure Gate (LiNGAM-SF)

**What it does:** LiNGAM-SF's key insight is distinguishing **latent confounders** (unobserved variables that uniformly inflate all features) from **direct mechanism violations** (specific features with anomalous residuals). Confounders are benign global events (e.g., system-wide load spike, NTP sync). Mechanism violations are attacks. We use this distinction to avoid excluding benign confounder windows from calibration.

**How:** Compute the z-score vector `z_j = residual_j / sigma_j` for all features. Then:
```
z_max  = max(|z_j|)          # peak violation
z_mean = mean(|z_j|)         # average elevation

spike_ratio = z_max / (z_mean + 1e-9)

if spike_ratio > spike_threshold:   # default = 3.0
    # One or few features dominate → mechanism violation → REJECT
    REJECT
else:
    # All features uniformly elevated → global confounder → ADMIT
    PASS to Stage 4
```

**Why this maps to LiNGAM-SF:** LiNGAM-SF proves (Property 1) that each edge in a streaming causal graph is one of three structures: direct causal, latent confounder, or both. Windows where a latent common cause drives ALL features produce uniformly elevated residuals (low spike_ratio). Windows where an attacker intervenes on specific edges produce selective spikes (high spike_ratio). This directly operationalizes LiNGAM-SF's confounder detection logic.

---

### Stage 4 — Transition Buffer (ModePlait)

**What it does:** ModePlait models time-evolving causality — detecting when the causal regime switches between distinct dynamical patterns. Windows during a regime transition do not represent any stable state and should not anchor the calibration distribution.

**How:** When `AdaptiveWindowDetector` fires a change-point, set a cooldown counter:
```
if change_point_detected:
    cooldown_remaining = transition_buffer    # default = 30 steps

if cooldown_remaining > 0:
    cooldown_remaining -= 1
    REJECT (in transition period — causal regime unstable)
else:
    ADMIT (pass to calibration queue)
```

**Why this maps to ModePlait:** ModePlait explicitly models "transitions of distinct dynamical patterns" as a core component of streaming causal discovery. During transitions, the causal structure is undefined — the old structure has been invalidated and the new one has not yet stabilized. Adding calibration scores from this period would be analogous to ModePlait including transition-regime data in a pattern model, which it explicitly avoids.

---

## Implementation Location in Codebase

| What to add | Where |
|-------------|-------|
| `RobustCalibrationFilter` class with 4-stage pipeline | `zerocausal_core.py` (new class after `ConformalCalibrator`) |
| Instantiate RCF in evaluation loop | `05_evaluate_zerocausal.py` and `09_evaluate_additional_datasets.py` |
| Add `--rcf` flag and per-stage enable/disable flags | Both evaluation scripts' argparse sections |
| Log filter admission rate per step | `steps_log` dict (add `rcf_admitted` field) |
| Add RCF stats to summary JSON | `summary['rcf_stats']` block |

### New class skeleton in `zerocausal_core.py`:
```python
class RobustCalibrationFilter:
    """
    Four-stage filter that gates what enters the conformal calibration queue.
    Stages: Causal Self-Gate, One-Class Energy Gate,
            Residual Structure Gate, Transition Buffer.
    """
    def __init__(self,
                 alpha_gate=0.10,        # Stage 1: causal p-value threshold
                 k_iqr=1.5,             # Stage 2: Tukey fence multiplier
                 w_iqr=500,             # Stage 2: rolling window for IQR
                 spike_threshold=3.0,   # Stage 3: spike ratio threshold
                 transition_buffer=30): # Stage 4: post-changepoint cooldown
        ...

    def update_energy_stats(self, h_cas_energy):
        # Update rolling Q1, Q3 for Stage 2

    def set_changepoint(self):
        # Called by AdaptiveWindowDetector on drift detection
        self.cooldown_remaining = self.transition_buffer

    def admit(self, p_min, h_cas_energy, z_scores) -> bool:
        # Returns True if window passes all 4 stages
        # Stage 1
        if p_min < self.alpha_gate:
            return False
        # Stage 2
        if h_cas_energy > self.q3 + self.k_iqr * self.iqr:
            return False
        # Stage 3
        z_max = np.max(np.abs(z_scores))
        z_mean = np.mean(np.abs(z_scores))
        if z_max / (z_mean + 1e-9) > self.spike_threshold:
            return False
        # Stage 4
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return False
        # Admitted — update energy stats
        self.update_energy_stats(h_cas_energy)
        return True
```

---

## Evaluation Plan

### Datasets
- **OpTC** (real, primary benchmark, 10 seeds)
- **TC3** (simulated, contamination sweep)
- **NODLINK** (simulated, drift experiment)
- **BETH** (real, cross-platform)

### Experiments

#### Experiment A — Core Metric Improvement (OpTC, 10 seeds)

| Configuration | AUC | Recall@5%FPR | FPR@95%Recall |
|--------------|-----|--------------|----------------|
| ZeroCausal (current paper) | 0.727 ± 0.071 | 24.21% | 93.40% |
| ZeroCausal + RCF (full) | *run* | *run* | *run* |

Report mean ± std over the same 10 seeds (42–51).

#### Experiment B — Contamination Sweep with and without RCF (TC3)

| Contamination | ZeroCausal | ZeroCausal + RCF | IF | AE |
|--------------|------------|------------------|----|-----|
| 0% | 0.982 | *run* | 0.866 | 1.000 |
| 1% | 0.674 | *run* | 0.855 | 0.975 |
| 5% | 0.659 | *run* | 0.808 | 0.843 |
| 10% | 0.612 | *run* | 0.749 | 0.691 |
| 20% | 0.552 | *run* | 0.679 | 0.604 |

**Goal:** ZeroCausal + RCF should degrade more gracefully, staying above IF at 5%+ contamination.

#### Experiment C — Stage Ablation (TC3, seed 42, 5% contamination)

Run each stage individually to measure contribution:

| Configuration | AUC @ 5% contamination |
|--------------|------------------------|
| ZeroCausal (no RCF) | 0.659 |
| + Stage 1 only (Causal Self-Gate) | *run* |
| + Stage 2 only (Energy Gate) | *run* |
| + Stage 3 only (Residual Structure Gate) | *run* |
| + Stage 4 only (Transition Buffer) | *run* |
| + All 4 stages (full RCF) | *run* |

#### Experiment D — Calibration Queue Health Analysis

At each evaluation step, log:
- `rcf_admitted` (bool): did this window enter calibration?
- `rcf_rejection_stage` (int 1–4): which stage rejected it?
- `calib_queue_size`: how many windows are currently in queue

This gives a visualization of how often each gate fires and whether the calibration queue starves (too many rejections) or floods (too few).

#### Experiment E — Drift Adaptation with RCF (TC3, simulate_drift)

Re-run the concept drift experiment comparing:
- ZeroCausal (no RCF): post-drift FPR = 9.2%
- ZeroCausal + RCF: expected lower post-drift FPR since Stage 4 (transition buffer) prevents unstable transition windows from polluting the queue during SCM refit

---

## Paper Section Plan — What to Write

### New subsection in Section IV (System Design)
**"IV-F: Robust Calibration Filter (RCF)"**

- Motivation: contamination vulnerability in H-CAS calibration CDF (cite own contamination experiment)
- Cite Causal-IDS for self-gating idea, OCR-APT for one-class energy bounding, TraceCluster for cluster density, LiNGAM-SF for confounder vs. mechanism discrimination, ModePlait for regime-transition exclusion
- Present the 4-stage pipeline with equations
- Note computational cost: O(1) per window, zero additional models trained

### New subsection in Section VI (Evaluation)
**"VI-K: RCF Impact on Contamination Robustness"**

- Table comparing with/without RCF on contamination sweep
- Stage ablation table (Experiment C)
- Calibration queue health figure (Experiment D)
- Subsection narrative explaining which gate fires most on real OpTC contaminated data

### Updated Table II
Add a row: "ZeroCausal + RCF" alongside existing "ZeroCausal (Linear SCM, Tuned)"

### Updated Discussion Section VII-A
Replace current "future work mitigation" paragraph about calibration contamination with "we address this via RCF" and forward-reference the new evaluation subsection.

---

## Success Criteria

The RCF is a net improvement if it achieves ALL of the following on OpTC (10-seed):

1. **AUC improves** from 0.727 to ≥ 0.760
2. **Recall at 5% FPR improves** from 24.21% to ≥ 35%
3. **Contamination at 1% AUC stays** ≥ 0.80 (vs. current 0.674)
4. **Calibration queue never starves** — admission rate stays ≥ 40% of windows on clean data
5. **Runtime overhead < 5ms per window** (must not break the 33.4ms/window budget)

If criteria 4 or 5 fail, tune `alpha_gate` up (less aggressive Stage 1) or increase `k_iqr` (wider energy fence) before declaring failure.

---

## Implementation Order (Priority)

1. **Stage 2 first** (Energy Gate) — simplest to implement, directly addresses the calibration inflation problem, expected largest single gain
2. **Stage 1 second** (Causal Self-Gate) — zero cost, just adds a condition using already-computed p_min
3. **Stage 4 third** (Transition Buffer) — one-line addition to existing change-point logic
4. **Stage 3 last** (Residual Structure Gate) — most complex, most novel, addresses the LiNGAM-SF insight about confounders vs. violations
5. **Full ablation** after all four stages work individually

---

# Part 2: Pure Novelty Extensions (2026-06-19)

Eight additional novel components implemented in `zerocausal_core.py` and evaluated in `14_novel_evaluation.py`.

---

## Novel Component Architecture

```
Raw Provenance Stream
        │
        ▼
[RobustParCorr]           ← N1: MCD-based contamination-aware causal discovery
        │
        ▼
[CausalGraphEvolutionDetector] ← N4: structural birth/death scoring (EWMA)
        │
[KalmanSCM]               ← N5: continuously evolving SCM coefficients
        │
        ▼
[MultiScaleCausalFusion]  ← N8: process/file/network scale fusion
        │
        ▼
[CausalInterventionScorer] ← N3: min-cut attacker effort estimate
        │
        ▼
[RobustCalibrationFilter] ← Existing: 4-stage gate
[SelfHealingCalibration]  ← N6: retroactive poisoning removal
        │
        ▼
[WeightedConformalCalibrator] ← N2: non-exchangeable conformal prediction
        │
        ▼
Alert + CausalRobustnessMetric ← N7: pre-deployment detectability certificate
```

---

## Theorem 1 — Contamination-Drift Robustness Guarantee

**Statement**: Let ε ∈ [0, 1/2) be the contamination rate of the calibration stream
and δ = TV(P_t, P_{t+1}) ≤ δ_max be the per-step total-variation drift rate.
Let λ be the exponential weight decay rate in WeightedConformalCalibrator.
Then ZeroCausal + RCF + WeightedConformalCalibrator achieves:

    FAR(α) ≤ α + 2ε/(1-ε) + δ_max/λ

**Proof Sketch**:

Step 1 (Contamination bound): Under Huber's ε-contamination model, the
empirical CDF Fˆ_n of calibration scores satisfies TV(Fˆ_n, F_clean) ≤ 2ε/(1-ε)
(Huber 1964, Theorem 2.1). This shifts the conformal p-value CDF by at most 2ε/(1-ε).

Step 2 (Drift bound): From Barber et al. (2022, Theorem 2), weighted conformal
prediction with decay rate λ satisfies:
    P(alarm on normal) ≤ α + Σ_t w_t · TV(P_t, P_test) / Σ_t w_t
Under geometric decay w_t = e^{-λΔt} and bounded drift δ per step:
    Σ_t w_t · TV / Σ_t w_t ≤ δ/λ

Step 3 (Combination): RCF Stage 1 (causal self-gate) rejects windows with
p_min < α_gate, which (under Bonferroni) are mechanism violations with probability
≥ 1 - α_gate. This reduces the effective contamination reaching the calibration
queue to ε' ≤ ε · α_gate. Substituting ε' yields the tighter bound:
    FAR ≤ α + 2ε·α_gate/(1-ε·α_gate) + δ/λ  □

**Empirical validation**: Run `python 14_novel_evaluation.py --contamination ε`
for ε ∈ {0, 0.01, 0.05, 0.10, 0.20} and verify FAR ≤ theorem bound.

---

## Novel Component Details

### N1 — RobustParCorr (zerocausal_core.py:RobustParCorr)
- **What**: Replaces PCMCI's standard ParCorr with MCD-based robust partial correlation
- **How**: Lag-embeds training data, fits MinCovDet, extracts precision matrix for partial correlations
- **Novelty claim**: First contamination-aware causal skeleton for security provenance streams
- **Key param**: support_fraction=0.85 → tolerates up to 15% contamination

### N2 — WeightedConformalCalibrator (zerocausal_core.py:WeightedConformalCalibrator)
- **What**: Conformal prediction for non-exchangeable streams with provable FAR bound
- **How**: Exponential decay weights w_t = exp(-λΔt); weighted conformal p-value
- **Novelty claim**: First conformal IDS with FAR ≤ α + δ/λ under concept drift
- **Key param**: lambda_decay=0.01

### N3 — CausalInterventionScorer (zerocausal_core.py:CausalInterventionScorer)
- **What**: Estimates minimum attacker effort from causal graph topology + residuals
- **How**: BFS boundary cut between violated/non-violated nodes; normalized by d
- **Novelty claim**: First "attacker cost" metric in causal IDS; interpretable for analysts
- **Output**: CIS ∈ [0,1], 0 = pinpoint, 1 = carpet-bomb

### N4 — CausalGraphEvolutionDetector (zerocausal_core.py:CausalGraphEvolutionDetector)
- **What**: Detects attacks from structural causal changes (edge birth/death/drift)
- **How**: EWMA of edge probabilities 1-p_ij_tau; Frobenius norm vs. baseline
- **Novelty claim**: Detects attacks as causal structure changes, not residual errors
- **Key param**: ewma_alpha=0.1, birth_threshold=0.80

### N5 — KalmanSCM (zerocausal_core.py:KalmanSCM)
- **What**: SCM with continuously evolving regression coefficients via Kalman filter
- **How**: Per-feature scalar Kalman filter on β_j; process noise Q controls drift speed
- **Novelty claim**: Eliminates discrete change-point resets; temporal provenance SCM
- **Key param**: process_noise_var=1e-4

### N6 — SelfHealingCalibration (zerocausal_core.py:SelfHealingCalibration)
- **What**: Retroactively removes poisoned calibration windows after alarm confirmed
- **How**: Cosine similarity between alarm z-vec and each calibration entry; removes if sim > threshold
- **Novelty claim**: Self-healing calibration that undoes past contamination in real time
- **Key param**: heal_threshold=0.85, max_heal_fraction=0.30

### N7 — CausalRobustnessMetric (zerocausal_core.py:CausalRobustnessMetric)
- **What**: Pre-deployment detectability certificate before any test data
- **How**: ρ(G,ε) = (1-2ε)·μ_strength/σ_max; ρ>1 → detectable
- **Novelty claim**: First pre-deployment causal IDS robustness certificate
- **Also computes**: Theorem 1 FAR bound for given (α, ε, δ, λ)

### N8 — MultiScaleCausalFusion (zerocausal_core.py:MultiScaleCausalFusion)
- **What**: Three independent CausalRegressionModels at process/file/network scales
- **How**: Partition edges by keyword; inverse-variance weighted CAS fusion
- **Novelty claim**: First multi-scale causal model for provenance graph IDS
- **Scales**: Process (1s-fast), File (5s-medium), Network (30s-slow)

---

## Files Modified in Part 2

| File | Change |
|------|--------|
| `zerocausal_core.py` | +MinCovDet import; 8 new classes (N1–N8) |
| `14_novel_evaluation.py` | New: evaluates all 8 components + contamination sweep |
| `todo.md` | New: tracks all novelty components |
| `postwork.md` | New: session log |
| `imp.md` | This update |
