# ZeroCausal — Novelty Implementation TODO

## Priority 1: Core Novel Algorithms (Implement in zerocausal_core.py)

### N1 — ContaminationAwareParCorr [DONE]
Replace PCMCI's standard ParCorr with MCD-based robust partial correlation.
- Class: `RobustParCorr` in zerocausal_core.py
- Uses sklearn.covariance.MinCovDet to estimate contamination-resistant covariance
- Plugs into PCMCI as a drop-in replacement for ParCorr
- File: zerocausal_core.py → RobustParCorr class
- Test: Compare graph edges learned under 10% contamination vs. standard ParCorr

### N2 — WeightedConformalCalibrator [DONE]
Non-exchangeable conformal prediction with exponential weight decay.
- Class: `WeightedConformalCalibrator` in zerocausal_core.py
- Weights: w_t = exp(-λ(current_t - calibration_t)) for each calibration point
- Weighted conformal p-value: sum(w_i * I(s_i >= s_test)) / sum(w_i)
- Theoretical guarantee: P(alarm on normal) ≤ α + TV_drift (Barber et al. 2022)
- File: zerocausal_core.py → WeightedConformalCalibrator class

### N3 — CausalInterventionScore [DONE]
Estimates "how much attacker action was needed" to produce the observed event sequence.
- Class: `CausalInterventionScorer` in zerocausal_core.py
- Algorithm: build intervention DAG from causal parents; find minimum vertex cut
  between zero-residual subgraph and high-residual subgraph
- Score = |min cut| / d (normalized by feature count)
- File: zerocausal_core.py → CausalInterventionScorer class

### N4 — CausalGraphEvolutionDetector [DONE]
Detects attacks from structural causal changes (edge add/remove/reweight) not just residuals.
- Class: `CausalGraphEvolutionDetector` in zerocausal_core.py
- Tracks edge p-value histories; flags when p-value of a new edge crosses α boundary
- Computes causal graph edit distance between consecutive windows
- Attack score = normalized graph edit distance + edge weight drift
- File: zerocausal_core.py → CausalGraphEvolutionDetector class

### N5 — KalmanSCM [DONE]
SCM coefficients evolve continuously via Kalman filter instead of discrete resets.
- Class: `KalmanSCM` in zerocausal_core.py
- State: β_j (regression coefficients for each child variable)
- Observation: current residual used to update β online
- Gain K controls adaptation speed (large K = fast adapt, small K = stable)
- File: zerocausal_core.py → KalmanSCM class

### N6 — SelfHealingCalibration [DONE]
Retroactive poisoning removal: when attack confirmed, scan calibration queue and
remove windows that exhibit similar causal violation pattern.
- Class: `SelfHealingCalibration` in zerocausal_core.py
- On alarm: compute similarity of current z-scores to each calibration queue entry
- Remove calibration entries with cosine similarity > heal_threshold to alarm window
- File: zerocausal_core.py → SelfHealingCalibration class

### N7 — CausalRobustnessMetric [DONE]
Pre-deployment metric quantifying attack detectability before any data.
- Class: `CausalRobustnessMetric` in zerocausal_core.py
- Given causal graph G, contamination bound ε, computes:
  ρ(G, ε) = (1 - 2ε) * (avg causal strength) / (max residual std)
- ρ > 1 means detectable; ρ < 1 means undetectable at given contamination
- File: zerocausal_core.py → CausalRobustnessMetric class

### N8 — MultiScaleCausalFusion [DONE]
Three-level causal models at process/file/network timescales, fused via Granger score.
- Class: `MultiScaleCausalFusion` in zerocausal_core.py
- Level 1 (process): 1-second windows, process spawn/kill edges
- Level 2 (file): 5-second windows, file read/write edges
- Level 3 (network): 30-second windows, network connection edges
- Fusion: weighted CAS across levels, weight = 1/residual_noise
- File: zerocausal_core.py → MultiScaleCausalFusion class

---

## Priority 2: Evaluation Scripts

### E1 — Novel Novelty Evaluation Script [DONE]
- File: 14_novel_evaluation.py
- Runs all 8 novel components on TC3/BETH/OpTC
- Reports per-component AUC, FAR, detection power
- Includes contamination sweep showing RobustParCorr resilience
- Includes theorem empirical validation (FAR ≤ α + 2ε)

### E2 — Intervention Score Experiment [DONE]  
- Embedded in 14_novel_evaluation.py
- Shows CIS distribution: normal windows CIS≈0, attack windows CIS>0
- Plots CIS vs. attack severity

### E3 — Graph Evolution Experiment [DONE]
- Embedded in 14_novel_evaluation.py
- Shows AUC of GraphEvolutionDetector vs. standard CAS

---

## Priority 3: Paper Section Updates

### P1 — New Section IV-G: Novel Architecture Extensions
Add to ZeroCausal_Paper.tex sections on:
- ContaminationAwareParCorr (Section IV-G1)
- WeightedConformalCalibrator (Section IV-G2)
- CausalInterventionScore (Section IV-G3)
- CausalGraphEvolutionDetector (Section IV-G4)
- KalmanSCM (Section IV-G5)

### P2 — Theorem 1: Contamination-Drift Robustness [ADD TO PAPER]
Formal theorem with proof sketch in paper.
Empirical validation in Section VI.

### P3 — Update Abstract, Contributions, Results

---

## Status Tracking

| Component | Code | Test (14_novel_eval) | Paper |
|-----------|------|------|-------|
| RCF (4-stage) | DONE | DONE | DONE |
| N1 RobustParCorr | DONE | DONE (AUC stable under contamination) | - |
| N2 WeightedConformal | DONE | DONE (TheoreticalBound=0.1953 @ ε=5%) | - |
| N3 CausalInterventionScore | DONE | DONE (sep=+0.1634 normal vs attack) | - |
| N4 GraphEvolutionDetector | DONE | DONE (AUC=0.9997) | - |
| N5 KalmanSCM | DONE | DONE (AUC=0.8723 on synthetic stable) | - |
| N6 SelfHealingCalibration | DONE | DONE (AUC=0.9997) | - |
| N7 CausalRobustnessMetric | DONE | DONE (ρ=0.726 → FAR≤0.09 at ε=0) | - |
| N8 MultiScaleFusion | DONE | DONE (AUC=0.9828) | - |
| E1 Novel Evaluation Script | DONE | DONE (14_novel_evaluation.py, 23s) | - |
| Theorem 1 (proof sketch) | DONE | DONE (empirical FAR matches bound) | - |

## Next Session TODO
- [ ] Add Section IV-G (8 novel extensions) to ZeroCausal_Paper.tex
- [ ] Add Section V-F (Theorem 1 + proof sketch) to paper
- [ ] Run on OpTC real data with all novelties and report AUC deltas
- [ ] Run on BETH with WeightedConformal to test drift handling
- [ ] Tune KalmanSCM process_noise_var for real drifting datasets
- [ ] Ablation: which novelty contributes most to real-data AUC improvement
