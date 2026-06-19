Viewed architecture.md:24-44

Here is the in-depth architectural breakdown of **ZeroCausal v2 (CausalMLHybrid)**, explaining how data flows through its four core processing layers:

```
[Raw Stream] ──> [Robust SCM Layer] ──> [1. 12D Causal Feature Extractor]
                                                │
       ┌────────────────────────────────────────┴────────────────────────────────────────┐
       ▼                                        ▼                                        ▼
[2a. CAS Scorer]                        [2b. LSTM Autoencoder]                  [2c. Causal Random Forest]
  (Instantaneous)                          (8-Window Sequence)                     (Supervised Mapping)
       │                                        │                                        │
       └────────────────────────────────────────┬────────────────────────────────────────┘
                                                ▼
                                   [3. Stacked Logistic Fusion]
                                                │
                                                ▼
                                    [4. Conformal Calibration]
                                                │
                                                ▼
                                          [Final Alarm]
```

---

### Layer 1: The 12-Dimensional Causal Feature Extractor
Instead of making decisions on a single anomaly metric, v2 maps the output of the Structural Causal Model (SCM) into a highly regularized **12-dimensional causal representation space**. For each time window $t$, the `CausalFeatureExtractor` computes:

1. **CAS (Causal Anomaly Score):** The baseline H-CAS score combining residual errors and p-values.
2. **$\log(\min p)$:** The logarithmic minimum p-value across all SCM feature regressions: $\log(\min_j p_j + 10^{-5})$.
3. **Residual Energy ($\chi^2$):** Normalized sum of squared prediction errors: $\sum_j \frac{(x_j - \hat{x}_j)^2}{\sigma_j^2}$.
4. **Novelty Count:** The count of active edges in window $t$ that had a zero-weight in the discovered causal graph adjacency matrix.
5. **Causal Violations ($n_{\text{violated}}$):** Count of features whose SCM prediction residual exceeds $2.5\sigma$.
6. **Maximum $z$-score ($z_{\max}$):** The largest individual standard deviation deviation: $\max_j \frac{|x_j - \hat{x}_j|}{\sigma_j}$.
7. **Mean $z$-score ($z_{\text{mean}}$):** The average deviation across all causal dimensions.
8. **Spike Ratio:** The ratio of max-to-mean deviation: $z_{\max} / (z_{\text{mean}} + 10^{-5})$.
9. **Causal Intervention Score (CIS):** Runs a BFS boundary calculation on the causal graph to measure how localized/pinpointed the violations are.
10. **Burstiness:** The variance of active edge counts in the current window.
11. **Entropy:** The Shannon entropy of the active edge distribution.
12. **Active Fraction:** The percentage of active edges out of the total edge pool.

---

### Layer 2: The Triple-Scoring Ensemble
The 12D feature vector $F_t$ is processed simultaneously by three parallel models, each specialized in a different anomaly signature:

#### A. Instantaneous Violations: CAS Scorer
* **Input:** $F_t[0]$ (the H-CAS score).
* **Role:** Detects immediate, high-magnitude violations of causal invariants (e.g., a massive process-spawning burst).

#### B. Temporal Sequence Violations: LSTM-Autoencoder
* **Input:** A sequence of 8 consecutive windows: $\{F_{t-7}, F_{t-6}, \dots, F_t\} \in \mathbb{R}^{8 \times 12}$.
* **Mechanism:** 
  * The **Encoder LSTM** compresses the sequence into a 32-dimensional latent vector.
  * The **Decoder LSTM** reconstructs the 12D vectors.
  * **Reconstruction Error:** The anomaly score is the mean squared error (MSE) of the reconstruction:
    $$s_{\text{lstm}} = \frac{1}{8} \sum_{i=0}^{7} \|F_{t-i} - \hat{F}_{t-i}\|_2^2$$
* **Role:** Catches slow, multi-stage, or stealthy APT activities (like low-and-slow lateral movement) that look normal in a single window but violate temporal patterns.

#### C. Pattern Discrimination: Causal Random Forest (CRF)
* **Input:** The 12D feature vector $F_t$.
* **Mechanism:** A Random Forest of 300 decision trees trained on the validation partition (which contains both normal and attack samples).
* **Output:** The probability that the feature vector represents an attack state:
    $$s_{\text{crf}} = P(\text{attack} | F_t)$$
* **Role:** Learns the specific combinations of causal features that correlate with attack behavior.

---

### Layer 3: Stacked Ensemble Fusion
The outputs of the three parallel scorers ($s_{\text{cas}}, s_{\text{lstm}}, s_{\text{crf}}$) are normalized and combined using a **logistic regression meta-learner**:

$$\text{Score}_{\text{final}} = \sigma(w_1 \cdot s_{\text{cas}} + w_2 \cdot s_{\text{lstm}} + w_3 \cdot s_{\text{crf}} + b)$$

Where:
* $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function.
* The weights $w = [w_1, w_2, w_3]$ and bias $b$ are trained on the validation set using gradient descent to maximize classification AUC.
* This meta-learner automatically learns which scorer to trust for each dataset (e.g., giving higher weight to the LSTM-AE on slow-moving datasets like StreamSpot, and higher weight to CAS on bursty datasets like TC3).

---

### Layer 4: Conformal Calibration
The raw fused score $\text{Score}_{\text{final}}$ is fed into the **`ConformalCalibrator`**:
1. It maintains a rolling history queue of the last $W$ benign fused scores.
2. It computes the empirical p-value of the current score against the queue.
3. It updates the conformal threshold dynamically:
   $$\alpha_{t+1} = \alpha_t + \eta \left( \mathbb{I}(\text{Score}_{\text{final}} > \alpha_t) - \alpha_{\text{target}} \right)$$
4. If the score exceeds $\alpha_t$, a conformal alarm is raised, guaranteeing that the false alarm rate stays tightly bounded to the target budget (e.g., 5%).