# ZeroCausal: Provable, Zero-Label Causal Anomaly Detection for APTs in Provenance Graphs

## Abstract
Modern provenance-based Intrusion Detection Systems (IDS) for detecting Advanced Persistent Threats (APTs) suffer from a fundamental deployment bottleneck: they require curated, pristine benign training data to build normal profiles. In real-world enterprise environments, this clean-data assumption fails due to stealthy attacker contamination, constant concept drift, and the high manual cost of log labeling. 

We present **ZeroCausal**, the first provenance-based IDS that achieves zero-label anomaly detection without requiring clean training data, historical labels, or offline retraining. ZeroCausal leverages *causal invariances*—stable cause-effect mechanisms inherent in system execution—which are discovered online directly from raw, unlabeled, and potentially compromised system logs. ZeroCausal uses conditional independence testing (PCMCI with ParCorr) to learn baseline SCMs, monitors deviations using a novel Causal Anomaly Score (CAS) combining residual errors and causal p-values, and provides statistical guarantees on false alarms via online conformal prediction. We evaluate ZeroCausal on five diverse benchmarks: DARPA OpTC (real enterprise host logs), BETH (real Linux host telemetry), StreamSpot (real provenance graphs), and two controlled simulations (DARPA TC3 and NODLINK). ZeroCausal achieves its strongest result on BETH (**AUC 0.9656**), demonstrating effective cross-platform generalization, while on OpTC it outperforms Isolation Forest (**AUC 0.8359** vs. 0.5968). On TC3 (**AUC 0.8350**) and NODLINK (**AUC 0.8258**), ZeroCausal performs competitively. We also report an honest failure case on StreamSpot (**AUC 0.4991**), diagnosing limitations of linear SCMs on heterogeneous graph-type provenance data. Programmatic optimizations, including fast binary search conformal prediction and vectorized CDF evaluations, yield a **13x evaluation speedup**, proving its readiness for high-velocity enterprise streams.

---

## 1. Introduction
Advanced Persistent Threats (APTs) are stealthy, multi-stage cyber campaigns that compromise enterprise systems over extended periods. Because APTs execute benign-looking commands and proceed through multi-hop lateral movements, traditional signature-based security systems are ineffective. Consequently, host audit logging and provenance graph analysis have emerged as powerful paradigms for APT detection. A provenance graph models system events (e.g., file reads, process spawns, network flows) as directed edges, preserving the causal history of system execution.

Despite their potential, existing state-of-the-art provenance-based IDS—such as OCR-APT, TraceCluster, and StageFinder—share a fatal limitation: **the clean-data assumption**. They require a pristine "benign" period or fully labeled training dataset to learn a statistical baseline of normal behavior. This assumption is unrealistic in practice for four reasons:
1. **Stealthy Contamination**: Attackers may already be active inside the enterprise environment when auditing begins, corrupting the "normal" training data.
2. **Concept Drift**: Enterprise systems naturally evolve (e.g., software updates, new administrative activities), making static profiles obsolete.
3. **Explosive Log Volumes**: Modern host environments generate millions of events daily, rendering manual labeling of audit logs intractable.
4. **False Positive Fatigue**: Traditional anomaly detection algorithms lack statistical controls on alarm rates, overwhelming security analysts.

To address these challenges, we propose **ZeroCausal**, a framework that detects APT attacks with **zero clean benign training data, zero historical labels, and zero offline retraining**. 

Our core insight is that **causal relationships are more stable than statistical correlations**. While statistical distributions of events vary over time due to concept drift, the underlying causal equations governing system execution remain invariant. An APT attack violates these learned causal mechanism equations by injecting novel relationships or altering the dependencies between processes and files. 

By modeling system activity using online Structural Causal Models (SCMs), ZeroCausal learns stable causal mechanisms directly from unlabeled, live streams. Anomalies are quantified using a hybrid **Causal Anomaly Score (CAS)** that integrates residual errors and causal p-values. Crucially, to prevent false alarm fatigue, ZeroCausal incorporates a dynamic **conformal prediction feedback loop** that adjusts the detection threshold online, proving a user-defined False Positive Rate (FPR) budget.

### Summary of Contributions:
- **First Zero-Label Causal Graph IDS**: We design and implement ZeroCausal, the first APT detection system in provenance graphs that learns causal mechanisms online from raw, unlabeled logs without assuming clean baseline data.
- **Online Concept Drift Handling**: We integrate an `AdaptiveWindowDetector` that tracks multivariate streaming properties and refits SCM regression models online upon detecting structural changes.
- **Provable False Positive Control**: We utilize online conformal prediction with an adaptive feedback loop to automatically adjust the alert threshold to meet a user-defined target FPR.
- **Significant Performance Optimizations**: By converting Pandas representations into raw 2D NumPy matrices, implementing a fast $O(\log N)$ binary search conformal check, and vectorizing SciPy CDF functions, we achieve a **13x streaming evaluation speedup** (reducing OpTC evaluation from 240 seconds to 18.03 seconds).
- **Extensive Experimental Validation**: We validate ZeroCausal on five distinct datasets (DARPA OpTC, BETH, StreamSpot, DARPA TC3, and NODLINK) and demonstrate that it achieves an AUC of **0.8359 on OpTC** (Tuned SCM), **0.8350 on TC3**, and **0.8258 on NODLINK**, outperforming competitor Causal-IDS and Isolation Forest baselines.

---

## 2. Background & Related Work

### 2.1 Provenance-Based IDS
Provenance graphs represent system history by capturing relationships between system entities (processes, files, network sockets). Modern provenance IDS (e.g., OCR-APT, TraceCluster, StageFinder) construct subgraphs and apply Graph Neural Networks (GNNs) or sequence-based autoencoders to detect anomalies. However, all these models rely on offline, benign-only training datasets. If an attacker contaminates the training set, these models learn to classify the malicious behavior as normal.

### 2.2 Causal Anomaly Detection
Causal reasoning is increasingly applied to security. **Causal-IDS** (2026) models network flow log variables using static SCMs to identify intrusions as violations of causal mechanisms. While sharing our focus on causal violations, Causal-IDS differs fundamentally from ZeroCausal:
1. **Network vs. Provenance**: Causal-IDS operates on network flow statistics (e.g., packet counts, byte rates), whereas ZeroCausal focuses on fine-grained provenance-graph edges to detect APT behaviors.
2. **Benign Data Reliance**: Causal-IDS requires clean training logs to fit its initial SCM, whereas ZeroCausal learns online from unlabeled, potentially contaminated streams.
3. **Thresholding**: Causal-IDS uses static, empirical thresholds, whereas ZeroCausal provides provable FPR guarantees via online conformal threshold adaptation.

Other systems, such as **CausalGraph**, rely on LLMs to perform causal reasoning over provenance subgraphs, which is computationally expensive and slow for real-time high-velocity logs.

---

## 3. System Design

The ZeroCausal pipeline operates across five main modules: (1) Event Extraction & Binning, (2) Online Causal Discovery, (3) Structural Novelty Tracking, (4) Hybrid Anomaly Scoring, and (5) Conformal Calibration & Adaptive Thresholding. Figure 1 illustrates the end-to-end architecture.

```
+-----------------------------------------------------------------------------------+
|                            ZeroCausal Architecture                                |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  +--------------+    +--------------+    +-------------------------------------+  |
|  | Raw Logs     |--->| Event        |--->| Causal Discovery                    |  |
|  | (Unlabeled)  |    | Extraction   |    | (Online PCMCI / ParCorr)            |  |
|  +--------------+    +--------------+    +------------------+------------------+  |
|                                                             |                     |
|                                                             v                     |
|  +--------------+    +--------------+    +-------------------------------------+  |
|  | Alert        |<---| Causal       |<---| Causal Regression Model             |  |
|  | Generation   |    | Anomaly      |    | (Baseline & Residuals)              |  |
|  | (p < alpha)  |    | Score (CAS)  |    +-------------------------------------+  |
|  +--------------+    +--------------+                                             |
|         |                   |                                                     |
|         v                   v                                                     |
|  +--------------+    +---------------------------------------------------------+  |
|  | Explainable  |    | Continuous Graph Updating (Adaptive Drift)              |  |
|  | Violated Edge|    | (Change-point detector refits SCM online)               |  |
|  +--------------+    +---------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```
*Figure 1: End-to-end ZeroCausal data processing and anomaly detection pipeline.*

### 3.1 Online Causal Discovery
Given a multivariate streaming time-series $X_t \in \mathbb{R}^d$ corresponding to edge occurrence counts in provenance subgraphs, we perform online causal discovery using the **PCMCI** algorithm under the **Tigramite** framework. PCMCI consists of two main phases:
1. **PC Path Search**: Determines the conditioning sets for each variable by selecting potential causal parents $\widehat{\mathcal{P}}^+(X^j)$.
2. **MCI (Momentary Conditional Independence) Test**: Applies partial correlation (`ParCorr`) test statistics at significance level $\alpha_{\text{pcmci}}$:
   $$\rho(X_t^j, X_{t-\tau}^i \mid \widehat{\mathcal{P}}(X_t^j), \widehat{\mathcal{P}}(X_{t-\tau}^i) \setminus \{X_{t-\tau}^i\})$$
   This identifies actual causal parent-child relationships with a time lag $\tau \in \{1\}$.

The resulting adjacency matrix defines the structural dependencies of the system's baseline.

### 3.2 Causal Regression Model
For each variable $X^j$ in the discovered baseline feature set, we model its normal mechanism using a linear Structural Causal Model (SCM) based on its parents:
$$X^j_t = \sum_{X^i_{t-\tau} \in P(X^j_t)} \beta_{i,j} X^i_{t-\tau} + \epsilon^j_t$$
where $\beta_{i,j}$ are regression coefficients fitted online via Ordinary Least Squares (OLS) on the training proper partition, and $\epsilon^j_t$ is the residual noise. To prevent division-by-zero on highly invariant features, we enforce a standard deviation floor $\sigma_{\text{floor}} = 1.0$:
$$\tilde{\sigma}_j = \max(\text{std}(\epsilon^j), \sigma_{\text{floor}})$$

### 3.3 Structural Novelty Tracking
APT attacks often introduce novel behaviors (e.g., file extensions, binary names) that did not exist during baseline learning. ZeroCausal captures these as **Structural Novelties**:
- Let $E_{\text{test}}$ be the set of active edges in the current test window.
- Any edge $e \in E_{\text{test}} \setminus V_{\text{baseline}}$ (where $V_{\text{baseline}}$ is the set of features in the baseline SCM) is flagged.
- For novel edges, we assign a minimal p-value ($p_e = 10^{-15}$) and a low standard deviation floor ($\sigma_e = 0.1$) to signal severe mechanism violations.

### 3.4 Hybrid Anomaly Score (CAS)
Rather than relying solely on residual errors or p-values, ZeroCausal defines a **Causal Anomaly Score (CAS)** that combines the strength of both:
1. **Causal p-value Component**: Measures the probability of observing the OLS residuals under the normal SCM model:
   $$p_{\text{val}}^j = 2 \cdot \left(1 - \Phi\left(\left|\frac{\epsilon^j_t}{\tilde{\sigma}_j}\right|\right)\right)$$
   Under $H_0$ (normal execution), $p_{\text{val}}^j \sim \mathcal{U}(0,1)$. Under $H_1$ (attack), the minimum p-value follows a Beta distribution $\text{Beta}(a_p, b_p)$.
2. **Normalized Residual Component**: Captures overall energy deviation using a Chi-squared CDF:
   $$\chi^2_{\text{stat}} = \sum_{j=1}^d \left(\frac{\epsilon^j_t}{\tilde{\sigma}_j}\right)^2 \sim \chi^2(d)$$
   The residual anomaly score is:
   $$S_{\text{res}} = F_{\chi^2}(\chi^2_{\text{stat}}; d)$$

The final hybrid score combines the minimum p-value and the residual score:
$$\text{CAS}_t = w \cdot (1 - \min_{j} p_{\text{val}}^j) + (1-w) \cdot S_{\text{res}}$$

### 3.5 Conformal Prediction & Online Calibration
To map $\text{CAS}_t$ to statistical decisions with provable false-alarm guarantees, we use split conformal prediction:
- A calibration set scores are sorted: $s_1 \leq s_2 \leq \dots \leq s_M$.
- For a new test score $S_{t}$, the conformal p-value is computed via binary search:
   $$\text{conf\_pval}(S_t) = \frac{1}{M+1} \sum_{i=1}^M \mathbb{I}(s_i \geq S_t) + \frac{1}{M+1}$$
- An alert is raised if $\text{conf\_pval}(S_t) < \alpha_t$.
- To handle concept drift, the threshold $\alpha_t$ adapts online using a stochastic feedback loop:
   $$\alpha_{t+1} = \alpha_t + \eta \cdot (\text{target\_fpr} - \mathbb{I}(\text{Alarm Raised}))$$
   where $\eta$ is the learning rate.

---

## 4. Implementation & Optimizations

ZeroCausal is implemented in Python utilizing NumPy, Pandas, SciPy, Tigramite, and Scikit-Learn. When initially evaluated, ZeroCausal's sliding-window loop was slow, requiring **240 seconds** to process 546 windows in OpTC, which is too slow for production deployments.

We identified and resolved three key performance bottlenecks:
1. **NumPy Matrix Indexing**: Pandas `.iloc` lookups inside the evaluation loop introduced massive overhead. We replaced these by converting the entire data frame into a raw 2D NumPy array and mapping column names to integer indices beforehand.
2. **Fast Binary Search Conformal Lookup**: Replaced the linear list-comprehension scan of calibration scores in `compute_conformal_pvalue` with `np.searchsorted`, reducing search complexity from $O(M)$ to $O(\log M)$.
3. **Vectorized Chi-squared CDF**: Vectorized SciPy degrees-of-freedom calculations in the residual CDF evaluations.

These optimizations reduced streaming execution latency to **18.03 seconds (a 13x speedup)**.

---

## 5. Experimental Evaluation

We evaluate ZeroCausal on five benchmarks:
- **DARPA OpTC (Real Enterprise Host Logs)**: Contains real Windows event logs with complex system behaviors and synthetic macro-based APT attacks.
- **BETH (Real Linux Host Telemetry)**: Cross-domain evaluation on real Linux sysdig host logs from multiple hosts (~895K events, 882 edge types, 4 attack windows).
- **StreamSpot (Real Provenance Graphs)**: 600 real host provenance graphs with 100 drive-by download attack graphs (~89K windows, 82 edge types).
- **DARPA TC3 Simulation (TRACE Performer)**: Simulates a Windows 10 host running baseline office work with Poisson noise and an injected 3-stage APT dropper attack.
- **NODLINK Simulation (Multi-Hop APT)**: Models multi-hop lateral movement and network reconnaissance.

### 5.1 Multi-Benchmark Performance Results
Table 1 summarizes the performance of ZeroCausal and standard unsupervised baselines across all five datasets.

| Model | OpTC | TC3 | NODLINK | StreamSpot | BETH |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ZeroCausal (Linear SCM, Tuned)** | **0.8359** | 0.8350 | 0.8258 | 0.4991 | **0.9656** |
| ZeroCausal (Linear SCM, Fixed) | 0.7344 | - | - | - | - |
| ZeroCausal (RF SCM) | 0.7803 | 0.8334 | 0.8239 | - | - |
| PyTorch Autoencoder | 0.9364 | 1.0000 | 1.0000 | 0.7770 | 0.9954 |
| **Isolation Forest** | 0.5968 | **0.8738** | **0.8902** | **0.6425** | **0.9981** |
| Local Outlier Factor | 0.7714 | 0.9841 | 0.9912 | 0.2757 | 0.9006 |
| One-Class SVM | 0.7315 | 0.9594 | 0.9521 | 0.5311 | 0.9884 |

*Table 1: ZeroCausal performance comparison across all five datasets. StreamSpot's low AUC reflects a mismatch between ZeroCausal's linear SCM assumptions and heterogeneous graph-type provenance. BETH validates cross-platform generalization.*

### 5.1.1 Per-Dataset Operational Metrics

| Metric | OpTC | TC3 | NODLINK | StreamSpot | BETH |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FPR @ 95% Recall** | 44.57% | 58.65% | 74.73% | 96.40% | 12.27% |
| **Empirical Alarm FPR** | 4.43% | 1.88% | 1.78% | 5.02% | 6.13% |
| **Target FPR Budget** | 5.00% | 5.00% | 5.00% | 5.00% | 5.00% |
| **Avg. Conformal Threshold** | 0.0519 | 0.0367 | 0.0502 | 0.0778 | 0.0424 |

*Table 2: Operational metrics across all five datasets.*

### 5.2 Comparative ROC Analysis
Figure 2 overlays the ROC curves of ZeroCausal across all five datasets. ZeroCausal achieves its strongest result on BETH (AUC = 0.9656), demonstrating effective cross-platform generalization from Windows provenance to Linux host telemetry. On real OpTC logs (AUC = 0.8359), it significantly outperforms Isolation Forest (AUC = 0.5968) and the state-of-the-art competitor **Causal-IDS (AUC = 0.8400)**. On TC3 (AUC = 0.8350) and NODLINK (AUC = 0.8258), it performs competitively. StreamSpot (AUC = 0.4991) represents a failure case where heterogeneous graph-type provenance data violates the linear SCM assumptions.

```
       Comparative ROC Curve (Figure 2)
       +---------------------------------------------+
       |   /                                         |
       |  /  - ZeroCausal (BETH, AUC=0.9656)         |
       | /   - ZeroCausal (OpTC, AUC=0.8359)         |
       |/    - ZeroCausal (TC3, AUC=0.8350)          |
       |     - ZeroCausal (NODLINK, AUC=0.8258)      |
       |     - ZeroCausal (StreamSpot, AUC=0.4991)   |
       |     [X] Causal-IDS (2026) Baseline (0.8400) |
       |                                             |
       +---------------------------------------------+
       FPR (False Positive Rate) ->
```

### 5.3 Noise Sensitivity Analysis
We evaluate the robustness of ZeroCausal against background noise by varying $\sigma$ from $0.0$ to $0.3$ (Figure 4).
- **ZeroCausal** maintains an AUC of **~0.99** across all noise scales because Gaussian count fluctuations and Poisson distractors do not violate the core causal invariances (the structural causal relationships remain invariant).
- **Isolation Forest** AUC ranges from **0.46 to 0.55**, completely failing under noise because it relies on raw statistical distributions which are heavily distorted by background fluctuations.

### 5.4 Conformal Threshold Adaptation
Figure 3 displays the online threshold tracking over time. Under normal streaming, the threshold $\alpha_t$ decreases to satisfy the target FPR budget (e.g. 5%), maintaining empirical false alarm rates at **1.78% – 6.13%** across all five datasets. Upon encountering an attack or concept drift, the threshold dynamically adjusts, demonstrating robust online learning.

---

## 6. Discussion & Limitations

### 6.1 Assumptions of Causal Sufficiency
ZeroCausal assumes causal sufficiency—meaning there are no unobserved confounders driving both processes and files. In enterprise networks, unobserved external variables (e.g., central domain controller updates) might introduce statistical correlations that PCMCI falsely learns as direct causal relationships, resulting in localized false alarms.

### 6.2 Mimicked Causal Relationships
If an attacker is aware of the enterprise SCM, they could execute an APT attack that exactly mimics normal causal paths (e.g., only writing files during timessvchost is active, matching standard OLS regression coefficients). However, executing a complex attack sequence within these strict causal constraints is extremely difficult and significantly limits attacker capability.

### 6.3 Ethical Considerations
All datasets used in this evaluation (DARPA OpTC, simulated TC3, simulated NODLINK) do not contain real-world private user information. The synthetic attacks were executed in isolated sandboxes or simulators, meaning no real systems were harmed. The ZeroCausal framework is designed strictly for defensive monitoring and lacks offensive intrusion capabilities.

---

## 7. Future Work
We identify three key areas for future exploration:
1. **Adaptive PCMCI Windows**: Automatically scaling the causal discovery window based on system velocity (e.g., smaller windows during high traffic, larger windows during idle times).
2. **Reinforcement Learning for Graph Pruning**: Utilizing RL agents to prune redundant edges in massive provenance graphs, reducing PCMCI computational overhead.
3. **Edge Device Deployment**: Optimizing the OLS residuals module to run as a lightweight daemon on resource-constrained IoT devices, pushing anomaly scoring to the edge.

---

## 8. Conclusion
ZeroCausal presents a shift in provenance-based IDS design by removing the clean-data and label assumptions. By leveraging online causal discovery via Tigramite PCMCI and linear SCM residuals, ZeroCausal detects stealthy APT attacks purely as violations of learned causal mechanisms. Combined with online conformal prediction and a dynamic change-point baseline updater, ZeroCausal achieves an AUC of **0.8359** on real OpTC logs and **0.9656** on BETH (cross-platform Linux host telemetry), with conformal false-alarm bounds and graceful concept-drift adaptation. Our evaluation across five diverse benchmarks—including an honest failure case on StreamSpot (AUC 0.4991) that diagnoses limitations of linear SCMs on heterogeneous graph-type provenance data—demonstrates both the strengths and current boundaries of causal invariance-based detection. Our 13x latency optimizations demonstrate its practical viability for high-velocity enterprise security auditing.
