# Submission Cover Letter

Dear Program Committee Chairs,

We are pleased to submit our manuscript, **"ZeroCausal: Provable, Zero-Label Causal Anomaly Detection for APTs in Provenance Graphs,"** for consideration for publication in the upcoming conference proceedings.

### Context & Core Contributions
Modern provenance-based host Intrusion Detection Systems (IDS) for Advanced Persistent Threats (APTs) are critical for enterprise security. However, all existing state-of-the-art provenance systems (e.g., OCR-APT, TraceCluster, StageFinder) make a fundamental, unviable assumption: they require **curated benign training data** or labeled attack samples to construct normal profiles. 

In this work, we present **ZeroCausal**, the first IDS framework that completely eliminates the clean training data and label assumptions. ZeroCausal operates by learning and monitoring *causal invariances*—stable cause-effect equations inherent in running system processes—discovered online directly from raw, unlabeled, and potentially contaminated system logs. 

Our core contributions are:
1. **Zero-Label, Zero-Benign Data Detection**: Discovers structural causal relationships online via conditional independence testing (PCMCI with ParCorr) directly on live, unlabeled streams.
2. **Online Concept Drift Updates**: Integrates an adaptive change-point detector that monitors sliding-window statistics to trigger automatic, local SCM refitting when normal system processes drift.
3. **Provable False Positive Control**: Applies online conformal prediction with a stochastic feedback loop to dynamically tune the anomaly decision threshold, satisfying a user-defined False Positive Rate (FPR) budget.
4. **Significant Scalability Optimizations**: Implements programmatic matrix conversions and binary search conformal p-value scans, delivering a **13x evaluation speedup** that reduces streaming latency on DARPA OpTC logs to 18.03 seconds (compared to 240 seconds in prior implementations).
5. **Rigorous Multi-Dataset Evaluation**: Evaluated across real DARPA OpTC host logs and simulated DARPA TC3 and NODLINK streams, achieving an AUC of **0.8359 on OpTC** (Tuned SCM), **0.8350 on TC3**, and **0.8258 on NODLINK**, vastly outperforming Isolation Forest baselines on OpTC and comparing favorably with state-of-the-art causal systems (Causal-IDS, 0.8400 AUC).

### Venue Alignment
Given its focus on host auditing, provenance graph representation, online causal reasoning, and practical deployment considerations, our work is a strong fit for the systems and network security tracks of your venue. 

We confirm that this manuscript is our original work, has not been published elsewhere, and is not currently under consideration for publication in any other venue. All authors have read and approved the manuscript for submission.

Thank you for your time and consideration of our work.

Sincerely,

The Authors of ZeroCausal
