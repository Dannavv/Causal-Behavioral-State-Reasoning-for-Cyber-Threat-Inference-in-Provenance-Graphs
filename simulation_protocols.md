# Data Simulation Protocols for ZeroCausal Evaluation

To thoroughly evaluate **ZeroCausal** under controlled and reproducible settings, we designed high-fidelity streaming log simulators for two distinct enterprise environments: **DARPA TC3 (Windows 10 Host)** and **NODLINK (Multi-Hop Network/Host)**. These simulators model realistic background noise, causal relationships, and stealthy APT attack behaviors.

---

## 1. DARPA TC3 (Windows 10 TRACE Baseline Simulation)

This protocol simulates a Windows 10 workstation running typical corporate activities, including web browsing, updates, background services, and development work.

### 1.1 Baseline Causal Graph (Normal Activity)
The simulation comprises 15 normal host provenance relationships (features):
- **User Activity**: Spawning chrome from explorer, chrome file reads, git repository interactions, and OneDrive sync activity.
- **System Services**: DNS requests (`svchost.exe` connecting to port 53), system event logging (`svchost.exe` writing to `System.evtx`), security auditing (`lsass.exe`), search indexing, and print spooling (`spoolsv.exe`).
- **Causal Invariance**: To simulate system causality, we inject an explicit cause-effect relationship:
  $$\text{spoolsv.exe reads printers.db} \leftarrow \text{svchost.exe connects to 8.8.8.8:53} > 2 \text{ events}$$
  This models a service trigger dependent on network lookup.

### 1.2 Noise and Contamination Injection
To stress-test ZeroCausal and baseline models:
- **Distractor Edges**: We inject 5 additional "distractor" columns that represent uncorrelated background noise (e.g., cmd spawning conhost, task manager opening explorer). These features follow a Poisson distribution scaled by a noise factor $\sigma$.
- **Gaussian Perturbation**: We apply zero-mean additive Gaussian noise to the counts of all features across all windows to simulate natural variations in logging frequency.
- **Clipping**: Final event counts are clipped at zero to prevent negative values.

### 1.3 APT Attack Vector
We inject 50 independent, stealthy multi-stage APT attacks in the latter 40% (test portion) of the stream. Each attack spans 3 contiguous 1-second windows:
1. **Initial Access / Execution** (Window $t$):
   `PROCESS:nginx.exe -> SPAWNS_PROCESS -> PROCESS:bash.exe`
2. **Dropper Payload** (Window $t+1$):
   `PROCESS:bash.exe -> WRITES_FILE -> FILE:malicious.elf`
3. **Privilege Escalation / Action on Objectives** (Window $t+2$):
   `PROCESS:malicious.elf -> MODIFY -> FILE:passwd`

---

## 2. NODLINK (Multi-Hop APT Simulation)

This protocol simulates a multi-hop enterprise scenario where normal activities occur alongside a highly stealthy lateral movement and reconnaissance campaign.

### 2.1 Baseline Causal Graph (Normal Activity)
The NODLINK baseline consists of 15 features:
- **Office / Productivity**: Outlook mail client operations, Word/Excel document reads.
- **Admin Activities**: Command prompt execution (`cmd.exe` spawning from explorer), lsass reading the SAM registry, and svchost file writes.
- **Causal Invariance**: We inject the following causal dependency:
  $$\text{cmd.exe reads LocalSettings} \leftarrow \text{explorer.exe spawns cmd.exe} > 0 \text{ events}$$

### 2.2 Noise and Contamination Injection
- **Distractor Edges**: We inject 5 distractor edges representing unrelated background actions (e.g., chrome reading history, dns connections, svchost modifying gpt.ini).
- **Gaussian Fluctuations**: Count-level Gaussian fluctuations are scaled by a noise factor $\sigma$ and added to the baseline.

### 2.3 APT Attack Vector
We inject 50 independent attack campaigns into the test stream, each spanning 3 seconds:
1. **Execution via Spear-phishing / Macro** (Window $t$):
   `PROCESS:outlook.exe -> SPAWNS_PROCESS -> PROCESS:cmd.exe`
2. **Reconnaissance** (Window $t+1$):
   `PROCESS:cmd.exe -> WRITES_FILE -> FILE:recon_results.txt`
3. **Lateral Movement / Data Exfiltration** (Window $t+2$):
   `PROCESS:cmd.exe -> CONNECTS_TO -> FLOW:10.0.0.15:445`

---

## 3. Simulator Parameter Summary

The following parameters are standard across the sensitivity evaluation scripts (`09_evaluate_additional_datasets.py` and `11_sensitivity_analysis.py`):

| Parameter | Value / Range | Description |
| :--- | :---: | :--- |
| `num_windows` | 1000 | Total streaming windows (1 second each) |
| `train_split` | 60% (first 600s) | Clean training baseline partition |
| `test_split` | 40% (last 400s) | Test partition with injected attacks |
| `num_attacks` | 50 | Total independent attack campaigns injected |
| `noise_level` ($\sigma$) | $0.0 \rightarrow 0.3$ | Scaling factor for Gaussian noise and distractor scaling |
| `target_fpr` | 5.0% | Target budget for the conformal calibration layer |
