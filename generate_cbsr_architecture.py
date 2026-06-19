import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Canvas ───────────────────────────────────────────────────────────────────
W, H = 26, 24
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Palette ──────────────────────────────────────────────────────────────────
BLUE   = "#1565C0"
PURPLE = "#6A1B9A"
INDIGO = "#1A237E"
ORANGE = "#BF360C"
TEAL   = "#00695C"
GREEN  = "#1B5E20"
PINK   = "#880E4F"
AMBER  = "#E65100"
RED    = "#B71C1C"
DARK   = "#263238"
MID    = "#546E7A"

# ── Primitives ───────────────────────────────────────────────────────────────
def box(ax, cx, cy, w, h, color, title, sub="", tsz=28, ssz=22):
    """Rounded rect with a solid left accent bar."""
    r = 0.28
    # drop shadow
    sh = FancyBboxPatch((cx-w/2+0.08, cy-h/2-0.08), w, h,
                        boxstyle=f"round,pad=0,rounding_size={r}",
                        fc="#DDDDDD", ec="none", zorder=1)
    ax.add_patch(sh)
    # background
    bg = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                        boxstyle=f"round,pad=0,rounding_size={r}",
                        fc="white", ec=color, lw=3.2, zorder=2)
    ax.add_patch(bg)
    # left accent bar
    bar = FancyBboxPatch((cx-w/2, cy-h/2), 0.22, h,
                         boxstyle=f"round,pad=0,rounding_size={r}",
                         fc=color, ec="none", zorder=3)
    ax.add_patch(bar)
    
    # title - shifted slightly lower (h*0.18 instead of h*0.22) for visual balance
    ty = cy + (h*0.18 if sub else 0)
    ax.text(cx+0.12, ty, title, ha="center", va="center",
            fontsize=tsz, fontweight="bold", color=color, zorder=4,
            multialignment="center")
    if sub:
        # If there are 3 lines, shift the subtext slightly lower to balance it
        n_lines = sub.count('\n') + 1
        sy_offset = -0.24 if n_lines >= 3 else -0.20
        ax.text(cx+0.12, cy + h*sy_offset, sub, ha="center", va="center",
                fontsize=ssz, color=MID, zorder=4,
                multialignment="center", linespacing=1.4)

def pill(ax, cx, cy, w, h, color, title, sub="", tsz=26, ssz=20):
    """Solid-fill pill for input nodes."""
    r = 0.28
    bg = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                        boxstyle=f"round,pad=0,rounding_size={r}",
                        fc=color+"1A", ec=color, lw=3.0, zorder=2)
    ax.add_patch(bg)
    ty = cy + (h*0.15 if sub else 0)
    ax.text(cx, ty, title, ha="center", va="center",
            fontsize=tsz, fontweight="bold", color=color, zorder=4)
    if sub:
        ax.text(cx, cy-h*0.22, sub, ha="center", va="center",
                fontsize=ssz, color=MID, style="italic", zorder=4)

def arrow(ax, x0, y0, x1, y1, color=MID, lw=3.2, label="", label_pos="side"):
    ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=25, connectionstyle="arc3,rad=0"),
                zorder=5)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        if label_pos == "side":
            ax.text(mx+0.25, my, label, fontsize=22, color=color, style="italic",
                    ha="left", va="center", zorder=6,
                    bbox=dict(fc="white", ec="none", pad=1.5))
        elif label_pos == "above":
            ax.text(mx, my+0.25, label, fontsize=22, color=color, style="italic",
                    ha="center", va="bottom", zorder=6,
                    bbox=dict(fc="white", ec="none", pad=1.5))

def phase_strip(ax, y, label, color):
    """Thin horizontal phase label strip."""
    ax.plot([0.4, W-0.4], [y, y], color=color, lw=0.6, ls="--", alpha=0.4, zorder=0)
    ax.text(0.4, y+0.08, label, fontsize=24, color=color,
            fontweight="bold", va="bottom")

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(W/2, 23.4,
        "Causal Behavioral State Reasoning — End-to-End Architecture",
        ha="center", va="center", fontsize=38, fontweight="bold", color=DARK)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA INGESTION
# ══════════════════════════════════════════════════════════════════════════════
phase_strip(ax, 22.7, "Phase 1 · Data Ingestion", BLUE)

IW, IH = 4.6, 1.3
ix = [3.1, 8.4, 13.7, 19.0]
ilabels = [("Audit Logs",     "host provenance stream"),
           ("Event Extractor","proc / file / net edges"),
           ("Time Binner",    "W = 1-s windows"),
           (r"Feature Matrix $X_t$", "edge-count vector (unlabeled)")]

for i, (x, (t, s)) in enumerate(zip(ix, ilabels)):
    pill(ax, x, 21.5, IW, IH, BLUE, t, s, tsz=24, ssz=19)
    if i < len(ix)-1:
        arrow(ax, x+IW/2, 21.5, ix[i+1]-IW/2, 21.5, BLUE, lw=3.2)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — CAUSAL DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════
phase_strip(ax, 20.0, "Phase 2 · Online Causal Discovery", PURPLE)

# Xt drops down
arrow(ax, 19.0, 20.85, 19.0, 19.4, BLUE, lw=3.2, label="$X_t$", label_pos="side")

box(ax, 7.0, 18.2, 8.5, 2.4, PURPLE,
    "PCMCI+ Discovery",
    "Online causal discovery via PCMCI\nRobust ParCorr (15% contamination)",
    tsz=28, ssz=22)

box(ax, 19.0, 18.2, 8.5, 2.4, PURPLE,
    "SCM Regression",
    r"$X^j_t = \sum \beta_{ij} X^i_{t-\tau} + \epsilon^j_t$" + "\n"
    "Residuals model deviation per variable",
    tsz=28, ssz=22)

arrow(ax, 11.25, 18.2, 14.75, 18.2, PURPLE, lw=3.2, label="$G_t$", label_pos="above")

# ══════════════════════════════════════════════════════════════════════════════
# STATE PROJECTION
# ══════════════════════════════════════════════════════════════════════════════
phase_strip(ax, 16.2, "Behavioral State Projection", INDIGO)

arrow(ax,  7.0, 17.0, 11.0, 15.6, PURPLE, lw=3.2)
arrow(ax, 19.0, 17.0, 15.0, 15.6, PURPLE, lw=3.2, label=r"$\epsilon_t$", label_pos="side")

box(ax, W/2, 14.4, 11.0, 2.4, INDIGO,
    r"State Projection  $\phi(X_t, G_t)$",
    r"$S_t \in \mathcal{S}_{\mathrm{struct}}^{3D} \times"
    r" \mathcal{S}_{\mathrm{dyn}}^{3D} \times \mathcal{S}_{\mathrm{beh}}^{6D}$"
    "\n12-D behavioral state representation",
    tsz=28, ssz=22)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — THREE PARALLEL OPERATORS
# ══════════════════════════════════════════════════════════════════════════════
phase_strip(ax, 12.2, "Phase 3 · Multi-Perspective State Reasoning", ORANGE)

ops = [
    (4.5,  ORANGE, "$R_L$ — Local Reasoner",
     "Instantaneous SCM violations\n"
     "Causal Anomaly Score (CAS)\n"
     r"$w(1-\min_j p_j) + (1-w)\hat{F}(\chi^2)$"),
    (W/2,  TEAL,   "$R_T$ — Temporal Reasoner",
     "LSTM Autoencoder ($k=8$ sequence)\n"
     "Learns state-transition dynamics\n"
     "High recon. error → anomaly"),
    (21.5, GREEN,  "$R_B$ — Behavioral Reasoner",
     "RF classifier (BPR) on 12-D states\n"
     "Learns attack manifold boundary\n"
     "Semi-supervised (sparse labels)"),
]

for ox, oc, ot, os in ops:
    box(ax, ox, 10.3, 7.8, 3.2, oc, ot, os, tsz=26, ssz=21)
    arrow(ax, W/2, 13.2, ox, 11.9, INDIGO, lw=3.2)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — RISK FUSION
# ══════════════════════════════════════════════════════════════════════════════
phase_strip(ax, 7.5, "Phase 4 · Risk Fusion", PINK)

for ox, oc, _ , _ in ops:
    if oc == ORANGE:
        arrow(ax, ox, 8.7, 10.5, 7.2, oc, lw=3.2)
    elif oc == TEAL:
        arrow(ax, ox, 8.7, 13.0, 7.2, oc, lw=3.2)
    elif oc == GREEN:
        arrow(ax, ox, 8.7, 15.5, 7.2, oc, lw=3.2)

box(ax, W/2, 6.0, 11.0, 2.4, PINK,
    "Risk Fusion — Stacked Meta-Learner",
    r"$\mathrm{Risk}_t = \sigma(w_1 R_L + w_2 R_T + w_3 R_B + b)$" + "\n"
    "Stacked Meta-Learner (val partition)",
    tsz=28, ssz=22)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — CONFORMAL CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
phase_strip(ax, 4.0, "Phase 5 · Conformal Calibration & Alerting", AMBER)

arrow(ax, W/2, 4.8, W/2, 3.6, PINK, lw=3.2, label=r"$\mathrm{Risk}_t$", label_pos="side")

box(ax, W/2, 2.4, 11.0, 2.4, AMBER,
    "Online Conformal Calibration",
    "Rolling queue of risk scores (size $M$)\n" +
    r"$\alpha_{t+1} = \alpha_t + \eta\,(\mathrm{FPR_{target}} - \mathbb{I}[\mathrm{alarm}])$" + "\n" +
    "FPR guarantee under concept drift",
    tsz=28, ssz=22)

# ══════════════════════════════════════════════════════════════════════════════
# ALERT
# ══════════════════════════════════════════════════════════════════════════════
arrow(ax, W/2, 1.2, W/2, 0.9, AMBER, lw=3.2)

al_w, al_h = 13.0, 0.8
al_bg = FancyBboxPatch((W/2-al_w/2, 0.3), al_w, al_h,
                       boxstyle="round,pad=0,rounding_size=0.22",
                       fc=RED, ec=RED, lw=3.0, zorder=2)
ax.add_patch(al_bg)
ax.text(W/2, 0.7, "⚠  APT ALERT  —  FPR ≤ 5% guaranteed  ⚠",
        ha="center", va="center", fontsize=28, fontweight="bold",
        color="white", zorder=4)

plt.tight_layout(pad=0.2)
out = "/DATA/shourya_2211mc14/Arp/work2/results/final/zerocausal_architecture.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print("Saved →", out)
