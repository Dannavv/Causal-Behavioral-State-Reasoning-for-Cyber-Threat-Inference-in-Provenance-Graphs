import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Canvas ───────────────────────────────────────────────────────────────────
W, H = 45, 16
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
def box(ax, cx, cy, w, h, color, title, sub="", tsz=30, ssz=24):
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
                        fc="white", ec=color, lw=3.0, zorder=2)
    ax.add_patch(bg)
    # left accent bar
    bar = FancyBboxPatch((cx-w/2, cy-h/2), 0.22, h,
                         boxstyle=f"round,pad=0,rounding_size={r}",
                         fc=color, ec="none", zorder=3)
    ax.add_patch(bar)
    # title - visually balanced using h*0.18 offset
    ty = cy + (h*0.18 if sub else 0)
    ax.text(cx+0.12, ty, title, ha="center", va="center",
            fontsize=tsz, fontweight="bold", color=color, zorder=4,
            multialignment="center")
    if sub:
        n_lines = sub.count('\n') + 1
        sy_offset = -0.24 if n_lines >= 3 else -0.20
        ax.text(cx+0.12, cy + h*sy_offset, sub, ha="center", va="center",
                fontsize=ssz, color=MID, zorder=4,
                multialignment="center", linespacing=1.4)

def pill(ax, cx, cy, w, h, color, title, sub="", tsz=28, ssz=22):
    """Solid-fill pill for input nodes."""
    r = 0.28
    bg = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                        boxstyle=f"round,pad=0,rounding_size={r}",
                        fc=color+"1A", ec=color, lw=2.8, zorder=2)
    ax.add_patch(bg)
    ty = cy + (h*0.15 if sub else 0)
    ax.text(cx, ty, title, ha="center", va="center",
            fontsize=tsz, fontweight="bold", color=color, zorder=4)
    if sub:
        ax.text(cx, cy-h*0.22, sub, ha="center", va="center",
                fontsize=ssz, color=MID, style="italic", zorder=4)

def arrow(ax, x0, y0, x1, y1, color=MID, lw=3.0, label="", label_pos="side"):
    ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=24, connectionstyle="arc3,rad=0"),
                zorder=5)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        if label_pos == "side":
            ax.text(mx+0.25, my, label, fontsize=20, color=color, style="italic",
                    ha="left", va="center", zorder=6,
                    bbox=dict(fc="white", ec="none", pad=1.5))
        elif label_pos == "above":
            ax.text(mx, my+0.25, label, fontsize=20, color=color, style="italic",
                    ha="center", va="bottom", zorder=6,
                    bbox=dict(fc="white", ec="none", pad=1.5))

def vertical_boundary(ax, x, color):
    """Vertical boundary divider."""
    ax.plot([x, x], [0.4, 13.5], color=color, lw=0.6, ls="--", alpha=0.4, zorder=0)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE & BOUNDARIES
# ══════════════════════════════════════════════════════════════════════════════
ax.text(W/2, 15.2,
        "Causal Behavioral State Reasoning — End-to-End Architecture",
        ha="center", va="center", fontsize=34, fontweight="bold", color=DARK)

# Vertical zone labels (centered on phase centers)
ax.text(3.6, 14.1, "Phase 1 · Data Ingestion", fontsize=20, color=BLUE, fontweight="bold", ha="center")
ax.text(11.5, 14.1, "Phase 2 · Causal Discovery", fontsize=20, color=PURPLE, fontweight="bold", ha="center")
ax.text(20.5, 14.1, "Behavioral State Projection", fontsize=20, color=INDIGO, fontweight="bold", ha="center")
ax.text(29.5, 14.1, "Phase 3 · Multi-Perspective Reasoning", fontsize=20, color=ORANGE, fontweight="bold", ha="center")
ax.text(38.5, 14.1, "Phase 4 & 5 · Risk Fusion & Calibration", fontsize=20, color=PINK, fontweight="bold", ha="center")

# Dividers (5 Zones)
vertical_boundary(ax, 7.0, BLUE)
vertical_boundary(ax, 16.0, PURPLE)
vertical_boundary(ax, 25.0, INDIGO)
vertical_boundary(ax, 34.0, ORANGE)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA INGESTION (Stacked)
# ══════════════════════════════════════════════════════════════════════════════
IW, IH = 6.5, 1.2
ix = 3.6
iy = [12.3, 9.5, 6.7, 3.9]
ilabels = [("Audit Logs",     "host provenance stream"),
           ("Event Extractor","proc / file / net edges"),
           ("Time Binner",    "W = 1-s windows"),
           (r"Feature Matrix $X_t$", "edge-count vector (unlabeled)")]

for i, (y, (t, s)) in enumerate(zip(iy, ilabels)):
    pill(ax, ix, y, IW, IH, BLUE, t, s)
    if i < len(iy)-1:
        arrow(ax, ix, y - IH/2, ix, iy[i+1] + IH/2, BLUE, lw=3.0)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — CAUSAL DISCOVERY (Stacked)
# ══════════════════════════════════════════════════════════════════════════════
# Feature Matrix Xt feeds both
arrow(ax, ix + IW/2, iy[3], 11.5 - 4.25, 11.2, BLUE, lw=3.0)
arrow(ax, ix + IW/2, iy[3], 11.5 - 4.25, 5.2, BLUE, lw=3.0, label="$X_t$", label_pos="side")

box(ax, 11.5, 11.2, 8.5, 2.4, PURPLE,
    "PCMCI+ Discovery",
    "Online causal discovery via PCMCI\nRobust ParCorr (15% contamination)")

box(ax, 11.5, 5.2, 8.5, 2.4, PURPLE,
    "SCM Regression",
    r"$X^j_t = \sum \beta_{ij} X^i_{t-\tau} + \epsilon^j_t$" + "\n"
    "Residuals model deviation per variable")

arrow(ax, 11.5, 10.0, 11.5, 6.4, PURPLE, lw=3.0, label="$G_t$", label_pos="side")

# ══════════════════════════════════════════════════════════════════════════════
# STATE PROJECTION
# ══════════════════════════════════════════════════════════════════════════════
arrow(ax, 11.5 + 4.25, 11.2, 20.5 - 4.25, 8.2, PURPLE, lw=3.0)
arrow(ax, 11.5 + 4.25, 5.2, 20.5 - 4.25, 8.2, PURPLE, lw=3.0, label=r"$\epsilon_t$", label_pos="side")

box(ax, 20.5, 8.2, 8.5, 2.4, INDIGO,
    r"State Projection  $\phi(X_t, G_t)$",
    r"$S_t \in \mathcal{S}_{\mathrm{struct}}^{3D} \times"
    r" \mathcal{S}_{\mathrm{dyn}}^{3D} \times \mathcal{S}_{\mathrm{beh}}^{6D}$"
    "\n12-D behavioral state representation")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — THREE PARALLEL OPERATORS (Stacked)
# ══════════════════════════════════════════════════════════════════════════════
ops = [
    (12.0, ORANGE, "$R_L$ — Local Reasoner",
     "Instantaneous SCM violations\n"
     "Causal Anomaly Score (CAS)\n"
     r"$w(1-\min_j p_j) + (1-w)\hat{F}(\chi^2)$"),
    (8.2,  TEAL,   "$R_T$ — Temporal Reasoner",
     "LSTM Autoencoder ($k=8$ sequence)\n"
     "Learns state-transition dynamics\n"
     "High recon. error → anomaly"),
    (4.4,  GREEN,  "$R_B$ — Behavioral Reasoner",
     "RF classifier (BPR) on 12-D states\n"
     "Learns attack manifold boundary\n"
     "Semi-supervised (sparse labels)"),
]

for oy, oc, ot, os in ops:
    box(ax, 29.5, oy, 8.5, 3.0, oc, ot, os, tsz=28, ssz=22)
    arrow(ax, 20.5 + 4.25, 8.2, 29.5 - 4.25, oy, INDIGO, lw=3.0)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 & 5 — RISK FUSION & CALIBRATION (Stacked in Column 5)
# ══════════════════════════════════════════════════════════════════════════════
# Operators feed Risk Fusion
for oy, oc, _ , _ in ops:
    if oc == ORANGE:
        arrow(ax, 29.5 + 4.25, oy, 38.5 - 4.25, 11.2, oc, lw=3.0)
    elif oc == TEAL:
        arrow(ax, 29.5 + 4.25, oy, 38.5 - 4.25, 10.5, oc, lw=3.0)
    elif oc == GREEN:
        arrow(ax, 29.5 + 4.25, oy, 38.5 - 4.25, 9.8, oc, lw=3.0)

box(ax, 38.5, 10.5, 8.5, 2.4, PINK,
    "Risk Fusion — Stacked Meta-Learner",
    r"$\mathrm{Risk}_t = \sigma(w_1 R_L + w_2 R_T + w_3 R_B + b)$" + "\n"
    "Stacked Meta-Learner (val partition)")

arrow(ax, 38.5, 9.3, 38.5, 6.7, PINK, lw=3.0, label=r"$\mathrm{Risk}_t$", label_pos="side")

box(ax, 38.5, 5.5, 8.5, 2.4, AMBER,
    "Online Conformal Calibration",
    "Rolling queue of risk scores (size $M$)\n" +
    r"$\alpha_{t+1} = \alpha_t + \eta\,(\mathrm{FPR_{target}} - \mathbb{I}[\mathrm{alarm}])$" + "\n" +
    "FPR guarantee under concept drift")

arrow(ax, 38.5, 4.3, 38.5, 2.4, AMBER, lw=3.0)

box(ax, 38.5, 1.8, 8.5, 1.2, RED,
    "⚠  APT ALERT  ⚠",
    "FPR ≤ 5% guaranteed", tsz=28, ssz=22)

plt.tight_layout(pad=0.2)
out = "/DATA/shourya_2211mc14/Arp/work2/results/final/presentation_architecture.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print("Saved →", out)
