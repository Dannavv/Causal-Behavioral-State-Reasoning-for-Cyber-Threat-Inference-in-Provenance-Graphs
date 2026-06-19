import os
import shutil

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

# ── canvas ────────────────────────────────────────────────────────────────────
W, H = 12.0, 7.2

# ── palette ──────────────────────────────────────────────────────────────────
GRN = ("#d1fae5", "#065f46")   # I/O nodes
BLU = ("#dbeafe", "#1e3a8a")   # data-processing
PRP = ("#ede9fe", "#4c1d95")   # causal core
AMB = ("#fef3c7", "#78350f")   # decision layer
RED = ("#fee2e2", "#7f1d1d")   # alert output
ARR = "#374151"                # default arrow / text colour


# ── helpers ──────────────────────────────────────────────────────────────────

def _rect(ax, x, y, w, h, fc, ec, lw=1.1, ls="-", zo=3):
    ax.add_patch(patches.Rectangle(
        (x, y), w, h, linewidth=lw, edgecolor=ec,
        facecolor=fc, linestyle=ls, zorder=zo))


def _txt(ax, x, y, s, fs=8.8, fw="bold", c="#0f172a",
         ha="center", va="center", lsp=1.3):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, fontweight=fw,
            color=c, linespacing=lsp, zorder=5)


def component(ax, x, y, w, h, line1, line2, fc, ec, fs1=8.8, fs2=7.6):
    """Two-line labelled box."""
    _rect(ax, x, y, w, h, fc, ec)
    _txt(ax, x + w / 2, y + h * 0.64, line1, fs=fs1)
    _txt(ax, x + w / 2, y + h * 0.26, line2, fs=fs2, fw="normal", c="#1e293b")


def harrow(ax, x1, x2, y, c=ARR, lbl="", lbl_dy=0.11):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=c,
                                lw=0.9, mutation_scale=9), zorder=6)
    if lbl:
        ax.text((x1 + x2) / 2, y + lbl_dy, lbl,
                ha="center", va="bottom", fontsize=6.5,
                color=c, style="italic", zorder=7)


def varrow(ax, x, y1, y2, c=ARR, lbl="", lside="right", ms=9):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=c,
                                lw=0.9, mutation_scale=ms), zorder=6)
    if lbl:
        xo = 0.11 if lside == "right" else -0.11
        ha = "left" if lside == "right" else "right"
        ax.text(x + xo, (y1 + y2) / 2, lbl,
                ha=ha, va="center", fontsize=6.5,
                color=c, style="italic", zorder=7)


def phase_group(ax, x, y, w, h, label, c):
    """Dashed group box with a phase label at the top-left corner."""
    _rect(ax, x, y, w, h, "#f8fafc", c, lw=0.85, ls="--", zo=1)
    ax.text(x + 0.15, y + h - 0.10, label,
            ha="left", va="top", fontsize=7.8, fontweight="bold",
            color=c, style="italic", zorder=2)


def phase_num(ax, y_center, num):
    ax.text(0.05, y_center, num, fontsize=10, fontweight="bold",
            color="#cbd5e1", ha="left", va="center", zorder=5)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── grid constants ────────────────────────────────────────────────────────
    # Phase I  – 4 boxes
    P1y, P1h = 5.55, 0.82
    bw1, g1 = 2.68, 0.25
    x1 = [0.25 + i * (bw1 + g1) for i in range(4)]
    # x1 ≈ [0.25, 3.18, 6.11, 9.04]   right edge of last: 9.04+2.68=11.72

    # Phase II – 3 boxes (same x-grid as Phase IV)
    P2y, P2h = 3.35, 1.05
    bw2, g2 = 3.62, 0.32
    x2 = [0.25 + i * (bw2 + g2) for i in range(3)]
    # x2 ≈ [0.25, 4.19, 8.13]   right edge: 8.13+3.62=11.75
    cx2 = [x2[i] + bw2 / 2 for i in range(3)]
    # cx2 ≈ [2.06, 6.00, 9.94]

    # CAS box – full width
    CASy, CASh = 2.22, 0.87

    # Phase IV – same x-grid as Phase II
    P4y, P4h = 0.34, 0.88
    x4 = x2

    # Xₜ data-bus y-coordinate (inside Phase II group, above component boxes)
    BUS_Y = 4.82

    # ── TITLE ────────────────────────────────────────────────────────────────
    _txt(ax, W / 2, 6.90, "ZeroCausal — End-to-End Architecture",
         fs=13.5, ha="center", va="center")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE I — DATA INGESTION
    # ═══════════════════════════════════════════════════════════════════════════
    phase_num(ax, P1y + P1h / 2, "I")

    component(ax, x1[0], P1y, bw1, P1h,
              "Audit Logs", "host provenance stream", *GRN)
    component(ax, x1[1], P1y, bw1, P1h,
              "Event Extractor", "proc / file / net edges", *BLU)
    component(ax, x1[2], P1y, bw1, P1h,
              "Time Binner", "W = 1-sec windows", *BLU)
    component(ax, x1[3], P1y, bw1, P1h,
              "Feature Matrix  Xₜ ∈ ℝᵈ",
              "edge-count vector (unlabeled)", *BLU)

    for i in range(3):
        harrow(ax, x1[i] + bw1, x1[i + 1], P1y + P1h / 2)

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE II — ONLINE CAUSAL MODELING  (dashed group)
    # ═══════════════════════════════════════════════════════════════════════════
    phase_group(ax, 0.18, 3.12, 11.62, 2.22,
                "Phase II — Online Causal Modeling", PRP[1])
    phase_num(ax, P2y + P2h / 2, "II")

    component(ax, x2[0], P2y, bw2, P2h,
              "PCMCI+ Discovery",
              "GPDC independence test,  τmax = 2 s", *PRP)
    component(ax, x2[1], P2y, bw2, P2h,
              "SCM Regression",
              "linear / RF  →  residuals  εʲₜ", *PRP)
    component(ax, x2[2], P2y, bw2, P2h,
              "Structural Novelty Tracker",
              "unseen edges  ▸  p = 10⁻¹⁵", *PRP)

    harrow(ax, x2[0] + bw2, x2[1], P2y + P2h / 2,
           PRP[1], lbl="causal parents")

    # ── Xₜ data bus: Feature Matrix ──► all three Phase II components ─────────
    # Vertical feed line from Feature Matrix bottom down to the bus
    ax.plot([cx2[2], cx2[2]], [P1y, BUS_Y],
            color=BLU[1], lw=1.0, zorder=4)
    ax.text(cx2[2] + 0.13, (P1y + BUS_Y) / 2, "Xₜ",
            ha="left", va="center", fontsize=8.2, fontweight="bold",
            color=BLU[1], style="italic", zorder=5)

    # Horizontal bus spanning all three Phase II box centres
    ax.plot([cx2[0], cx2[2]], [BUS_Y, BUS_Y],
            color=BLU[1], lw=1.0, zorder=4)

    # Drop arrows: bus → each Phase II box top
    for cx in cx2:
        varrow(ax, cx, BUS_Y, P2y + P2h, BLU[1], ms=8)

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE III — HYBRID ANOMALY SCORE
    # ═══════════════════════════════════════════════════════════════════════════
    _rect(ax, 0.25, CASy, 11.50, CASh, "#f5f3ff", PRP[1], lw=1.35)
    _txt(ax, 6.00, CASy + CASh * 0.67,
         "Causal Anomaly Score  (CAS)", fs=11.0)
    _txt(ax, 6.00, CASy + CASh * 0.26,
         "CASₜ  =  w · (1 − p_min)  +  (1 − w) · Ŝres",
         fs=9.5, fw="normal", c="#3b0764")

    phase_num(ax, CASy + CASh / 2, "III")

    # SCM residuals → CAS
    varrow(ax, cx2[1], P2y, CASy + CASh, PRP[1], lbl="ε residuals")
    # Structural Novelty → CAS
    varrow(ax, cx2[2], P2y, CASy + CASh, PRP[1], lbl="p_novel")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE IV — CONFORMAL DECISION LAYER  (dashed group)
    # ═══════════════════════════════════════════════════════════════════════════
    phase_group(ax, 0.18, 0.12, 11.62, 1.93,
                "Phase IV — Conformal Decision Layer", AMB[1])
    phase_num(ax, P4y + P4h / 2, "IV")

    component(ax, x4[0], P4y, bw2, P4h,
              "Calibration Queue",
              "M = 245 sorted scores", *AMB)
    component(ax, x4[1], P4y, bw2, P4h,
              "Conformal p-value\n& Adaptive αₜ",
              "FPR-bounded threshold update", *AMB)
    component(ax, x4[2], P4y, bw2, P4h,
              "⚠  APT Alert",
              "FPR-controlled detection", *RED, fs1=10.0)

    harrow(ax, x4[0] + bw2, x4[1], P4y + P4h / 2, AMB[1])
    harrow(ax, x4[1] + bw2, x4[2], P4y + P4h / 2,
           AMB[1], lbl="conf_pval < αₜ ?", lbl_dy=0.10)

    # CAS score → Calibration Queue
    varrow(ax, 6.00, CASy, P4y + P4h, AMB[1], lbl="CASₜ")

    # ── Drift-refit feedback arrow (left-side L-shape) ────────────────────────
    lx = 0.05
    fb_bot = P4y + P4h * 0.35        # exit on left of Phase IV
    fb_top = P2y + P2h * 0.50        # entry on left of Phase II

    ax.plot([x4[0], lx, lx],
            [fb_bot, fb_bot, fb_top],
            color="#dc2626", lw=0.9,
            linestyle=(0, (5, 2)),
            solid_capstyle="round", zorder=4)
    ax.annotate("", xy=(x2[0], fb_top), xytext=(lx + 0.01, fb_top),
                arrowprops=dict(arrowstyle="->", color="#dc2626",
                                lw=0.9, mutation_scale=8), zorder=6)
    ax.text(lx - 0.03, (fb_bot + fb_top) / 2,
            "drift\nrefit", ha="right", va="center",
            fontsize=6.5, color="#dc2626", style="italic", zorder=5)

    # ── save ─────────────────────────────────────────────────────────────────
    plt.tight_layout(pad=0.15)

    out_dir = os.path.join(os.getcwd(), "results", "final")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "zerocausal_architecture.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Architecture diagram saved to {out_path}")

    os.makedirs("plots", exist_ok=True)
    shutil.copy(out_path, "plots/zerocausal_architecture.png")
    print("Copied to plots/zerocausal_architecture.png")

    artifact_path = (
        "/DATA/shourya_2211mc14/.gemini/antigravity-ide/brain/"
        "b7d7d095-4573-49cf-933c-6f4730b480d7/plots/zerocausal_architecture.png"
    )
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    shutil.copy(out_path, artifact_path)
    print(f"Copied to artifact path {artifact_path}")


if __name__ == "__main__":
    main()
