import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# DDIM-LI Manuscript Figures

Generates every main-text figure for the manuscript (see `MANUSCRIPT_PLAN.md`
Section 2). One figure per section below, each self-contained.

**Before running:** edit the `PATHS` dict in the Setup cell to point at your
actual output files (this notebook was built and tested against synthetic
data matching the real schemas in this sandboxed environment, which has no
access to your actual training outputs). Journal-style white theme matches
`baseline_cnn/evaluate_cnn.py`'s `_make_plots` for visual consistency with
every other plot already generated in this project.

Run this from the `manuscript/` directory (paths below are relative to it).""")

code('''import os, sys, csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from scipy.ndimage import maximum_filter

REPO_ROOT = os.path.abspath("..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "baseline_cnn"))

# ---- Journal-style white theme, matching baseline_cnn/evaluate_cnn.py's _make_plots ----
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black", "text.color": "black",
    "grid.color": "#cccccc", "grid.linestyle": "--", "grid.linewidth": 0.5,
    "legend.framealpha": 0.9, "legend.edgecolor": "#cccccc", "font.size": 10,
})

def styled_ax(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors="black")
    for s in ax.spines.values():
        s.set_edgecolor("black"); s.set_linewidth(0.8)
    ax.grid(True, color="#cccccc", linestyle="--", linewidth=0.5, zorder=0)

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

# ==== EDIT THESE to your actual output paths before running for real ====
PATHS = {
    "diffusion_npz":          "../outputs/nature_256_T36_ir_li_only/eval_ens_50/plot_data.npz",
    "diffusion_csv":          "../outputs/nature_256_T36_ir_li_only/eval_ens_50/metrics_per_step.csv",
    "persistence_npz":        "../outputs/persistence_baseline/persistence_pr_curves.npz",
    "pysteps_li_npz":         "../pysteps_li/optical_flow_li_pr_curves.npz",
    "pysteps_ir_npz":         "../pysteps_ir/optical_flow_ir_pr_curves.npz",
    "cnn_npz":                "../baseline_cnn/baseline_cnn_pr_curves.npz",
    "cnn_csv":                "../baseline_cnn/baseline_cnn_metrics.csv",
    "lightgbm_npz":           "../baseline_lightgbm/baseline_lightgbm_pr_curves.npz",
    "lightgbm_csv":           "../baseline_lightgbm/baseline_lightgbm_metrics.csv",
    "positional_ceiling_csv": "../positional_ceiling.csv",
    "diffusion_npz_n10":      "../outputs/nature_256_T36_ir_li_only/eval/plot_data.npz",
    "diffusion_npz_n30":      "../outputs/nature_256_T36_ir_li_only/eval_ens_30/plot_data.npz",
    "diffusion_npz_n50":      "../outputs/nature_256_T36_ir_li_only/eval_ens_50/plot_data.npz",
    "example_cases_npz":      "../example_cases.npz",
}
DT_MIN, T_OUT = 10, 6
LEAD_TIMES = [(t + 1) * DT_MIN for t in range(T_OUT)]
LEAD_KM_SCALES = [1, 2, 4, 8, 16, 32]
PIXEL_SIZE_KM = 4.0

def have(key):
    p = PATHS[key]
    ok = os.path.isfile(p)
    if not ok:
        print(f"[missing] {key} -> {p}  (edit PATHS or generate this file first)")
    return ok
''')

md("""## Figure 1 — Study overview

(a) Map of the study regions. (b) Model architecture schematic.

Per `MANUSCRIPT_PLAN.md`: the architecture schematic is typically a
hand-built conceptual diagram, not something generated from data — if you
already have one, just copy it into `figures/fig1_overview.png` directly.
The region-map panel below uses **placeholder coordinates** — replace
`TEST_REGIONS` with the real bounding boxes from `configs/evaluate.yaml`
before trusting this.""")

code('''EXISTING_FIG1_PATH = None  # e.g. "/path/to/existing/architecture_figure.png"

if EXISTING_FIG1_PATH and os.path.exists(EXISTING_FIG1_PATH):
    import shutil
    shutil.copy(EXISTING_FIG1_PATH, os.path.join(FIGDIR, "fig1_overview.png"))
    print(f"Copied existing figure -> {FIGDIR}/fig1_overview.png")
else:
    # PLACEHOLDER bounds -- verify against configs/evaluate.yaml's test_roots
    TEST_REGIONS = {
        "region_1": (10, 20, -5, 5),
        "region_2": (20, 30, -5, 5),
        "region_3": (10, 20, -15, -5),
        "region_4": (20, 30, -15, -5),
    }
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (lon0, lon1, lat0, lat1) in TEST_REGIONS.items():
        ax.add_patch(plt.Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0,
                                   fill=False, edgecolor="#1f77b4", linewidth=2))
        ax.text((lon0 + lon1) / 2, (lat0 + lat1) / 2, name, ha="center", fontsize=8)
    ax.set_xlim(0, 40); ax.set_ylim(-25, 15)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Study regions (PLACEHOLDER bounds -- verify before use)")
    styled_ax(ax)
    fig.savefig(os.path.join(FIGDIR, "fig1_overview.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print("WARNING: placeholder region bounds in TEST_REGIONS -- replace with real coordinates")
''')

md("""## Figure 2 — Headline performance

PR-AUC vs. lead time, model vs. the three physical baselines, with 95% CI
shaded bands. Reuses `bootstrap_pr_auc_ci.py`'s validated `load_run`/
`bootstrap_ci`/`pr_auc` functions directly rather than reimplementing the
sequence-level block bootstrap (a real risk of subtle inconsistency
otherwise — see that script's own docstring on why a naive resampling
shortcut understated variance).""")

code('''from bootstrap_pr_auc_ci import load_run, bootstrap_ci, pr_auc as _pr_auc_fn

def marginal_ci_by_lead(npz_path, n_boot=1000, seed=0):
    run = load_run(npz_path)
    rng = np.random.default_rng(seed)
    means, los, his = [], [], []
    for t in sorted(run.keys()):
        prob, lbl, seqid = run[t]
        result = bootstrap_ci(prob, lbl, seqid, n_boot, rng)  # dict: point/ci_lo/ci_hi/n_boot_valid
        means.append(result["point"])
        los.append(result["ci_lo"])
        his.append(result["ci_hi"])
    return np.array(means), np.array(los), np.array(his)

if have("diffusion_npz") and have("persistence_npz") and have("pysteps_li_npz") and have("pysteps_ir_npz"):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    series = [
        ("Diffusion model", PATHS["diffusion_npz"], "#1f77b4"),
        ("Persistence",     PATHS["persistence_npz"], "#7f7f7f"),
        ("Optical flow (LI)", PATHS["pysteps_li_npz"], "#2ca02c"),
        ("Optical flow (IR)", PATHS["pysteps_ir_npz"], "#d62728"),
    ]
    for label, path, color in series:
        m, lo, hi = marginal_ci_by_lead(path)
        ax.plot(LEAD_TIMES, m, color=color, linewidth=2, marker="o", markersize=5, label=label)
        ax.fill_between(LEAD_TIMES, lo, hi, color=color, alpha=0.15)
    ax.set_xlabel("Lead time (min)"); ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 1)
    ax.set_title("Model vs. physical baselines", fontweight="bold")
    styled_ax(ax)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig2_headline_performance.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
''')

md("""## Figure 3 — Displacement is incoherent, not advective

Loads `diagnose_positional_ceiling.py`'s output. That script only printed
to console until this session — it now supports `--output_csv`. Generate
the data first:

```bash
python diagnose_positional_ceiling.py --config configs/evaluate.yaml \\
    --n_seq 60 --n_members 10 --pool_radii 1 2 3 --max_shift 4 \\
    --output_csv positional_ceiling.csv
```""")

code('''if have("positional_ceiling_csv"):
    with open(PATHS["positional_ceiling_csv"]) as f:
        rows = list(csv.DictReader(f))
    lead = [float(r["lead_min"]) for r in rows]
    exact = [float(r["exact"]) for r in rows]
    best_shift = [float(r["best_shift"]) for r in rows]
    near_cols = [k for k in rows[0].keys() if k.startswith("near_")]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(lead, exact, color="#7f7f7f", linewidth=2, marker="o", label="Exact (pixelwise)")
    for i, col in enumerate(sorted(near_cols, key=lambda c: int(c.split("_")[1][:-2]))):
        vals = [float(r[col]) for r in rows]
        km = col.split("_")[1]
        ax.plot(lead, vals, linewidth=1.5, marker="s", markersize=4,
                color=mcm.get_cmap("viridis")(i / max(len(near_cols) - 1, 1)),
                label=f"Neighbourhood-relaxed ({km})")
    ax.plot(lead, best_shift, color="#d62728", linewidth=2, marker="^", label="Best global shift")
    ax.axhline(0.7, color="black", linestyle=":", linewidth=1, label="0.7 reference")
    ax.set_xlabel("Lead time (min)"); ax.set_ylabel("PR-AUC")
    ax.set_title("Displacement is incoherent, not advective", fontweight="bold")
    styled_ax(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig3_incoherent_displacement.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
''')

md("""## Figure 4 — Example nowcasts (qualitative)

**No existing script produces this data.** Needs a small companion script
(not yet written) that loads a trained checkpoint, runs inference on a
few chosen test sequences, and saves an npz with per-case arrays:
`context_ir`, `context_li` (T_in,H,W), `target_li` (T_out,H,W), and
`ensemble_li` (n_members,T_out,H,W) — physical-space, post
`_li_to_physical`. Case selection needs real data access (not available
in this sandboxed environment) — pick 2-3 representative storms:
a clear success, a genuinely uncertain/multimodal case, and honestly,
a failure case.""")

code('''if have("example_cases_npz"):
    data = np.load(PATHS["example_cases_npz"], allow_pickle=True)
    case_names = [k.split("_context_ir")[0] for k in data.files if k.endswith("_context_ir")]

    for case in case_names:
        ctx_ir = data[f"{case}_context_ir"]        # (T_in, H, W)
        tgt_li = data[f"{case}_target_li"]          # (T_out, H, W)
        ens_li = data[f"{case}_ensemble_li"]        # (n_members, T_out, H, W)
        n_show_members = min(3, ens_li.shape[0])

        n_cols = 2 + n_show_members
        fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.2))
        axes[0].imshow(ctx_ir[-1], cmap="turbo"); axes[0].set_title("Last context frame (IR)")
        axes[1].imshow(tgt_li[-1], cmap="turbo", vmin=0, vmax=1); axes[1].set_title("Ground truth (+60min)")
        for m in range(n_show_members):
            axes[2 + m].imshow(ens_li[m, -1], cmap="turbo", vmin=0, vmax=1)
            axes[2 + m].set_title(f"Ensemble member {m+1}")
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Example case: {case}", fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, f"fig4_example_{case}.png"), dpi=300, bbox_inches="tight", facecolor="white")
        plt.show()
''')

md("""## Figure 5 — Baseline-verification methodology (the F4 story)

Two panels: FSS-vs-scale (diffusion vs. CNN, reusing the same computation
as `compare_diffusion_vs_cnn_fss.py`) and the ensemble-size sensitivity
result (PR-AUC gap vs. `n_members`, the diminishing-returns curve from
`FINDINGS.md` F4).""")

code('''# --- Panel A: FSS vs scale, diffusion vs CNN ---
def cnn_fss_by_threshold_scale(csv_path, thresholds, scales):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for thr in thresholds:
        for s in scales:
            col = f"fss_{thr}_scale{s}"
            out[(thr, s)] = (np.mean([float(r[col]) for r in rows])
                             if col in rows[0] else np.nan)
    return out

if have("diffusion_npz") and have("cnn_csv"):
    diff_data = np.load(PATHS["diffusion_npz"], allow_pickle=True)
    thresholds = [0.1, 0.3, 0.5]
    cnn_fss = cnn_fss_by_threshold_scale(PATHS["cnn_csv"], thresholds, LEAD_KM_SCALES)

    fig, axes = plt.subplots(1, len(thresholds), figsize=(5.5 * len(thresholds), 5))
    scale_km = [(2 * s + 1) * PIXEL_SIZE_KM for s in LEAD_KM_SCALES]
    for col, thr in enumerate(thresholds):
        ax = axes[col]
        d_vals = [float(np.mean(diff_data[f"fss_thr{thr}_s{s}"]))
                 if f"fss_thr{thr}_s{s}" in diff_data else np.nan for s in LEAD_KM_SCALES]
        c_vals = [cnn_fss[(thr, s)] for s in LEAD_KM_SCALES]
        ax.plot(scale_km, d_vals, color="#1f77b4", linewidth=2, marker="o", label="Diffusion model")
        ax.plot(scale_km, c_vals, color="#d62728", linewidth=2, marker="s", label="CNN baseline")
        ax.axhline(0.5, color="#999999", linestyle="--", linewidth=1)
        ax.set_title(f"FSS (p > {thr})", fontweight="bold")
        ax.set_xlabel("Scale (km)"); ax.set_ylabel("FSS")
        ax.set_ylim(-0.05, 1.05)
        styled_ax(ax)
        ax.legend(fontsize=8)
    fig.suptitle("FSS vs spatial scale: diffusion vs. CNN baseline", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig5a_fss_vs_scale.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

# --- Panel B: ensemble-size (n_members) sensitivity ---
n_labels, n_paths = ["n=10", "n=30", "n=50"], ["diffusion_npz_n10", "diffusion_npz_n30", "diffusion_npz_n50"]
available = [(lbl, PATHS[k]) for lbl, k in zip(n_labels, n_paths) if os.path.isfile(PATHS[k])]
if available:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for lbl, path in available:
        run = load_run(path)
        aucs = [_pr_auc_fn(*run[t][:2]) for t in sorted(run.keys())]
        ax.plot(LEAD_TIMES[:len(aucs)], aucs, marker="o", linewidth=2, label=lbl)
    ax.set_xlabel("Lead time (min)"); ax.set_ylabel("PR-AUC")
    ax.set_title("Ensemble-size sensitivity", fontweight="bold")
    styled_ax(ax)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig5b_n_members_sensitivity.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
else:
    print("[missing] none of the n_members=10/30/50 eval directories found -- edit PATHS")
''')

md("""## Figure 6 — Genuine uncertainty quantification

Calibration/reliability diagram (from `cal_prob_t`/`cal_label_t` raw pairs
already saved in `plot_data.npz`) alongside CRPS and spread-skill vs. lead
time (already computed per-lead by `evaluate.py`, saved in
`metrics_per_step.csv`'s `crps`/`spread_skill` columns).""")

code('''from sklearn.calibration import calibration_curve

if have("diffusion_npz"):
    data = np.load(PATHS["diffusion_npz"], allow_pickle=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # --- Reliability diagram ---
    ax1.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.2, zorder=5)
    cmap = mcm.get_cmap("turbo")
    for t in range(T_OUT):
        pk, lk = f"cal_prob_{t}", f"cal_label_{t}"
        if pk in data and lk in data:
            frac_pos, mean_pred = calibration_curve(data[lk], data[pk], n_bins=10, strategy="uniform")
            ax1.plot(mean_pred, frac_pos, color=cmap(t / max(T_OUT - 1, 1)), linewidth=1.5,
                    marker="o", markersize=3, label=f"+{(t+1)*DT_MIN}m")
    ax1.set_xlabel("Mean predicted probability"); ax1.set_ylabel("Observed frequency")
    ax1.set_title("Reliability diagram", fontweight="bold")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    styled_ax(ax1)
    ax1.legend(fontsize=7, ncol=2)

    # --- CRPS + spread-skill vs lead time ---
    if have("diffusion_csv"):
        with open(PATHS["diffusion_csv"]) as f:
            rows = list(csv.DictReader(f))
        lead = [float(r["lead_min"]) for r in rows]
        crps = [float(r["crps"]) for r in rows if "crps" in r]
        ss = [float(r["spread_skill"]) for r in rows if "spread_skill" in r]
        ax2b = ax2.twinx()
        l1, = ax2.plot(lead, crps, color="#1f77b4", linewidth=2, marker="o", label="CRPS")
        l2, = ax2b.plot(lead, ss, color="#2ca02c", linewidth=2, marker="s", label="Spread-skill ratio")
        ax2.set_xlabel("Lead time (min)"); ax2.set_ylabel("CRPS", color="#1f77b4")
        ax2b.set_ylabel("Spread-skill ratio", color="#2ca02c")
        ax2.set_title("CRPS and spread-skill vs. lead time", fontweight="bold")
        styled_ax(ax2)
        ax2.legend(handles=[l1, l2], fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig6_uncertainty_quantification.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

nbf.validate(nb)
with open("/home/claude/DDIM-LI/manuscript/generate_figures.ipynb", "w") as f:
    nbf.write(nb, f)
print("Notebook written and validated OK")
