"""Plot beam selection results: entropy vs uniform vs random baselines."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(_REPO_ROOT, "results", "beam_eval")
BASELINE_PATH = os.path.join(OUT_ROOT, "det", "CRN_r18_256x704_128x128_4key", "metrics_summary.json")
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

BUDGETS = [0, 20, 40, 60, 80, 100]  # pct values on x-axis (0 = no radar)
STRATEGIES = ["entropy", "uniform", "random"]

COLORS = {
    "entropy": "#2196F3",   # blue
    "uniform": "#FF9800",   # orange
    "random":  "#9E9E9E",   # grey
    "baseline": "#4CAF50",  # green
}
MARKERS = {"entropy": "o", "uniform": "s", "random": "^"}
LABELS  = {"entropy": "Entropy (ours)", "uniform": "Uniform", "random": "Random"}


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def mean_ap_for_class(d, cls):
    return sum(d["label_aps"][cls].values()) / 4


def get_path(strategy, pct):
    if pct == 0:
        return os.path.join(OUT_ROOT, "beam_filtered_r18_budget0pct", "metrics_summary.json")
    if pct == 100 and strategy == "entropy":
        return os.path.join(OUT_ROOT, "beam_filtered_r18_budget100pct", "metrics_summary.json")
    suffix = "" if strategy == "entropy" else f"_{strategy}"
    return os.path.join(OUT_ROOT, f"beam_filtered_r18_budget{pct}pct{suffix}", "metrics_summary.json")


def collect():
    baseline = load_metrics(BASELINE_PATH)
    no_radar = load_metrics(get_path("entropy", 0))

    data = {}  # strategy -> pct -> metrics dict
    for strategy in STRATEGIES:
        data[strategy] = {}
        for pct in BUDGETS:
            path = get_path(strategy, pct)
            if os.path.exists(path):
                data[strategy][pct] = load_metrics(path)
            # uniform/random don't have pct=0 or pct=100 separately — they share with entropy
            elif pct == 0:
                data[strategy][pct] = no_radar
            elif pct == 100:
                data[strategy][pct] = baseline

    return baseline, no_radar, data


def style_ax(ax, title, xlabel, ylabel, ylim=None):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(BUDGETS)
    ax.grid(True, linestyle="--", alpha=0.4)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_metric(ax, data, baseline_val, key_fn, ylabel, title):
    """key_fn(metrics_dict) -> scalar value."""
    ax.axhline(baseline_val, color=COLORS["baseline"], linestyle="--",
               linewidth=1.8, label="Baseline (100% radar, no filter)", zorder=1)

    for strategy in STRATEGIES:
        xs, ys = [], []
        for pct in BUDGETS:
            m = data[strategy].get(pct)
            if m is not None:
                xs.append(pct)
                ys.append(key_fn(m))
        ax.plot(xs, ys, color=COLORS[strategy], marker=MARKERS[strategy],
                linewidth=2, markersize=7, label=LABELS[strategy], zorder=2)

    style_ax(ax, title, "Radar budget (%)", ylabel)


def plot_relative(ax, data, baseline_val, no_radar_val, key_fn, ylabel, title):
    """Normalised so 0% radar = 0, 100% (baseline) = 1."""
    span = baseline_val - no_radar_val

    ax.axhline(1.0, color=COLORS["baseline"], linestyle="--",
               linewidth=1.8, label="Baseline (100% radar, no filter)", zorder=1)
    ax.axhline(0.0, color="#555", linestyle=":", linewidth=1.2, zorder=1)

    for strategy in STRATEGIES:
        xs, ys = [], []
        for pct in BUDGETS:
            m = data[strategy].get(pct)
            if m is not None:
                val = key_fn(m)
                xs.append(pct)
                ys.append((val - no_radar_val) / span if span > 0 else 0.0)
        ax.plot(xs, ys, color=COLORS[strategy], marker=MARKERS[strategy],
                linewidth=2, markersize=7, label=LABELS[strategy], zorder=2)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    style_ax(ax, title, "Radar budget (%)", ylabel)


def main():
    baseline, no_radar, data = collect()

    b_nds   = baseline["nd_score"]
    b_map   = baseline["mean_ap"]
    b_car   = mean_ap_for_class(baseline, "car")
    b_truck = mean_ap_for_class(baseline, "truck")

    nr_nds   = no_radar["nd_score"]
    nr_map   = no_radar["mean_ap"]
    nr_car   = mean_ap_for_class(no_radar, "car")
    nr_truck = mean_ap_for_class(no_radar, "truck")

    # ── Figure 1: Absolute NDS & mAP ────────────────────────────────────────
    fig1, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig1.suptitle("Entropy-guided Beam Selection vs Baselines", fontsize=14, fontweight="bold")

    plot_metric(axes[0], data, b_nds,
                lambda m: m["nd_score"], "NDS", "NDS vs Radar Budget")
    plot_metric(axes[1], data, b_map,
                lambda m: m["mean_ap"], "mAP", "mAP vs Radar Budget")

    fig1.tight_layout()
    fig1.savefig(os.path.join(PLOT_DIR, "fig1_nds_map.png"), dpi=150)
    print("Saved fig1_nds_map.png")

    # ── Figure 2: Car & Truck AP ─────────────────────────────────────────────
    fig2, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle("Per-class AP: Car & Truck", fontsize=14, fontweight="bold")

    plot_metric(axes[0], data, b_car,
                lambda m: mean_ap_for_class(m, "car"), "AP (mean over thresholds)", "Car AP vs Radar Budget")
    plot_metric(axes[1], data, b_truck,
                lambda m: mean_ap_for_class(m, "truck"), "AP (mean over thresholds)", "Truck AP vs Radar Budget")

    fig2.tight_layout()
    fig2.savefig(os.path.join(PLOT_DIR, "fig2_car_truck_ap.png"), dpi=150)
    print("Saved fig2_car_truck_ap.png")

    # ── Figure 3: Relative improvement (0% radar = 0, baseline = 100%) ──────
    fig3, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig3.suptitle("Relative Recovery vs Radar Budget\n(0% = no radar, 100% = full baseline)",
                  fontsize=14, fontweight="bold")

    plot_relative(axes[0, 0], data, b_nds, nr_nds,
                  lambda m: m["nd_score"], "Relative NDS recovery", "NDS")
    plot_relative(axes[0, 1], data, b_map, nr_map,
                  lambda m: m["mean_ap"], "Relative mAP recovery", "mAP")
    plot_relative(axes[1, 0], data, b_car, nr_car,
                  lambda m: mean_ap_for_class(m, "car"), "Relative AP recovery", "Car AP")
    plot_relative(axes[1, 1], data, b_truck, nr_truck,
                  lambda m: mean_ap_for_class(m, "truck"), "Relative AP recovery", "Truck AP")

    fig3.tight_layout()
    fig3.savefig(os.path.join(PLOT_DIR, "fig3_relative_improvement.png"), dpi=150)
    print("Saved fig3_relative_improvement.png")

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Strategy':<20} {'Budget':>7}  {'NDS':>6}  {'mAP':>6}  {'Car AP':>7}  {'Truck AP':>8}")
    print("=" * 90)
    print(f"{'Baseline (no filter)':<20} {'100%':>7}  {b_nds:.4f}  {b_map:.4f}  {b_car:.4f}   {b_truck:.4f}")
    print(f"{'No radar':<20} {'0%':>7}  {nr_nds:.4f}  {nr_map:.4f}  {nr_car:.4f}   {nr_truck:.4f}")
    print("-" * 90)
    for strategy in STRATEGIES:
        for pct in [20, 40, 60, 80, 100]:
            m = data[strategy].get(pct)
            if m is None:
                continue
            nds   = m["nd_score"]
            mmap  = m["mean_ap"]
            car   = mean_ap_for_class(m, "car")
            truck = mean_ap_for_class(m, "truck")
            print(f"{LABELS[strategy]:<20} {pct:>6}%  {nds:.4f}  {mmap:.4f}  {car:.4f}   {truck:.4f}")
        print()
    print("=" * 90)


if __name__ == "__main__":
    main()
