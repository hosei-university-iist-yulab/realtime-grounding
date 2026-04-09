"""
Visualization utilities for TGP experiments.

Generates publication-quality figures for IEEE VTC2026-Spring paper.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MaxNLocator
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Visualization disabled.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# IEEE conference style settings
IEEE_STYLE = {
    "figure.figsize": (3.5, 2.5),  # Single column width
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "axes.grid": True,
    "grid.alpha": 0.3,
}

# Color palette (colorblind-friendly)
COLORS = {
    "tgp": "#2ecc71",      # Green - our method
    "cloud": "#e74c3c",    # Red - cloud baselines
    "local": "#3498db",    # Blue - local baselines
    "ablation": "#9b59b6", # Purple - ablation variants
    "baseline": "#95a5a6", # Gray - other baselines
}


def setup_style():
    """Apply IEEE publication style."""
    if MATPLOTLIB_AVAILABLE:
        plt.rcParams.update(IEEE_STYLE)
        if SEABORN_AVAILABLE:
            sns.set_palette("colorblind")


def plot_latency_comparison(
    results: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    title: str = "Latency Comparison"
) -> Optional[Any]:
    """
    Plot latency comparison bar chart.

    Args:
        results: Dict mapping method name to {"mean_ms": x, "std_ms": y}
        output_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    setup_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    methods = list(results.keys())
    means = [results[m].get("mean_ms", 0) for m in methods]
    stds = [results[m].get("std_ms", 0) for m in methods]

    # Color based on method type
    colors = []
    for m in methods:
        if "tgp" in m.lower() or "ours" in m.lower():
            colors.append(COLORS["tgp"])
        elif "cloud" in m.lower() or "gpt" in m.lower() or "claude" in m.lower():
            colors.append(COLORS["cloud"])
        else:
            colors.append(COLORS["baseline"])

    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Method")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.annotate(f"{mean:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=6)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved to {output_path}")

    return fig


def plot_accuracy_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ["accuracy", "f1"],
    output_path: Optional[str] = None,
    title: str = "Accuracy Comparison"
) -> Optional[Any]:
    """
    Plot accuracy/F1 comparison grouped bar chart.

    Args:
        results: Dict mapping method to metric values
        metrics: List of metrics to plot
        output_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    setup_style()
    fig, ax = plt.subplots(figsize=(4.5, 2.5))

    methods = list(results.keys())
    n_methods = len(methods)
    n_metrics = len(metrics)

    x = np.arange(n_methods)
    width = 0.8 / n_metrics

    for i, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) for m in methods]
        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=metric.upper())

    ax.set_ylabel("Score")
    ax.set_xlabel("Method")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")

    return fig


def plot_ablation_study(
    results: Dict[str, Dict[str, float]],
    baseline_key: str = "full_system",
    output_path: Optional[str] = None,
    title: str = "Ablation Study"
) -> Optional[Any]:
    """
    Plot ablation study showing contribution of each component.

    Args:
        results: Dict mapping config name to metrics
        baseline_key: Key for full system baseline
        output_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))

    configs = [k for k in results.keys() if k != baseline_key]
    baseline = results.get(baseline_key, {})

    # Latency subplot
    ax1 = axes[0]
    latencies = [results[c].get("latency_ms", 0) for c in configs]
    baseline_lat = baseline.get("latency_ms", 80)

    colors = [COLORS["ablation"] if lat > baseline_lat * 1.1 else COLORS["tgp"] for lat in latencies]
    ax1.barh(configs, latencies, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axvline(baseline_lat, color=COLORS["tgp"], linestyle="--", label=f"Full ({baseline_lat:.0f}ms)")
    ax1.set_xlabel("Latency (ms)")
    ax1.set_title("Latency Impact")
    ax1.legend(fontsize=6)

    # Accuracy subplot
    ax2 = axes[1]
    accuracies = [results[c].get("accuracy", 0) for c in configs]
    baseline_acc = baseline.get("accuracy", 0.95)

    colors = [COLORS["ablation"] if acc < baseline_acc * 0.9 else COLORS["tgp"] for acc in accuracies]
    ax2.barh(configs, accuracies, color=colors, edgecolor="black", linewidth=0.5)
    ax2.axvline(baseline_acc, color=COLORS["tgp"], linestyle="--", label=f"Full ({baseline_acc:.0%})")
    ax2.set_xlabel("Accuracy")
    ax2.set_xlim(0, 1)
    ax2.set_title("Accuracy Impact")
    ax2.legend(fontsize=6)

    plt.suptitle(title, fontsize=10)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")

    return fig


def plot_staleness_roc(
    fpr: List[float],
    tpr: List[float],
    auc: float,
    output_path: Optional[str] = None,
    title: str = "Staleness Detection ROC"
) -> Optional[Any]:
    """
    Plot ROC curve for staleness detection.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: Area under curve
        output_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    setup_style()
    fig, ax = plt.subplots(figsize=(3.5, 3))

    ax.plot(fpr, tpr, color=COLORS["tgp"], lw=2, label=f"TGP (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")

    return fig


def plot_scalability(
    sensor_counts: List[int],
    latencies: Dict[str, List[float]],
    output_path: Optional[str] = None,
    title: str = "Scalability Analysis"
) -> Optional[Any]:
    """
    Plot latency vs. number of sensors.

    Args:
        sensor_counts: X-axis values (number of sensors)
        latencies: Dict mapping method to list of latencies
        output_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    setup_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for method, lats in latencies.items():
        color = COLORS["tgp"] if "tgp" in method.lower() else COLORS["baseline"]
        marker = "o" if "tgp" in method.lower() else "s"
        ax.plot(sensor_counts, lats, marker=marker, color=color, label=method, linewidth=1.5)

    ax.set_xlabel("Number of Sensors")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(title)
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")

    return fig


def generate_latex_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    caption: str = "Experimental Results",
    label: str = "tab:results"
) -> str:
    """
    Generate LaTeX table from results.

    Args:
        results: Dict mapping method to metrics
        metrics: List of metric names to include
        caption: Table caption
        label: LaTeX label

    Returns:
        LaTeX table string
    """
    methods = list(results.keys())

    # Build header
    header = "Method & " + " & ".join(m.replace("_", " ").title() for m in metrics) + " \\\\"

    # Build rows
    rows = []
    for method in methods:
        values = []
        for metric in metrics:
            val = results[method].get(metric, 0)
            if isinstance(val, float):
                if "accuracy" in metric or "f1" in metric:
                    values.append(f"{val:.2%}")
                elif "ms" in metric or "latency" in metric:
                    values.append(f"{val:.1f}")
                else:
                    values.append(f"{val:.2f}")
            else:
                values.append(str(val))
        rows.append(f"{method} & " + " & ".join(values) + " \\\\")

    table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{l{'c' * len(metrics)}}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    return table


def save_all_figures(
    results: Dict[str, Any],
    output_dir: str = "output/figures"
) -> List[str]:
    """
    Generate and save all figures for paper.

    Args:
        results: Experiment results dict
        output_dir: Output directory

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # Latency comparison
    if "latency" in results:
        path = output_dir / "fig_latency.pdf"
        plot_latency_comparison(results["latency"], str(path))
        saved.append(str(path))

    # Accuracy comparison
    if "accuracy" in results:
        path = output_dir / "fig_accuracy.pdf"
        plot_accuracy_comparison(results["accuracy"], output_path=str(path))
        saved.append(str(path))

    # Ablation study
    if "ablation" in results:
        path = output_dir / "fig_ablation.pdf"
        plot_ablation_study(results["ablation"], output_path=str(path))
        saved.append(str(path))

    # Scalability
    if "scalability" in results:
        path = output_dir / "fig_scalability.pdf"
        data = results["scalability"]
        plot_scalability(
            data.get("sensor_counts", []),
            data.get("latencies", {}),
            output_path=str(path)
        )
        saved.append(str(path))

    return saved


if __name__ == "__main__":
    # Test visualizations
    print("Testing visualization module...")

    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping tests.")
    else:
        # Test latency comparison
        test_results = {
            "TGP (Ours)": {"mean_ms": 78, "std_ms": 12},
            "Cloud GPT-4": {"mean_ms": 650, "std_ms": 150},
            "PostgreSQL+LLM": {"mean_ms": 130, "std_ms": 25},
            "Prompt-only": {"mean_ms": 55, "std_ms": 8}
        }

        fig = plot_latency_comparison(test_results, title="Latency Comparison")
        plt.show()

        # Generate LaTeX table
        latex = generate_latex_table(
            test_results,
            ["mean_ms", "std_ms"],
            caption="Latency Results",
            label="tab:latency"
        )
        print("\nGenerated LaTeX:")
        print(latex)
