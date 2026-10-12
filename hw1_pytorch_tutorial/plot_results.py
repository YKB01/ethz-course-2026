"""Create presentation figures from recorded metrics, without modifying notebooks."""

import csv
import base64
import json
import os
import uuid
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hw1_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "results"


def embed_figure(exercise, filename):
    """Attach a standard Jupyter image output; keep plotting imports out of cells."""
    path = ROOT.parent / "src" / f"ex{exercise}.ipynb"
    notebook = json.loads(path.read_text())
    for entry in notebook["cells"]:
        entry.setdefault("id", uuid.uuid4().hex[:8])
    cell = [c for c in notebook["cells"] if c["cell_type"] == "code"][-1]
    # Replace a previously attached figure rather than accumulating duplicates.
    cell["outputs"] = [o for o in cell.get("outputs", []) if o.get("output_type") != "display_data"]
    cell["outputs"].append({
        "output_type": "display_data", "metadata": {},
        "data": {"image/png": base64.b64encode((ROOT / filename).read_bytes()).decode("ascii"),
                 "text/plain": [f"Recorded ex{exercise} experiment curves"]},
    })
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")


def smooth(values, window=25):
    values = np.asarray(values)
    window = min(window, len(values))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_ex3():
    metrics = json.loads((ROOT / "ex3_metrics.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    losses = metrics["train_losses"]
    smoothed = smooth(losses)
    axes[0].plot(np.arange(len(losses) - len(smoothed) + 1, len(losses) + 1), smoothed)
    axes[0].set(xlabel="Optimizer update", ylabel="Cross-entropy", title="MLP training loss (25-step mean)")
    epochs = np.arange(1, len(metrics["test_losses"]) + 1)
    axes[1].plot(epochs, metrics["train_epoch_losses"], "o-", label="Train")
    axes[1].plot(epochs, metrics["test_losses"], "o-", label="Test")
    axes[1].set(xlabel="Epoch", ylabel="Cross-entropy", title=f"Final test accuracy: {metrics['test_accs'][-1]:.2%}")
    axes[1].set_xticks(epochs)
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(ROOT / "ex3_loss_curves.png", dpi=180)
    plt.close(fig)
    embed_figure(3, "ex3_loss_curves.png")


def plot_ex4():
    metrics = json.loads((ROOT / "ex4_metrics.json").read_text())
    results = metrics["results"]
    variants = list(dict.fromkeys(r["variant"] for r in results))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    summary = []
    for variant in variants:
        runs = [r for r in results if r["variant"] == variant]
        loss_curves = np.array([smooth(r["train_losses"]) for r in runs])
        steps = np.arange(len(runs[0]["train_losses"]) - loss_curves.shape[1] + 1,
                          len(runs[0]["train_losses"]) + 1)
        axes[0].plot(steps, loss_curves.mean(0), label=variant.upper())
        accuracy = np.array([r["test_accs"] for r in runs]) * 100
        mean = accuracy.mean(0)
        std = accuracy.std(0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
        epochs = np.arange(1, len(mean) + 1)
        line, = axes[1].plot(epochs, mean, "o-", label=variant.upper())
        axes[1].fill_between(epochs, mean - std, mean + std, color=line.get_color(), alpha=.15)
        finals = np.array([r["final_test_accuracy"] for r in runs])
        bests = np.array([r["best_test_accuracy"] for r in runs])
        summary.append({
            "variant": variant, "seeds": len(runs), "parameters": runs[0]["num_parameters"],
            "final_accuracy_mean": finals.mean(),
            "final_accuracy_sample_std": finals.std(ddof=1) if len(runs) > 1 else 0.,
            "best_accuracy_mean": bests.mean(),
        })
    axes[2].bar([s["variant"].upper() for s in summary], [s["final_accuracy_mean"] * 100 for s in summary],
                yerr=[s["final_accuracy_sample_std"] * 100 for s in summary], capsize=5)
    axes[0].set(xlabel="Optimizer update", ylabel="Cross-entropy", title="Training loss (25-step mean, averaged seeds)")
    axes[1].set(xlabel="Epoch", ylabel="Test accuracy (%)", title="Mean +/- sample SD across seeds")
    axes[1].set_xticks(epochs)
    axes[2].set(ylabel="Final test accuracy (%)", title="Final mean +/- sample SD")
    axes[2].set_ylim(0, 100)
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
    axes[0].legend()
    axes[1].legend()
    fig.savefig(ROOT / "ex4_comparison.png", dpi=180)
    plt.close(fig)
    embed_figure(4, "ex4_comparison.png")
    with (ROOT / "ex4_summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    for entry in summary:
        print(f"{entry['variant']}: parameters={entry['parameters']}, "
              f"final={entry['final_accuracy_mean']:.2%} +/- {entry['final_accuracy_sample_std']:.2%}, "
              f"mean best={entry['best_accuracy_mean']:.2%}")
    # Paired differences are descriptive, not a significance test.
    baseline = {r['seed']: r['final_test_accuracy'] for r in results if r['variant'] == 'ffn'}
    for variant in variants:
        if variant == 'ffn':
            continue
        differences = [(r['final_test_accuracy'] - baseline[r['seed']]) * 100
                       for r in results if r['variant'] == variant and r['seed'] in baseline]
        print(f"{variant} - FFN paired final differences (percentage points): "
              + ", ".join(f"{value:+.2f}" for value in differences))


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "ex3_metrics.json").exists():
        plot_ex3()
    if (ROOT / "ex4_metrics.json").exists():
        plot_ex4()
    print(f"Figures saved under {ROOT}")
