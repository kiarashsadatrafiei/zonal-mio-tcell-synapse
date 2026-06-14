"""
Q1 extension for zonal_mio_simulation.py
=========================================

Place this file in the same directory as your original file:
    zonal_mio_simulation.py

Then run:
    python zonal_mio_q1_extension.py --mode standard

What it adds to the original pipeline
-------------------------------------
1. Keeps your original scenarios:
   - separable
   - amplitude_overlap
2. Adds a harder feature-level stress scenario:
   - stress_overlap
3. Re-evaluates amplitude-only, amplitude+carrier, alignment-only, and combined models.
4. Saves publication-ready CSV/PDF/PNG outputs.
5. Adds cross-scenario model comparison and a severity-sweep robustness curve.

Scientific use in manuscript
----------------------------
Use the original scenario as a formal behavior demonstration, and use stress_overlap
as a robustness/sensitivity analysis. Do not call this empirical validation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import zonal_mio_simulation as base
except ImportError as exc:
    raise ImportError(
        "Could not import zonal_mio_simulation.py. Put this file in the same folder as "
        "zonal_mio_simulation.py and run it from that folder."
    ) from exc


@dataclass(frozen=True)
class StressConfig:
    source_scenario: str = "amplitude_overlap"
    output_scenario: str = "stress_overlap"
    severity: float = 0.75
    noise_scale: float = 0.55
    zone_blur: float = 0.35
    amplitude_noise_scale: float = 0.10
    random_state: int = 2026


REGIME_ORDER = base.REGIMES
REGIME_LABELS = base.REGIME_LABELS
FEATURE_SETS = base.feature_sets()
MODEL_ORDER = [
    "amplitude_only",
    "amplitude_plus_carrier_mass",
    "alignment_only",
    "combined",
]
MODEL_LABELS = {
    "amplitude_only": "Amplitude",
    "amplitude_plus_carrier_mass": "Amplitude\n+ carrier mass",
    "alignment_only": "Alignment",
    "combined": "Combined",
}

ALIGNMENT_FEATURES = FEATURE_SETS["alignment_only"]
AMPLITUDE_PLUS_CARRIER_FEATURES = FEATURE_SETS["amplitude_plus_carrier_mass"]


def set_q1_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 500,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_original_scenario(config: base.SimulationConfig, scenario: str, rerun: bool = False) -> Path:
    outdir = Path(config.output_root) / scenario
    feature_file = outdir / "synthetic_features.csv"
    summary_file = outdir / "model_comparison_summary.csv"

    if rerun or not feature_file.exists() or not summary_file.exists():
        print(f"Running original scenario via zonal_mio_simulation.py: {scenario}")
        base.run_scenario(scenario, config)
    else:
        print(f"Found existing scenario outputs: {outdir}")

    return outdir


def make_stress_features(df: pd.DataFrame, cfg: StressConfig) -> pd.DataFrame:
    """
    Degrade the feature space without altering regime labels.

    The perturbation is intentionally conservative and interpretable:
    - alignment descriptors are pulled toward the pooled mean;
    - Gaussian measurement noise is added;
    - d/p/c zonal descriptors are mixed within each sample to mimic boundary-registration uncertainty;
    - amplitude and carrier mass remain weakly discriminative.
    """
    rng = np.random.default_rng(cfg.random_state)
    out = df.copy()

    for col in ALIGNMENT_FEATURES:
        x = out[col].astype(float).to_numpy()
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        out[col] = (
            (1.0 - cfg.severity) * x
            + cfg.severity * mu
            + rng.normal(0.0, cfg.noise_scale * sd, size=len(out))
        )

    zone_triplets = [
        ["A_d_mean", "A_p_mean", "A_c_mean"],
        ["A_d_var", "A_p_var", "A_c_var"],
        ["A_d_early", "A_p_middle", "A_c_late"],
    ]
    for triplet in zone_triplets:
        X = out[triplet].astype(float).to_numpy()
        row_mean = X.mean(axis=1, keepdims=True)
        out[triplet] = (1.0 - cfg.zone_blur) * X + cfg.zone_blur * row_mean

    for col in ALIGNMENT_FEATURES:
        if col == "P_C":
            out[col] = np.clip(out[col], 0.0, 1.0)
        else:
            out[col] = np.clip(out[col], -1.0, 1.0)

    for col in AMPLITUDE_PLUS_CARRIER_FEATURES:
        x = out[col].astype(float).to_numpy()
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        out[col] = x + rng.normal(0.0, cfg.amplitude_noise_scale * sd, size=len(out))

    out["scenario"] = cfg.output_scenario
    return out


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_q1_pca(df: pd.DataFrame, outdir: Path, title: str, name: str = "03_feature_space_pca_q1") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.ravel()

    for ax, model_name in zip(axes, MODEL_ORDER):
        features = FEATURE_SETS[model_name]
        X = df[features].values
        X = SimpleImputer(strategy="median").fit_transform(X)
        X = StandardScaler().fit_transform(X)
        coords = PCA(n_components=2, random_state=2026).fit_transform(X)

        for regime in REGIME_ORDER:
            idx = df["regime"].values == regime
            ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                s=14,
                alpha=0.70,
                label=REGIME_LABELS[regime],
            )

        ax.set_title(MODEL_LABELS[model_name].replace("\n", " "))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="center right")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0.02, 0.02, 0.86, 0.95])
    savefig(fig, outdir, name)


def plot_q1_model_comparison(summary: pd.DataFrame, outdir: Path, scenario_name: str, name: str = "04_model_comparison_q1") -> None:
    classifiers = ["logistic", "random_forest"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True, constrained_layout=True)

    for ax, classifier in zip(axes, classifiers):
        sub = summary[summary["classifier"] == classifier].set_index("model")
        means = [float(sub.loc[m, "macro_AUROC_mean"]) for m in MODEL_ORDER]
        sds = [float(sub.loc[m, "macro_AUROC_sd"]) for m in MODEL_ORDER]
        x = np.arange(len(MODEL_ORDER))

        ax.bar(x, means, yerr=sds, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=20, ha="right")
        ax.set_ylim(0.40, 1.03)
        ax.set_ylabel("Cross-validated macro-AUROC")
        ax.set_title("Logistic" if classifier == "logistic" else "Random forest")

        for i, value in enumerate(means):
            ax.text(i, min(value + 0.025, 1.015), f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Model comparison under {scenario_name.replace('_', ' ')}", fontsize=14)
    savefig(fig, outdir, name)


def plot_q1_confusions(pooled_predictions: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], outdir: Path, name: str = "05_confusion_matrices_q1") -> None:
    keys = [
        "logistic__amplitude_only",
        "logistic__amplitude_plus_carrier_mass",
        "logistic__alignment_only",
        "logistic__combined",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), constrained_layout=True)

    for ax, key, model_name in zip(axes, keys, MODEL_ORDER):
        y_true, y_pred, classes = pooled_predictions[key]
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        im = ax.imshow(cm, vmin=0, vmax=1, aspect="auto")
        ax.set_title(MODEL_LABELS[model_name].replace("\n", " "))
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels([str(c).replace("_", "\n") for c in classes], rotation=45, ha="right")
        ax.set_yticklabels([str(c).replace("_", "\n") for c in classes])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle("Cross-validated confusion matrices, logistic classifier", fontsize=14)
    savefig(fig, outdir, name)


def run_stress_scenario(config: base.SimulationConfig, stress_cfg: StressConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_dir = Path(config.output_root) / stress_cfg.source_scenario
    outdir = Path(config.output_root) / stress_cfg.output_scenario
    outdir.mkdir(parents=True, exist_ok=True)

    source_file = source_dir / "synthetic_features.csv"
    if not source_file.exists():
        raise FileNotFoundError(f"Missing source feature file: {source_file}")

    print(f"\nCreating stress scenario from: {source_file}")
    df_source = pd.read_csv(source_file)
    df_stress = make_stress_features(df_source, stress_cfg)

    performance, pooled_predictions = base.evaluate_models(df_stress, config)
    summary = base.summarize_performance(performance)
    increments = base.incremental_value(summary)

    df_stress.to_csv(outdir / "synthetic_features.csv", index=False)
    performance.to_csv(outdir / "model_comparison_folds.csv", index=False)
    summary.to_csv(outdir / "model_comparison_summary.csv", index=False)
    increments.to_csv(outdir / "incremental_value.csv", index=False)

    with open(outdir / "stress_parameters.json", "w", encoding="utf-8") as f:
        json.dump(asdict(stress_cfg), f, indent=2)

    plot_q1_pca(df_stress, outdir, "Feature-space structure under stress-overlap perturbation")
    plot_q1_model_comparison(summary, outdir, stress_cfg.output_scenario)
    plot_q1_confusions(pooled_predictions, outdir)

    print("\nStress scenario summary:")
    print(summary.to_string(index=False))
    print("\nStress incremental value:")
    print(increments.to_string(index=False))
    print(f"\nSaved stress outputs to: {outdir}")

    return df_stress, summary, increments


def load_summary(config: base.SimulationConfig, scenario: str) -> pd.DataFrame:
    path = Path(config.output_root) / scenario / "model_comparison_summary.csv"
    df = pd.read_csv(path)
    df.insert(0, "scenario", scenario)
    return df


def plot_cross_scenario_comparison(config: base.SimulationConfig, scenarios: List[str]) -> None:
    root = Path(config.output_root)
    combined_dir = root / "cross_scenario_summary"
    combined_dir.mkdir(parents=True, exist_ok=True)

    summaries = [load_summary(config, s) for s in scenarios]
    summary_all = pd.concat(summaries, ignore_index=True)
    summary_all.to_csv(combined_dir / "model_comparison_summary_all_scenarios.csv", index=False)

    classifiers = ["logistic", "random_forest"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True, constrained_layout=True)
    x = np.arange(len(MODEL_ORDER))
    width = 0.24

    for ax, classifier in zip(axes, classifiers):
        for k, scenario in enumerate(scenarios):
            sub = summary_all[(summary_all["classifier"] == classifier) & (summary_all["scenario"] == scenario)].set_index("model")
            means = [float(sub.loc[m, "macro_AUROC_mean"]) for m in MODEL_ORDER]
            sds = [float(sub.loc[m, "macro_AUROC_sd"]) for m in MODEL_ORDER]
            ax.bar(x + (k - 1) * width, means, width=width, yerr=sds, capsize=3, label=scenario.replace("_", " "))

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=20, ha="right")
        ax.set_ylim(0.40, 1.03)
        ax.set_ylabel("Macro-AUROC")
        ax.set_title("Logistic" if classifier == "logistic" else "Random forest")
        ax.legend(frameon=False)

    fig.suptitle("Descriptor-family performance across synthetic scenarios", fontsize=14)
    savefig(fig, combined_dir, "06_cross_scenario_model_comparison")


def quick_logistic_auroc(df: pd.DataFrame, config: base.SimulationConfig, repeats: int = 5) -> pd.DataFrame:
    y = df["regime"].values
    cv = RepeatedStratifiedKFold(n_splits=config.cv_splits, n_repeats=repeats, random_state=config.seed)
    rows = []

    for model_name in MODEL_ORDER:
        features = FEATURE_SETS[model_name]
        X = df[features].values
        X = SimpleImputer(strategy="median").fit_transform(X)
        X = StandardScaler().fit_transform(X)
        classifier = LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced")

        scores = []
        for train_idx, test_idx in cv.split(X, y):
            clf = clone(classifier)
            clf.fit(X[train_idx], y[train_idx])
            probability = clf.predict_proba(X[test_idx])
            scores.append(roc_auc_score(y[test_idx], probability, multi_class="ovr", average="macro"))

        rows.append({"model": model_name, "macro_AUROC_mean": np.mean(scores), "macro_AUROC_sd": np.std(scores)})

    return pd.DataFrame(rows)


def plot_stress_severity_curve(config: base.SimulationConfig, source_scenario: str = "amplitude_overlap") -> None:
    source_path = Path(config.output_root) / source_scenario / "synthetic_features.csv"
    df_source = pd.read_csv(source_path)
    outdir = Path(config.output_root) / "cross_scenario_summary"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    severities = np.linspace(0.0, 0.90, 10)
    for sev in severities:
        cfg = StressConfig(severity=float(sev), source_scenario=source_scenario, output_scenario="severity_sweep")
        df_stress = make_stress_features(df_source, cfg)
        small = quick_logistic_auroc(df_stress, config, repeats=5)
        small["severity"] = sev
        rows.append(small)

    curve = pd.concat(rows, ignore_index=True)
    curve.to_csv(outdir / "stress_severity_curve_logistic.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for model_name in MODEL_ORDER:
        sub = curve[curve["model"] == model_name]
        ax.errorbar(
            sub["severity"],
            sub["macro_AUROC_mean"],
            yerr=sub["macro_AUROC_sd"],
            marker="o",
            capsize=3,
            label=MODEL_LABELS[model_name].replace("\n", " "),
        )

    ax.set_xlabel("Stress severity")
    ax.set_ylabel("Macro-AUROC, logistic classifier")
    ax.set_ylim(0.40, 1.03)
    ax.set_title("Robustness of descriptor families under increasing spatial-zonal overlap")
    ax.legend(frameon=False)
    savefig(fig, outdir, "07_stress_severity_curve")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="standard", choices=["quick", "standard", "extended"])
    parser.add_argument("--rerun-original", action="store_true", help="Rerun separable/amplitude_overlap even if outputs exist.")
    parser.add_argument("--severity", type=float, default=0.75, help="Stress severity. Recommended: 0.70 to 0.80.")
    parser.add_argument("--noise-scale", type=float, default=0.55)
    parser.add_argument("--zone-blur", type=float, default=0.35)
    parser.add_argument("--skip-curve", action="store_true", help="Skip severity-sweep curve to save time.")
    return parser.parse_args()


def main() -> None:
    set_q1_plot_style()
    args = parse_args()
    config = base.make_config(args.mode)
    Path(config.output_root).mkdir(parents=True, exist_ok=True)

    print("Q1 extension for Zonal MIO simulation")
    print(f"Mode: {args.mode}")
    print(f"Output root: {config.output_root}")
    print(f"Stress severity: {args.severity}")

    ensure_original_scenario(config, "separable", rerun=args.rerun_original)
    ensure_original_scenario(config, "amplitude_overlap", rerun=args.rerun_original)

    stress_cfg = StressConfig(
        source_scenario="amplitude_overlap",
        output_scenario="stress_overlap",
        severity=args.severity,
        noise_scale=args.noise_scale,
        zone_blur=args.zone_blur,
        random_state=config.seed,
    )
    run_stress_scenario(config, stress_cfg)

    scenarios = ["separable", "amplitude_overlap", "stress_overlap"]
    plot_cross_scenario_comparison(config, scenarios)

    if not args.skip_curve:
        plot_stress_severity_curve(config, source_scenario="amplitude_overlap")

    print("\nCompleted Q1 extension.")
    print(f"Check: {Path(config.output_root) / 'stress_overlap'}")
    print(f"Check: {Path(config.output_root) / 'cross_scenario_summary'}")


if __name__ == "__main__":
    main()
