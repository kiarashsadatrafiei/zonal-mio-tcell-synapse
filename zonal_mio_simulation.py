from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 2026

    grid_size: int = 128
    n_time: int = 30
    n_per_regime: int = 140

    eta_c: float = 0.30
    eta_p: float = 0.68

    texture_smoothing: float = 5.0
    fine_texture_smoothing: float = 1.6
    carrier_smoothing: float = 1.2

    field_noise_sigma: float = 0.055
    measurement_noise_sigma: float = 0.030

    signal_mean: float = 0.42
    signal_sd_overlap: float = 0.003
    signal_sd_separable: float = 0.045

    carrier_mean: float = 0.35
    carrier_sd_overlap: float = 0.003
    carrier_sd_separable: float = 0.040

    cv_splits: int = 5
    cv_repeats: int = 15

    output_root: str = "zonal_mio_outputs"


REGIMES = [
    "productive_activation",
    "checkpoint_suppression",
    "unstable_engagement",
    "assembly_failure",
]

REGIME_LABELS = {
    "productive_activation": "Productive activation",
    "checkpoint_suppression": "Checkpoint suppression",
    "unstable_engagement": "Unstable engagement",
    "assembly_failure": "Assembly failure",
}

SCENARIOS = ["separable", "amplitude_overlap"]


def make_config(mode: str) -> SimulationConfig:
    if mode == "quick":
        return SimulationConfig(
            grid_size=88,
            n_time=18,
            n_per_regime=35,
            cv_repeats=3,
            output_root="zonal_mio_outputs_quick",
        )

    if mode == "standard":
        return SimulationConfig()

    if mode == "extended":
        return SimulationConfig(
            grid_size=160,
            n_time=36,
            n_per_regime=180,
            cv_repeats=20,
            output_root="zonal_mio_outputs_extended",
        )

    raise ValueError("mode must be one of: quick, standard, extended")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 450,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def normalize_inside(field: np.ndarray, mask: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    out = np.full_like(field, np.nan, dtype=float)
    values = field[mask]
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    out[mask] = (values - lo) / (hi - lo + eps)
    return out


def robust_normalize_inside(field: np.ndarray, mask: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    out = np.full_like(field, np.nan, dtype=float)
    values = field[mask]
    lo = np.nanpercentile(values, 1)
    hi = np.nanpercentile(values, 99)
    out[mask] = np.clip((values - lo) / (hi - lo + eps), 0.0, 1.0)
    return out


def match_mean(field: np.ndarray, mask: np.ndarray, target: float, eps: float = 1e-9) -> np.ndarray:
    out = field.copy()
    current = np.nanmean(out[mask])
    out[mask] = out[mask] * target / (current + eps)
    out[mask] = np.clip(out[mask], 0.0, 1.0)
    out[~mask] = np.nan
    return out


def match_p95(field: np.ndarray, mask: np.ndarray, target: float, eps: float = 1e-9) -> np.ndarray:
    out = field.copy()
    current = np.nanpercentile(out[mask], 95)
    out[mask] = out[mask] * target / (current + eps)
    out[mask] = np.clip(out[mask], 0.0, 1.0)
    out[~mask] = np.nan
    return out


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)

    if np.sum(valid) < 20:
        return 0.0

    av = a[valid]
    bv = b[valid]

    if np.std(av) < 1e-10 or np.std(bv) < 1e-10:
        return 0.0

    r, _ = spearmanr(av, bv)

    if not np.isfinite(r):
        return 0.0

    return float(r)


def make_domain(config: SimulationConfig) -> Dict[str, np.ndarray]:
    n = config.grid_size
    xs = np.linspace(-1.0, 1.0, n)
    ys = np.linspace(-1.0, 1.0, n)
    x, y = np.meshgrid(xs, ys)

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    mask = r <= 1.0

    zone_c = mask & (r <= config.eta_c)
    zone_p = mask & (r > config.eta_c) & (r <= config.eta_p)
    zone_d = mask & (r > config.eta_p)

    return {
        "x": x,
        "y": y,
        "r": r,
        "theta": theta,
        "mask": mask,
        "d": zone_d,
        "p": zone_p,
        "c": zone_c,
    }


def smooth_texture(
    rng: np.random.Generator,
    domain: Dict[str, np.ndarray],
    sigma: float,
) -> np.ndarray:
    raw = rng.normal(0.0, 1.0, size=domain["r"].shape)
    smoothed = gaussian_filter(raw, sigma=sigma)
    return normalize_inside(smoothed, domain["mask"])


def annular_field(
    domain: Dict[str, np.ndarray],
    center: float,
    width: float,
    angular_mode: int,
    phase: float,
    angular_strength: float,
) -> np.ndarray:
    r = domain["r"]
    theta = domain["theta"]

    field = np.exp(-((r - center) ** 2) / (2 * width**2))
    field *= 1.0 + angular_strength * np.cos(angular_mode * theta + phase)
    field[~domain["mask"]] = np.nan

    return normalize_inside(field, domain["mask"])


def patchy_field(
    rng: np.random.Generator,
    domain: Dict[str, np.ndarray],
    config: SimulationConfig,
    center: float,
    width: float,
    n_patches: int,
    patch_sigma: float,
) -> np.ndarray:
    x = domain["x"]
    y = domain["y"]
    field = np.zeros_like(x, dtype=float)

    for _ in range(n_patches):
        angle = rng.uniform(0, 2 * math.pi)
        radius = np.clip(rng.normal(center, width), 0.02, 0.98)
        cx = radius * math.cos(angle)
        cy = radius * math.sin(angle)
        amplitude = rng.uniform(0.65, 1.35)

        field += amplitude * np.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2 * patch_sigma**2)
        )

    field = gaussian_filter(field, sigma=config.fine_texture_smoothing)
    return normalize_inside(field, domain["mask"])


def blend(fields: List[np.ndarray], weights: List[float], mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(fields[0], dtype=float)

    for field, weight in zip(fields, weights):
        out[mask] += weight * field[mask]

    return robust_normalize_inside(out, mask)


def gaussian_time(tau: float, center: float, width: float, amplitude: float = 1.0) -> float:
    return amplitude * math.exp(-((tau - center) ** 2) / (2 * width**2))


def zone_weights(regime: str, tau: float, rng: np.random.Generator) -> Dict[str, float]:
    if regime == "productive_activation":
        return {
            "d": 0.18 + gaussian_time(tau, 0.18, 0.14, 0.90),
            "p": 0.20 + gaussian_time(tau, 0.52, 0.22, 0.95),
            "c": 0.12 + gaussian_time(tau, 0.82, 0.18, 0.55),
        }

    if regime == "checkpoint_suppression":
        return {
            "d": 0.20 + gaussian_time(tau, 0.20, 0.17, 0.75),
            "p": 0.16 + gaussian_time(tau, 0.52, 0.22, 0.62),
            "c": 0.10 + gaussian_time(tau, 0.78, 0.20, 0.30),
        }

    if regime == "unstable_engagement":
        pulse = rng.choice([0.0, 0.15, 0.35], p=[0.45, 0.35, 0.20])

        return {
            "d": float(np.clip(rng.normal(0.42, 0.25) + pulse, 0.05, 0.95)),
            "p": float(np.clip(rng.normal(0.40, 0.26) + 0.5 * pulse, 0.05, 0.95)),
            "c": float(np.clip(rng.normal(0.35, 0.22), 0.04, 0.85)),
        }

    if regime == "assembly_failure":
        common = rng.uniform(0.25, 0.55)

        return {
            "d": float(np.clip(common + rng.normal(0.0, 0.06), 0.05, 0.75)),
            "p": float(np.clip(common + rng.normal(0.0, 0.06), 0.05, 0.75)),
            "c": float(np.clip(common + rng.normal(0.0, 0.06), 0.05, 0.75)),
        }

    raise ValueError(f"Unknown regime: {regime}")


def compose_zonal_map(
    domain: Dict[str, np.ndarray],
    textures: Dict[str, np.ndarray],
    weights: Dict[str, float],
    base: np.ndarray,
    base_weight: float,
) -> np.ndarray:
    mask = domain["mask"]
    out = np.zeros_like(domain["r"], dtype=float)

    for zone in ["d", "p", "c"]:
        out[domain[zone]] += weights[zone] * textures[zone][domain[zone]]

    out[mask] += base_weight * base[mask]
    return robust_normalize_inside(out, mask)


def generate_I_mu_frame(
    regime: str,
    t: int,
    n_time: int,
    domain: Dict[str, np.ndarray],
    rng: np.random.Generator,
    config: SimulationConfig,
    stable_textures: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    tau = t / max(n_time - 1, 1)
    mask = domain["mask"]
    phase = rng.uniform(0, 2 * math.pi)

    d_wave = annular_field(domain, 0.84, 0.10, 5, phase, 0.25)
    p_wave = annular_field(domain, 0.52, 0.13, 6, phase + 0.7, 0.22)
    c_wave = annular_field(domain, 0.21, 0.10, 3, phase + 1.4, 0.18)

    d_patch = patchy_field(rng, domain, config, 0.84, 0.08, 16, 0.050)
    p_patch = patchy_field(rng, domain, config, 0.52, 0.09, 18, 0.055)
    c_patch = patchy_field(rng, domain, config, 0.22, 0.06, 8, 0.050)

    I_textures = {
        "d": blend([d_wave, d_patch, stable_textures["d"]], [0.45, 0.35, 0.20], mask),
        "p": blend([p_wave, p_patch, stable_textures["p"]], [0.45, 0.35, 0.20], mask),
        "c": blend([c_wave, c_patch, stable_textures["c"]], [0.40, 0.35, 0.25], mask),
    }

    I_weights = zone_weights(regime, tau, rng)
    base = smooth_texture(rng, domain, sigma=config.texture_smoothing)

    I_map = compose_zonal_map(
        domain=domain,
        textures=I_textures,
        weights=I_weights,
        base=base,
        base_weight=0.20,
    )

    if regime == "productive_activation":
        mu_textures = {
            "d": blend([I_textures["d"], stable_textures["d"]], [0.87, 0.13], mask),
            "p": blend([I_textures["p"], stable_textures["p"]], [0.90, 0.10], mask),
            "c": blend([I_textures["c"], stable_textures["c"]], [0.70, 0.30], mask),
        }

        mu_weights = {
            "d": 0.16 + 0.95 * I_weights["d"],
            "p": 0.20 + 1.05 * I_weights["p"],
            "c": 0.10 + 0.70 * I_weights["c"],
        }

    elif regime == "checkpoint_suppression":
        independent_p = patchy_field(rng, domain, config, 0.52, 0.10, 16, 0.065)
        independent_c = smooth_texture(rng, domain, sigma=config.texture_smoothing)

        mu_textures = {
            "d": blend([I_textures["d"], stable_textures["d"]], [0.60, 0.40], mask),
            "p": blend([independent_p, stable_textures["p"]], [0.78, 0.22], mask),
            "c": independent_c,
        }

        mu_weights = {
            "d": 0.16 + 0.65 * I_weights["d"],
            "p": 0.10 + 0.25 * I_weights["p"],
            "c": 0.08 + 0.20 * I_weights["c"],
        }

    elif regime == "unstable_engagement":
        if rng.random() < 0.25:
            mu_textures = I_textures
        else:
            mu_textures = {
                "d": smooth_texture(rng, domain, sigma=config.texture_smoothing),
                "p": smooth_texture(rng, domain, sigma=config.texture_smoothing),
                "c": smooth_texture(rng, domain, sigma=config.texture_smoothing),
            }

        mu_weights = zone_weights(regime, tau, rng)

    elif regime == "assembly_failure":
        mu_textures = {
            "d": smooth_texture(rng, domain, sigma=config.texture_smoothing),
            "p": smooth_texture(rng, domain, sigma=config.texture_smoothing),
            "c": smooth_texture(rng, domain, sigma=config.texture_smoothing),
        }

        mu_weights = zone_weights(regime, tau, rng)

    else:
        raise ValueError(f"Unknown regime: {regime}")

    mu_map = compose_zonal_map(
        domain=domain,
        textures=mu_textures,
        weights=mu_weights,
        base=smooth_texture(rng, domain, sigma=config.texture_smoothing),
        base_weight=0.20,
    )

    I_map[mask] += rng.normal(0.0, config.field_noise_sigma, size=np.sum(mask))
    mu_map[mask] += rng.normal(0.0, config.field_noise_sigma, size=np.sum(mask))

    I_map[mask] = np.clip(I_map[mask], 0.0, 1.0)
    mu_map[mask] = np.clip(mu_map[mask], 0.0, 1.0)

    I_map[~mask] = np.nan
    mu_map[~mask] = np.nan

    return I_map, mu_map


def generate_carrier_frame(
    regime: str,
    t: int,
    n_time: int,
    domain: Dict[str, np.ndarray],
    rng: np.random.Generator,
    config: SimulationConfig,
    memory: Dict[str, np.ndarray],
) -> np.ndarray:
    x = domain["x"]
    y = domain["y"]
    mask = domain["mask"]
    tau = t / max(n_time - 1, 1)

    if regime == "productive_activation":
        n_clusters = 26
        radial_center = 0.86 - 0.54 * tau
        radial_jitter = 0.055
        persistence = 0.76
        inward_step = 0.022

    elif regime == "checkpoint_suppression":
        n_clusters = 22
        radial_center = 0.82 - 0.34 * tau
        radial_jitter = 0.095
        persistence = 0.46
        inward_step = 0.010

    elif regime == "unstable_engagement":
        n_clusters = int(rng.integers(16, 28))
        radial_center = rng.uniform(0.35, 0.88)
        radial_jitter = 0.17
        persistence = 0.16
        inward_step = 0.000

    elif regime == "assembly_failure":
        n_clusters = int(rng.integers(14, 26))
        radial_center = rng.uniform(0.35, 0.92)
        radial_jitter = 0.20
        persistence = 0.08
        inward_step = -0.002

    else:
        raise ValueError(f"Unknown regime: {regime}")

    previous = memory.get("centers", np.empty((0, 2), dtype=float))
    centers: List[Tuple[float, float]] = []

    n_persistent = int(min(len(previous), round(persistence * n_clusters)))

    if n_persistent > 0:
        selected = previous[rng.choice(len(previous), size=n_persistent, replace=False)]

        for cx, cy in selected:
            radius = math.sqrt(cx**2 + cy**2)
            angle = math.atan2(cy, cx)

            new_radius = np.clip(
                radius - inward_step + rng.normal(0.0, 0.010),
                0.04,
                0.98,
            )
            new_angle = angle + rng.normal(0.0, 0.08)

            centers.append(
                (
                    new_radius * math.cos(new_angle),
                    new_radius * math.sin(new_angle),
                )
            )

    while len(centers) < n_clusters:
        angle = rng.uniform(0, 2 * math.pi)
        radius = np.clip(rng.normal(radial_center, radial_jitter), 0.04, 0.98)
        centers.append((radius * math.cos(angle), radius * math.sin(angle)))

    centers_array = np.asarray(centers, dtype=float)
    memory["centers"] = centers_array

    field = np.zeros_like(x, dtype=float)

    for cx, cy in centers_array:
        sigma = rng.uniform(0.024, 0.060)
        amplitude = rng.uniform(0.70, 1.30)

        field += amplitude * np.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2)
        )

    field = gaussian_filter(field, sigma=config.carrier_smoothing)
    field = normalize_inside(field, mask)

    field[mask] += rng.normal(0.0, config.measurement_noise_sigma, size=np.sum(mask))
    field[mask] = np.clip(field[mask], 0.0, 1.0)
    field[~mask] = np.nan

    return field


def harmonize_time_series(
    synapse: Dict[str, np.ndarray],
    scenario: str,
    domain: Dict[str, np.ndarray],
    rng: np.random.Generator,
    config: SimulationConfig,
) -> Dict[str, np.ndarray]:
    mask = domain["mask"]

    I = synapse["I"].copy()
    C = synapse["C"].copy()

    n_time = I.shape[0]

    if scenario == "separable":
        signal_sd = config.signal_sd_separable
        carrier_sd = config.carrier_sd_separable
        p95_sd = 0.060

    elif scenario == "amplitude_overlap":
        signal_sd = config.signal_sd_overlap
        carrier_sd = config.carrier_sd_overlap
        p95_sd = 0.003

    else:
        raise ValueError("scenario must be separable or amplitude_overlap")

    base_signal = np.clip(
        rng.normal(config.signal_mean, signal_sd),
        0.25,
        0.62,
    )

    base_carrier = np.clip(
        rng.normal(config.carrier_mean, carrier_sd),
        0.22,
        0.55,
    )

    phase = rng.uniform(0, 2 * math.pi)

    envelope = np.asarray(
        [
            1.0 + 0.025 * math.sin(2 * math.pi * t / max(n_time - 1, 1) + phase)
            for t in range(n_time)
        ]
    )

    if scenario == "amplitude_overlap":
        envelope = np.ones(n_time)

    for t in range(n_time):
        target_mean_I = float(np.clip(base_signal * envelope[t], 0.30, 0.55))
        target_p95_I = float(np.clip(rng.normal(0.78, p95_sd), 0.70, 0.88))

        I[t] = match_mean(I[t], mask, target_mean_I)
        I[t] = match_p95(I[t], mask, target_p95_I)
        I[t] = match_mean(I[t], mask, target_mean_I)

        target_mean_C = float(np.clip(base_carrier * envelope[t], 0.24, 0.50))
        target_p95_C = float(np.clip(rng.normal(0.74, p95_sd), 0.66, 0.86))

        C[t] = match_mean(C[t], mask, target_mean_C)
        C[t] = match_p95(C[t], mask, target_p95_C)
        C[t] = match_mean(C[t], mask, target_mean_C)

    return {
        "I": I,
        "mu": synapse["mu"],
        "C": C,
    }


def generate_synapse(
    regime: str,
    scenario: str,
    domain: Dict[str, np.ndarray],
    rng: np.random.Generator,
    config: SimulationConfig,
) -> Dict[str, np.ndarray]:
    n_time = config.n_time
    n = config.grid_size

    I_series = np.zeros((n_time, n, n), dtype=float)
    mu_series = np.zeros((n_time, n, n), dtype=float)
    C_series = np.zeros((n_time, n, n), dtype=float)

    stable_textures = {
        "d": smooth_texture(rng, domain, sigma=config.texture_smoothing),
        "p": smooth_texture(rng, domain, sigma=config.texture_smoothing),
        "c": smooth_texture(rng, domain, sigma=config.texture_smoothing),
    }

    carrier_memory: Dict[str, np.ndarray] = {}

    for t in range(n_time):
        I_map, mu_map = generate_I_mu_frame(
            regime=regime,
            t=t,
            n_time=n_time,
            domain=domain,
            rng=rng,
            config=config,
            stable_textures=stable_textures,
        )

        C_map = generate_carrier_frame(
            regime=regime,
            t=t,
            n_time=n_time,
            domain=domain,
            rng=rng,
            config=config,
            memory=carrier_memory,
        )

        I_series[t] = I_map
        mu_series[t] = mu_map
        C_series[t] = C_map

    synapse = {
        "I": I_series,
        "mu": mu_series,
        "C": C_series,
    }

    return harmonize_time_series(
        synapse=synapse,
        scenario=scenario,
        domain=domain,
        rng=rng,
        config=config,
    )


def alignment_timeseries(
    I_series: np.ndarray,
    mu_series: np.ndarray,
    domain: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    alignment = {"d": [], "p": [], "c": []}

    for t in range(I_series.shape[0]):
        for zone in ["d", "p", "c"]:
            alignment[zone].append(
                safe_spearman(I_series[t][domain[zone]], mu_series[t][domain[zone]])
            )

    return {
        zone: np.asarray(values, dtype=float)
        for zone, values in alignment.items()
    }


def carrier_persistence(C_series: np.ndarray, mask: np.ndarray) -> float:
    scores = []

    for t in range(C_series.shape[0] - 1):
        current = C_series[t][mask]
        next_frame = C_series[t + 1][mask]

        if np.std(current) < 1e-10 or np.std(next_frame) < 1e-10:
            scores.append(0.0)
            continue

        corr = np.corrcoef(current, next_frame)[0, 1]

        if np.isfinite(corr):
            scores.append(max(0.0, corr))
        else:
            scores.append(0.0)

    return float(np.clip(np.mean(scores), 0.0, 1.0))


def carrier_inward_routing(C_series: np.ndarray, domain: Dict[str, np.ndarray]) -> float:
    mask = domain["mask"]
    radial_coordinate = domain["r"][mask]

    weighted_radius = []

    for t in range(C_series.shape[0]):
        carrier = C_series[t][mask]
        weighted_radius.append(
            float(np.sum(carrier * radial_coordinate) / (np.sum(carrier) + 1e-9))
        )

    weighted_radius = np.asarray(weighted_radius)
    slope = np.polyfit(np.arange(len(weighted_radius)), weighted_radius, 1)[0]

    return float(-slope)


def peak_order_score(alignment: Dict[str, np.ndarray]) -> float:
    td = int(np.argmax(alignment["d"]))
    tp = int(np.argmax(alignment["p"]))
    tc = int(np.argmax(alignment["c"]))

    score = 0.0
    score += 0.5 if td <= tp else 0.0
    score += 0.5 if tp <= tc else 0.0

    return float(score)


def extract_features(
    synapse: Dict[str, np.ndarray],
    regime: str,
    scenario: str,
    synapse_id: int,
    domain: Dict[str, np.ndarray],
) -> Dict[str, float | str | int]:
    mask = domain["mask"]

    I = synapse["I"]
    mu = synapse["mu"]
    C = synapse["C"]

    alignment = alignment_timeseries(I, mu, domain)

    n_time = I.shape[0]
    early = slice(0, n_time // 3)
    middle = slice(n_time // 3, 2 * n_time // 3)
    late = slice(2 * n_time // 3, n_time)

    I_values = I[:, mask]
    C_values = C[:, mask]

    temporal_signal = np.nanmean(I_values, axis=1)
    temporal_carrier = np.nanmean(C_values, axis=1)

    zone_means = [np.mean(alignment[zone]) for zone in ["d", "p", "c"]]
    zone_vars = [np.var(alignment[zone]) for zone in ["d", "p", "c"]]

    return {
        "synapse_id": synapse_id,
        "scenario": scenario,
        "regime": regime,

        "A_d_mean": float(np.mean(alignment["d"])),
        "A_p_mean": float(np.mean(alignment["p"])),
        "A_c_mean": float(np.mean(alignment["c"])),

        "A_d_var": float(np.var(alignment["d"])),
        "A_p_var": float(np.var(alignment["p"])),
        "A_c_var": float(np.var(alignment["c"])),

        "A_d_early": float(np.mean(alignment["d"][early])),
        "A_p_middle": float(np.mean(alignment["p"][middle])),
        "A_c_late": float(np.mean(alignment["c"][late])),

        "A_global_mean": float(np.mean(zone_means)),
        "A_global_var": float(np.mean(zone_vars)),
        "A_zonal_contrast": float(np.max(zone_means) - np.min(zone_means)),
        "A_peak_order_score": peak_order_score(alignment),

        "P_C": carrier_persistence(C, mask),
        "carrier_inward_routing": carrier_inward_routing(C, domain),

        "Y_total_mean": float(np.nanmean(I_values)),
        "Y_peak_mean": float(np.nanmean(np.nanpercentile(I_values, 95, axis=1))),
        "Y_total_integral": float(np.nansum(I_values) / n_time),
        "Y_temporal_sd": float(np.std(temporal_signal)),

        "C_total_mean": float(np.nanmean(C_values)),
        "C_peak_mean": float(np.nanmean(np.nanpercentile(C_values, 95, axis=1))),
        "C_temporal_sd": float(np.std(temporal_carrier)),
        "contact_area": float(np.sum(mask)),
    }


def simulate_dataset(
    scenario: str,
    config: SimulationConfig,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    seed_offset = 1000 if scenario == "amplitude_overlap" else 0
    rng = np.random.default_rng(config.seed + seed_offset)

    domain = make_domain(config)

    rows = []
    representatives: Dict[str, Dict[str, np.ndarray]] = {}

    synapse_id = 0

    for regime in REGIMES:
        for i in range(config.n_per_regime):
            synapse = generate_synapse(
                regime=regime,
                scenario=scenario,
                domain=domain,
                rng=rng,
                config=config,
            )

            rows.append(
                extract_features(
                    synapse=synapse,
                    regime=regime,
                    scenario=scenario,
                    synapse_id=synapse_id,
                    domain=domain,
                )
            )

            if i == 0:
                representatives[regime] = synapse

            synapse_id += 1

    return pd.DataFrame(rows), representatives


def feature_sets() -> Dict[str, List[str]]:
    amplitude_only = [
        "Y_total_mean",
        "Y_total_integral",
        "contact_area",
    ]

    amplitude_plus_carrier_mass = [
        "Y_total_mean",
        "Y_total_integral",
        "contact_area",
        "C_total_mean",
    ]

    alignment_only = [
        "A_d_mean",
        "A_p_mean",
        "A_c_mean",
        "A_d_var",
        "A_p_var",
        "A_c_var",
        "A_d_early",
        "A_p_middle",
        "A_c_late",
        "A_global_mean",
        "A_global_var",
        "A_zonal_contrast",
        "A_peak_order_score",
        "P_C",
        "carrier_inward_routing",
    ]

    return {
        "amplitude_only": amplitude_only,
        "amplitude_plus_carrier_mass": amplitude_plus_carrier_mass,
        "alignment_only": alignment_only,
        "combined": amplitude_plus_carrier_mass + alignment_only,
    }


def make_classifier(kind: str, seed: int):
    if kind == "logistic":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=8000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=450,
            max_depth=5,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown classifier: {kind}")


def evaluate_models(
    df: pd.DataFrame,
    config: SimulationConfig,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    sets = feature_sets()

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["regime"].values)

    cv = RepeatedStratifiedKFold(
        n_splits=config.cv_splits,
        n_repeats=config.cv_repeats,
        random_state=config.seed,
    )

    records = []
    pooled_predictions: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for classifier_kind in ["logistic", "random_forest"]:
        for model_name, features in sets.items():
            X = df[features].values

            true_labels = []
            predicted_labels = []

            for fold_id, (train_index, test_index) in enumerate(cv.split(X, y)):
                classifier = make_classifier(classifier_kind, seed=config.seed + fold_id)
                classifier.fit(X[train_index], y[train_index])

                probability = classifier.predict_proba(X[test_index])
                prediction = np.argmax(probability, axis=1)

                balanced_accuracy = balanced_accuracy_score(y[test_index], prediction)

                fold_log_loss = log_loss(
                    y[test_index],
                    probability,
                    labels=np.arange(len(encoder.classes_)),
                )

                try:
                    macro_auc = roc_auc_score(
                        y[test_index],
                        probability,
                        multi_class="ovr",
                        average="macro",
                    )
                except ValueError:
                    macro_auc = np.nan

                records.append(
                    {
                        "classifier": classifier_kind,
                        "model": model_name,
                        "fold": fold_id,
                        "balanced_accuracy": balanced_accuracy,
                        "macro_AUROC": macro_auc,
                        "log_loss": fold_log_loss,
                        "n_features": len(features),
                        "features": ",".join(features),
                    }
                )

                true_labels.append(y[test_index])
                predicted_labels.append(prediction)

            pooled_predictions[f"{classifier_kind}__{model_name}"] = (
                np.concatenate(true_labels),
                np.concatenate(predicted_labels),
                encoder.classes_,
            )

    return pd.DataFrame(records), pooled_predictions


def summarize_performance(performance: pd.DataFrame) -> pd.DataFrame:
    return (
        performance.groupby(["classifier", "model"])
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            macro_AUROC_mean=("macro_AUROC", "mean"),
            macro_AUROC_sd=("macro_AUROC", "std"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_sd=("log_loss", "std"),
            n_features=("n_features", "first"),
        )
        .reset_index()
    )


def incremental_value(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for classifier in sorted(summary["classifier"].unique()):
        sub = summary[summary["classifier"] == classifier]

        amplitude = sub[sub["model"] == "amplitude_only"].iloc[0]
        amplitude_carrier = sub[sub["model"] == "amplitude_plus_carrier_mass"].iloc[0]
        alignment = sub[sub["model"] == "alignment_only"].iloc[0]
        combined = sub[sub["model"] == "combined"].iloc[0]

        rows.append(
            {
                "classifier": classifier,
                "delta_AUROC_alignment_minus_amplitude": alignment["macro_AUROC_mean"] - amplitude["macro_AUROC_mean"],
                "delta_AUROC_combined_minus_amplitude": combined["macro_AUROC_mean"] - amplitude["macro_AUROC_mean"],
                "delta_AUROC_alignment_minus_amplitude_plus_carrier": alignment["macro_AUROC_mean"] - amplitude_carrier["macro_AUROC_mean"],
                "delta_log_loss_alignment_minus_amplitude": alignment["log_loss_mean"] - amplitude["log_loss_mean"],
                "delta_log_loss_combined_minus_amplitude": combined["log_loss_mean"] - amplitude["log_loss_mean"],
            }
        )

    return pd.DataFrame(rows)


def savefig(fig: plt.Figure, output_dir: str, name: str) -> None:
    fig.savefig(os.path.join(output_dir, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_spatial_montage(
    representatives: Dict[str, Dict[str, np.ndarray]],
    config: SimulationConfig,
    output_dir: str,
) -> None:
    domain = make_domain(config)
    mask = domain["mask"]

    n_time = config.n_time

    timepoints = {
        "early": int(0.20 * (n_time - 1)),
        "middle": int(0.52 * (n_time - 1)),
        "late": int(0.82 * (n_time - 1)),
    }

    fig, axes = plt.subplots(9, 4, figsize=(14, 20), constrained_layout=True)

    row_labels = []

    for phase in ["early", "middle", "late"]:
        row_labels.extend(
            [
                rf"$\tilde I(\mathbf{{r}}, t_{{{phase}}})$",
                rf"$\mu(\mathbf{{r}}, t_{{{phase}}})$",
                rf"$C(\mathbf{{r}}, t_{{{phase}}})$",
            ]
        )

    for col, regime in enumerate(REGIMES):
        axes[0, col].set_title(REGIME_LABELS[regime])

        row = 0

        for _, idx in timepoints.items():
            for key in ["I", "mu", "C"]:
                ax = axes[row, col]

                image = representatives[regime][key][idx].copy()
                image[~mask] = np.nan

                im = ax.imshow(image, origin="lower", vmin=0, vmax=1)

                ax.set_xticks([])
                ax.set_yticks([])

                if col == 0:
                    ax.set_ylabel(row_labels[row])

                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.015)

                row += 1

    fig.suptitle("Time-resolved synthetic synaptic fields", fontsize=15)
    savefig(fig, output_dir, "01_spatial_montage")


def plot_alignment_trajectories(
    representatives: Dict[str, Dict[str, np.ndarray]],
    config: SimulationConfig,
    output_dir: str,
) -> pd.DataFrame:
    domain = make_domain(config)

    records = []

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 10),
        sharex=True,
        constrained_layout=True,
    )

    for ax, regime in zip(axes, REGIMES):
        alignment = alignment_timeseries(
            representatives[regime]["I"],
            representatives[regime]["mu"],
            domain,
        )

        t = np.arange(config.n_time)

        ax.plot(t, alignment["d"], linewidth=2.0, linestyle="-", label=r"$\mathcal{A}_d(t)$")
        ax.plot(t, alignment["p"], linewidth=2.0, linestyle="--", label=r"$\mathcal{A}_p(t)$")
        ax.plot(t, alignment["c"], linewidth=2.2, linestyle=":", label=r"$\mathcal{A}_c(t)$")

        ax.axhline(0.0, linewidth=1.0)
        ax.set_ylim(-0.65, 1.02)
        ax.set_ylabel("Alignment")
        ax.set_title(REGIME_LABELS[regime])
        ax.legend(frameon=False, ncol=3, loc="lower right")

        for i in range(config.n_time):
            records.append(
                {
                    "regime": regime,
                    "time": i,
                    "A_d": float(alignment["d"][i]),
                    "A_p": float(alignment["p"][i]),
                    "A_c": float(alignment["c"][i]),
                }
            )

    axes[-1].set_xlabel("Time frame")

    fig.suptitle("Representative zone-resolved alignment trajectories", fontsize=15)
    savefig(fig, output_dir, "02_alignment_trajectories")

    return pd.DataFrame(records)


def plot_feature_pca(df: pd.DataFrame, config: SimulationConfig, output_dir: str) -> None:
    sets = feature_sets()

    panels = [
        ("Amplitude-only", sets["amplitude_only"]),
        ("Amplitude and carrier mass", sets["amplitude_plus_carrier_mass"]),
        ("Alignment-only", sets["alignment_only"]),
        ("Combined", sets["combined"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.ravel()

    for ax, (title, features) in zip(axes, panels):
        X = StandardScaler().fit_transform(df[features].values)
        coordinates = PCA(n_components=2, random_state=config.seed).fit_transform(X)

        for regime in REGIMES:
            index = df["regime"].values == regime

            ax.scatter(
                coordinates[index, 0],
                coordinates[index, 1],
                s=14,
                alpha=0.70,
                label=REGIME_LABELS[regime],
            )

        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    axes[-1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Feature-space structure across descriptor families", fontsize=14)

    savefig(fig, output_dir, "03_feature_space_pca")


def plot_model_comparison(summary: pd.DataFrame, output_dir: str) -> None:
    model_order = [
        "amplitude_only",
        "amplitude_plus_carrier_mass",
        "alignment_only",
        "combined",
    ]

    model_labels = [
        "Amplitude",
        "Amplitude\n+ carrier mass",
        "Alignment",
        "Combined",
    ]

    classifiers = sorted(summary["classifier"].unique())

    fig, axes = plt.subplots(
        1,
        len(classifiers),
        figsize=(12, 4.8),
        sharey=True,
        constrained_layout=True,
    )

    if len(classifiers) == 1:
        axes = [axes]

    for ax, classifier in zip(axes, classifiers):
        sub = summary[summary["classifier"] == classifier]

        means = []
        sds = []

        for model in model_order:
            row = sub[sub["model"] == model].iloc[0]
            means.append(float(row["macro_AUROC_mean"]))
            sds.append(float(row["macro_AUROC_sd"]))

        x = np.arange(len(model_order))

        ax.bar(x, means, yerr=sds, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=20, ha="right")
        ax.set_ylim(0.45, 1.03)
        ax.set_ylabel("Cross-validated macro-AUROC")
        ax.set_title(classifier.replace("_", " ").title())

        for i, value in enumerate(means):
            ax.text(i, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Cross-validated model comparison", fontsize=14)

    savefig(fig, output_dir, "04_model_comparison")


def plot_confusion_matrices(
    pooled_predictions: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: str,
) -> None:
    keys = [
        "logistic__amplitude_only",
        "logistic__amplitude_plus_carrier_mass",
        "logistic__alignment_only",
        "logistic__combined",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)

    for ax, key in zip(axes, keys):
        y_true, y_pred, classes = pooled_predictions[key]
        matrix = confusion_matrix(y_true, y_pred, normalize="true")

        image = ax.imshow(matrix, vmin=0, vmax=1)

        ax.set_title(key.replace("logistic__", "").replace("_", " ").title())
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))

        ax.set_xticklabels([label.replace("_", "\n") for label in classes], rotation=45, ha="right")
        ax.set_yticklabels([label.replace("_", "\n") for label in classes])

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle("Cross-validated confusion matrices", fontsize=14)

    savefig(fig, output_dir, "05_confusion_matrices")


def diagnostic_text(summary: pd.DataFrame, scenario: str) -> str:
    sub = summary[summary["classifier"] == "logistic"]

    amplitude = float(sub[sub["model"] == "amplitude_only"]["macro_AUROC_mean"].iloc[0])
    amplitude_carrier = float(sub[sub["model"] == "amplitude_plus_carrier_mass"]["macro_AUROC_mean"].iloc[0])
    alignment = float(sub[sub["model"] == "alignment_only"]["macro_AUROC_mean"].iloc[0])
    combined = float(sub[sub["model"] == "combined"]["macro_AUROC_mean"].iloc[0])

    lines = [
        f"Scenario: {scenario}",
        f"Logistic amplitude-only macro-AUROC: {amplitude:.3f}",
        f"Logistic amplitude and carrier-mass macro-AUROC: {amplitude_carrier:.3f}",
        f"Logistic alignment-only macro-AUROC: {alignment:.3f}",
        f"Logistic combined macro-AUROC: {combined:.3f}",
    ]

    return "\n".join(lines)


def run_scenario(scenario: str, config: SimulationConfig) -> None:
    output_dir = os.path.join(config.output_root, scenario)
    ensure_dir(output_dir)

    print("")
    print(f"Running scenario: {scenario}")
    print(f"Output directory: {output_dir}")

    df, representatives = simulate_dataset(scenario, config)

    performance, pooled_predictions = evaluate_models(df, config)
    summary = summarize_performance(performance)
    increments = incremental_value(summary)

    df.to_csv(os.path.join(output_dir, "synthetic_features.csv"), index=False)
    performance.to_csv(os.path.join(output_dir, "model_comparison_folds.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "model_comparison_summary.csv"), index=False)
    increments.to_csv(os.path.join(output_dir, "incremental_value.csv"), index=False)

    timeseries = plot_alignment_trajectories(representatives, config, output_dir)
    timeseries.to_csv(os.path.join(output_dir, "representative_timeseries.csv"), index=False)

    plot_spatial_montage(representatives, config, output_dir)
    plot_feature_pca(df, config, output_dir)
    plot_model_comparison(summary, output_dir)
    plot_confusion_matrices(pooled_predictions, output_dir)

    with open(os.path.join(output_dir, "parameters.json"), "w", encoding="utf-8") as file:
        payload = asdict(config)
        payload["scenario"] = scenario
        json.dump(payload, file, indent=2)

    print("")
    print("Model-comparison summary:")
    print(summary.to_string(index=False))

    print("")
    print("Incremental value:")
    print(increments.to_string(index=False))

    print("")
    print(diagnostic_text(summary, scenario))

    print("")
    print("Files:")
    for filename in sorted(os.listdir(output_dir)):
        print(f"  - {filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["quick", "standard", "extended"],
        help="Simulation scale.",
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="both",
        choices=["separable", "amplitude_overlap", "both"],
        help="Synthetic scenario.",
    )

    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    set_plot_style()

    args = parse_args()
    config = make_config(args.mode)

    ensure_dir(config.output_root)

    print("Zonal MIO simulation")
    print(f"Mode: {args.mode}")
    print(f"Scenario: {args.scenario}")
    print(f"Output root: {config.output_root}")

    if args.scenario == "both":
        scenarios = SCENARIOS
    else:
        scenarios = [args.scenario]

    for scenario in scenarios:
        run_scenario(scenario, config)

    print("")
    print("Completed.")
    print(f"Output root: {config.output_root}")


if __name__ == "__main__":
    main()