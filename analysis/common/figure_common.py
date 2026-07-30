#!/usr/bin/env python3
"""Shared strict loaders and helpers for the result figures.

The module intentionally fails on incomplete or ambiguous evidence instead of
silently plotting a partial matrix
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

OFFLOAD_LEVELS = (0, 2, 4, 8, 12, 16)
BASE_CONCURRENCIES = (1, 2, 4, 8, 12, 16)
PLOT_CONCURRENCIES = (4, 8)
PROFILE_ORDER = ("128/32", "256/64", "512/128")
MODEL_ORDER = ("llama", "qwen")
MODEL_LABELS = {"llama": "Llama", "qwen": "Qwen"}


class EvidenceError(RuntimeError):
    """Raised when the evidence matrix is incomplete or ambiguous."""


def find_repo_root(start: Path | None = None) -> Path:
    """Find the artifact repository root."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (
            (candidate / "data" / "derived").is_dir()
            and (candidate / "experiments").is_dir()
            and (candidate / "analysis").is_dir()
        ):
            return candidate
    raise EvidenceError(
        "Artifact repository root not found. "
        "Provide --repo-root explicitly."
    )


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise EvidenceError(f"{description} fehlt: {path}")
    return path.resolve()


def require_dir(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise EvidenceError(f"{description} fehlt: {path}")
    return path.resolve()


def choose_path(
    repo_root: Path,
    explicit: Path | None,
    preferred: Sequence[Path],
    globs: Sequence[str],
    description: str,
    *,
    kind: str = "file",
    latest_ok: bool = False,
) -> Path:
    """Resolve a source path deterministically.

    Preferred paths win. Fallback globbing is deliberately strict unless
    ``latest_ok`` is enabled for timestamped analysis directories.
    """
    checker = require_file if kind == "file" else require_dir
    if explicit is not None:
        path = explicit if explicit.is_absolute() else repo_root / explicit
        return checker(path, description)

    for relative in preferred:
        path = relative if relative.is_absolute() else repo_root / relative
        if (path.is_file() if kind == "file" else path.is_dir()):
            return path.resolve()

    candidates: list[Path] = []
    for pattern in globs:
        candidates.extend(repo_root.glob(pattern))
    candidates = sorted(
        {p.resolve() for p in candidates if (p.is_file() if kind == "file" else p.is_dir())}
    )
    if not candidates:
        expected = "\n  - ".join(str(p) for p in preferred)
        raise EvidenceError(
            f"{description} nicht gefunden. Bevorzugte Pfade:\n  - {expected}"
        )
    if len(candidates) > 1 and not latest_ok:
        shown = "\n  - ".join(str(p) for p in candidates)
        raise EvidenceError(
            f"Mehrere Kandidaten für {description} gefunden. "
            f"Bitte den Pfad explizit angeben:\n  - {shown}"
        )
    return candidates[-1]


def model_alias(value: str) -> str:
    lowered = str(value).lower()
    if "llama" in lowered:
        return "llama"
    if "qwen" in lowered:
        return "qwen"
    raise EvidenceError(f"Unbekannte Modellbezeichnung: {value!r}")


def require_columns(df: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise EvidenceError(f"Fehlende Spalten in {source}: {missing}")


def numeric(df: pd.DataFrame, columns: Iterable[str], source: Path) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        if out[column].isna().any():
            bad = out.index[out[column].isna()].tolist()[:10]
            raise EvidenceError(
                f"Nichtnumerische/fehlende Werte in {source}, Spalte {column}, Zeilen {bad}"
            )
    return out


def _validate_exact_cells(
    df: pd.DataFrame,
    keys: Sequence[str],
    expected_per_cell: int,
    source: Path,
) -> None:
    counts = df.groupby(list(keys), dropna=False).size()
    bad = counts[counts != expected_per_cell]
    if not bad.empty:
        raise EvidenceError(
            f"Unerwartete Zellbesetzung in {source}; erwartet {expected_per_cell} pro Zelle:\n{bad}"
        )


def _validate_completed(df: pd.DataFrame, source: Path) -> None:
    if "completed" in df.columns and not (pd.to_numeric(df["completed"]) == 20).all():
        raise EvidenceError(f"Nicht alle Läufe in {source} haben completed=20")
    if "failed" in df.columns and not (pd.to_numeric(df["failed"]) == 0).all():
        raise EvidenceError(f"Mindestens ein Lauf in {source} hat failed != 0")


def load_base_runs(
    repo_root: Path,
    llama_csv: Path | None = None,
    qwen_csv: Path | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Load and validate the two complete 180-run baseline matrices."""
    llama_path = choose_path(
        repo_root,
        llama_csv,
        preferred=[Path("data/derived/baseline/llama/runs_summary.csv")],
        globs=[],
        description="Llama baseline CSV",
    )
    qwen_path = choose_path(
        repo_root,
        qwen_csv,
        preferred=[Path("data/derived/baseline/qwen/runs_summary.csv")],
        globs=[],
        description="Qwen baseline CSV",
    )

    frames: list[pd.DataFrame] = []
    for expected_model, path in (("llama", llama_path), ("qwen", qwen_path)):
        frame = pd.read_csv(path)
        required = [
            "model_name",
            "offload_gb",
            "run_concurrency",
            "run_id",
            "input_len",
            "output_len",
            "temperature",
            "median_ttft_ms",
            "median_tpot_ms",
            "median_itl_ms",
            "median_e2el_ms",
        ]
        require_columns(frame, required, path)
        frame = numeric(
            frame,
            [
                "offload_gb",
                "run_concurrency",
                "run_id",
                "input_len",
                "output_len",
                "temperature",
                "median_ttft_ms",
                "median_tpot_ms",
                "median_itl_ms",
                "median_e2el_ms",
            ],
            path,
        )
        frame["model"] = frame["model_name"].map(model_alias)
        if set(frame["model"].unique()) != {expected_model}:
            raise EvidenceError(
                f"{path} enthält nicht ausschließlich {expected_model}: "
                f"{sorted(frame['model'].unique())}"
            )
        if set(frame["offload_gb"].astype(int)) != set(OFFLOAD_LEVELS):
            raise EvidenceError(f"Falsche Offload-Stufen in {path}")
        if set(frame["run_concurrency"].astype(int)) != set(BASE_CONCURRENCIES):
            raise EvidenceError(f"Falsche Concurrency-Stufen in {path}")
        if len(frame) != 180:
            raise EvidenceError(f"{path} enthält {len(frame)} statt 180 Läufen")
        _validate_exact_cells(frame, ["offload_gb", "run_concurrency"], 5, path)
        if set(frame["input_len"].astype(int)) != {256} or set(frame["output_len"].astype(int)) != {64}:
            raise EvidenceError(f"Baseline-Profil in {path} ist nicht 256/64")
        if not np.allclose(frame["temperature"], 0.0):
            raise EvidenceError(f"temperature != 0 in {path}")
        _validate_completed(frame, path)
        if frame.duplicated(["offload_gb", "run_concurrency", "run_id"]).any():
            raise EvidenceError(f"Doppelte Baseline-Läufe in {path}")
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    return combined, [llama_path, qwen_path]


def load_profile_runs(
    repo_root: Path,
    profile_csv: Path | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Load and validate the 72-run profile-robustness matrix."""
    path = choose_path(
        repo_root,
        profile_csv,
        preferred=[Path("data/derived/profile_robustness/runs_summary.csv")],
        globs=[],
        description="Profile robustness CSV",
    )
    frame = pd.read_csv(path)
    required = [
        "model_name",
        "offload_gb",
        "run_concurrency",
        "run_id",
        "input_len",
        "output_len",
        "median_ttft_ms",
        "median_tpot_ms",
        "median_itl_ms",
    ]
    require_columns(frame, required, path)
    frame = numeric(
        frame,
        [
            "offload_gb",
            "run_concurrency",
            "run_id",
            "input_len",
            "output_len",
            "median_ttft_ms",
            "median_tpot_ms",
            "median_itl_ms",
        ],
        path,
    )
    frame["model"] = frame["model_name"].map(model_alias)
    frame["profile"] = (
        frame["input_len"].astype(int).astype(str)
        + "/"
        + frame["output_len"].astype(int).astype(str)
    )
    if len(frame) != 72:
        raise EvidenceError(f"{path} enthält {len(frame)} statt 72 Läufen")
    if set(frame["model"]) != set(MODEL_ORDER):
        raise EvidenceError(f"Profilkampagne in {path} enthält nicht beide Modelle")
    if set(frame["offload_gb"].astype(int)) != {0, 12}:
        raise EvidenceError(f"Profilkampagne in {path} enthält nicht Offload 0/12")
    if set(frame["run_concurrency"].astype(int)) != set(PLOT_CONCURRENCIES):
        raise EvidenceError(f"Profilkampagne in {path} enthält nicht Concurrency 4/8")
    if set(frame["profile"]) != set(PROFILE_ORDER):
        raise EvidenceError(
            f"Profilkampagne in {path} enthält unerwartete Profile: {sorted(frame['profile'].unique())}"
        )
    _validate_exact_cells(
        frame,
        ["model", "offload_gb", "profile", "run_concurrency"],
        3,
        path,
    )
    _validate_completed(frame, path)
    if frame.duplicated(["model", "offload_gb", "profile", "run_concurrency", "run_id"]).any():
        raise EvidenceError(f"Doppelte Profilrobustheitsläufe in {path}")
    return frame, [path]


def load_fixd_main(
    repo_root: Path,
    fixd_csv: Path | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Load and validate the 36-run generic GPU-load control summary."""
    path = choose_path(
        repo_root,
        fixd_csv,
        preferred=[
            Path("results/tables/gpu_load_control/fixd_main_summary.csv")
        ],
        globs=[],
        description="Generic GPU load control CSV",
    )
    frame = pd.read_csv(path)
    required = [
        "model",
        "condition",
        "concurrency",
        "run_no",
        "median_ttft_ms",
        "median_itl_ms",
    ]
    require_columns(frame, required, path)
    frame = numeric(
        frame,
        ["concurrency", "run_no", "median_ttft_ms", "median_itl_ms"],
        path,
    )
    frame["model"] = frame["model"].map(model_alias)
    expected_conditions = {"gpu_only_normal", "gpu_only_loaded", "cpu_offload12"}
    if len(frame) != 36:
        raise EvidenceError(f"{path} enthält {len(frame)} statt 36 Hauptläufen")
    if set(frame["model"]) != set(MODEL_ORDER):
        raise EvidenceError(f"Fix-D-Hauptmatrix in {path} enthält nicht beide Modelle")
    if set(frame["condition"]) != expected_conditions:
        raise EvidenceError(f"Unerwartete Bedingungen in {path}: {sorted(frame['condition'].unique())}")
    if set(frame["concurrency"].astype(int)) != set(PLOT_CONCURRENCIES):
        raise EvidenceError(f"Fix-D-Hauptmatrix in {path} enthält nicht Concurrency 4/8")
    _validate_exact_cells(frame, ["model", "condition", "concurrency"], 3, path)
    if frame.duplicated(["model", "condition", "concurrency", "run_no"]).any():
        raise EvidenceError(f"Doppelte Fix-D-Läufe in {path}")
    return frame, [path]


def _load_d2_model(root: Path, expected_model: str) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted(root.glob("**/main/gpu_only_loaded/*.json"))
    if not files:
        raise EvidenceError(f"Keine D2-Probe-JSONs unter {root} gefunden")
    rows: list[dict] = []
    used: list[Path] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            obj = json.load(handle)
        alias = model_alias(obj.get("model_alias") or obj.get("model_name") or obj.get("model_id"))
        if alias != expected_model:
            continue
        bg_label = str(obj.get("bg_label", ""))
        if bg_label != "c64i1024o512":
            continue
        completed = int(obj.get("completed", -1))
        failed = int(obj.get("failed", -1))
        if completed != 20 or failed != 0:
            raise EvidenceError(f"Ungültiger D2-Lauf {path}: completed={completed}, failed={failed}")
        row = {
            "model": alias,
            "condition": "kv_vram_pressure",
            "concurrency": int(obj["concurrency"]),
            "run_no": int(obj["run_no"]),
            "median_ttft_ms": float(obj["median_ttft_ms"]),
            "median_itl_ms": float(obj["median_itl_ms"]),
            "gpu_memory_utilization": 0.75 if alias == "llama" else 0.65,
            "file": str(path),
        }
        rows.append(row)
        used.append(path.resolve())
    frame = pd.DataFrame(rows)
    if len(frame) != 6:
        raise EvidenceError(
            f"D2-{expected_model} unter {root} enthält {len(frame)} statt 6 gültigen Probe-Läufen"
        )
    if set(frame["concurrency"]) != set(PLOT_CONCURRENCIES):
        raise EvidenceError(f"D2-{expected_model} enthält nicht Concurrency 4/8")
    _validate_exact_cells(frame, ["model", "condition", "concurrency"], 3, root)
    if frame.duplicated(["model", "concurrency", "run_no"]).any():
        raise EvidenceError(f"Doppelte D2-Läufe unter {root}")
    return frame, used


def load_d2_runs(
    repo_root: Path,
    llama_root: Path | None = None,
    qwen_root: Path | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Load and validate both six-run near-boundary D2 campaigns."""
    llama_path = choose_path(
        repo_root,
        llama_root,
        preferred=[Path("data/raw/kv_vram_control/llama")],
        globs=[],
        description="Llama KV/VRAM control directory",
        kind="dir",
    )
    qwen_path = choose_path(
        repo_root,
        qwen_root,
        preferred=[Path("data/raw/kv_vram_control/qwen")],
        globs=[],
        description="Qwen KV/VRAM control directory",
        kind="dir",
    )
    llama, llama_files = _load_d2_model(llama_path, "llama")
    qwen, qwen_files = _load_d2_model(qwen_path, "qwen")
    return pd.concat([llama, qwen], ignore_index=True), llama_files + qwen_files


@dataclass(frozen=True)
class FitResult:
    model: str
    concurrency: int
    affine_intercept_ms: float
    affine_slope_ms_per_gib: float
    affine_r2: float
    global_linear_intercept_ms: float
    global_linear_slope_ms_per_gib: float
    global_linear_r2: float
    saturating_intercept_ms: float
    saturating_slope_ms_per_gib: float
    g_sat_gib: float
    saturating_r2: float


def r_squared(y: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total <= 0:
        raise EvidenceError("R² ist für konstante Zielwerte nicht definiert")
    return 1.0 - residual / total


def fit_affine_saturating(
    model: str,
    concurrency: int,
    x: np.ndarray,
    y: np.ndarray,
) -> FitResult:
    """Fit the exact paper models to six cell medians."""
    order = np.argsort(x)
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    if tuple(x.astype(int)) != OFFLOAD_LEVELS:
        raise EvidenceError(f"Fit benötigt exakt {OFFLOAD_LEVELS}, erhalten {tuple(x)}")

    mask = x <= 12
    affine_slope, affine_intercept = np.polyfit(x[mask], y[mask], 1)
    affine_pred = affine_intercept + affine_slope * x[mask]
    affine_r2 = r_squared(y[mask], affine_pred)

    global_slope, global_intercept = np.polyfit(x, y, 1)
    global_pred = global_intercept + global_slope * x
    global_r2 = r_squared(y, global_pred)

    def residual(theta: np.ndarray) -> np.ndarray:
        intercept, slope, g_sat = theta
        return intercept + slope * np.minimum(x, g_sat) - y

    optimized = least_squares(
        residual,
        x0=np.asarray([affine_intercept, affine_slope, 12.5]),
        bounds=(np.asarray([-np.inf, -np.inf, 0.0]), np.asarray([np.inf, np.inf, 16.0])),
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
        max_nfev=100_000,
    )
    if not optimized.success:
        raise EvidenceError(f"Sättigungsfit fehlgeschlagen: {optimized.message}")
    sat_intercept, sat_slope, g_sat = map(float, optimized.x)
    sat_pred = sat_intercept + sat_slope * np.minimum(x, g_sat)
    sat_r2 = r_squared(y, sat_pred)

    return FitResult(
        model=model,
        concurrency=int(concurrency),
        affine_intercept_ms=float(affine_intercept),
        affine_slope_ms_per_gib=float(affine_slope),
        affine_r2=float(affine_r2),
        global_linear_intercept_ms=float(global_intercept),
        global_linear_slope_ms_per_gib=float(global_slope),
        global_linear_r2=float(global_r2),
        saturating_intercept_ms=sat_intercept,
        saturating_slope_ms_per_gib=sat_slope,
        g_sat_gib=g_sat,
        saturating_r2=float(sat_r2),
    )


def base_cell_medians(base: pd.DataFrame) -> pd.DataFrame:
    columns = ["median_ttft_ms", "median_tpot_ms", "median_itl_ms", "median_e2el_ms"]
    return (
        base.groupby(["model", "run_concurrency", "offload_gb"], as_index=False)[columns]
        .median()
        .sort_values(["model", "run_concurrency", "offload_gb"])
        .reset_index(drop=True)
    )


def profile_ratios(profile: pd.DataFrame) -> pd.DataFrame:
    cells = (
        profile.groupby(["model", "profile", "run_concurrency", "offload_gb"], as_index=False)[
            ["median_tpot_ms", "median_itl_ms", "median_ttft_ms"]
        ]
        .median()
    )
    rows: list[dict] = []
    for (model, profile_name, concurrency), group in cells.groupby(
        ["model", "profile", "run_concurrency"], sort=True
    ):
        by_offload = group.set_index("offload_gb")
        for metric in ("median_tpot_ms", "median_itl_ms", "median_ttft_ms"):
            normal = float(by_offload.loc[0, metric])
            off12 = float(by_offload.loc[12, metric])
            rows.append(
                {
                    "model": model,
                    "profile": profile_name,
                    "concurrency": int(concurrency),
                    "metric": metric.removeprefix("median_").removesuffix("_ms").upper(),
                    "normal_ms": normal,
                    "offload12_ms": off12,
                    "ratio_offload12_over_normal": off12 / normal,
                }
            )
    result = pd.DataFrame(rows)
    result["profile"] = pd.Categorical(result["profile"], PROFILE_ORDER, ordered=True)
    return result.sort_values(["model", "metric", "concurrency", "profile"]).reset_index(drop=True)


def control_cells(fixd: pd.DataFrame, d2: pd.DataFrame) -> pd.DataFrame:
    fixd_cells = (
        fixd.groupby(["model", "condition", "concurrency"], as_index=False)[
            ["median_ttft_ms", "median_itl_ms"]
        ]
        .median()
    )
    d2_cells = (
        d2.groupby(["model", "condition", "concurrency"], as_index=False)[
            ["median_ttft_ms", "median_itl_ms", "gpu_memory_utilization"]
        ]
        .median()
    )
    combined = pd.concat([fixd_cells, d2_cells], ignore_index=True, sort=False)
    rows: list[dict] = []
    for (model, concurrency), group in combined.groupby(["model", "concurrency"], sort=True):
        indexed = group.set_index("condition")
        expected = {"gpu_only_normal", "gpu_only_loaded", "cpu_offload12", "kv_vram_pressure"}
        if set(indexed.index) != expected:
            raise EvidenceError(
                f"Kontrollzellen für {model}, c={concurrency} unvollständig: {sorted(indexed.index)}"
            )
        normal_ttft = float(indexed.loc["gpu_only_normal", "median_ttft_ms"])
        normal_itl = float(indexed.loc["gpu_only_normal", "median_itl_ms"])
        offload_itl = float(indexed.loc["cpu_offload12", "median_itl_ms"])
        for condition, row in indexed.iterrows():
            rows.append(
                {
                    "model": model,
                    "concurrency": int(concurrency),
                    "condition": condition,
                    "median_ttft_ms": float(row["median_ttft_ms"]),
                    "median_itl_ms": float(row["median_itl_ms"]),
                    "ttft_over_normal": float(row["median_ttft_ms"]) / normal_ttft,
                    "itl_over_normal": float(row["median_itl_ms"]) / normal_itl,
                    "itl_over_offload12_percent": 100.0 * float(row["median_itl_ms"]) / offload_itl,
                    "gpu_memory_utilization": (
                        float(row["gpu_memory_utilization"])
                        if pd.notna(row.get("gpu_memory_utilization", np.nan))
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "concurrency", "condition"]).reset_index(drop=True)


def fit_table_from_base(base: pd.DataFrame) -> pd.DataFrame:
    cells = base_cell_medians(base)
    rows: list[dict] = []
    for model in MODEL_ORDER:
        for concurrency in PLOT_CONCURRENCIES:
            group = cells[(cells["model"] == model) & (cells["run_concurrency"] == concurrency)]
            fit = fit_affine_saturating(
                model,
                concurrency,
                group["offload_gb"].to_numpy(),
                group["median_tpot_ms"].to_numpy(),
            )
            rows.append(fit.__dict__)
    return pd.DataFrame(rows)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    return paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_provenance(
    output_dir: Path,
    stem: str,
    repo_root: Path,
    input_files: Sequence[Path],
    extra: dict | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo_root": ".",
        "git_commit": git_commit(repo_root),
        "inputs": [
            {
                "path": str(path.resolve().relative_to(repo_root.resolve())),
                "sha256": sha256(path),
            } for path in sorted(set(input_files))
        ],
    }
    if extra:
        payload.update(extra)
    path = output_dir / f"{stem}_provenance.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
