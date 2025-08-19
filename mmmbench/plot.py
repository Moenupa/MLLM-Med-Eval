from .constants import ModelFreeMetric, ModelBasedMetric
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


METRICS = [c.value for c in ModelBasedMetric] + [c.value for c in ModelFreeMetric]
N_METRICS = len(METRICS)
LINGSHU_BASELINE = {
    "mimic-cxr-vqa_test": {
        ModelFreeMetric.ROUGE: 30.8,
        ModelFreeMetric.CIDEr: 109.4 / 4,
    },
    "mmmu_medical": {ModelFreeMetric.ROUGE: 54.0},
    "vqa-rad_test": {ModelFreeMetric.ROUGE: 67.9},
    "slake_test": {ModelFreeMetric.ROUGE: 83.1},
    "path-vqa_test": {ModelFreeMetric.ROUGE: 61.9},
}


def normalize_scores(scale: tuple[int, int], value: float) -> float:
    # scale to [0, 100]
    _min, _max = scale
    assert _min < _max
    return (value - _min) / (_max - _min) * 100


def plot_radar(
    df: pd.DataFrame,
    scaler: dict[ModelBasedMetric | ModelFreeMetric, tuple[int, int]] = {
        ModelBasedMetric.Acc: (1, 5),
        ModelBasedMetric.InstFollow: (1, 5),
        ModelFreeMetric.ROUGE: (0, 100),
        ModelFreeMetric.CIDEr: (0, 400),
        ModelFreeMetric.ExactMatch: (0, 100),
    },
    baseline: dict[
        str, dict[ModelBasedMetric | ModelFreeMetric, float]
    ] = LINGSHU_BASELINE,
    model_include: list[str] | None = None,
    model_exclude: list[str] | None = None,
):
    assert all(c in df.columns for c in METRICS)

    angles = np.linspace(0, 2 * np.pi, N_METRICS, endpoint=False).tolist()
    angles += angles[:1]  # close the circle
    # for each json path, plot a radar chart
    if model_include is None:
        model_include = df.index.get_level_values("model").unique().tolist()

    for json_path, group in df.groupby("json_path"):
        for metric in METRICS:
            if metric not in scaler:
                continue

            group[metric] = group[metric].apply(
                lambda x: normalize_scores(scaler[metric], x)
            )

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        sns.set_style("whitegrid")

        # Plot models
        for model in model_include:
            if model_exclude is not None and model in model_exclude:
                continue

            try:
                vals = group.xs(model, level="model")[METRICS].iloc[0].tolist()
            except Exception:
                continue
            vals += vals[:1]
            ax.plot(angles, vals, label=model)
            ax.fill(angles, vals, alpha=0.15)
            ax.set_ylim(0, 100)

        # Plot baseline if provided
        if baseline is not None and json_path in baseline:
            baseline_scores = baseline[json_path]

            base_vals = [baseline_scores.get(m, 0) for m in METRICS]
            base_vals += base_vals[:1]
            ax.plot(
                angles, base_vals, "--", color="gray", linewidth=2, label="Baseline"
            )

        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(METRICS, fontsize=12)
        ax.set_title(json_path, size=14, y=1.08)
        ax.legend(loc="lower left")

        fig.savefig(f"radar/{json_path}.png", bbox_inches="tight")
