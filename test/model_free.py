import os
import pickle as pkl
from glob import glob
import sys

from tqdm.contrib.concurrent import process_map

from mmmbench.constants import PATH_RE_TO_MODEL_MAPPING
from mmmbench.metric_nlp import ModelFreeMetric, compute_nlp_metrics


def test_compute_metrics():
    # model free metrics
    results: list[dict] = process_map(
        compute_nlp_metrics,
        [
            (each_json, model_name, metric_name)
            for metric_name in ModelFreeMetric
            for json_regex, model_names in PATH_RE_TO_MODEL_MAPPING.items()
            for each_json in glob(json_regex)
            for model_name in model_names
        ],
        max_workers=os.cpu_count() // 2 or 1,
        chunksize=1,
        desc="Computing metrics",
        unit="metric",
        leave=True,
        file=sys.stdout,
    )

    out_file = "nlp_results.pkl"
    with open(out_file, "wb") as f:
        pkl.dump(results, f)

    print(f"computed metrics to {out_file}")
    for metric in ModelFreeMetric:
        print(f"- {metric.value}")


if __name__ == "__main__":
    test_compute_metrics()
