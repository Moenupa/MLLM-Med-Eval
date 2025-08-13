import pickle as pkl
from glob import glob

from tqdm.contrib.concurrent import process_map

from mmmbench.constants import MAX_WORKER_PER_JSON, PATH_RE_TO_MODEL_MAPPING
from mmmbench.metric_llm import ModelBasedMetric, compute_model_based_metric

if __name__ == "__main__":
    args_list = [
        (each_json, model_name, metric_name, i)
        for metric_name in ModelBasedMetric
        for json_regex, model_names in PATH_RE_TO_MODEL_MAPPING.items()
        for each_json in glob(json_regex)
        for model_name in model_names
        for i in range(MAX_WORKER_PER_JSON)
    ]

    results = process_map(
        compute_model_based_metric,
        args_list,
        max_workers=64,
        chunksize=1,
        desc="Computing model-based metrics",
        unit="metric",
    )

    with open("llm_results.pkl", "wb") as f:
        pkl.dump(results, f)
