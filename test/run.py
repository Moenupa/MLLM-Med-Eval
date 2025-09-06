import argparse
import pickle as pkl
import sys
from glob import glob

from tqdm.contrib.concurrent import process_map

from mmmbench.constants import MAX_WORKER_PER_JSON
from mmmbench.metric_llm import compute_gpt_metrics
from mmmbench.metric_nlp import compute_nlp_metrics
from mmmbench.metrics import GPTMetric, ModelBasedMetric, ModelFreeMetric

PATH_RE_TO_MODEL_MAPPING: dict[str, list[str]] = {
    # "output/*.json": [
    #     "step4k",
    #     "step6k",
    #     "step8k",
    #     "step10k",
    # ],
    # "output_as2/*.json": [
    #     "Qwen2.5-VL-7B-Instruct-iter4K",
    # ],
    # "output_ascend/*.json": [
    #     "Qwen2.5-VL-7B-Instruct",
    #     "Qwen2.5-VL-7B-Instruct-iter3557",
    #     "Lingshu-7B",
    #     "Lingshu-7B-iter1447",
    # ],
    # "output_lingshu_trained/*.json": [
    #     "step4k",
    #     "lingshu",
    #     "lingshu-0819",
    # ],
    "output/forget-7b-geo3k-grpo/*.jsonl": [
        "Qwen2.5-VL-7B-grpo-mmmath",
        "Qwen2.5-VL-7B-grpo-geo3k",
        "Qwen2.5-VL-7B-Instruct",
    ],
    "output/forget-3b-geo3k-grpo/*.jsonl": [
        "Qwen2.5-VL-3B-grpo-geo3k",
        "Qwen2.5-VL-3B-Instruct",
    ],
}


def process_parallel(
    json_regex_to_keys: dict[str, list[str]],
    function: callable,
    metrics: list | ModelFreeMetric | ModelBasedMetric | GPTMetric,
    save_to: str,
    max_workers: int = 1,
    max_jobs_per_json: int = 1,
):
    jobs = [
        (each_json, model_name, metric_name, i)
        for metric_name in metrics
        for json_regex, model_names in json_regex_to_keys.items()
        for each_json in glob(json_regex)
        for model_name in model_names
        for i in range(max(max_jobs_per_json, 1))  # divide json into segments
    ]

    results: list[dict] = process_map(
        function,
        jobs,
        max_workers=max_workers,
        chunksize=1,
        desc="Computing metrics",
        leave=True,
        file=sys.stdout,
    )

    with open(save_to, "wb") as f:
        pkl.dump(results, f)

    print(f"computed metrics to {save_to}")
    for metric in metrics:
        print(f"- {metric.value}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nlp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute NLP metrics",
    )
    parser.add_argument(
        "--gpt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute GPT metrics",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.gpt:
        process_parallel(
            json_regex_to_keys=PATH_RE_TO_MODEL_MAPPING,
            function=compute_gpt_metrics,
            metrics=[GPTMetric.Acc],
            save_to="llm_results.pkl",
            max_workers=MAX_WORKER_PER_JSON,
            max_jobs_per_json=MAX_WORKER_PER_JSON // 2,
        )
    elif args.nlp:
        process_parallel(
            json_regex_to_keys=PATH_RE_TO_MODEL_MAPPING,
            function=compute_nlp_metrics,
            metrics=ModelFreeMetric,
            save_to="nlp_results.pkl",
            max_workers=MAX_WORKER_PER_JSON,
            max_jobs_per_json=1,
        )
    else:
        raise ValueError
