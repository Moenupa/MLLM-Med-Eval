import os.path as osp
import pickle as pkl
from collections import defaultdict
from glob import glob

import pandas as pd

from mmmbench.constants import ModelBasedMetric, ModelFreeMetric
from mmmbench.plot import plot_radar


def prefix_stepn(s: str) -> str:
    # simple transform
    # step10k -> Qwen2.5-VL-7B-Instruct-step10k
    if s.startswith("step"):
        return f"Qwen2.5-VL-7B-Instruct-{s}"
    return s


def form_dataframe(results: list[dict]) -> pd.DataFrame:
    out = defaultdict(dict)
    for result in results:
        # multi-index
        json_name = osp.basename(result.pop("json_path")).replace(".json", "")
        model = result.pop("model")
        model = prefix_stepn(model)

        for metric, value in result.items():
            out[(json_name, model)][metric] = value

    for k in out.keys():
        # ensure all metrics are present for each (json_path, model) pair
        for metric in ModelFreeMetric:
            if metric.value not in out[k]:
                out[k][metric.value] = None

        for metric in ModelBasedMetric:
            if metric.value not in out[k]:
                out[k][metric.value] = None

    # sort keys by len
    out = dict(sorted(out.items(), key=lambda x: (x[0][0], len(x[0][1]))))

    # use na if a metric is not available
    df = pd.DataFrame.from_dict(out, orient="index")
    df.index.names = ["json_path", "model"]

    # *100 like Lingshu paper
    df = df * 100
    df = df.round(1)

    return df


if __name__ == "__main__":
    results = []
    for path in glob("*.pkl"):
        with open(path, "rb") as file:
            results.extend(pkl.load(file))

    # results is like this: [
    #    {'json_path': 'output/bronchoscopy_sysu_test.json', 'model': 'step4k', 'ROUGE-L': 0.123},
    #    {'json_path': 'output/mimic-cxr-vqa_test.json', 'model': 'step6k', 'CIDEr': 0.456},
    # ]

    df = form_dataframe(results)
    # arrange column in order
    df = df.reindex(
        columns=[
            ModelFreeMetric.ExactMatch.value,
            ModelFreeMetric.ROUGE.value,
            ModelFreeMetric.CIDEr.value,
            ModelBasedMetric.Acc.value,
            ModelBasedMetric.InstFollow.value,
        ]
    )
    # scale back modelbasedmetrics to 1-5 range
    df[ModelBasedMetric.Acc.value] = df[ModelBasedMetric.Acc.value] / 100
    df[ModelBasedMetric.InstFollow.value] = df[ModelBasedMetric.InstFollow.value] / 100
    # df.sort_index(
    #     ascending=True,
    #     key=lambda x: x.map(lambda y: (len(y[0]), len(y[1]))),
    #     inplace=True,
    # )

    df.to_json("readable_mllm_med_results.json", indent=4)

    # use with to print dataframe without max width length
    with pd.option_context("display.max_colwidth", None, "display.max_rows", None):
        print(df)

    plot_radar(
        df,
        model_include=[
            "ascend_Lingshu-7B",
            "lingshu_trained_step10k",
            "ascend_Qwen2.5-VL-7B-Instruct",
            "Qwen2.5-VL-7B-Instruct-step10k",
        ],
        model_exclude=[
            "step4k",
            "step6k",
            "step8k",
            "ascend_Lingshu-7B-iter1447",
            "ascend_Qwen2.5-VL-7B-Instruct-iter3557",
            "lingshu_trained_step4k",
            "lingshu_trained_step6k",
            "lingshu_trained_step8k",
        ],
    )
