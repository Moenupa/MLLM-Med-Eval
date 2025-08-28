import pickle as pkl
import sys
from glob import glob

import pandas as pd

from mmmbench.io import form_dataframe
from mmmbench.metrics import GPTMetric, ModelFreeMetric
from mmmbench.plot import plot_radar

if __name__ == "__main__":
    results = []

    files = sys.argv[1:] if len(sys.argv) > 1 else glob("*.pkl")
    for path in files:
        with open(path, "rb") as file:
            results.extend(pkl.load(file))

    df = form_dataframe(results)
    # arrange column in order
    df = df.reindex(
        columns=[
            ModelFreeMetric.ExactMatch,
            ModelFreeMetric.ROUGE,
            ModelFreeMetric.CIDEr,
            GPTMetric.Acc,
            GPTMetric.InstFollow,
        ]
    )

    df.to_json("readable_mllm_med_results.json", indent=4)

    # use with to print dataframe without max width length
    with pd.option_context("display.max_colwidth", None, "display.max_rows", None):
        print(df)

    plot_radar(
        df,
        # model_include=[
        #     "ascend_Lingshu-7B",
        #     "lingshu_trained_step10k",
        #     "ascend_Qwen2.5-VL-7B-Instruct",
        #     "Qwen2.5-VL-7B-Instruct-step10k",
        # ],
        model_exclude=[
            r"ascend_.*-iter\d+",
            "Qwen2.5-VL-7B-Instruct-step4k",
            "Qwen2.5-VL-7B-Instruct-step8k",
            "Qwen2.5-VL-7B-Instruct-step10k",
        ],
    )
