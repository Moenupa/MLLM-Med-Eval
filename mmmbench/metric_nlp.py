import os.path as osp
import warnings

import pandas as pd
from cidereval import cider, ciderD
from rouge import Rouge
from sklearn.metrics import accuracy_score

from .constants import GT_KEY, PROBLEM_KEY, ModelFreeMetric

warnings.filterwarnings("ignore", category=UserWarning)


def compute_cider(gts: list[str], preds: list[str]) -> float:
    return cider(predictions=preds, references=[[g.lower()] for g in gts])["avg_score"]


def compute_rouge(gts: list[str], preds: list[str]) -> float:
    scorer = Rouge()
    rougeL = scorer.get_scores(
        hyps=preds,
        refs=gts,
        avg=True,
        ignore_empty=True,
    )

    return rougeL["rouge-l"]["f"]


def lingshu_cleaning(text: str) -> str:
    if len(text) == 2 and text.endswith("."):
        return text[:-1]
    return text


def compute_nlp_metrics(
    args: tuple[str, str, ModelFreeMetric],
) -> dict:
    json_path, model_name, metric_name = args

    assert metric_name is not None and metric_name in ModelFreeMetric

    # column -> models
    # row -> data samples
    # get avg score for each of the models
    df = pd.read_json(json_path, lines=False)

    assert GT_KEY in df.columns
    assert model_name in df.columns

    df[model_name] = df[model_name].apply(str.lower)
    df[GT_KEY] = df[GT_KEY].apply(str.lower)
    if "mmmu_medical.json" in json_path:
        df[model_name] = df[model_name].apply(lingshu_cleaning)

    meta_info = {
        "json_path": json_path,
        "model": f"{osp.dirname(json_path)}_{model_name}".replace("output_", ""),
    }
    if metric_name == ModelFreeMetric.ExactMatch:
        return meta_info | {
            metric_name.value: accuracy_score(df[GT_KEY], df[model_name]),
        }

    if metric_name == ModelFreeMetric.ROUGE:
        return meta_info | {
            metric_name.value: compute_rouge(
                df[GT_KEY].tolist(), df[model_name].tolist()
            )
        }

    if metric_name == ModelFreeMetric.CIDEr:
        return meta_info | {
            metric_name.value: compute_cider(
                df[GT_KEY].tolist(), df[model_name].tolist()
            )
        }

    # not implemented
    raise NotImplementedError
    return meta_info | {metric_name.value: 0}
