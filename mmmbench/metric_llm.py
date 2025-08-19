import pandas as pd
import os.path as osp

from .constants import GT_KEY, MAX_WORKER_PER_JSON, PROBLEM_KEY, ModelBasedMetric
from .llm_rater import LLMRater, append_cache, cache_file, load_cache, make_key


def lingshu_cleaning(text: str) -> str:
    if len(text) == 2 and text.endswith("."):
        return text[:-1]
    return text


def compute_model_based_metric(
    args: tuple[str, str, ModelBasedMetric, int | None],
) -> dict:
    json_path, model_name, metric_name, worker_id = args

    assert metric_name is not None and metric_name in ModelBasedMetric

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

    cache_path = cache_file(json_path, model_name, metric_name)
    cache_map = load_cache(cache_path)

    rater = LLMRater()
    scores = []
    for idx, row in df.iterrows():
        # for parallel processing
        if worker_id is not None and idx % MAX_WORKER_PER_JSON != worker_id:
            continue

        gt = row[GT_KEY]
        pred = row[model_name]
        context_val = row[PROBLEM_KEY]

        # cached score, unique by make_key()
        key = make_key(int(idx), str(gt), str(pred), metric_name)
        if key in cache_map:
            score = int(cache_map[key]["score"])
            reason = str(cache_map[key].get("reason", ""))
        else:
            rating = rater.rate(str(gt), str(pred), metric_name, context_val)
            score = int(rating.score)
            reason = rating.reason
            append_cache(
                cache_path,
                {
                    "key": key,
                    "index": int(idx),
                    "metric": metric_name.value,
                    "score": score,
                    "reason": reason,
                    "gt": gt,
                    "pred": pred,
                },
            )
        scores.append(score)

    if not scores:
        return meta_info | {metric_name.value: 0.0}

    avg_score = float(sum(scores) / len(scores))

    if metric_name == ModelBasedMetric.Acc:
        return meta_info | {metric_name.value: avg_score}

    if metric_name == ModelBasedMetric.InstFollow:
        return meta_info | {metric_name.value: avg_score}

    # not implemented
    raise NotImplementedError
