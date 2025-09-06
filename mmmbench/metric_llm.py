from .constants import GT_KEY, MAX_WORKER_PER_JSON, PROBLEM_KEY
from .io import read_answers, unique_model_id
from .llm_rater import LLMRater, append_cache, cache_file, load_cache, make_key
from .metrics import GPTMetric, ScoreBinary


def compute_gpt_metrics(
    args: tuple[str, str, GPTMetric, int | None],
) -> dict:
    json_path, model_name, metric_name, worker_id, *_ = args
    assert metric_name is not None and metric_name in GPTMetric

    df = read_answers(json_path)

    meta_info = {
        "json_path": json_path,
        "model": unique_model_id(json_path, model_name),
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
        context = row[PROBLEM_KEY]

        # cached score, unique by make_key()
        key = make_key(int(idx), str(gt), str(pred), metric_name)
        if key in cache_map:
            score = int(cache_map[key]["score"])
            reason = str(cache_map[key].get("reason", ""))
        else:
            rating = rater.rate(
                gt=str(gt),
                pred=str(pred),
                metric=metric_name,
                score_template=ScoreBinary,
                context=context,
            )
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
                    "context": context,
                    "gt": gt,
                    "pred": pred,
                },
            )
        scores.append(score)

    if not scores:
        return meta_info | {metric_name.value: 0.0}

    avg_score = float(sum(scores) / len(scores))

    return meta_info | {metric_name.value: avg_score}
