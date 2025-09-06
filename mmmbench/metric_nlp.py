import sys
import warnings

from cidereval import cider
from rouge import Rouge
from sklearn.metrics import accuracy_score

from .constants import GT_KEY
from .io import read_answers, unique_model_id
from .metrics import ModelFreeMetric

warnings.filterwarnings("ignore", category=UserWarning)


def filter_natural_lang(pred: str, gt: str, incl: bool = False) -> str:
    pass


def compute_cider(gts: list[str], preds: list[str]) -> float:
    return cider(predictions=preds, references=[[g.lower()] for g in gts])["avg_score"]


def compute_rouge(gts: list[str], preds: list[str]) -> float | None:
    # a fix to 'maximum recursion depth exceeded in comparison'
    # set this larger if you encounter the same error
    # see https://github.com/pltrdy/rouge/issues/19
    sys.setrecursionlimit(2000 * 2000 + 10)

    scorer = Rouge(metrics=["rouge-l"])
    try:
        rougeL = scorer.get_scores(
            hyps=preds,
            refs=gts,
            avg=True,
            ignore_empty=True,
        )

        return rougeL["rouge-l"]["f"]
    except Exception as e:
        print(e)

    return None


def compute_nlp_metrics(
    args: tuple[str, str, ModelFreeMetric],
) -> dict:
    json_path, model_name, metric_name, *_ = args

    assert metric_name is not None and metric_name in ModelFreeMetric

    # column -> models
    # row -> data samples
    # get avg score for each of the models
    df = read_answers(json_path, model_name)

    meta_info = {
        "json_path": json_path,
        "model": unique_model_id(json_path, model_name),
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
