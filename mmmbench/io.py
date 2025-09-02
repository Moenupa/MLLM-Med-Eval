import os.path as osp
import re
from collections import defaultdict

import pandas as pd

from .constants import GT_KEY, PROBLEM_KEY
from .metrics import GPTMetric, ModelFreeMetric


def get_content_between(text: str, start_tag: str, end_tag: str) -> str:
    # regex pattern: match start_tag ... end_tag with non-greedy capture
    pattern = rf"{start_tag}(.*?){end_tag}"

    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches[0]


def extract_html_tag(text: str, tag: str, hard: bool = True):
    if not text:
        return ""

    target_str = get_content_between(f"<{tag}>", f"</{tag}>", text)
    if target_str:
        return target_str
    elif hard:
        return text
    else:
        return ""


# borrowed from mathruler.grader
def extract_boxed_content(text: str) -> str:
    """
    Extracts answers in \\boxed{}.
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return text


def answer_cleaning(inp: str) -> str:
    return inp.replace(".", "").replace("\n", "").strip()


def parse_response(response):
    response = response.lower()
    if "boxed" in response:
        response = extract_boxed_content(response)
    elif "<answer>" in response:
        response = extract_html_tag(response, "answer")
    answer_patterns = [
        "**answer**:",
        "**answer**",
        "*answer*:",
        "**answer:**",
        "answer is",
        "answer:",
        "答案:",
        "final answer",
        "final answer is",
    ]
    for answer_pattern in answer_patterns:
        if answer_pattern in response:
            response = response.split(answer_pattern)[-1]

    return response


def read_answers(path: str, model_name: str | None = None) -> pd.DataFrame:
    if path.endswith(".json"):
        df = pd.read_json(path)
    elif path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        raise NotImplementedError

    assert GT_KEY in df.columns
    assert PROBLEM_KEY in df.columns

    if model_name is not None:
        df[model_name] = df[model_name].apply(str.lower)
        df[GT_KEY] = df[GT_KEY].apply(str.lower)

        df[model_name] = df[model_name].apply(answer_cleaning)

    return df


def beautify_model_name(model_name: str) -> str:
    if model_name.startswith("step"):
        return f"Qwen2.5-VL-7B-Instruct-{model_name}"
    return model_name


def form_dataframe(results: list[dict]) -> pd.DataFrame:
    """
    Fill cell values into a metric-to-model 2d matrix.

    Parameters
    ----------
    results : list[dict]
        A list of sparse cell values, e.g. `[{'json_path': 'xx.json', 'model': 'step4k', 'ROUGE-L': 0.123}]`

    Returns
    -------
    pd.DataFrame
        A dataframe, column=Metric, index=model
    """
    out = defaultdict(dict)
    for result in results:
        # dataset name, i.e. json filename w/o extension
        dataset_name = osp.basename(result.pop("json_path")).split(".")[0]
        model_name = beautify_model_name(result.pop("model"))

        for metric, value in result.items():
            out[(dataset_name, model_name)][metric] = value

    for k in out.keys():
        # ensure all metrics are present for each (json_path, model) pair
        for metric in ModelFreeMetric:
            if metric.value not in out[k]:
                out[k][metric.value] = None

        for metric in GPTMetric:
            if metric.value not in out[k]:
                out[k][metric.value] = None

    # sort keys by 1. dataset name, 2. model prefix, 3. model name length
    out = dict(sorted(out.items(), key=lambda x: (x[0][0], x[0][1][:10], len(x[0][1]))))

    # use na if a metric is not available
    df = pd.DataFrame.from_dict(out, orient="index")
    df.index.names = ["json_path", "model"]

    # *100 like Lingshu paper
    df = df * 100
    df = df.round(1)

    return df


def unique_model_id(path: str, model_name: str) -> str:
    # make a unique id from path and model_name
    # path is unique, model_name maybe not
    return f"{path.split('/')[-2]}/{model_name}"
