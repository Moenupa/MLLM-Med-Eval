from enum import StrEnum

from pydantic import BaseModel, Field


# %% NLP metrics
class ModelFreeMetric(StrEnum):
    ExactMatch = "ExactMatch"
    CIDEr = "CIDEr"
    ROUGE = "ROUGE-L"


# %% metrics like BERTscore
class ModelBasedMetric(StrEnum):
    pass


# %% LLM as a judge
class GPTMetric(StrEnum):
    Acc = "GPT-Accuracy"
    InstFollow = "GPT-InstructionFollowing"


_GPT_PROMPT: dict[GPTMetric, str] = {
    GPTMetric.Acc: "Task: Rate factual accuracy of the prediction relative to the ground truth.",
    GPTMetric.InstFollow: "Task: Rate instruction following and adherence to expected answer format based on the ground truth. Consider wether the prediction directly and concisely provides the final answer in the same format (e.g., single letter, number, unit), without extra unrelated text.",
}


def get_prompt(gpt_metric_name: GPTMetric) -> str:
    """
    Returns the prompt for the given GPTMetric

    Parameters
    ----------
    gpt_metric_name : GPTMetric
        metric name, e.g. GPTMetric.Acc

    Returns
    -------
    str
        task prompt, e.g. "Task: Rate factual accuracy..."

    Raises
    ------
    TypeError
        if gpt_metric_name is not in GPTMetric
    """
    if gpt_metric_name not in GPTMetric:
        raise TypeError

    return _GPT_PROMPT[gpt_metric_name]


_BASE_REASON_DESC = "Brief justification for the score"
_SCORE_LIKERT_DESC = "Integer score from 1 (worst) to 5 (best)"
_SCORE_BINARY_DESC = "Integer score 0 = incorrect, 1 = correct"


class BaseScore(BaseModel):
    score: int
    reason: str | None


class ScoreLikert(BaseScore):
    score: int = Field(ge=1, le=5, description=_SCORE_LIKERT_DESC)
    reason: None = None


class ScoreLikertWithReason(BaseScore):
    score: int = Field(ge=1, le=5, description=_SCORE_LIKERT_DESC)
    reason: str = Field(min_length=1, description=_BASE_REASON_DESC)


class ScoreBinary(BaseScore):
    score: int = Field(ge=0, le=1, description=_SCORE_BINARY_DESC)
    reason: None = None


class ScoreBinaryWithReason(BaseScore):
    score: int = Field(ge=0, le=1, description=_SCORE_BINARY_DESC)
    reason: str = Field(min_length=1, description=_BASE_REASON_DESC)


_HEADER_LIKERT = "You are a strict grader. Compare the model prediction to the ground truth. Return a structured assessment with a score from 1 (worst) to 5 (best)"
_HEADER_BINARY = "You are a strict grader. Compare the model prediction to the ground truth. Return a structured assessment with a score from 0 = incorrect to 1 = correct"


def get_header(score_template: BaseScore):
    if score_template not in BaseScore.__subclasses__():
        raise TypeError

    if score_template == ScoreLikert:
        return f"{_HEADER_LIKERT}."
    elif score_template == ScoreLikertWithReason:
        return f"{_HEADER_LIKERT} and a brief reason."
    elif score_template == ScoreBinary:
        return f"{_HEADER_BINARY}."
    elif score_template == ScoreBinaryWithReason:
        return f"{_HEADER_BINARY} and a brief reason."
    else:
        raise NotImplementedError
