PATH_RE_TO_MODEL_MAPPING = {
    "output/*.json": [
        "step4k",
        "step6k",
        "step8k",
        "step10k",
    ],
    "output_ascend/*.json": [
        "Qwen2.5-VL-7B-Instruct",
        "Qwen2.5-VL-7B-Instruct-iter3557",
        "Lingshu-7B",
        "Lingshu-7B-iter1447",
    ],
}

GT_KEY = "answer"
PROBLEM_KEY = "problem"

from enum import StrEnum


class ModelBasedMetric(StrEnum):
    Acc = "GPT-Accuracy"
    InstFollow = "GPT-InstructionFollowing"


class ModelFreeMetric(StrEnum):
    ROUGE = "ROUGE-L"
    CIDEr = "CIDEr"
    ExactMatch = "ExactMatch"


MAX_WORKER_PER_JSON = 4
