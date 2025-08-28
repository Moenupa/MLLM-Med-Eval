PATH_RE_TO_MODEL_MAPPING = {
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
    "output_forgetting/*.jsonl": ["grpo-geo3k", "Qwen2.5-VL-7B-Instruct"]
}

GT_KEY = "answer"
PROBLEM_KEY = "problem"

# this should only work for ModelBasedMetric
MAX_WORKER_PER_JSON = 8
