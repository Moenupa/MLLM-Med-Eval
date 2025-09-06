from os import cpu_count

GT_KEY = "answer"
PROBLEM_KEY = "problem"

# this should only work for ModelBasedMetric
MAX_WORKER_PER_JSON = max(cpu_count(), 8)
