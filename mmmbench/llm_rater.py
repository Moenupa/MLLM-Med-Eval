import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from .constants import ModelBasedMetric

load_dotenv()

assert os.getenv("OPENAI_BASE_URL")
assert os.getenv("OPENAI_API_KEY")


class Rating(BaseModel):
    score: int = Field(
        ge=1, le=5, description="Integer score from 1 (worst) to 5 (best)"
    )
    reason: str = Field(min_length=1, description="Brief justification for the score")


def _build_prompt(
    metric: ModelBasedMetric, gt: str, pred: str, context: Optional[str]
) -> str:
    header = (
        "You are a strict grader. Compare the model prediction to the ground truth. "
        "Return a structured assessment with a score from 1 (worst) to 5 (best) and a brief reason."
    )
    if metric == ModelBasedMetric.Acc:
        task = (
            "Task: Rate the factual accuracy of the prediction relative to the ground truth. "
            "5 = exactly correct or semantically equivalent; 3 = partially correct; 1 = incorrect."
        )
    elif metric == ModelBasedMetric.InstFollow:
        task = (
            "Task: Rate instruction following and adherence to expected answer format based on the ground truth. "
            "Consider whether the prediction directly and concisely provides the final answer in the same format (e.g., single letter, number, unit), without extra unrelated text."
        )
    else:
        raise NotImplementedError
        task = "Task: Rate instruction how well the prediction follows the expected format and instructions. "

    parts = [header, task]
    if context:
        parts.append(f"Context:\n{context}")
    parts.append(f"Ground truth:\n{gt}")
    parts.append(f"Prediction:\n{pred}")
    parts.append("Only provide the structured result as requested (score and reason).")
    return "\n\n".join(parts)


def make_key(idx: int, gt: str, pred: str, metric: ModelBasedMetric) -> str:
    payload = f"{idx}||{metric.value}||{gt}||{pred}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def cache_file(json_path: str, model_name: str, metric: ModelBasedMetric) -> Path:
    cache_dir = Path(json_path).parent / ".cache_gpt_metrics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{Path(json_path).stem}.{model_name}.{metric.value}.jsonl"
    return cache_dir / fname


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = obj.get("key")
                if key:
                    cache[key] = obj
            except Exception:
                continue
    return cache


def append_cache(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class LLMRater:

    def __init__(self, llm: str = "gpt-5-mini"):
        self.llm = llm
        self.client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"))

    def rate(
        self,
        gt: str,
        pred: str,
        metric: ModelBasedMetric,
        context: Optional[str],
        n_attempts: int = 3,
    ) -> Rating:
        prompt = _build_prompt(metric, gt, pred, context)
        last_err: Optional[Exception] = None
        for attempt in range(n_attempts):
            try:
                resp = self.client.responses.parse(
                    model=self.llm,
                    input=prompt,
                    text_format=Rating,
                    temperature=0.0,
                )
                # newer SDKs expose output_parsed; fall back if needed
                parsed = getattr(resp, "output_parsed", None) or getattr(
                    resp, "parsed", None
                )
                if isinstance(parsed, Rating):
                    return parsed
                # If not directly a Rating, try constructing it from dict
                if hasattr(parsed, "model_dump"):
                    return Rating(**parsed.model_dump())
                if isinstance(parsed, dict):
                    return Rating(**parsed)
                # As a last resort, try reading the first output's content
                outputs = getattr(resp, "output", None) or getattr(
                    resp, "outputs", None
                )
                if outputs and isinstance(outputs, list):
                    text = (
                        outputs[0].get("content", "")
                        if isinstance(outputs[0], dict)
                        else str(outputs[0])
                    )
                    # Let the model produce JSON; try to parse
                    try:
                        data = json.loads(text)
                        return Rating(**data)
                    except Exception:
                        pass
                raise RuntimeError("Failed to parse rating response")
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (2**attempt))
        raise last_err if last_err else RuntimeError("Unknown OpenAI error")
