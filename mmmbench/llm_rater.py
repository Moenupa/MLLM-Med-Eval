import hashlib
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .metrics import (
    BaseScore,
    GPTMetric,
    get_header,
    get_prompt,
)

load_dotenv()

# assert os.getenv("OPENAI_BASE_URL")
assert os.getenv("OPENAI_API_KEY")


def _build_prompt(
    metric: GPTMetric,
    score_template: BaseScore,
    gt: str,
    pred: str,
    context: str | None,
) -> str:
    header = get_header(score_template)
    task = get_prompt(metric)

    parts = [header, task]
    if context:
        parts.append(f"Context:\n{context}")
    parts.append(f"Ground truth:\n{gt}")
    parts.append(f"Prediction:\n{pred}")
    return "\n\n".join(parts)


# cache operations, reduces API calls
def make_key(idx: int, gt: str, pred: str, metric: GPTMetric) -> str:
    payload = f"{idx}||{metric.value}||{gt}||{pred}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def cache_file(json_path: str, model_name: str, metric: GPTMetric) -> Path:
    cache_dir = Path(json_path).parent / ".cache_gpt_metrics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{Path(json_path).stem}.{model_name}.{metric.value}.jsonl"
    return cache_dir / fname


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
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


def append_cache(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class LLMRater:
    def __init__(self, llm: str = "gpt-5-mini"):
        self.llm = llm
        self.client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"))

    @retry(wait=wait_exponential(multiplier=1.5), stop=stop_after_attempt(3))
    def rate(
        self,
        gt: str,
        pred: str,
        metric: GPTMetric,
        score_template: BaseScore,
        context: str | None = None,
    ) -> BaseScore:
        """
        Rate prediction based on ground truth and context.
        Use prompt to obtain metric on the score scale.
        E.g. get accuracy score either 0 or 1.

        Parameters
        ----------
        gt : str
            Ground truth
        pred : str
            Prediction
        metric : GPTMetric
            The metric to use, affect the prompt
        score_template : BaseScore
            The score scale to use, affect the prompt, e.g. Binary, Likert
        context : str | None, optional
            Context, by default None

        Returns
        -------
        BaseScore
            A subclass of BaseScore containing a LLM rated score and reason (optional)

        Raises
        ------
        RuntimeError
            If LLM fails to rate
        """
        prompt = _build_prompt(
            metric=metric,
            score_template=score_template,
            gt=gt,
            pred=pred,
            context=context,
        )
        resp = self.client.responses.parse(
            model=self.llm,
            input=prompt,
            text_format=score_template,
            temperature=0.0,
        )
        parsed = resp.output_parsed
        if isinstance(parsed, score_template):
            return parsed

        # If the model produced JSON, try to parse
        if hasattr(parsed, "model_dump"):
            return score_template(**parsed.model_dump())
        if isinstance(parsed, dict):
            return score_template(**parsed)

        # As a last resort, try reading the first output's content
        if (outputs := resp.output) and isinstance(outputs, list):
            text = (
                outputs[0].get("content", "")
                if isinstance(outputs[0], dict)
                else str(outputs[0])
            )
            # Let the model produce JSON; try to parse
            try:
                data = json.loads(text)
                return score_template(**data)
            except Exception:
                pass
        raise RuntimeError("Failed to parse score response")
