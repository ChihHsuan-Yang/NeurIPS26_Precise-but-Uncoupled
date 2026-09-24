from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Dict, Optional


def extract_boxed(text: str) -> str:
    if not text:
        return ""
    raw = str(text)
    marker = r"\boxed{"
    last_match = ""
    start = 0

    while True:
        idx = raw.find(marker, start)
        if idx == -1:
            break

        cursor = idx + len(marker)
        depth = 1
        chunks: list[str] = []

        while cursor < len(raw) and depth > 0:
            char = raw[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            chunks.append(char)
            cursor += 1

        if depth == 0:
            last_match = "".join(chunks).strip()
        start = idx + 1

    return last_match


def extract_final_answer(text: str) -> str:
    if not text:
        return ""

    boxed = extract_boxed(text)
    if boxed:
        return boxed

    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return str(text).strip()

    return lines[-1].rstrip(".")


def _to_decimal(value: str) -> Optional[Decimal]:
    if value is None:
        return None

    cleaned = str(value).strip()
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.rstrip(".")

    try:
        return Decimal(cleaned).normalize()
    except (InvalidOperation, ValueError):
        return None


def numeric_equal(pred_str: str, gt_str: str, tol: str = "0") -> bool:
    pred = _to_decimal(pred_str)
    gt = _to_decimal(gt_str)

    if pred is not None and gt is not None:
        if tol and tol != "0":
            return abs(pred - gt) <= Decimal(tol)
        return pred == gt

    return str(pred_str).strip() == str(gt_str).strip()


def exact_match(prediction: str, ground_truth: str, tol: str = "0") -> bool:
    pred = extract_final_answer(prediction)
    gt = extract_final_answer(str(ground_truth))
    return bool(pred) and numeric_equal(pred, gt, tol=tol)


def normalize_omni_judge_report_text(report: str) -> str:
    text = str(report or "")
    text = text.replace("Ġ", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def parse_omni_judge_report(report: str) -> Dict[str, str]:
    """
    Parse an Omni-Judge markdown report. The local model output can contain
    tokenizer artifacts like spaced-out headers, so we normalize first and then
    extract the canonical sections robustly.
    """
    normalized = normalize_omni_judge_report_text(report)
    data: Dict[str, str] = {}
    verdict_match = re.search(r"TRUE|FALSE", normalized, flags=re.IGNORECASE)
    verdict = verdict_match.group(0).upper() if verdict_match else ""

    answer_match = re.search(
        r"## Student Final Answer\s*(.*)",
        normalized,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if answer_match:
        answer_text = answer_match.group(1).strip()
        if verdict_match:
            answer_text = answer_text[: verdict_match.start() - answer_match.start(1)].strip()
        answer_text = re.split(r"\s*#\s*#.*$", answer_text, maxsplit=1)[0].strip()
        if answer_text:
            data["Student Final Answer"] = answer_text

    if verdict:
        data["Equivalence Judgement"] = verdict

    justification = ""
    justification_patterns = [
        r"J\s*u\s*s\s*f?\s*i?\s*c?\s*a\s*t\s*i\s*o\s*n\s*(.*?)(?:=== report over ===|$)",
        r"J\s*u\s*s\s*t\s*i\s*f\s*i\s*c\s*a\s*t\s*i\s*o\s*n\s*(.*?)(?:=== report over ===|$)",
    ]
    for pattern in justification_patterns:
        match = re.search(pattern, normalized, flags=re.DOTALL | re.IGNORECASE)
        if match:
            justification = match.group(1).strip(" :#-\n")
            break
    if justification:
        data["Justification"] = justification

    return data
