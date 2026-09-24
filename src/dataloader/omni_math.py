import json
from typing import Any, Dict

from .dataloader import DataLoader
from . import dataloader_registry


def _normalize_record(line: Dict[str, Any], default_dataset_name: str) -> Dict[str, Any]:
    question = line.get("question", line.get("problem", ""))
    answer = line.get("answer_number")
    if answer is None or str(answer).strip() == "":
        answer = line.get("answer", "")
    if answer is None or str(answer).strip() == "":
        answer = line.get("label", "")
    normalized_answer = str(answer).strip()

    record = {
        "input": question,
        "answer": normalized_answer,
        "question": question,
        "answer_number": normalized_answer,
        "dataset_name": line.get("dataset_name", default_dataset_name),
        "source": line.get("source", default_dataset_name),
    }

    for key in (
        "difficulty",
        "difficulty_tier",
        "domain",
        "judge_type",
        "raw_difficulty",
    ):
        if key in line:
            record[key] = line[key]

    if "answer" in line:
        record["raw_answer"] = line["answer"]
    if "label" in line and line["label"] is not None:
        record["label"] = str(line["label"]).strip()
    for solution_key in ("equation_solution", "equation_answer", "solution"):
        if solution_key in line and line[solution_key] is not None:
            record[solution_key] = line[solution_key]
            if "reference_solution" not in record:
                record["reference_solution"] = line[solution_key]

    return record


@dataloader_registry.register("tasksolving/omni_math/alcf_per")
@dataloader_registry.register("tasksolving/omni_math/sophia_120b_per")
@dataloader_registry.register("tasksolving/omni_math/sophia_120b_per_rule")
@dataloader_registry.register("tasksolving/omni_math/sophia_120b_per_verifier")
@dataloader_registry.register("tasksolving/omni_math/sophia_120b_per_judge")
@dataloader_registry.register("tasksolving/omni_math/sophia_120b_per_oss_120b_eval")
@dataloader_registry.register("tasksolving/omni_math/sophia_120b_broadcast_deliberation")
@dataloader_registry.register("tasksolving/omni_math/sophia_120b_broadcast_deliberation_hint")
@dataloader_registry.register("tasksolving/omni_math/sophia_120b_broadcast_deliberation_hint_3agent_per_mirror")
@dataloader_registry.register("tasksolving/omni_math/sophia_120b_broadcast_deliberation_plain")
@dataloader_registry.register("tasksolving/omni_math/sophia_120b_single_agent_reflect_hint")
@dataloader_registry.register("tasksolving/omni_math/sophia_120b_single_agent_reflect_plain")
@dataloader_registry.register("tasksolving/omni_math/sophia_120b_single_llm_hint")
@dataloader_registry.register("tasksolving/omni_math/sophia_120b_single_llm_plain")
@dataloader_registry.register("tasksolving/omni_math_rule/sophia_120b_per")
class OmniMathLoader(DataLoader):
    """
    Supports both:
      1. preprocessed MGSM-style Omni-MATH JSONL:
         {"question": ..., "answer_number": ..., ...}
      2. raw Omni-MATH / omni-math-rule JSONL:
         {"problem": ..., "answer": ..., "difficulty": ..., ...}
    """

    def load(self):
        path_lower = self.path.lower()
        if "omni-math-2-filtered" in path_lower or "omni_math_2_filtered" in path_lower:
            default_dataset_name = "omni-math-2-filtered"
        elif "omni-math-rule" in path_lower or "omni_math_rule" in path_lower:
            default_dataset_name = "omni-math-rule"
        else:
            default_dataset_name = "omni-math"
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(
                    _normalize_record(json.loads(line), default_dataset_name)
                )
