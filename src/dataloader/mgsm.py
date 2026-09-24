from .dataloader import DataLoader
from . import dataloader_registry
import json
import re


@dataloader_registry.register("tasksolving/mgsm/gpt-4")
@dataloader_registry.register("tasksolving/mgsm/gpt-3.5")
@dataloader_registry.register("tasksolving/mgsm/alcf_sophia")
@dataloader_registry.register("tasksolving/mgsm/alcf_sophia_light_model")
@dataloader_registry.register("tasksolving/mgsm/alcf_per")
@dataloader_registry.register("tasksolving/mgsm/per_light")
class MGSMLoader(DataLoader):
    def __init__(self, path: str):
        self.answer_pat = re.compile(r"#### (-?\d+)")
        super().__init__(path)

    def load(self):
        with open(self.path) as f:
            for line in f:
                line = json.loads(line)
                normalized_answer = str(line["answer_number"]).strip()
                record = {
                    "input": line["question"],
                    "answer": normalized_answer,
                    "question": line["question"],
                    "answer_number": normalized_answer,
                    "dataset_name": line.get("dataset_name", "mgsm"),
                    "source": line.get("source", "mgsm"),
                }

                for key in (
                    "equation_solution",
                    "language",
                    "split",
                    "difficulty",
                    "difficulty_tier",
                    "judge_type",
                ):
                    if key in line:
                        record[key] = line[key]

                # Preserve the raw free-form answer field without overwriting the
                # normalized benchmark ground truth stored in `answer`.
                if "answer" in line:
                    record["raw_answer"] = line["answer"]

                self.examples.append(record)
