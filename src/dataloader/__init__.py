# RELEASE NOTE: upstream AgentVerse demo loaders (responsegen/humaneval/commongen/
# logic_grid) were removed from this release. Paper runs use `omni-math`.
from agentverse.registry import Registry

dataloader_registry = Registry(name="dataloader")

from .gsm8k import GSM8KLoader
from .mgsm import MGSMLoader
from .omni_math import OmniMathLoader
