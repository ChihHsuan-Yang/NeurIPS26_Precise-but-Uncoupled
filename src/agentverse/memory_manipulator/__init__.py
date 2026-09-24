from agentverse.registry import Registry

memory_manipulator_registry = Registry(name="Memory_Manipulator_Registry")

from .base import BaseMemoryManipulator
from .basic import BasicMemoryManipulator
from .plan import Plan

# RELEASE NOTE: the simulation-only `reflection` manipulator (scikit-learn/numpy)
# was removed. Paper agents use memory_manipulator_type `basic` (the default).
