from agentverse.registry import Registry

executor_registry = Registry(name="ExecutorRegistry")

from .base import BaseExecutor, NoneExecutor
# RELEASE NOTE: the upstream code_test / tool_using / coverage_test executors were
# removed (they pull spacy/aiohttp/httpx and a local tool server). No paper config
# selects them; every paper protocol uses the `none` executor.
