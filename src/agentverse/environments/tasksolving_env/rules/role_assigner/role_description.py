# AgentVerse/agentverse/environments/tasksolving_env/rules/role_assigner/role_description.py
from __future__ import annotations

from typing import TYPE_CHECKING, List

from . import role_assigner_registry
from .base import BaseRoleAssigner

if TYPE_CHECKING:
    from agentverse.message import RoleAssignerMessage
    from agentverse.agents import CriticAgent, RoleAssignerAgent


@role_assigner_registry.register("role_description")
class DescriptionAssigner(BaseRoleAssigner):
    """
    Generates descriptions for each agent.

    IMPORTANT:
    - Do NOT overwrite member.name (keep stable identity for trace).
    - Only update member.role_description (existing field).
    """

    async def astep(
        self,
        role_assigner: RoleAssignerAgent,
        group_members: List[CriticAgent],
        advice: str = "No advice yet.",
        task_description: str = "",
        *args,
        **kwargs,
    ) -> List[CriticAgent]:
        assert task_description != ""
        assert len(group_members) > 0

        roles = await role_assigner.astep(advice, task_description, len(group_members))
        if len(roles.content) < len(group_members):
            raise ValueError(
                f"Number of roles ({len(roles.content)}) and number of group members ({len(group_members)}) do not match."
            )

        for role, member in zip(roles.content[: len(group_members)], group_members):
            description = role.strip().strip(".")
            member.role_description = description

            # DO NOT do: member.name = description
            # DO NOT add new attributes via setattr (Pydantic forbids it)

        return group_members

    def reset(self):
        pass


@role_assigner_registry.register("role_description_name")
class DescriptionNameAssigner(BaseRoleAssigner):
    """
    Generates description and name for each agent.

    IMPORTANT:
    - Do NOT overwrite member.name (keep stable identity for trace).
    - Only update member.role_description.
    """

    async def astep(
        self,
        role_assigner: RoleAssignerAgent,
        group_members: List[CriticAgent],
        advice: str = "No advice yet.",
        task_description: str = "",
        *args,
        **kwargs,
    ) -> List[CriticAgent]:
        assert task_description != ""
        assert len(group_members) > 0

        # roles: [{'name': 'xxx', 'description': 'xxx'}, ...]
        roles = await role_assigner.astep(advice, task_description, len(group_members))

        if len(group_members) >= 2 and len(roles.content) != len(group_members):
            raise ValueError(
                f"Number of roles ({len(roles.content)}) and number of group members ({len(group_members)}) do not match."
            )

        for role_dict, member in zip(roles.content, group_members):
            description = role_dict["description"].strip().strip(".")
            member.role_description = description

            # Ignore role_dict["name"] on purpose to keep member.name stable for tracing
            # DO NOT do: member.name = role_dict["name"].strip()

        return group_members