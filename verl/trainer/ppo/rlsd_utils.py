"""
RLSD (Reinforcement Learning with Self-Distillation) utilities.

This module provides skill-based privileged information loading and
token-level advantage computation for the RLSD algorithm.
All task types and keyword mappings are loaded from skill_mapping.json,
making this module environment-agnostic.
"""

import json
import os
from typing import Dict, List, Optional

import torch
from omegaconf import OmegaConf


def load_skill_mapping(skills_dir: str) -> dict:
    mapping_path = os.path.join(skills_dir, "skill_mapping.json")
    with open(mapping_path, "r") as f:
        return json.load(f)


def load_skill_content(skills_dir: str, skill_mapping: dict) -> Dict[str, str]:
    """Load all skill markdown files into a dict keyed by skill name."""
    contents = {}
    for skill_name, filename in skill_mapping["skill_files"].items():
        filepath = os.path.join(skills_dir, filename)
        with open(filepath, "r") as f:
            contents[skill_name] = f.read().strip()
    return contents


def _normalize_task_name(task_name: Optional[str]) -> Optional[str]:
    if task_name is None:
        return None
    task_name = str(task_name).lower()
    if "alfworld" in task_name:
        return "alfworld"
    if "webshop" in task_name:
        return "webshop"
    if "search" in task_name:
        return "search"
    return task_name


def _to_plain_dict(value) -> dict:
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else dict(value)


class SkillProvider:
    """Loads and caches skill files, provides privileged info per task type.

    Everything is driven by skill_mapping.json:
      - skill_files: maps skill names to markdown filenames
      - task_to_skill: maps task type strings to skill names
      - task_keywords (optional): ordered list of (task_type -> keywords) for
        inferring task type from prompt text. Keywords are matched in order;
        the first task type whose ALL keywords appear in the text wins.
    """

    def __init__(self, skills_dir: str, skill_all: bool = False, skills_dirs: Optional[Dict[str, str]] = None):
        self.skills_dir = skills_dir
        self.skill_all = skill_all

        self.task_skill_providers = {}
        self.default_provider = None
        if skills_dirs:
            skills_dirs = _to_plain_dict(skills_dirs)
            self.task_skill_providers = {
                _normalize_task_name(task_name): SkillProvider(skills_dir=task_skills_dir, skill_all=skill_all)
                for task_name, task_skills_dir in skills_dirs.items()
            }
            self.default_provider = next(iter(self.task_skill_providers.values()))
            self.skill_mapping = self.default_provider.skill_mapping
            self.skill_contents = self.default_provider.skill_contents
            self.task_to_skill = self.default_provider.task_to_skill
            self.task_keywords = self.default_provider.task_keywords
            return

        self.skill_mapping = load_skill_mapping(skills_dir)
        self.skill_contents = load_skill_content(skills_dir, self.skill_mapping)
        self.task_to_skill = self.skill_mapping["task_to_skill"]
        # task_keywords: ordered dict of task_type -> list of keywords (all must match)
        self.task_keywords: Dict[str, List[str]] = self.skill_mapping.get("task_keywords", {})

        if self.skill_all:
            self._all_skills_text = self._build_all_skills_text()

    def _provider_for_task(self, task_name: Optional[str]):
        if not self.task_skill_providers:
            return self
        normalized = _normalize_task_name(task_name)
        return self.task_skill_providers.get(normalized, self.default_provider)

    def _build_all_skills_text(self) -> str:
        """Concatenate general_skills + all task-specific skills."""
        general = self.skill_contents.get("general_skills", "")
        parts = [general]
        for skill_name, content in self.skill_contents.items():
            if skill_name != "general_skills":
                parts.append(content)
        return "\n\n".join(parts)

    def _get_skill_text(self, task_type: Optional[str]) -> str:
        """Assemble general_skills + task-specific skill text."""
        general = self.skill_contents.get("general_skills", "")
        parts = [general]
        if task_type:
            mapped_name = self.task_to_skill.get(task_type)
            if mapped_name and mapped_name in self.skill_contents:
                parts.append(self.skill_contents[mapped_name])
        return "\n\n".join(parts)

    def get_privileged_info(self, gamefile: str) -> str:
        """Return skills for a gamefile path (task type appears as substring)."""
        if self.task_skill_providers:
            return self.default_provider.get_privileged_info(gamefile)
        if self.skill_all:
            return self._all_skills_text
        matched_task = None
        for task_type in self.task_to_skill:
            if task_type in gamefile:
                matched_task = task_type
                break
        return self._get_skill_text(matched_task)

    def get_privileged_info_from_prompt(self, prompt_text: str) -> str:
        """Infer task type from prompt text using keyword rules from skill_mapping.json.

        Uses ``any`` matching: a task type is matched if ANY of its keywords
        appear in the prompt.  When multiple task types match, all of their
        skill texts are concatenated (general_skills is included only once).
        """
        if self.task_skill_providers:
            return self.default_provider.get_privileged_info_from_prompt(prompt_text)
        if self.skill_all:
            return self._all_skills_text
        text_lower = prompt_text.lower()
        matched_tasks = []
        for task_type, keywords in self.task_keywords.items():
            if keywords and any(kw in text_lower for kw in keywords):
                matched_tasks.append(task_type)

        if not matched_tasks:
            return self._get_skill_text(None)

        general = self.skill_contents.get("general_skills", "")
        parts = [general]
        for task_type in matched_tasks:
            mapped_name = self.task_to_skill.get(task_type)
            if mapped_name and mapped_name in self.skill_contents:
                parts.append(self.skill_contents[mapped_name])
        return "\n\n".join(parts)

    def get_privileged_info_from_data_source(self, data_source: str, prompt_text: str) -> str:
        """Infer task type using data_source field and prompt content.

        Matching rules:
          - data_source == 'popqa' -> entity_attribute_lookup
          - data_source in ('nq', 'triviaqa') -> direct_retrieval
          - prompt contains 'which' + 'or' (without 'for') -> compare
          - data_source == 'hotpotqa' -> multi_hop_reasoning
          - otherwise -> general skills only (unknown)
        """
        if self.task_skill_providers:
            return self.default_provider.get_privileged_info_from_data_source(data_source, prompt_text)
        if self.skill_all:
            return self._all_skills_text

        task_type = None
        if data_source == "popqa":
            task_type = "entity_attribute_lookup"
        elif data_source in ("nq", "triviaqa"):
            task_type = "direct_retrieval"
        elif data_source == "hotpotqa":
            task_type = "multi_hop_reasoning"
        else:
            text_lower = prompt_text.lower()
            if "which" in text_lower and "or" in text_lower and "for" not in text_lower:
                task_type = "compare"
            elif data_source in ("2wikimultihopqa", "musique", "bamboogle"):
                task_type = "multi_hop_reasoning"

        return self._get_skill_text(task_type)

    def get_privileged_info_for_sample(
        self,
        task_name: Optional[str] = None,
        gamefile: Optional[str] = None,
        data_source: Optional[str] = None,
        prompt_text: str = "",
    ) -> str:
        """Return skill text, routing by task_name when multitask skills are configured."""
        provider = self._provider_for_task(task_name)
        if provider is not self:
            normalized_task = _normalize_task_name(task_name)
            if gamefile is not None:
                return provider.get_privileged_info(str(gamefile))
            if normalized_task == "search" and data_source is not None:
                return provider.get_privileged_info_from_data_source(str(data_source), prompt_text)
            return provider.get_privileged_info_from_prompt(prompt_text)

        if gamefile is not None:
            return self.get_privileged_info(str(gamefile))
        if data_source is not None:
            return self.get_privileged_info_from_data_source(str(data_source), prompt_text)
        return self.get_privileged_info_from_prompt(prompt_text)


def compute_rlsd_token_advantage(
    seq_advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    rlsd_lambda: float = 0.5,
    rlsd_clip_eps: float = 0.2,
) -> torch.Tensor:
    """
    Compute token-level RLSD advantages.

    Args:
        seq_advantages: (bs, response_length) — standard GRPO sequence-level advantage
            broadcast to all tokens (same value per sequence).
        student_log_probs: (bs, response_length) — log π_θ(y_t | x, y_<t).
        teacher_log_probs: (bs, response_length) — log π_θ(y_t | x, r, y_<t).
        response_mask: (bs, response_length) — mask for valid response tokens.
        rlsd_lambda: mixing coefficient λ. When 0, degrades to standard GRPO.
        rlsd_clip_eps: clipping bound ε_w for token weights.

    Returns:
        token_advantages: (bs, response_length) — token-level advantage Â_t.
    """
    with torch.no_grad():
        first_valid = response_mask.int().argmax(dim=-1)  # (bs,)
        batch_indices = torch.arange(seq_advantages.size(0), device=seq_advantages.device)
        A_seq = seq_advantages[batch_indices, first_valid]  # (bs,)

        delta_t = teacher_log_probs - student_log_probs  # (bs, response_length)

        sign_A = torch.sign(A_seq).unsqueeze(-1)  # (bs, 1)
        w_t = torch.exp(sign_A * delta_t)  # (bs, response_length)
        w_t = torch.clamp(w_t, 1.0 - rlsd_clip_eps, 1.0 + rlsd_clip_eps)

        A_seq_expanded = A_seq.unsqueeze(-1)  # (bs, 1)
        token_advantages = A_seq_expanded * ((1.0 - rlsd_lambda) + rlsd_lambda * w_t)
        token_advantages = token_advantages * response_mask

    return token_advantages
