from pathlib import Path
from typing import Dict, Optional, Any
import logging
from functools import lru_cache

import yaml

from aiasearch.log import PROJECT_NAME


class PromptManager:
    def __init__(self, model_name: str, prompt_path: str = None):
        self._logger = logging.getLogger(PROJECT_NAME)
        self.model_name = model_name.lower()
        if prompt_path is None:
            self.prompt_dir = Path(__file__).parent
        else:
            self.prompt_dir = Path(prompt_path)
        self.prompts = self._load_prompts()

    @lru_cache(maxsize=None)
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts for specified model with caching."""
        try:
            prompt_file = self.prompt_dir / f"{self.model_name}.yaml"
            if not prompt_file.exists():
                self._logger.warning(f"No prompts found for {self.model_name}, using default prompts")
                prompt_file = self.prompt_dir / "default.yaml"

            with open(prompt_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self._logger.error(f"Error loading prompts: {e}")
            return {}

    def get_prompt(self,
                   task: str,
                   role: str = "user") -> Optional[str]:
        """
        Get a prompt for a specific task and role.

        Args:
            task: The prompt task (e.g., 'summarize', 'analyze')
            role: The prompt role ('system' or 'user')

        Returns:
            Formatted prompt string or None if not found
        """
        try:
            if task not in self.prompts:
                self._logger.warning(f"Task {task} not found, using query_grounded")
                task = "query_grounded"

            if role not in self.prompts[task]:
                self._logger.warning(f"Role {role} not found for task {task}")
                return None

            prompt_template = self.prompts[task][role]
            # return prompt_template.format()
            return prompt_template

        except KeyError as e:
            self._logger.error(f"Missing key in prompt template: {e}")
            return None
        except Exception as e:
            self._logger.error(f"Error formatting prompt: {e}")
            return None

    def list_available_tasks(self) -> list:
        """List all available prompt tasks."""
        return list(self.prompts.keys())

    def get_full_prompt(self, task: str) -> Dict[str, str]:
        """Get both system and user prompts for a task."""
        return {
            "system": self.get_prompt(task, "system"),
            "user": self.get_prompt(task, "user")
        }
