from typing import Any
import json
from pathlib import Path
import logging
import accelerate

import tqdm

logger = logging.getLogger(__name__)


class CurriculumPromptLoader:
    def __init__(self, prompt_path: str) -> None:
        self.accelerator: None | accelerate.Accelerator = None
        self.difficulty_to_prompts: dict[int, list[dict[str, Any]]] = {}
        self.difficulty_to_prompts_idx: dict[int, int] = {}
        self.prompt_path = Path(prompt_path)
        self.current_difficulty = 1
        self.sample_num_batches_per_epoch = 0
        self.t: tqdm.tqdm | None = None
        self.difficulty_range: tuple[int, int] | None = None

    def get_sample_num_batches_per_epoch(self) -> int:
        return self.sample_num_batches_per_epoch

    def difficulty_range_getter(self) -> tuple[int, int]:
        assert self.difficulty_range, "need init"
        return self.difficulty_range

    def init(self, accelerator: accelerate.Accelerator, batch_size: int):
        self.accelerator = accelerator
        total = 0
        logger.info(f"initial index: {self.accelerator.process_index}, num process: {self.accelerator.num_processes}")
        for difficulty_str, prompts in json.loads(self.prompt_path.read_text()).items():
            total += len(prompts)
            self.difficulty_to_prompts[self._extract_difficulty(difficulty_str)] = prompts
            self.difficulty_to_prompts_idx[self._extract_difficulty(difficulty_str)] = self.accelerator.process_index
        self.t = tqdm.tqdm(total=total, desc="dataloader")
        self.sample_num_batches_per_epoch = total // (self.accelerator.num_processes * batch_size)
        self.difficulty_range = (min(self.difficulty_to_prompts), max(self.difficulty_to_prompts))

    def _extract_difficulty(self, difficulty_str: str) -> int:
        return int(difficulty_str.split("_")[-1])

    def next(self) -> tuple[str, Any]:
        assert self.accelerator and self.t, "not initialize"
        self.t.update(self.accelerator.num_processes)
        if self.difficulty_to_prompts_idx[self.current_difficulty] >= len(
            self.difficulty_to_prompts[self.current_difficulty]
        ):
            logger.warning(f"difficulty {self.current_difficulty} has no more prompts, reset to 0")
            self.difficulty_to_prompts_idx[self.current_difficulty] = self.accelerator.process_index
        prompt = self.difficulty_to_prompts[self.current_difficulty][
            self.difficulty_to_prompts_idx[self.current_difficulty]
        ]
        self.difficulty_to_prompts_idx[self.current_difficulty] += self.accelerator.num_processes
        return prompt["prompt"], prompt

    def set_difficulty(self, difficulty: int) -> None:
        logger.info(f"set difficulty to {difficulty}")
        self.current_difficulty = difficulty
