import numpy as np
from typing import Any, Callable
import random


class Curriculum:
    def __init__(
        self,
        sample_num_batches_per_epoch_getter: Callable[[], int],
        difficulty_range_getter: Callable[[], tuple[int, int]],
        eta: float,
        alpha: float,
        beta: float,
        strategy: str = "random",
    ):
        self.strategy = strategy
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.strategy = strategy
        self.sample_num_batches_per_epoch_getter = sample_num_batches_per_epoch_getter
        self.difficulty_range_getter = difficulty_range_getter

    def infer_target_difficulty(self, metadata: dict[str, Any]) -> int:
        min_difficulty, max_difficulty = self.difficulty_range_getter()
        if self.strategy == "random":
            # TODO: 需要传进来难度scale，或者我们后面定好
            return random.randint(min_difficulty, max_difficulty)
        elif self.strategy == "reward":
            return self._reward_based_infer(metadata)
        elif self.strategy == "timestep":
            return self._timestep_based_infer(metadata)
        else:
            raise NotImplementedError("Not implemented yet")

    def _reward_based_infer(self, metadata: dict[str, Any]) -> int:
        min_difficulty, max_difficulty = self.difficulty_range_getter()
        result_difficulty = metadata["difficulty"] + self.eta * np.tanh(self.alpha * metadata["reward"] - self.beta)
        return min(max(0, result_difficulty) + min_difficulty, max_difficulty)

    def _timestep_based_infer(self, metadata: dict[str, Any]) -> int:
        min_difficulty, max_difficulty = self.difficulty_range_getter()
        return min(
            max(
                0,
                metadata["current_step"]
                // (self.sample_num_batches_per_epoch_getter() / (max_difficulty - min_difficulty + 1)),
            )
            + min_difficulty,
            max_difficulty,
        )
