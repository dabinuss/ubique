import logging
import random
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class ProviderRouter:
    """
    Manages available execution and data providers for Ubique v3.
    Incorporates a lightweight curiosity heuristic to favor novel or less-frequented
    provider paths, preventing premature convergence and supporting continuous exploration.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.registry: Dict[str, Any] = {}
        self.exploration_counts: Dict[str, int] = {}
        self.curiosity_weight: float = float(self.config.get("curiosity_weight", 0.15))

    def register(self, name: str, provider: Any) -> None:
        self.registry[name] = provider
        if name not in self.exploration_counts:
            self.exploration_counts[name] = 0
        logger.info(f"Registered provider: {name}")

    def select(self, task_type: str, candidate_names: List[str]) -> Optional[str]:
        if not candidate_names:
            return None

        valid_candidates = [c for c in candidate_names if c in self.registry]
        if not valid_candidates:
            return None

        # Apply a curiosity bonus inversely proportional to past selection counts
        # to encourage broader experiential learning and hypothesis testing.
        def score_candidate(name: str) -> float:
            count = self.exploration_counts.get(name, 0)
            # Inverse frequency bonus with smoothing
            novelty_bonus = 1.0 / (1.0 + float(count))
            base_score = 1.0
            return base_score + (self.curiosity_weight * novelty_bonus)

        scored = [(name, score_candidate(name)) for name in valid_candidates]
        total_score = sum(score for _, score in scored)

        if total_score <= 0:
            selected = random.choice(valid_candidates)
        else:
            r = random.uniform(0, total_score)
            cumulative = 0.0
            selected = valid_candidates[0]
            for name, score in scored:
                cumulative += score
                if r <= cumulative:
                    selected = name
                    break

        self.exploration_counts[selected] = self.exploration_counts.get(selected, 0) + 1
        logger.debug(f"Selected provider {selected} for task {task_type} (counts: {self.exploration_counts})")
        return selected

    def get(self, name: str) -> Optional[Any]:
        return self.registry.get(name)
