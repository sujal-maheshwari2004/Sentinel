import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional

from core.config import ModelConfig

log = logging.getLogger(__name__)


class ModelState(IntEnum):
    WAITING     = 0
    TRAINING    = 1
    INFERENCING = 2
    RETRAINING  = 3
    ERROR       = 4


@dataclass
class ModelInstance:
    """
    Runtime state for a single model managed by Sentinel.
    Created when a model is registered and lives for the duration of the process.
    """
    config: ModelConfig
    state: ModelState = ModelState.WAITING
    rows_collected: int = 0
    waiting_since: float = field(default_factory=time.time)
    last_trained: Optional[float] = None
    last_inferred: Optional[float] = None
    artifact_path: Optional[str] = None
    consecutive_errors: int = 0
    mlflow_run_id: Optional[str] = None


class ModelRegistry:
    """
    Owns all model instances and their state.
    All state transitions go through here - nothing else mutates model state directly.
    """

    def __init__(self):
        self._models: Dict[str, ModelInstance] = {}

    def register(self, config: ModelConfig) -> ModelInstance:
        """Register a new model. Always starts in WAITING state."""
        instance = ModelInstance(config=config)
        self._models[config.name] = instance
        log.info("Registered model: %s", config.name)
        return instance

    def get(self, name: str) -> Optional[ModelInstance]:
        """Return a model instance by name, or None if not found."""
        return self._models.get(name)

    def get_all(self) -> List[ModelInstance]:
        """Return all registered model instances."""
        return list(self._models.values())

    def get_by_state(self, state: ModelState) -> List[ModelInstance]:
        """Return all models currently in a given state."""
        return [m for m in self._models.values() if m.state == state]

    def transition(self, model: ModelInstance, new_state: ModelState) -> None:
        """
        Move a model to a new state and log the transition.
        This is the only place state should change.
        """
        old_state = model.state
        model.state = new_state
        log.info(
            "Model %s: %s -> %s",
            model.config.name,
            old_state.name,
            new_state.name,
        )