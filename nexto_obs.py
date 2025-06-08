from typing import Any, Iterable
import numpy as np


BOOST_LOCATIONS: Iterable[np.ndarray] = []


class NextoObsBuilder:
    """Placeholder observation builder for the Nexto agent."""

    def __init__(self, field_info: Any = None):
        self.field_info = field_info

    def build_obs(self, player: Any, game_state: Any, prev_action: np.ndarray):
        """Return a minimal observation for the agent.

        This implementation simply returns an empty numpy array. It exists so
        that :class:`bot.Nexto` can instantiate and call into the object without
        requiring the heavy rlgym dependencies used by the original project.
        """
        return np.array([])
