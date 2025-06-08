import numpy as np


class Agent:
    """Simple placeholder agent returning zero actions."""

    def act(self, obs, beta: float = 1.0):
        """Return an action and attention weights.

        Parameters
        ----------
        obs : Any
            Observation of the current game state.
        beta : float, optional
            Unused randomness parameter, by default 1.0.

        Returns
        -------
        tuple
            A tuple ``(action, weights)`` where ``action`` is an 8 element
            numpy array matching the controller format and ``weights`` is ``None``.
        """
        action = np.zeros(8, dtype=float)
        weights = None
        return action, weights
