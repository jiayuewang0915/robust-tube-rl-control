import numpy as np


def scale_error(error: np.ndarray) -> np.ndarray:
    """Map the error interval [-4, 4] to [-1, 1]."""
    return -1.0 + 0.25 * (np.asarray(error) + 4.0)


def scale_state(state: np.ndarray) -> np.ndarray:
    """Map the tank level interval [0, 4] to [-1, 1]."""
    return -1.0 + 0.5 * np.asarray(state)


def unscale_action(scaled_action: float, low: float = 0.0, high: float = 15.0) -> float:
    """Map an actor output in [-1, 1] to the physical input range."""
    return float((scaled_action + 1.0) * (high - low) / 2.0 + low)


def combine_scaled_actions(model_action: float, error_action: float, low: float = 0.0, high: float = 15.0) -> float:
    """Average two scaled actor outputs and map the result to the physical input range.

    The robust controller combines the nominal model action and the learned error
    correction before converting back to the tank input units.
    """
    return float(((model_action + error_action) + 2.0) * (high - low) / 4.0 + low)
