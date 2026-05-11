"""Robust reinforcement learning for spherical tank level control."""

__all__ = ["SphericalTank"]


def __getattr__(name):
    if name == "SphericalTank":
        from robust_rl_tank.env import SphericalTank

        return SphericalTank
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
