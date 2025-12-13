"""
Matt AI - Self-Training Large Language Model System

A self-improving AI system with built-in ethical standards and constraints.
"""

__version__ = "0.1.0"
__author__ = "Matthew J Picone"

# Lazy imports to avoid requiring heavy dependencies unless actually used
__all__ = [
    "EthicalController",
    "SelfTrainingLLM",
    "IterativeTrainer",
]

def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies unnecessarily."""
    if name == "EthicalController":
        from .ethical_controller import EthicalController
        return EthicalController
    elif name == "SelfTrainingLLM":
        from .model import SelfTrainingLLM
        return SelfTrainingLLM
    elif name == "IterativeTrainer":
        from .trainer import IterativeTrainer
        return IterativeTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
