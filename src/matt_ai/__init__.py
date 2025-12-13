"""
Matt AI - Self-Training Large Language Model System

A self-improving AI system with built-in ethical standards and constraints.
"""

__version__ = "0.1.0"
__author__ = "Matthew J Picone"

from .ethical_controller import EthicalController
from .model import SelfTrainingLLM
from .trainer import IterativeTrainer

__all__ = [
    "EthicalController",
    "SelfTrainingLLM",
    "IterativeTrainer",
]
