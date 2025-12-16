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
    "WebScraper",
    "InstructionHandler",
    "ChatInterface",
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
    elif name == "WebScraper":
        from .web_scraper import WebScraper
        return WebScraper
    elif name == "InstructionHandler":
        from .instruction_handler import InstructionHandler
        return InstructionHandler
    elif name == "ChatInterface":
        from .chat_interface import ChatInterface
        return ChatInterface
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
