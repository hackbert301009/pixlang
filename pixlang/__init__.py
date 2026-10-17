# pixlang/__init__.py
"""
PixLang — A minimal, human-readable DSL for computer vision pipelines.
"""
__version__ = "0.5.0"
__author__ = "PixLang Contributors"

from .commands import registry
from .executor import Executor
from .parser import parse

__all__ = ["parse", "Executor", "registry"]
