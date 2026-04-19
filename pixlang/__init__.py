# pixlang/__init__.py
"""
PixLang — A minimal, human-readable DSL for computer vision pipelines.
"""
__version__ = "0.4.0"
__author__ = "PixLang Contributors"

from .parser import parse
from .executor import Executor
from .commands import registry

__all__ = ["parse", "Executor", "registry"]
