# pixlang/commands/__init__.py
from pixlang.batch import register_batch_commands

from . import builtin, lowlevel
from .registry import (
    ArgumentError,
    CommandInfo,
    CommandRegistry,
    ConflictError,
    UnknownCommandError,
)

registry = CommandRegistry()
builtin.register_all(registry)
builtin.register_v03(registry)
builtin.register_v04(registry)
lowlevel.register_v05(registry)
register_batch_commands(registry)

__all__ = [
    "registry", "CommandRegistry", "CommandInfo", "ConflictError",
    "UnknownCommandError", "ArgumentError",
]
