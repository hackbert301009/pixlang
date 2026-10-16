# pixlang/commands/__init__.py
from .registry import (
    ArgumentError, CommandInfo, CommandRegistry, ConflictError, UnknownCommandError,
)
from . import builtin
from pixlang.batch import register_batch_commands

registry = CommandRegistry()
builtin.register_all(registry)
builtin.register_v03(registry)
builtin.register_v04(registry)
register_batch_commands(registry)

__all__ = [
    "registry", "CommandRegistry", "CommandInfo", "ConflictError",
    "UnknownCommandError", "ArgumentError",
]
