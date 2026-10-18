# pixlang/plugins/__init__.py
from .loader import (
    ENTRY_POINT_GROUP,
    PluginLoader,
    PluginManifest,
    config_trust_request,
    local_plugin_path,
    trust_from_config,
)
from .trust import (
    add_trusted,
    is_trusted,
    overbroad_warning,
    remove_trusted,
    trust_file,
    trust_hint_for,
    trusted_paths,
)

__all__ = [
    "PluginLoader", "PluginManifest", "ENTRY_POINT_GROUP",
    "local_plugin_path", "trust_from_config", "config_trust_request",
    "add_trusted", "remove_trusted", "is_trusted", "trusted_paths",
    "trust_file", "trust_hint_for", "overbroad_warning",
]
