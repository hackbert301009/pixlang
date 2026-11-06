"""`$name` substitution inside ASSERT messages."""
from __future__ import annotations

import re
from typing import Any, Callable, Tuple

# A spec must end in a type letter, so "slot $i: empty" keeps its colon as text.
_FORMAT_SPEC = r"[<>^]?[+-]?0?\d*[,_]?(?:\.\d+)?[bcdeEfFgGnosxX%]"
_PLACEHOLDER = re.compile(
    r"\$(?:"
    r"(?P<escaped>\$)"
    rf"|\{{(?P<braced>[A-Za-z_]\w*)(?::(?P<braced_spec>{_FORMAT_SPEC}))?\}}"
    rf"|(?P<bare>[A-Za-z_]\w*)(?::(?P<bare_spec>{_FORMAT_SPEC}))?"
    r")"
)


def interpolate(template: str, lookup: Callable[[str], Tuple[bool, Any]],
                render: Callable[[Any], str]) -> str:
    """Replace ``$name``, ``${name}``, ``$name:.2f`` and ``$$`` in *template*.

    A name that is not defined is left as written so the original failure
    stays visible; a spec the value cannot take falls back to *render*.
    """
    if "$" not in template:
        return template

    def substitute(match: re.Match) -> str:
        if match.group("escaped"):
            return "$"
        name = match.group("braced") or match.group("bare")
        spec = match.group("braced_spec") or match.group("bare_spec")
        found, value = lookup(name)
        if not found:
            return match.group(0)
        if spec:
            try:
                return format(value, spec)
            except (TypeError, ValueError):
                pass
        return render(value)

    return _PLACEHOLDER.sub(substitute, template)
