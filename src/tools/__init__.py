"""Tool package — importing triggers self-registration of all built-in tools.

Import order matters: registry first, then tool modules.
"""

from .registry import ToolRegistry, get_registry, register

# Importing these modules triggers their register() calls at module-level.
from . import file_ops  # noqa: F401
from . import coding  # noqa: F401
from . import interpreter  # noqa: F401
from . import search  # noqa: F401
from . import plan_tools  # noqa: F401

__all__ = ["ToolRegistry", "get_registry", "register"]
