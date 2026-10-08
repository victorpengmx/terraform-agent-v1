"""Semantic terrain ops, registered via the op registry (DESIGN §5.5).

Importing this package registers every op module listed below.
"""

from terrain.ops import detail, erode, materials, noise, peaks, raise_region, smooth, terrace, valleys, water  # noqa: F401  (registers ops)
from terrain.ops.registry import (
    STAGES,
    OpError,
    OpParams,
    OpSpec,
    apply_op,
    get_op,
    list_ops,
    register_op,
    tool_schemas,
)

__all__ = ["STAGES", "OpError", "OpParams", "OpSpec", "apply_op", "get_op", "list_ops", "register_op", "tool_schemas"]
