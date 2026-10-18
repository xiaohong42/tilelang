"""HIP lowering rules for vectorized float32 math intrinsics.

TVM's default rules map scalar ``float32`` calls such as ``T.exp2`` to the device
math library (``exp2f``) but leave vector calls untouched, so a ``T.Parallel``
loop that vectorizes over ``T.exp2`` fails in the HIP codegen with ``Unresolved
call tirx.exp2``. Lower vector calls to the same extern function; the HIP codegen
already scalarizes vector extern calls lane by lane. Scalar calls fall through to
the default rules.
"""

from tvm import DataType, tirx
from tvm.ir import register_intrin_lowering

_UNARY_FLOAT_OPS = ("exp", "exp2", "exp10", "log", "log2", "log10", "sqrt", "tanh", "sin", "cos")


def _lower_vector_float32(op):
    dtype = DataType(op.dtype)
    if dtype.lanes <= 1 or dtype.with_lanes(1) != "float32":
        return op
    name = op.op.name.split(".")[-1]
    return tirx.call_pure_extern(op.dtype, f"{name}f", *op.args)


for _name in _UNARY_FLOAT_OPS:
    register_intrin_lowering(f"tirx.{_name}", target="hip", f=_lower_vector_float32, level=99)
