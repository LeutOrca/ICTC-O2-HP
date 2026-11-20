"""Native lane-zero ascending FP32 sum without version-specific tl.gather."""
import triton
import triton.language as tl


@triton.jit
def sum32_ascending(values):
    rows: tl.constexpr = values.shape[0]
    # Lane zero of shfl_down offsets 1,2,4,8,16 is the adjacent-pair tree.
    # Keep this tree explicit; tl.sum may choose another FP32 addition order.
    for level in tl.static_range(5):
        pairs = tl.reshape(values, (rows, 16 >> level, 2))
        lower, upper = tl.split(pairs)
        values = lower + upper
    return tl.reshape(values, (rows,))
