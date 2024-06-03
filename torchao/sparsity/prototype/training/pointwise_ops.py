from functools import partial
import torch
from torch.sparse import SparseSemiStructuredTensor 
from torchao.sparsity.prototype.training.autograd import semi_sparse_sparsify_like

def _semi_sparse_pointwise_op(func, types, args=(), kwargs=None, sparsify_like_args_list=()):
    """
    adds pointwise op support for semi-structured tensors. 

    Assumes that at least one of the arguments in arg is a SparseSemiStructuredTensor.
    The last instance of a SparseSemiStructuredTensor is used as the reference mask to sparsify the others tensors passed in args. 
    sparsify_like_args_list is used to specify which arguments to sparsify like the reference tensor.
    """
    reference_sparse_tensor = None
    for tensor in args:
        if isinstance(tensor, SparseSemiStructuredTensor):
            reference_sparse_tensor = tensor
    assert reference_sparse_tensor is not None

    def handle_arg(i, tensor):
        if isinstance(tensor, torch.Tensor):
            if not isinstance(tensor, SparseSemiStructuredTensor):
                if i in sparsify_like_args_list:
                    tensor = semi_sparse_sparsify_like(tensor, reference_sparse_tensor)
                else:
                    raise ValueError(
                        f"Operation {func.__module__}.{func.__name__} on {type(reference_sparse_tensor)} requires all operands to "
                        f"be {type(reference_sparse_tensor)}, but operand {i} is a {type(tensor)}"
                    )
            else:
                if (
                    tensor.compressed_swizzled_bitmask is None
                    or reference_sparse_tensor.compressed_swizzled_bitmask is None
                    or tensor.compressed_swizzled_bitmask.data_ptr() != reference_sparse_tensor.compressed_swizzled_bitmask.data_ptr()
                    or tensor.compressed_swizzled_bitmask.stride() != reference_sparse_tensor.compressed_swizzled_bitmask.stride()
                ):
                    raise ValueError(
                        f"Operation {func.__module__}.{func.__name__} on {type(reference_sparse_tensor)} requires all operands to be "
                        f"{type(reference_sparse_tensor)} with the same sparsity pattern"
                    )
        return tensor

    args_updated = [ handle_arg(i, tensor) for i, tensor in enumerate(args) ]

    return reference_sparse_tensor.__class__(
        reference_sparse_tensor.shape,
        func(*[
                x.packed if isinstance(x, SparseSemiStructuredTensor) else x
                for x in args_updated
            ]),
        reference_sparse_tensor.meta,
        func(
            *[
                x.packed_t if isinstance(x, SparseSemiStructuredTensor)
                else x for x in args_updated
            ]
        ),
        reference_sparse_tensor.meta_t,
        reference_sparse_tensor.compressed_swizzled_bitmask,
    )

# Add pointwise ops to the dispatch table
CUTLASS_POINTWISE_OP_DISPATCH_TABLE = {
    torch.ops.aten.relu: _semi_sparse_pointwise_op,
    torch.ops.aten.gelu: _semi_sparse_pointwise_op,
    torch.ops.aten.silu: _semi_sparse_pointwise_op,
    torch.ops.aten.mul: partial(
        # `mul` BW in swiglu
        _semi_sparse_pointwise_op,
        sparsify_like_args_list=(0, 1),
    ),
    torch.ops.aten.add: _semi_sparse_pointwise_op,
    # Note: for these ops, we allow the gradient to come in as a `torch.Tensor`
    # and we will run the sparsification right before calling the BW aten func
    torch.ops.aten.gelu_backward: partial(
        _semi_sparse_pointwise_op,
        sparsify_like_args_list=(0,)
    ),
    torch.ops.aten.silu_backward: partial(
        _semi_sparse_pointwise_op,
        sparsify_like_args_list=(0, 1)
    ),
    torch.ops.aten.threshold_backward: partial(  # relu BW
        _semi_sparse_pointwise_op,
        sparsify_like_args_list=(0,),
    ),
}
