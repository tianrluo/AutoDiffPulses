import torch
from torch import Tensor


def pen_null(rf: Tensor) -> Tensor:
    """
    *INPUTS*
    - `rf`  (1, xy, nT, (nCoils))
    *OUTPUTS*
    - `pen` (1,)
    """
    return rf.new_zeros([])


def pen_l2(rf: Tensor) -> Tensor:
    pen = torch.norm(rf)**2
    return pen
