r"""Basis for parameterizing waveforms
"""

import torch
from torch import Tensor


def bspline(
        order: int, nB: int, nT: int, isHead: bool = True,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32) -> Tensor:
    r"""BSpine basis for waveform parameterization

    Usage:
        ``basis = bspline(order, nB, nT)``

    Inputs:
        - ``order``: `(1,)`, int, order of the B-spline.
        - ``nB``: `(1,)`, int, number of basis.
        - ``nT``: `(1,)`, int, length of a basis.
    Optionals:
        - ``isHead``: [T/f], 1-less node basis at head or tail
    Outputs:
        - ``basis``: `(1, nB, nT)`, Tensor, BSpline basis.

    .. warning:
        Future version will output ``torch.SparseTensor``, once SparseTensor
        supports ``reshape``, ``transpose``, and ``matmul``.
    """

    basis = torch.zeros([1, nB, nT], device=device, dtype=dtype)

    # divide `nT` into `nB` pieces, `nTB * (nB-1) + nXtra == nT`
    nTB = (nT // nB)
    nXtra = nT - (nTB * nB)  # nT = (nB-nXtra)*nTB + nXtra*(nTB+1)

    if isHead:
        split = (nB-nXtra)*nTB
        ibB = list(range(0, split, nTB))+list(range(split, nT, nTB+1))+[nT]
    else:
        split = nXtra*(nTB+1)
        ibB = list(range(0, split, nTB+1))+list(range(split, nT, nTB))+[nT]

    if order == 0:
        for iB in range(nB):
            basis[0, iB, ibB[iB]:ibB[iB+1]] = 1.0
    else:
        raise ValueError("Only 0th order is supported in this verion")

    return basis
