from typing import Optional, Union
from torch import Tensor


def err_null(
    Mr_: Tensor,
    Md_: Tensor,
    idx: Union[slice, list] = slice(None),
    w_: Optional[Tensor] = None
) -> Tensor:
    """
    *INPUTS*
    - `Mr_` (1, nM, xyz)
    - `Md_` (1, nM, xyz)
    *OPTIONALS*
    - `w_`  (1, nM)
    *OUTPUTS*
    - `err` (1,)
    """
    return Mr_.new_zeros([])


def err_l2_(
    Mr_: Tensor,
    Md_: Tensor,
    idx: Union[slice, list] = slice(None),
    w_: Optional[Tensor] = None
) -> Tensor:
    """
    *INPUTS*
    - `Mr_` (1, nM, xyz)
    - `Md_` (1, nM, xyz)
    *OPTIONALS*
    - `idx` indices for slicing `(Mr_ - Md_)`, e.g.:
      `x[..., 1:] is x[..., slice(1, None)] is x[..., [1,2]]`.
    - `w_`  (1, nM)
    *OUTPUTS*
    - `err` (1,)
    """
    Me_ = (Mr_ - Md_)[..., idx]
    err = (Me_ if w_ is None else Me_*w_[..., None]).norm()**2
    return err


def err_ml2_(
    Mr_: Tensor,
    Md_: Tensor,
    idx: Union[slice, list] = slice(None),
    w_: Optional[Tensor] = None
) -> Tensor:
    """
    *INPUTS*
    - `Mr_` (1, nM, xyz)
    - `Md_` (1, nM, xyz)
    *OPTIONALS*
    - `idx` indices for slicing `(Mr_ - Md_)`, e.g.:
      `x[..., 1:] is x[..., slice(1, None)] is x[..., [1,2]]`.
    - `w_`  (1, nM)
    *OUTPUTS*
    - `err` (1,)
    """
    Me_ = Mr_[..., idx].norm(dim=-1) - Md_[..., idx].norm(dim=-1)
    err = (Me_ if w_ is None else Me_*w_[..., None]).norm()**2
    return err
