from typing import Tuple, Optional

import scipy.io as spio  # matlab/python file io is a pain in the a**
import numpy as np
import torch
from torch import tensor
from mrphy.mobjs import SpinCube, Pulse


__all__ = ['m2p', 'p2m']


def m2p(m2pName: str,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
        ) -> Tuple[Optional[dict], Optional[SpinCube], Optional[Pulse],
                   Optional[dict]]:
    r"""load m2p matfile into python

    Usage:
        ``target, cube, pulse, arg = m2p(m2pName; device, dtype)
    INPUTS:
        - ``m2pName`` str, name of the m2p matfile:\
          The file stores matlab structure variables: ``target``, ``cube``, \
          ``pulse``, and ``arg``.
    OUTPUTS:
        - ``target`` dict: \
          ``.d_`` (1, nM, xyz) \
          ``.weight_`` (1, nM)
        - ``cube`` mrphy.mobjs.SpinCube
        - ``pulse`` mrphy.mobjs.Pulse
        - ``arg`` dict: Everything else
    """
    # util functions
    dkw = {'device': device, 'dtype': dtype}
    f_st2dic = lambda x: {k: x[k].item() for k in x.dtype.names}  # noqa:E731
    f_c2r_np = lambda x, a: np.stack((x.real, x.imag), axis=a)  # noqa:E731
    f_t = (lambda x, device=device, dtype=dtype:
           tensor(x[None, ...], device=device, dtype=dtype))  # noqa:E731

    # call `mfile['st_name'][member_name'].item()` to visit struct members.
    mfile = spio.loadmat(m2pName)

    # %% cube and its preprocessing
    if 'cube_st' in mfile.keys():
        tmp = f_st2dic(mfile['cube_st'])
        shape, fov = ((1,)+tuple(tmp['dim'].flatten()),
                      f_t(tmp['fov'].flatten()))
        cube_d = {k: f_t(tmp[k]) for k in ('T1', 'T2', 'M')}
        cube_d['mask'] = f_t(tmp['mask'], dtype=torch.bool)
        cube_d['ofst'] = f_t(tmp['ofst'].flatten())
        cube_d['Δf'] = f_t(tmp['b0Map']) if tmp['b0Map'].size > 0 else None
        cube_d['γ'] = f_t(tmp['gam']) if tmp['gam'].size > 0 else None

        cube = SpinCube(shape, fov, **cube_d, **dkw)
    else:
        cube = None

    # %% target and its preprocessing
    if 'target' in mfile.keys():
        tmp, target = f_st2dic(mfile['target']), {}
        # (1, nM, xyz); (1, nM)
        if 'd_' in tmp.keys():
            target['d_'] = tmp['d_']
        elif 'd' in tmp.keys():
            target['d_'] = (cube.extract(f_t(tmp['d'])) if cube
                            else f_t(tmp['d']))
        else:
            raise KeyError('Missing desired profile in dict `target`')

        if 'weight_' in tmp.keys():
            target['weight_'] = tmp['weight_']
        elif 'weight' in tmp.keys():
            target['weight_'] = (cube.extract(f_t(tmp['weight'])) if cube
                                 else f_t(tmp['weight']))
        else:
            raise KeyError('Missing weighting in dict `target`')

    else:
        target = None

    if 'pulse_st' in mfile.keys():
        # %% pulse and its preprocessing
        tmp, pulse_d = f_st2dic(mfile['pulse_st']), {}
        pulse_d['rf'] = f_t(f_c2r_np(tmp['rf'], 1)[0, ...])  # (1, xy, nT)
        pulse_d['gr'] = f_t(tmp['gr'])
        pulse_d['gmax'] = f_t(tmp['gmax'].astype(np.float).flatten())
        pulse_d['smax'] = f_t(tmp['smax'].astype(np.float).flatten())
        pulse_d['rfmax'] = f_t(tmp['rfmax'].flatten())

        # dt should have dim `()` ⊻ `(1 ⊻ N,)`
        pulse_d['dt'] = tensor(tmp['dt'].flatten(), **dkw)

        pulse = Pulse(**pulse_d, **dkw)
    else:
        pulse = None

    if 'arg' in mfile.keys():
        # %% `arg`: everything else
        arg = f_st2dic(mfile['arg'])
        # handle single coil b1Map case, as matlab ignores dim of size 1
        if 'b1Map' in arg.keys() and arg['b1Map'].ndim == cube.ndim-1:
            arg['b1Map'] = arg['b1Map'][..., None]
        if 'b1Map_' in arg.keys() and arg['b1Map_'].ndim == 1:
            arg['b1Map_'] = arg['b1Map_'][..., None]
    else:
        arg = None

    return target, cube, pulse, arg


def p2m(p2mName: str, pulse: Optional[Pulse] = None,
        arg: dict = dict()):

    if pulse:
        pulse_d = pulse.asdict(toNumpy=True)
        pulse_d['rf'] = pulse_d['rf'][0, ...]
        pulse_d['gr'] = pulse_d['gr'][0, ...]
        pulse_d.update({k: str(pulse_d[k]) for k in ('device', 'dtype')})
    else:
        pulse_d = dict()

    spio.savemat(p2mName, {'pulse_st': pulse_d, 'arg': arg})
    return
