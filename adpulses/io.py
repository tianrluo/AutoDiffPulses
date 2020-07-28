from typing import Tuple

import scipy.io as spio  # matlab/python file io is a pain in the a**
import numpy as np
import torch
from torch import tensor
from mrphy.mobjs import SpinCube, Pulse

from adpulses import metrics, penalties


def m2p(m2pName: str,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
        ) -> Tuple[dict, SpinCube, Pulse, dict]:
    """
    *INPUTS*:
    - m2pName str, name of the m2p matfile:
      The file stores matlab structure variables: `target`, `cube`, `pulse`,
      and `arg`.
    *OUTPUTS*:
    - target dict:
      .d_ (1, nM, xyz)
      .weight_ (1, nM)
    - cube (1,) mrphy.mobjs.SpinCube
    - pulse (1,) mrphy.mobjs.Pulse
    - arg dict: Everything else
    """
    # util functions
    dkw = {'device': device, 'dtype': dtype}
    f_st2dic = lambda x: {k: x[k].item() for k in x.dtype.names}  # noqa:E731
    f_c2r_np = lambda x, a: np.stack((x.real, x.imag), axis=a)  # noqa:E731
    f_tensor = (lambda x, device=device, dtype=dtype:
                tensor(x[None, ...], device=device, dtype=dtype))  # noqa:E731

    # call `mfile['st_name'][member_name'].item()` to visit struct members.
    mfile = spio.loadmat(m2pName)

    # %% cube and its preprocessing
    tmp = f_st2dic(mfile['cube_st'])
    shape, fov = ((1,)+tuple(tmp['dim'].flatten()),
                  f_tensor(tmp['fov'].flatten()))
    cube_d = {k: f_tensor(tmp[k]) for k in ('T1', 'T2', 'M')}
    cube_d['mask'] = f_tensor(tmp['mask'], dtype=torch.bool)
    cube_d['ofst'] = f_tensor(tmp['ofst'].flatten())
    cube_d['Δf'] = f_tensor(tmp['b0Map']) if tmp['b0Map'].size > 0 else None
    cube_d['γ'] = f_tensor(tmp['gam']) if tmp['gam'].size > 0 else None

    cube = SpinCube(shape, fov, **cube_d, **dkw)

    # %% target and its preprocessing
    tmp, target = f_st2dic(mfile['target']), {}
    # (1, nM, xyz); (1, nM)
    target['d_'] = cube.extract(f_tensor(tmp['d']))  # (1,nM,xyz)
    target['weight_'] = cube.extract(f_tensor(tmp['weight'].astype(np.float)))

    # %% pulse and its preprocessing
    tmp, pulse_d = f_st2dic(mfile['pulse_st']), {}
    pulse_d['rf'] = f_tensor(f_c2r_np(tmp['rf'], 1)[0, ...])  # (1, xy, nT)
    pulse_d['gr'] = f_tensor(tmp['gr'])
    pulse_d['dt'] = f_tensor(tmp['dt'].flatten())
    pulse_d['gmax'] = f_tensor(tmp['gmax'].astype(np.float).flatten())
    pulse_d['smax'] = f_tensor(tmp['smax'].astype(np.float).flatten())
    pulse_d['rfmax'] = f_tensor(tmp['rfmax'].flatten())

    pulse = Pulse(**pulse_d, **dkw).to(device=cube.device, dtype=cube.dtype)

    # %% `arg`: everything else
    arg = f_st2dic(mfile['arg'])

    return target, cube, pulse, arg


def p2m(p2mName: str, pulse: Pulse, optInfos: dict):
    pulse_dict = pulse.asdict(toNumpy=True)
    pulse_dict['rf'] = pulse_dict['rf'][0, ...]
    pulse_dict['gr'] = pulse_dict['gr'][0, ...]
    pulse_dict.update({k: str(pulse_dict[k]) for k in ('device', 'dtype')})
    spio.savemat(p2mName, {'pulse_st': pulse_dict, 'optInfos': optInfos})
    return
