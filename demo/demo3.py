from typing import Tuple
from numbers import Number
import torch
import mrphy

from numpy import ndarray, pi
from torch import tensor, Tensor
from mrphy import mobjs

from adpulses import metrics, penalties, optimizers


def demo3():
    msg = (
        '============================ demo3.py ==============================='
        '\n'
        'This tutorial demos an design of a b0/b1 robust inversion pulse using'
        '\n'
        'the AutoDiffPulses tool with an adiabatic full passage RF as init.'
        '\n'
        'The design assums a max b1 of 0.25 Gauss.'
        '\n'
        'It accounts for b0 of range [-200, 200] Hz, and 10-percent b1 loss.'
        '\n'
        '============================ demo3.py ==============================='
        '\n'
    )
    print(msg)

    flim, b1lim = (-200, 200), (0.9, 1)
    nf, nb = 201, 10
    b1max = 0.25  # Gauss

    device = torch.device('cuda:0')
    dtype = torch.float64

    adiabatic_opt(flim, b1lim, nf, nb, b1max,
                  device=device, dtype=dtype)
    return


def adiabatic_opt(
    flim: Tuple, b1lim: Tuple, nf: int, nb: int, b1max: Number,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32
):
    target, cube, pIni, b1map = init(flim, b1lim, nf, nb, b1max, device, dtype)

    err_meth, pen_meth = 'l2z', 'null'

    idx_hash = {'': slice(None), 'xyz': slice(None),
                'x':  [0],    'y': [1],     'z': [2],
                'xy': [0, 1], 'yz': [1, 2], 'xz': [0, 2]}

    if err_meth == 'null':
        fn_err = (lambda Mr_, Md_, w_:
                  metrics.err_null(Mr_, Md_, [], w_))  # NOQA: E731
    else:
        name, xyz = err_meth.split('2')
        if name == 'l':
            fn = metrics.err_l2_
        elif name == 'ml':
            fn = metrics.err_ml2_

        fn_err = (lambda Mr_, Md_, w_:
                  fn(Mr_, Md_, idx_hash[xyz], w_))  # NOQA: E731

    pen_hash = {'null': penalties.pen_null, 'l2': penalties.pen_l2}
    fn_pen = pen_hash[pen_meth]

    pIni0 = pIni
    for _ in range(1):
        pIni1, _ = optimizers.arctanLBFGS(target, cube, pIni0,
                                          fn_err=fn_err, fn_pen=fn_pen,
                                          niter=10, niter_gr=0, niter_rf=2,
                                          b1Map=b1map)
        rf = pIni1.rf  # (1, xy, nT)
        rf = (rf + torch.flip(rf, dims=[2]))/2
        pIni1.rf = rf
        pIni0 = pIni1

    return


def init(
    flim: Tuple, b1lim: Tuple, nf: int, nb: int, b1max: Number,
    device: torch.device, dtype: torch.dtype
) -> Tuple[dict, mobjs.SpinCube, mobjs.Pulse, Tensor]:

    dkw = {'device': device, 'dtype': dtype}

    fov, ofst = tensor([[0., 0., 0.]], **dkw), tensor([[0., 0., 0.]], **dkw)
    imsize = (1,) + (1, nb, nf)

    tmp1 = torch.ones(imsize, **dkw)
    b0map = tmp1 * torch.linspace(*flim, nf, **dkw)
    b1map = tmp1 * torch.linspace(*b1lim, nb, **dkw)[..., None]
    b1map = torch.stack((b1map, torch.zeros(b1map.shape, **dkw)), dim=-1)
    weight = torch.ones(imsize, **dkw)

    cube = mobjs.SpinCube(imsize, fov, ofst=ofst, Δf=b0map, **dkw)

    fn_target = lambda d_, weight_: {'d_': d_, 'weight_': weight_}  # noqa:E731

    d = torch.stack((torch.zeros(imsize, **dkw),
                     torch.zeros(imsize, **dkw),
                     -torch.ones(imsize, **dkw)), axis=-1)

    target = fn_target(cube.extract(d), cube.extract(weight))

    dt = mrphy.dt0.to(**dkw)  # Sec, torch.Tensor
    beta = 5.3

    rf_peak = 0.8 * b1max

    fn_adiabatic = fullpassage

    tp = 1.5e-3  # Sec
    bw = 0.81e3  # Hz

    rf_c = rf_peak * fn_adiabatic(tp, beta, bw, dt.item())[None, None, ...]
    rf_c = rf_c.to(device=dkw['device'])
    rf = torch.cat((torch.real(rf_c), torch.imag(rf_c)), dim=1)
    gr = torch.zeros((1, 3, rf_c.shape[2]), **dkw)

    pulse = mobjs.Pulse(rf, gr, rfmax=b1max, dt=dt, **dkw)

    return target, cube, pulse, b1map


def fullpassage(tp: Number, beta: Number, bw: Number, dt: Number) -> ndarray:
    tn = torch.linspace(-1, 1, 2*round(tp/dt/2)+1, dtype=torch.float64)
    cSub = (2*round(tp/dt/2)+1)//2
    amp = 1 / torch.cosh(beta * tn)
    amp = amp/torch.max(amp)

    frq = bw/2 * torch.tanh(beta*-tn)

    phs = torch.cumsum(frq * dt * pi, dim=0)
    phs = phs - phs[cSub] + 0*pi/2

    rf_n = amp * torch.exp(1j * phs)
    return rf_n


if __name__ == '__main__':
    demo3()
