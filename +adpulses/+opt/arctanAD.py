# arctan.py

import numpy as np
import torch
from torch import tensor

import adpulses

from adpulses import io, optimizers, metrics, penalties

if __name__ == "__main__":
    import sys

    if len(sys.argv) <= 1:  # mode DEBUG
        import os
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    m2pName = ('m2p.mat' if len(sys.argv) <= 1 else sys.argv[1])
    p2mName = ('p2m.mat' if len(sys.argv) <= 2 else sys.argv[2])
    gpuID = ('0' if len(sys.argv) <= 3 else sys.argv[3])

    # %% load
    if gpuID == '-1':
        device, dtype = torch.device('cpu'), torch.float32
    else:
        device, dtype = torch.device('cuda:'+gpuID), torch.float32

    target, cube, pulse, arg = io.m2p(m2pName, device=device, dtype=dtype)

    def dflt_arg(k, v, fn):
        return (fn(k) if ((k in arg.keys()) and (arg[k].size > 0)) else v)

    f_c2r_np = lambda x, a: np.stack((x.real, x.imag), axis=a)  # noqa:E731
    f_t = (lambda x, device=device, dtype=dtype:
                tensor(x[None, ...], device=device, dtype=dtype))  # noqa:E731

    arg['doRelax'] = dflt_arg('doRelax', True, lambda k: bool(arg[k].item()))

    b1Map = dflt_arg('b1Map', None, lambda k: f_t(f_c2r_np(arg[k], -2)))
    b1Map_ = dflt_arg('b1Map_', None, lambda k: f_t(f_c2r_np(arg[k], -2)))
    assert ((b1Map_ is None) or (b1Map is None))

    arg['b1Map_'] = (b1Map_ if b1Map is None else cube.extract(b1Map))

    arg['niter'] = dflt_arg('niter', 8, lambda k: arg[k].item())
    arg['niter_gr'] = dflt_arg('niter_gr', 2, lambda k: arg[k].item())
    arg['niter_rf'] = dflt_arg('niter_rf', 2, lambda k: arg[k].item())

    eta = dflt_arg('eta', 4, lambda k: float(arg[k].item()))
    print('eta: ', eta)

    err_meth = dflt_arg('err_meth', 'l2xy', lambda k: arg[k].item())
    pen_meth = dflt_arg('pen_meth', 'l2', lambda k: arg[k].item())

    err_hash = {'null': metrics.err_null, 'l2': metrics.err_l2,
                'l2xy': metrics.err_l2xy, 'ml2xy': metrics.err_ml2xy,
                'l2z': metrics.err_l2z}
    pen_hash = {'null': penalties.pen_null, 'l2': penalties.pen_l2}

    fn_err, fn_pen = err_hash[err_meth], pen_hash[pen_meth]

    # %% pulse design
    kw = {k: arg[k] for k in ('b1Map_', 'niter', 'niter_gr', 'niter_rf',
                              'doRelax')}

    pulse, optInfos = optimizers.arctanLBFGS(target, cube, pulse,
                                             fn_err, fn_pen, eta=eta, **kw)

    # %% saving
    io.p2m(p2mName, pulse, {'optInfos': optInfos})
