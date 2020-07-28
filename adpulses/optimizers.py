from typing import Tuple, Callable, Optional
from time import time
from numbers import Number

import numpy as np
from torch import optim, Tensor
import mrphy
from mrphy.mobjs import SpinCube, Pulse


def arctanLBFGS(
        target: dict, cube: SpinCube, pulse: Pulse,
        fn_err: Callable[[Tensor, Tensor, Optional[Tensor]], Tensor],
        fn_pen: Callable[[Tensor], Tensor],
        niter: int = 10, niter_gr: int = 2, niter_rf: int = 2,
        eta: Number = 4., b1Map_: Optional[Tensor] = None, doRelax: bool = True
        ) -> Tuple[Pulse, dict]:
    r"""Joint RF/GR optimization via direct arctan trick

    Usage:
        ``arctanLBFGS(target, cube, pulse, fn_err, fn_pen; eta=eta)``

    Inputs:
        - ``target``: dict, with fields:
            ``d_``: `(1, nM, xy)`, desired excitation;
            ``weight_``: `(1, nM)`.
        - ``cube``: mrphy.mobjs.SpinCube.
        - ``pulse``: mrphy.mobjs.Pulse.
        - ``fn_err``: error metric function. See :mod:`~adpulses.metrics`.
        - ``fn_pen``: penalty function. See :mod:`~adpulses.penalties`.
    Optionals:
        - ``niter``: int, number of iterations.
        - ``niter_gr``: int, number of LBFGS iters for updating *gradients*.
        - ``niter_rf``: int, number of LBFGS iters for updating *RF*.
        - ``eta``: `(1,)`, penalization term weighting coefficient.
        - ``b1Map_``: `(1, nM, xy,(nCoils))`, a.u., transmit sensitivity.
        - ``doRelax``: [T/f], whether accounting relaxation effects in simu.
    Outputs:
        - ``pulse``: mrphy.mojbs.Pulse, optimized pulse.
        - ``optInfos``: dict, optimization informations.
    """
    # Set up: Interior mapping
    tρ, θ = mrphy.utils.rf2tρθ(pulse.rf, pulse.rfmax)
    tsl = mrphy.utils.s2ts(mrphy.utils.g2s(pulse.gr, pulse.dt), pulse.smax)

    opt_rf = optim.LBFGS([tρ, θ], lr=3., max_iter=10, history_size=100,
                         tolerance_change=1e-4,
                         line_search_fn='strong_wolfe')

    opt_sl = optim.LBFGS([tsl], lr=3., max_iter=20, history_size=100,
                         tolerance_change=1e-6,
                         line_search_fn='strong_wolfe')

    tρ.requires_grad = θ.requires_grad = tsl.requires_grad = True

    # Set up: optimizer
    loss_hist = np.full((niter*(niter_gr+niter_rf),), np.nan)
    Md_, w_ = target['d_'], target['weight_'].sqrt()  # (1, nM, xy), (1, nM)

    rfmax, smax = pulse.rfmax, pulse.smax

    def fn_loss(cube, pulse):
        Mr_ = cube.applypulse(pulse, b1Map_=b1Map_, doRelax=doRelax)
        loss_err, loss_pen = fn_err(Mr_, Md_, w_=w_), fn_pen(pulse.rf)
        return loss_err, loss_pen

    log_col = '\n#iter\t ‖ elapsed time\t ‖ error\t ‖ penalty\t ‖ total loss'

    def logger(i, t0, loss_err, loss_pen):
        loss = loss_err + eta*loss_pen
        print("%i\t | %.1f  \t | %.3f\t | %.3f\t | %.3f" %
              (i, time()-t0, loss_err.item(), loss_pen.item(), loss.item()))
        return loss

    # Optimization
    t0 = time()
    for i in range(niter):

        if not (i % 5):
            print(log_col)

        log_ind = 0

        def closure():
            opt_rf.zero_grad()
            opt_sl.zero_grad()

            pulse.rf = mrphy.utils.tρθ2rf(tρ, θ, rfmax)
            pulse.gr = mrphy.utils.s2g(mrphy.utils.ts2s(tsl, smax), pulse.dt)

            loss_err, loss_pen = fn_loss(cube, pulse)
            loss = loss_err + eta*loss_pen
            loss.backward()
            return loss

        print('rf-loop: ', niter_rf)
        for _ in range(niter_rf):
            opt_rf.step(closure)

            loss_err, loss_pen = fn_loss(cube, pulse)
            loss = logger(i, t0, loss_err, loss_pen)

            loss_hist[i*(niter_gr+niter_rf)+log_ind] = loss.item()
            log_ind += 1

        print('gr-loop: ', niter_gr)
        for _ in range(niter_gr):
            opt_sl.step(closure)

            loss_err, loss_pen = fn_loss(cube, pulse)
            loss = logger(i, t0, loss_err, loss_pen)

            loss_hist[i*(niter_gr+niter_rf)+log_ind] = loss.item()
            log_ind += 1

    print('\n== Results: ==')
    print(log_col)
    logger(i, t0, loss_err, loss_pen)

    optInfos = {'loss_hist': loss_hist}
    return pulse, optInfos
