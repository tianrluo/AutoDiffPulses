# Auto-Differentiation Based MRI Pulse Design

Reference implementation of:\
[Joint Design of RF and Gradient Waveforms via Auto-Differentiation for 3D Tailored Exitation in MRI](https://ieeexplore.ieee.org/document/9439482)\
(arXiv: [https://arxiv.org/abs/2008.10594](https://arxiv.org/abs/2008.10594))

cite as:

```bib
@article{luo2021joint,
  author={Luo, Tianrui and Noll, Douglas C. and Fessler, Jeffrey A. and Nielsen, Jon-Fredrik},
  journal={IEEE Transactions on Medical Imaging},
  title={Joint Design of RF and gradient waveforms via auto-differentiation for 3D tailored excitation in MRI},
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2021.3083104}}
```

For the `interpT` feature, consider citing:
```bib
@inproceedings{luo2021MultiScale,
  title={Multi-scale Accelerated Auto-differentiable Bloch-simulation based joint design of excitation RF and gradient waveforms},
  booktitle={ISMRM},
  pages={3958},
  author={Tianrui Luo and Douglas C. Noll and Jeffrey A. Fessler and Jon-Fredrik Nielsen},
  year={2021}
}
```

## Steady-state sequence optimization

Besides regular tailored-excitation design, the pulse can be optimized against
the steady-state magnetization of an SPGR
sequence. Use `adpulses.optimizers.arctanLBFGS_spgr` (Python) or
`adpulses.opt.arctanAD_spgr` (MATLAB) — the steady-state counterparts of
`arctanLBFGS` / `adpulses.opt.arctanAD`. `demo/demo_ss.m` gives a worked
comparison of a regular design against a steady-state design.

This feature accompanies a technical note currently under review at
*Magnetic Resonance in Medicine*; a citation will be added once available.

## System Requirements:
- Python `≥3.8`.

Tested on Ubuntu 18.04 / 20.04; other operating systems are untested but not
explicitly excluded.

## General comments

`setup_AutoDiffPulses.m` does the configurations for Matlab.\
For the python part, in your command line, navigate to the repo's root directory, type:

```sh
pip install -e .
```

Demos are provided in `./demo`. In particular, `demo/demo_ss.m` uses the
included example dataset (`demo/IniVars.mat`) to compare a regular design
against a steady-state design under steady-state Bloch simulation.

## Dependencies

This work requires Python (`≥v3.8`) and PyTorch (`≥v1.6`).

### Python
`pip install -e .` (see above) installs all Python dependencies, including the
steady-state-enabled MRphy.py (pinned in `setup.py`):

- `mrphy`: [tianrluo/MRphy.py](https://github.com/tianrluo/MRphy.py.git) (`≥v0.2.1`) — with steady-state (SPGR) Bloch simulation support (`mrphy.steady_state.spgr`).

Other Python dependencies (`numpy`, `scipy`, `torch`) are installed automatically.

### MATLAB
`setup_AutoDiffPulses.m` bootstraps the MATLAB dependencies automatically: it
clones the packages below into `./matlab_deps` (git-ignored) at the pinned
commits, compiles the C Bloch simulator (`mex blochcim.c`), and adds them to the
path. This requires `git` on the system `PATH`. To force a fresh checkout,
delete `./matlab_deps` and re-run the script.

- `+mrphy`: [tianrluo/MRphy.mat](https://github.com/tianrluo/MRphy.mat.git) — the MATLAB counterpart, extended with steady-state Bloch simulation support.
- `+attr`: [fmrilab/attr.mat](https://github.com/fmrilab/attr.mat).

MATLAB must also be configured (via `pyenv`) to use a Python environment that
has the `mrphy` installation described above.
