# Auto-Differentiation Based MRI Pulse Design

Reference implementation of:\
[Joint Design of RF and Gradient Waveforms via Auto-Differentiation for 3D Tailored Exitation in MRI](https://arxiv.org/abs/2008.10594)

<!--\ doi: []())-->

cite as:

```bib
@misc{luo2020joint,
  title={Joint Design of RF and gradient waveforms via auto-differentiation for 3D tailored excitation in MRI},
  author={Tianrui Luo and Douglas C. Noll and Jeffrey A. Fessler and Jon-Fredrik Nielsen},
  year={2020},
  eprint={2008.10594},
  archivePrefix={arXiv},
  primaryClass={eess.IV},
  url={https://arxiv.org/abs/2008.10594}
}
```

For the `interpT` feature, consider citing:
```bib
@inproceedings{luo2021MultiScale,
  title={Multi-scale Accelerated Auto-differentiable Bloch-simulation based joint design of excitation RF and gradient waveforms},
  booktitle={ISMRM},
  pages={0000},
  author={Tianrui Luo and Douglas C. Noll and Jeffrey A. Fessler and Jon-Fredrik Nielsen},
  year={2021}
}
```

## System Requirements:
- Ubuntu 18.04, Python 3

The implementation may fail with other configurations.

## General comments

`setup_AutoDiffPulses.m` does the configurations for Matlab.\
For the python part, in your command line, navigate to the repo's root directory, type:

```sh
pip install .
```

Demos are provided in `./demo`.

This repo has included binary test data files for basic accessibility in certain regions.\
Future binary data files will be added to: <https://drive.google.com/drive/folders/1EyKhA_d74OC4KADMuTd1kRTEMoVqWdIY>.

## Dependencies

This work requries Python (`≥v3.5`), PyTorch (`≥v1.3`) with CUDA.

- `MRphy`: Python, Github [link](https://github.com/tianrluo/MRphy.py) (`≥v0.1.5`).
- `+mrphy`: Matlab, Github [link](https://github.com/tianrluo/MRphy.mat).
- `+attr`: Matlab, Github [link](https://github.com/fmrilab/attr.mat).

Other Python dependencies include:\
`scipy`, `numpy`, `PyTorch`.
