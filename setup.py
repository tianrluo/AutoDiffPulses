from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    'torch>=1.6',
    'numpy',
    'scipy',
    # MRphy.py with steady-state (SPGR) Bloch simulation support
    # (`mrphy.steady_state.spgr`), first released in v0.2.1.
    'mrphy @ git+https://github.com/tianrluo/MRphy.py.git@v0.2.1',
]

with open("README.md", "r") as h:
    long_description = h.read()

setup(
    name="adpulses",
    version="0.3.0",
    author="Tianrui Luo, Yongli He",
    author_email="tianrluo@umich.edu, yonglihe@umich.edu",
    description="A Pytorch based tool for MR pulse design, with support for steady-state sequence optimization",
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
