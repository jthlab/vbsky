from setuptools import find_packages, setup

setup(
    name="vbsky",
    author="Caleb Ki and Jonathan Terhorst",
    author_email="jonth@umich.edu",
    description="Variational inference for Bayesian phylodynamic models",
    install_requires=[
        "biopython>=1.79",
        "jax>=0.2.25",
        "jaxlib>=0.1.74",
        "numpy>=1.17.0",
        "scipy>=1.5.0",
        "matplotlib>=3.0.0",
        "msprime>=1.1.0",
        "tqdm>=4.64.0",
        "sklearn>=0.0",
        "networkx>2.7.1",
    ],
    packages=find_packages(),
)
