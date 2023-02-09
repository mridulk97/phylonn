from setuptools import setup, find_packages

setup(
    name='phylonn',
    version='0.0.1',
    description='Discovering Novel Biological Traits From Images Using Phylogeny-Guided Neural Networks',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
