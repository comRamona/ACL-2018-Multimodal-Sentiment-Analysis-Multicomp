""" Setup script for mlp package. """

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

setup(
    name="mmdata",
    author="G25",
    description=("MLP Coursework University of Edinburgh "
                 "School of Informatics"),
    url="https://github.com/comRamona/MLP_Multimodal",
    packages=['mmdata']
)

