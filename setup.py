""" Setup script for mlp package. """

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

setup(
    name="mmdata",
    author="Ramona Comanescu",
    description=("MLP COursework University of Edinburgh "
                 "School of Informatics"),
    url="https://github.com/comRamona/Honours-LDA",
    packages=['mmdata']
)

