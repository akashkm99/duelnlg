"""Defines setuptools metadata."""

import setuptools
from setuptools.command.install import install

import subprocess
import os

with open("README.md", "r") as readme_file:
    LONG_DESCRIPTION = readme_file.read()

with open("requirements.txt") as req_file:
    install_requires = req_file.read().splitlines()

install_requires += ["bleurt@file://localhost/" + os.getcwd() + "/duelnlg/direct_eval/metrics/bleurt#egg=bleurt"]


setuptools.setup(
    name="duelnlg",
    version="0.0.1",
    author="Akash Kumar M",
    author_email="makashkumar99@gmail.com",
    description="Algorithms discribed in ACL 2022 Active Evaluation Paper",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/akashkm99/duelnlg",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
