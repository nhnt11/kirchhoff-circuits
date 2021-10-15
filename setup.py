# @Author:  Felix Kramer
# @Date:   2021-05-23T23:25:28+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-09-05T00:06:04+02:00
# @License: MIT

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kirchhoff", 
    version="0.1.8",
    author="felixk1990",
    author_email="felixuwekramer@protonmail.com",
    description="Collection of routines for creation and manipulation of Kirchhoff circuits based on resistor-only networks, together with 2D/3D spatial embeddings. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/felixk1990/kirchhoff-circuit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
