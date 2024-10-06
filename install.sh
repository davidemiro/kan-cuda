#!/bin/bash

pip install "pybind11>=2.12"
pip install "numpy<2"

cd cpp

pip install .

#python cuda/setup.py