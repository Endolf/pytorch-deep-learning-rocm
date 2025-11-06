#!/bin/bash

pip freeze > pytorch-rocm-requirements.txt
python -m venv venv
source venv/bin/activate
pip install -r pytorch-rocm-requirements.txt
rm pytorch-rocm-requirements.txt
pip install -r requirements.txt
