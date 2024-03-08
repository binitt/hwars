#!/bin/bash

rm -rf dist/*.whl build &&
python setup.py bdist_wheel &&
rm -rf build
