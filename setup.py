#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import platform

import os
from setuptools import setup, find_packages

__location__ = os.path.join(os.getcwd(), os.path.dirname(inspect.getfile(inspect.currentframe())))

py_major_version, py_minor_version, _ = (int(v.rstrip('+')) for v in platform.python_version_tuple())


def get_install_requirements(path):
    content = open(os.path.join(__location__, path)).read()
    requires = [req for req in content.split('\\n') if req != '']
    if py_major_version == 2 or (py_major_version == 3 and py_minor_version < 4):
        requires.append('pathlib')
    return requires


setup(
    name='titachric',
    author='qwert',
    version='1.0',
    packages=find_packages(),
    package_data={
        'config': ['*.cfg', '*.ini']
    },
    install_requires=get_install_requirements('requirements.txt'),
)
