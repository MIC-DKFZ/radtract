<!--
Copyright © 2023 German Cancer Research Center (DKFZ), Division of Medical Image Computing

SPDX-License-Identifier: Apache-2.0
-->

# Radiomic Tractometry (RadTract)

![Build Status](https://git.dkfz.de/mic/personal/group6/neuro/radtract/badges/main/pipeline.svg?ignore_skipped=true)
![Coverage](https://git.dkfz.de/mic/personal/group6/neuro/radtract/badges/main/coverage.svg)

Copyright © German Cancer Research Center (DKFZ), [Division of Medical Image Computing (MIC)](https://www.dkfz.de/en/mic/index.php). Please make sure that your usage of this code is in compliance with the code [license](https://github.com/MIC-DKFZ/radtract/blob/master/LICENSE).

## Overview

Python package for radiomic tractometry (RadTract), a method for the extraction of radiomics features along white matter tracts.

## Installation

### Requirements

- No specific hardware requirements. A state-of-the-art desktop computer should be sufficient.
- Tested on Ubuntu 22.04 but should run on other systems as well.
- Tested with Python 3.8 and higher
- Numpy should be installed prior to the RadTract setup, all other dependencies will be installed automatically. 
- It is recommended to use a virtual environment for the installation. 

See `.gitlab-ci.yml` for the currently tested configurations.

### Installation

- Install from source: navigate to the root directory of RadTract and run `pip install .`
- Install from PyPI: run `pip install radtract`

Installation should complete within a few seconds.

## Example usage

See `tests\test_radtract.py` for examples of how to use RadTract. Test data is included in `tests\test_data`.

### Expected runtimes

All tests should complete within a couple of minutes on a standard desktop computer.