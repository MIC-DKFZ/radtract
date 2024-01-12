<!--
Copyright © 2023 German Cancer Research Center (DKFZ), Division of Medical Image Computing

SPDX-License-Identifier: Apache-2.0
-->

# Radiomic Tractometry (RadTract)

Copyright © German Cancer Research Center (DKFZ), [Division of Medical Image Computing (MIC)](https://www.dkfz.de/en/mic/index.php). Please make sure that your usage of this code is in compliance with the code [license](https://github.com/MIC-DKFZ/radtract/blob/master/LICENSE).

If you use RadTract, please cite our [paper](https://www.nature.com/articles/s41467-023-44591-3): `Neher, P., Hirjak, D. & Maier-Hein, K. Radiomic tractometry reveals tract-specific imaging biomarkers in white matter. Nat Commun 15, 303 (2024). https://doi.org/10.1038/s41467-023-44591-3`


## Overview

RadTract is a python implementation of radiomic tractometry or "Tractomics". It is designed for tract-specific microstructural analysis of the brain’s white matter using diffusion MRI. It enhances traditional tractometry, which often misses valuable information due to its reliance on bare summary statistics and scalar values. RadTract incorporates radiomics, a method that analyzes a multitude of quantitative image features beyond visual perception, into tractometry. This integration allows for improved predictive modeling while maintaining the localization capability of tractometry.

RadTract has demonstrated its effectiveness in diagnosing disease subgroups across various datasets and estimating demographic and clinical parameters in multiple clinical populations. It holds the potential to pioneer a new generation of tract-specific imaging biomarkers, benefiting a wide range of applications from basic neuroscience to medical research.

For details about the approach, please refer to our [paper](https://www.nature.com/articles/s41467-023-44591-3): `Neher, P., Hirjak, D. & Maier-Hein, K. Radiomic tractometry reveals tract-specific imaging biomarkers in white matter. Nat Commun 15, 303 (2024). https://doi.org/10.1038/s41467-023-44591-3`. An overview of the method is shown in Figure 1.

![](resources/radtract_overview.png)_Figure 1: Illustration of the complete RadTract process. The points of a statically resampled tract (a) can be seen as samples of partly overlapping classes that are not linearly separable. We are aiming at finding the hyperplanes, superimposed as white lines on the tract in (a), that optimally separate the classes with the smallest amount of errors. This task can be solved using large-margin classifiers such as SVMs. This enables us to create parcellations directly in voxel-space (b) that do not suffer from projection-induced misassignments, as is the case in the centerline-based approach (d). For visualization purposes, the tract parcellation in voxel-space is projected back on the original streamlines (e). The proposed tract parcellation in voxel-space (b) is used to calculate a multitude of radiomics features per parcel, visualized in (c). Exemplary feature classes and image filters available when using [pyradiomics](https://pyradiomics.readthedocs.io/en/latest/) as calculation engine are listed in (f). RadTract currently supports [MIRP](https://github.com/oncoray/mirp) as an alternative engine for calculating radiomics features.

## Installation

### Requirements

- No specific hardware requirements. A state-of-the-art desktop computer should be sufficient.
- Tested on Ubuntu 22.04 but should run on other systems as well.
- Tested with Python 3.8 and higher
- Numpy should be installed prior to the RadTract setup (pyradiomics requirement), all other dependencies will be installed automatically. 
- Should the pyradiomics setup fail with a missing numpy error despite installed numpy, see section "Pyradiomics installation issues" below.
- It is recommended to use a virtual environment for the installation. 

See `.gitlab-ci.yml` for the currently tested configurations.

### Installation

Installation via anaconda is not supported currently!

1. virtual environment
   - Create a virtual environment: `python -m venv myvenv`
   - Activate the virtual environment: `source myvenv/bin/activate`
2. Installation
   - Install from source: navigate to the root directory of RadTract and run `pip install .`
   - Install from PyPI: run `pip install radtract`

Installation should complete within a few seconds.

### Pyradiomics installation issues

If the pyradiomics installation fails with a missing numpy error despite numpy being installed, a workaround is to install pyradimics directly from source:

1. Checkout the pyradiomics repo: `git clone git://github.com/Radiomics/pyradiomics`
2. Activate your virtual environment (if you use one): `source myvenv/bin/activate`
3. Navigate to the pyradiomics source and install from there: `pip install .`
4. Then run pip `pip install radtract` again.

## Examples

A complete pipeline example can be found in [example.ipynb](https://github.com/MIC-DKFZ/radtract/blob/main/example.ipynb). 

Further examples can be found in the RadTract test script `tests\test_radtract.py`. Test data is included in `tests\test_data`.


### Expected runtimes

RadTract parcellation and feature calculation should complete within a couple of minutes on a standard desktop computer.
