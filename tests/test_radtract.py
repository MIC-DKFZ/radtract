# Copyright © 2023 German Cancer Research Center (DKFZ), Division of Medical Image Computing
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
from radtract import parcellation, features, tractdensity
import nibabel as nib
import os


def load_data():
    data_folder = os.path.dirname(__file__) + '/test_data/'
    beginnings = nib.load(data_folder + 'test_tract_b.nii.gz')
    envelope = nib.load(data_folder + 'test_tract_envelope.nii.gz')
    centerline_parcellation = nib.load(data_folder + 'centerline_parcellation.nii.gz')
    hyperplane_parcellation = nib.load(data_folder + 'hyperplane_parcellation.nii.gz')
    sl_file = nib.streamlines.load(data_folder + 'test_tract.trk')
    streamlines = sl_file.streamlines
    return streamlines, beginnings, envelope, centerline_parcellation, hyperplane_parcellation


def get_results_path():
    p = os.path.dirname(__file__) + '/test_results/'
    os.makedirs(p, exist_ok=True)
    return p


def test_envelope():
    streamlines, beginnings, envelope, _, _ = load_data()

    new_envelope = tractdensity.tract_envelope(streamlines, reference_image=beginnings, out_image_filename=get_results_path() + 'test_tract_envelope.nii.gz')

    assert np.equal(new_envelope.get_fdata(), envelope.get_fdata()).all(), 'envelope test 1 failed'
    new_envelope = nib.load(get_results_path() + 'test_tract_envelope.nii.gz')
    assert np.equal(new_envelope.get_fdata(), envelope.get_fdata()).all(), 'envelope test 2 failed'


def test_centerline_parcellation():
    streamlines, beginnings, envelope, centerline_parcellation, _ = load_data()
    new_parcellation, _, _, _ = parcellation.parcellate_tract(streamlines=streamlines, parcellation_type='centerline', binary_envelope=envelope, num_parcels=17, start_region=beginnings, out_parcellation_filename=get_results_path() + 'test_tract_parcellation-centerline.nii.gz')

    assert np.equal(new_parcellation.get_fdata(), centerline_parcellation.get_fdata()).all(), 'centerline parcellation test 1 failed'
    new_parcellation = nib.load(get_results_path() + 'test_tract_parcellation-centerline.nii.gz')
    assert np.equal(new_parcellation.get_fdata(), centerline_parcellation.get_fdata()).all(), 'centerline parcellation test 2 failed'


def test_hyperplane_parcellation():
    streamlines, beginnings, envelope, _, hyperplane_parcellation = load_data()
    new_parcellation, _, _, _ = parcellation.parcellate_tract(streamlines=streamlines, parcellation_type='hyperplane', binary_envelope=envelope, num_parcels=17, start_region=beginnings, out_parcellation_filename=get_results_path() + 'test_tract_parcellation-hyperplane.nii.gz')

    assert np.equal(new_parcellation.get_fdata(), hyperplane_parcellation.get_fdata()).all(), 'hyperplane parcellation test 1 failed'
    new_parcellation = nib.load(get_results_path() + 'test_tract_parcellation-hyperplane.nii.gz')
    assert np.equal(new_parcellation.get_fdata(), hyperplane_parcellation.get_fdata()).all(), 'hyperplane parcellation test 2 failed'


def test_num_parcel_esimation():
    streamlines, beginnings, _, _, _ = load_data()

    num = parcellation.estimate_num_parcels(streamlines=streamlines,
                                            reference_image=beginnings,
                                            num_voxels=5)
    assert num == 18, 'num parcel estimation test 1 failed'


def test_hyperplane_features():
    data_folder = os.path.dirname(__file__) + '/test_data/'
    features_df = pd.read_csv(data_folder + 'hyperplane_features.csv')
    new_features = features.calc_radiomics(parcellation_file_name=data_folder + 'hyperplane_parcellation.nii.gz',
                                           parameter_map_file_name=data_folder + 'test_map.nii.gz',
                                           out_csv_file=get_results_path() + 'hyperplane_features.csv',
                                           num_parcels=17,
                                           remove_paths=True
                                           )

    new_features_df = pd.DataFrame(new_features)
    pd.testing.assert_frame_equal(new_features_df, features_df, check_dtype=False)


def test_centerline_features():
    data_folder = os.path.dirname(__file__) + '/test_data/'
    features_df = pd.read_csv(data_folder + 'centerline_features.csv')
    new_features = features.calc_radiomics(parcellation_file_name=data_folder + 'centerline_parcellation.nii.gz',
                                           parameter_map_file_name=data_folder + 'test_map.nii.gz',
                                           out_csv_file=get_results_path() + 'centerline_features.csv',
                                           num_parcels=17,
                                           remove_paths=True
                                           )

    new_features_df = pd.DataFrame(new_features)
    pd.testing.assert_frame_equal(new_features_df, features_df, check_dtype=False)
