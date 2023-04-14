# Copyright © 2023 German Cancer Research Center (DKFZ), Division of Medical Image Computing
#
# SPDX-License-Identifier: Apache-2.0

from radiomics import featureextractor
import pandas as pd
import os
import nibabel as nib
import numpy as np
import argparse
import joblib
from scipy.ndimage import map_coordinates
from nibabel.affines import apply_affine


def map_check(map: str,
              extractor_settings: dict = None):
    '''
    Check if the map is normalized to a range of 0-1. If not, print a warning.
    :param map: Path to the map
    :param extractor_settings: Settings for the pyradiomics feature extractor
    '''
    img = nib.load(map)
    data = img.get_fdata()
    data = np.nan_to_num(data)
    range = np.max(data) - np.min(data)
    robust_range = np.percentile(data, 99) - np.percentile(data, 1)

    print('Checking map: ', map)

    if extractor_settings is not None and extractor_settings['normalize'] == 'true':
        print('Map is being normalized. Please make sure that your bin with is set accordingly.')
        return

    num_bins = range / extractor_settings['binWidth']
    print('Map range: ', range)
    print('Robust map range (99 percentile): ', robust_range)
    print('Bin width: ', extractor_settings['binWidth'])
    print('Number of bins: ', num_bins)

    if num_bins < 16 or num_bins > 128:
        print('Number of bins is not in the recommended range of 16-128.')
        print('Consider normalizing the map or adjust the binWidth in the pyrad.yaml parameter file.')
        print('Suggestion for bin width (50 bins): ' + str(range / 50) + '.')
    else:
        print('Map range is OK: ', range)


def batch_calc_radiomics(parcellation_file_names: list,
                         parameter_map_file_names: list,
                         out_csv_files: list):
    """
    Calculate radiomics features for a list of parcellations and parameter maps
    :param parcellation_file_names: list of parcellation files
    :param parameter_map_file_names: list of parameter maps
    :param out_csv_files: save features to these files
    """
    assert len(parcellation_file_names) == len(parameter_map_file_names)
    assert len(parcellation_file_names) == len(out_csv_files)
    for i in range(len(parcellation_file_names)):
        calc_radiomics(parcellation_file_names[i], parameter_map_file_names[i], out_csv_files[i])


def calc_radiomics(parcellation_file_name: str,
                   parameter_map_file_name: str,
                   out_csv_file: str,
                   num_parcels: int = None,
                   features: dict = None,
                   pyrad_params=None,
                   remove_paths: bool = False):
    """
    Calculate radiomics features for a parcellation and a matching parameter map using pyradiomcs
    :param parcellation_file_name:
    :param parameter_map_file_name:
    :param out_csv_file: save features to this file
    :param num_parcels: optional to ensure that all labels are present in the parcellation
    :param features: append new features to this dict, if none, create empty dict
    :param pyrad_params: if none, use default parameter file (designed for FA maps)
    :param remove_paths: remove paths from feature file
    :return: features dict
    """
    if pyrad_params is not None:
        extractor = featureextractor.RadiomicsFeatureExtractor(pyrad_params)
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor(os.path.dirname(__file__) + '/pyrad.yaml')
    print('Pyradiomics settings:', extractor.settings)
    print('Enabled image types:', extractor.enabledImagetypes)
    print('Enables features:', extractor.enabledFeatures)
    map_check(parameter_map_file_name, extractor.settings)

    if features is None:
        features = dict()
        features['map'] = []
        features['parcellation'] = []
        features['label'] = []

    parcellation_data = nib.load(parcellation_file_name).get_fdata().astype('int64')

    labels = np.unique(parcellation_data)
    print('Found labels', labels)
    if num_parcels is not None:
        assert num_parcels == len(labels)-1, 'Number of parcels does not match number of labels in ' + parcellation_file_name
    for label in labels:
        label = int(label)
        if label == 0:
            continue
        print('pyradiomics processing label ' + str(label))
        feature_vector = extractor.execute(imageFilepath=parameter_map_file_name, maskFilepath=parcellation_file_name, label=label)
        print('pyradiomics formatting results ...')
        if remove_paths:
            features['map'].append(os.path.basename(parameter_map_file_name))
            features['parcellation'].append(os.path.basename(parcellation_file_name))
        else:
            features['map'].append(parameter_map_file_name)
            features['parcellation'].append(parcellation_file_name)
        features['label'].append(label)
        for featureName in feature_vector.keys():
            try:
                val = float(feature_vector[featureName])
                if featureName not in features.keys():
                    features[featureName] = []
                features[featureName].append(val)
            except Exception:
                pass

    if out_csv_file is not None:
        if not out_csv_file.endswith('.csv'):
            out_csv_file += '.csv'
        print('pyradiomics saving results ...')
        features = pd.DataFrame(features)
        features.to_csv(out_csv_file, index=False)
    print('pyradiomics finished processing')

    return features


def calc_tractometry(point_label_file_name: str,
                     parameter_map_file_name: str,
                     out_csv_file: str,
                     num_parcels: int = None):
    """
    Calculate tractometry features using points and corresponding parcel labels
    :param point_label_file_name:
    :param parameter_map_file_name:
    :param out_csv_file:
    :param num_parcels: optional to ensure that all labels are present in the parcellation
    :return:
    """
    streamline_point_parcels = joblib.load(point_label_file_name)
    map = nib.load(parameter_map_file_name)
    map_data = map.get_fdata()
    print('Calculating tractometry ...')
    points = apply_affine(np.linalg.inv(map.affine), streamline_point_parcels['points'])
    values = map_coordinates(map_data, points.T, order=1)
    vals_per_parcel = dict()
    points_per_parcel = dict()
    if num_parcels is not None:
        assert num_parcels == np.unique(streamline_point_parcels['parcels']).shape[0], 'Number of parcels does not match number of labels in ' + point_label_file_name
    for parcel, val, p in zip(streamline_point_parcels['parcels'], values, streamline_point_parcels['points']):
        if parcel not in vals_per_parcel.keys():
            vals_per_parcel[parcel] = []
        vals_per_parcel[parcel].append(val)
        if parcel not in points_per_parcel.keys():
            points_per_parcel[parcel] = []
        points_per_parcel[parcel].append(p)

    # for parcel in points_per_parcel.keys():
    #     text = ''
    #     text += '<?xml version="1.0" encoding="UTF-8"?><point_set_file><file_version>0.1</file_version><point_set><time_series><time_series_id>0</time_series_id><Geometry3D ImageGeometry="false" FrameOfReferenceID="0">'
    #     text += '<IndexToWorld type="Matrix3x3" m_0_0="1" m_0_1="0" m_0_2="0" m_1_0="0" m_1_1="1" m_1_2="0" m_2_0="0" m_2_1="0" m_2_2="1"/><Offset type="Vector3D" x="0" y="0" z="0"/><Bounds>'
    #     text += '<Min type="Vector3D" x="89.933372497558594" y="98.688766479492188" z="-0.39603650569915771"/><Max type="Vector3D" x="127.03989410400391" y="165.80229187011719" z="141.04673767089844"/></Bounds></Geometry3D>'
    #     i = 0
    #     for p in points_per_parcel[parcel]:
    #         text += '<point><id>' + str(i) + '</id><specification>0</specification>'
    #         text += '<x>' + str(p[0]) + '</x>'
    #         text += '<y>' + str(p[1]) + '</y>'
    #         text += '<z>' + str(p[2]) + '</z>'
    #         text += '</point>'
    #         i += 1
    #     text += '</time_series></point_set></point_set_file>'
    #     with open(os.path.join(os.path.dirname(point_label_file_name), 'tractometry_centerline_points_' + str(parcel) + '.mps'), 'w') as f:
    #         f.write(text)

    features = dict()
    features['map'] = []
    features['parcellation'] = []
    features['label'] = []
    features['tractometry-mean'] = []
    for parcel in sorted(vals_per_parcel.keys()):
        features['map'].append(parameter_map_file_name)
        features['parcellation'].append(point_label_file_name)
        features['label'].append(parcel)
        features['tractometry-mean'].append(np.nanmean(vals_per_parcel[parcel]))

    if out_csv_file is not None:
        if not out_csv_file.endswith('.csv'):
            out_csv_file += '.csv'
        print('tractometry saving results ...')
        features = pd.DataFrame(features)
        features.to_csv(out_csv_file, index=False)
    print('tractometry finished processing')

    return features


def load_features(feature_file_names: list, feature_filter: str = None, expected_parcels: int = None):
    """
    Load features from files
    :param feature_file_names: list of feature file names
    :param feature_filter: only use columns containing this string
    :return: pandas dataframe of selected features, one row per file
    """

    out_df = None
    for feature_file_name in feature_file_names:
        feature_names = []

        feature_df = pd.read_csv(feature_file_name)

        parcels = feature_df['label'].tolist()
        if expected_parcels is not None and len(parcels) != expected_parcels:
            raise Exception('ERROR: Feature file does not contain ' + str(expected_parcels) + ' parcels:', feature_file_name)

        feature_df = feature_df.drop(columns=['map', 'parcellation', 'label'])

        col_list = [col for col in feature_df if not col.startswith('diagnostic')]

        if feature_filter not in ['all', 'tractometry', None]:
            col_list = [col for col in col_list if col.__contains__(feature_filter)]
        if len(col_list) == 0:
            print('ERROR: No features left after filtering with \'' + feature_filter + '\'', feature_file_name)
            exit(1)

        # print('shape', len(feature_df.loc[:, feature_df.columns.str.contains('shape')].columns.tolist()))
        # print('firstorder', len(feature_df.loc[:, feature_df.columns.str.contains('firstorder')].columns.tolist()) // 12)
        # print('glcm', len(feature_df.loc[:, feature_df.columns.str.contains('glcm')].columns.tolist()) // 12)
        # print('glrlm', len(feature_df.loc[:, feature_df.columns.str.contains('glrlm')].columns.tolist()) // 12)
        # print('glszm', len(feature_df.loc[:, feature_df.columns.str.contains('glszm')].columns.tolist()) // 12)
        # print('gldm', len(feature_df.loc[:, feature_df.columns.str.contains('gldm')].columns.tolist()) // 12)
        # print('ngtdm', len(feature_df.loc[:, feature_df.columns.str.contains('ngtdm')].columns.tolist()) // 12)

        feature_df = feature_df[col_list]

        for feature in feature_df:
            for parcel in parcels:
                feature_names.append(str(parcel) + '_' + str(feature))

        features = feature_df.to_numpy()
        features = features.flatten(order='F')

        if out_df is None:
            out_df = pd.DataFrame([features], columns=feature_names)
        else:
            out_df = pd.concat([out_df, pd.DataFrame([features], columns=feature_names)], axis=0)

    return out_df


def main():
    parser = argparse.ArgumentParser(description='RadTract Feature Calculation')
    parser.add_argument('--parcellation', type=str, help='Input parcellation file')
    parser.add_argument('--map', type=str, help='Parameter map file (e.g. fractional anisotropy)')
    parser.add_argument('--output', type=str, help='Output feature file (.csv)')
    args = parser.parse_args()

    calc_radiomics(parcellation_file_name=args.parcellation,
                   parameter_map_file_name=args.map,
                   out_csv_file=args.output
                   )


if __name__ == '__main__':
    main()
