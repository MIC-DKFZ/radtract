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
import SimpleITK as sitk
from cmdint.Utils import ProgressBar


def map_check(data: np.ndarray,
              extractor_settings: dict = None):
    '''
    :param map: Path to the map
    :param extractor_settings: Settings for the pyradiomics feature extractor
    '''
    data = np.nan_to_num(data)
    range = np.max(data) - np.min(data)
    robust_range = np.percentile(data, 99) - np.percentile(data, 1)

    print('Checking map ...')

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
        print('Suggestion for bin width (50 bins), based on the map range: ' + str(range / 50) + '.')
        print('Suggestion for bin width (50 bins), based on the robust map range: ' + str(robust_range / 50) + '.')
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

    if features is None:
        features = dict()
        features['map'] = []
        features['parcellation'] = []
        features['parcel'] = []

    map_sitk, parc_sitk = extractor.loadImage(parameter_map_file_name, parcellation_file_name)
    parcellation_data = sitk.GetArrayFromImage(parc_sitk)

    map_check(sitk.GetArrayFromImage(map_sitk), extractor.settings)

    labels = np.unique(parcellation_data)
    print('Found labels', labels)
    if num_parcels is not None:
        assert num_parcels == len(
            labels) - 1, 'Number of parcels does not match number of labels in ' + parcellation_file_name

    # get global features
    print('pyradiomics generating global tract features')
    parcellation_data = sitk.GetArrayFromImage(parc_sitk)
    parcellation_data[parcellation_data > 0] = 1
    env_sitk = sitk.GetImageFromArray(parcellation_data)
    env_sitk.CopyInformation(parc_sitk)
    feature_vector = extractor.execute(imageFilepath=map_sitk, maskFilepath=env_sitk)
    print('pyradiomics formatting results ...')
    if remove_paths:
        features['map'].append(os.path.basename(parameter_map_file_name))
        features['parcellation'].append(os.path.basename(parcellation_file_name))
    else:
        features['map'].append(parameter_map_file_name)
        features['parcellation'].append(parcellation_file_name)
    features['parcel'].append('GLOBAL')
    for featureName in feature_vector.keys():
        try:
            val = float(feature_vector[featureName])
            if featureName not in features.keys():
                features[featureName] = []
            features[featureName].append(val)
        except Exception:
            pass

    # features per label
    for label in labels:
        label = int(label)
        if label == 0:
            continue
        print('pyradiomics processing label ' + str(label))
        feature_vector = extractor.execute(imageFilepath=map_sitk, maskFilepath=parc_sitk, label=label)
        print('pyradiomics formatting results ...')
        if remove_paths:
            features['map'].append(os.path.basename(parameter_map_file_name))
            features['parcellation'].append(os.path.basename(parcellation_file_name))
        else:
            features['map'].append(parameter_map_file_name)
            features['parcellation'].append(parcellation_file_name)
        features['parcel'].append('P' + str(label))
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
        features_df = pd.DataFrame(features)
        features_df.to_csv(out_csv_file, index=False)
    print('pyradiomics finished processing')

    return features


def calc_tractometry(point_label_file_name: str,
                     parameter_map_file_name: str,
                     out_csv_file: str,
                     features: dict = None,
                     num_parcels: int = None):
    """
    Calculate tractometry features using points and corresponding parcel labels
    :param point_label_file_name:
    :param parameter_map_file_name:
    :param out_csv_file:
    :param features: append new features to this dict, if none, create empty dict
    :param num_parcels: optional to ensure that all labels are present in the parcellation
    :return:
    """

    if features is None:
        features = dict()
        features['map'] = []
        features['parcellation'] = []
        features['parcel'] = []
        features['tractometry-mean'] = []

    streamline_point_parcels = joblib.load(point_label_file_name)
    map = nib.load(parameter_map_file_name)
    map_data = map.get_fdata()
    print('Calculating tractometry ...')
    points = apply_affine(np.linalg.inv(map.affine), streamline_point_parcels['points'])
    values = map_coordinates(map_data, points.T, order=1)
    vals_per_parcel = dict()
    points_per_parcel = dict()
    if num_parcels is not None:
        assert num_parcels == np.unique(streamline_point_parcels['parcels']).shape[
            0], 'Number of parcels does not match number of labels in ' + point_label_file_name
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
    #     with open(os.path.join(os.path.dirname(point_label_file_name), 'tractometry_points_' + str(parcel) + '.mps'), 'w') as f:
    #         f.write(text)

    for parcel in sorted(vals_per_parcel.keys()):
        features['map'].append(parameter_map_file_name)
        features['parcellation'].append(point_label_file_name)
        features['parcel'].append('P' + str(parcel))
        features['tractometry-mean'].append(np.nanmean(vals_per_parcel[parcel]))

    if out_csv_file is not None:
        if not out_csv_file.endswith('.csv'):
            out_csv_file += '.csv'
        print('tractometry saving results ...')
        features_df = pd.DataFrame(features)
        features_df.to_csv(out_csv_file, index=False)
    print('tractometry finished processing')

    return features


def load_features(feature_file_names: list, select=[], drop=[], expected_parcels: int = None, verbose: bool = False,
                  remove_map_substrings=[]):
    """
    Load features from files
    :param feature_file_names: list of feature file names
    :param select: select features containing these substrings
    :param drop: drop features containing these substrings
    :param expected_parcels: check that each feature file contains this number of parcels
    :param verbose: print stats
    :param remove_map_substrings: remove these substrings from map names (map names are appended to feature names in flattened feature matrix), paths are removed by default
    :return: pandas dataframe of selected features, one row per file
    """

    if verbose:
        print('Loading features ...')
        print('Select:', select)
        print('Drop:', drop)

    out_df = None
    bar = ProgressBar(len(feature_file_names))
    for feature_file_name in feature_file_names:
        feature_names = []

        feature_df = pd.read_csv(feature_file_name)
        # check for nan values
        if feature_df.isna().values.any():
            print('Feature file contains NaN values:', feature_file_name)

        parcels = feature_df['parcel'].tolist()
        maps = feature_df['map'].tolist()

        c = 0
        for map in maps:
            for substring in remove_map_substrings:
                map = map.replace(substring, '')
            map = map.replace('.nii.gz', '')
            map = map.split('/')[-1]
            maps[c] = map
            parcels[c] = str(parcels[c]) + '_' + map
            c += 1

        if expected_parcels is not None and len(
                parcels) != expected_parcels + 1:  # +1 accounts for tract-global features
            raise Exception('ERROR: Feature file does not contain ' + str(expected_parcels) + ' parcels:',
                            feature_file_name)

        # remove columns that are not features
        feature_df = feature_df.drop(columns=['map', 'parcellation', 'parcel'])
        col_list = [col for col in feature_df if not col.startswith('diagnostic')]

        feature_df = feature_df[col_list]

        for feature in feature_df:
            for parcel in parcels:
                feature_names.append(str(parcel) + '_' + str(feature))

        # flatten feature matrix (column-major order) to match loop above
        features = feature_df.to_numpy()
        features = features.flatten(order='F')

        tmp_df = pd.DataFrame([features], columns=feature_names)
        # select columns containing any of these strings
        if len(select) > 0:
            tmp_df = tmp_df.filter(regex='|'.join(select))
            if tmp_df.shape[1] == 0:
                print('NO FEATURES LEFT AFTER SELECTION', select)

        # drop columns containing any of these strings
        if len(drop) > 0:
            tmp_df = tmp_df.drop(tmp_df.filter(regex='|'.join(drop)).columns, axis=1)
            if tmp_df.shape[1] == 0:
                print('NO FEATURES LEFT AFTER DROP', drop)

        feature_file_names = tmp_df.columns.tolist()

        if out_df is None:
            out_df = tmp_df
        else:
            out_df = pd.concat([out_df, tmp_df], axis=0)

        bar.next()

    if verbose and len(feature_file_names) > 0:
        print('Loaded samples:', out_df.shape[0])
        print('Stats per sample:')
        print('Total number of features:', out_df.shape[1])
        print('Maps:', list(np.unique(maps)))
        # print('Number of parcels:', out_df.shape[0] // len(np.unique(maps)))
        # print('Number of features per map and parcel:', out_df.shape[1])

        print('Number of shape features:', len([col for col in out_df if col.__contains__('shape')]))
        print('Number of firstorder features:', len([col for col in out_df if col.__contains__('firstorder')]))
        print('Number of glcm features:', len([col for col in out_df if col.__contains__('glcm')]))
        print('Number of glrlm features:', len([col for col in out_df if col.__contains__('glrlm')]))
        print('Number of glszm features:', len([col for col in out_df if col.__contains__('glszm')]))
        print('Number of gldm features:', len([col for col in out_df if col.__contains__('gldm')]))
        print('Number of ngtdm features:', len([col for col in out_df if col.__contains__('ngtdm')]))

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
