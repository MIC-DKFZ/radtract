# Copyright © 2023 German Cancer Research Center (DKFZ), Division of Medical Image Computing
#
# SPDX-License-Identifier: Apache-2.0

import radiomics
import mirp
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
import sys


def get_version():
    path = os.path.dirname(os.path.abspath(__file__))
    with open(path + '/__init__.py', 'r') as file:
        content = file.read()
        version = content.split('__version__ = ')[1].split('\n')[0].replace("'", '')
        return version


class Extractor:

    def __init__(self, num_parcels: int = None) -> None:
        self.num_parcels = num_parcels
        
        self.features = pd.DataFrame({'map': [], 
                                        'parcellation': [], 
                                        'parcel': [], 
                                        'filter': [], 
                                        'feature': [], 
                                        'value': [], 
                                        'extractor': [], 
                                        'extractor_version': [],
                                        'radtract_version': []})
        
    def init_features(self):
        print('Initializing features dataframe ...')
        self.features = pd.DataFrame({'map': [], 
                                        'parcellation': [], 
                                        'parcel': [], 
                                        'filter': [], 
                                        'feature': [], 
                                        'value': [], 
                                        'extractor': [], 
                                        'extractor_version': [],
                                        'radtract_version': []})

    def calc_radiomics(self, parcellation_file_name: str, parameter_map_file_name: str):
        raise NotImplementedError


class PyradiomicsExtractor(Extractor):

    remove_paths = False

    def __init__(self, pyrad_params=None, remove_paths = False) -> None:
        super().__init__()
        if pyrad_params is not None:
            self.extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(pyrad_params)
        else:
            self.extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(os.path.dirname(__file__) + '/pyrad.yaml')
        
        self.remove_paths = remove_paths

    def map_check(self, data: np.ndarray, extractor_settings: dict = None):
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

    def output_to_df(self, features, parcellation_file_name: str, parameter_map_file_name: str, parcel: str):

        # feature_vector is a dict with all features
        # convert this dict to a df with the dict keys as one column and the values as another column
        # then split the key column into two columns (filter and feature)
        # then add the parcel column
        # then append to self.features
        # then return self.features

        f = dict()
        f['filter'] = []
        f['feature'] = []
        f['value'] = []
        for key, value in features.items():
            filter = key.split('_')[0]
            feature = key.replace(filter + '_', '')
            f['filter'].append(filter)
            f['feature'].append(feature)
            f['value'].append(value)

        features_df = pd.DataFrame(f)
        features_df['extractor'] = 'pyradiomics'
        features_df['extractor_version'] = radiomics.__version__
        features_df['radtract_version'] = get_version()

        if self.remove_paths:
            features_df['map'] = os.path.basename(parameter_map_file_name)
            features_df['parcellation'] = os.path.basename(parcellation_file_name)
        else:
            features_df['map'] = parameter_map_file_name
            features_df['parcellation'] = parcellation_file_name
        features_df['parcel'] = parcel
        self.features = pd.concat([self.features, features_df], axis=0)


    def calc_radiomics(self, parcellation_file_name: str, parameter_map_file_name: str):
            
            self.init_features()
            
            print('Pyradiomics settings:', self.extractor.settings)
            print('Enabled image types:', self.extractor.enabledImagetypes)
            print('Enables features:', self.extractor.enabledFeatures)

            map_sitk, parc_sitk = self.extractor.loadImage(parameter_map_file_name, parcellation_file_name)
            parcellation_data = sitk.GetArrayFromImage(parc_sitk)

            self.map_check(sitk.GetArrayFromImage(map_sitk), self.extractor.settings)

            labels = np.unique(parcellation_data)
            print('Found labels', labels)
            if self.num_parcels is not None:
                assert self.num_parcels == len(labels) - 1, 'Number of parcels does not match number of labels in ' + parcellation_file_name

            # get global features
            print('pyradiomics calculating global tract features')
            parcellation_data = sitk.GetArrayFromImage(parc_sitk)
            parcellation_data[parcellation_data > 0] = 1
            env_sitk = sitk.GetImageFromArray(parcellation_data)
            env_sitk.CopyInformation(parc_sitk)

            self.output_to_df(self.extractor.execute(imageFilepath=map_sitk, maskFilepath=env_sitk),
                              parcellation_file_name, parameter_map_file_name, 'GLOBAL')
            

            # features per label
            for label in labels:
                label = int(label)
                if label == 0:
                    continue
                print('pyradiomics processing parcel ' + str(label))
                self.output_to_df(self.extractor.execute(imageFilepath=map_sitk, maskFilepath=parc_sitk, label=label),
                                  parcellation_file_name, parameter_map_file_name, 'P' + str(label))

            print('pyradiomics finished processing')

            return self.features
    

class MirpExtractor(Extractor):

    def calc_radiomics(self, parcellation_file_name: str, parameter_map_file_name: str, features: dict = None):

        if features is None:
            features = dict()
            features['map'] = []
            features['parcellation'] = []
            features['parcel'] = []

        
        parcellation_data = nib.load(parcellation_file_name).get_fdata()

        labels = np.unique(parcellation_data)
        print('Found labels', labels)
        if self.num_parcels is not None:
            assert self.num_parcels == len(labels) - 1, 'Number of parcels does not match number of labels in ' + parcellation_file_name

        # features per label
        # for label in labels:
        #     label = int(label)
        #     if label == 0:
        #         continue

        #     mask = np.zeros(parcellation_data.shape)
        #     mask[parcellation_data == label] = 1

        #     print('pyradiomics processing label ' + str(label))
        #     features = extract_features(image=map_data, mask=mask, base_discretisation_method="fixed_bin_number", base_discretisation_n_bins=32, export_features=True, )
        features = mirp.extract_features(image=parameter_map_file_name, 
                                         mask=parcellation_file_name, 
                                         base_discretisation_method="fixed_bin_number", 
                                         base_discretisation_n_bins=32, 
                                         export_features=True, 
                                         num_cpus=16)
        print('mirp finished processing')

        return features


def calc_tractometry(point_label_file_name: str,
                     parameter_map_file_name: str,
                     out_csv_file: str = None,
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

    extractor = PyradiomicsExtractor()
    bla = extractor.calc_radiomics(parcellation_file_name='/home/neher/Downloads/bmbf_c0_046/tractseg_output/parcellations/CST_left_hyperplane.nii.gz',
                                   parameter_map_file_name='/home/neher/Downloads/bmbf_c0_046/Diffusion_MNI_tensors_fa.nii.gz',)

    # for el in bla:
    #     print(el.columns)
    print(bla.shape)
    bla.to_csv('/home/neher/Downloads/test.csv')
    bla.to_pickle('/home/neher/Downloads/test.pkl')


    return

    parser = argparse.ArgumentParser(description='RadTract Feature Calculation')
    parser.add_argument('--parcellation', type=str, help='Input parcellation file')
    parser.add_argument('--map', type=str, help='Parameter map file (e.g. fractional anisotropy)')
    parser.add_argument('--pyrad_params', type=str, help='Pyradiomics parameter file (e.g. pyrad.yaml)', default=None)
    parser.add_argument('--output', type=str, help='Output feature file (.csv)')

    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()

    if args.parcellation.endswith('.nii.gz'):
        calc_radiomics(parcellation_file_name=args.parcellation,
                    parameter_map_file_name=args.map,
                    pyrad_params=args.pyrad_params,
                    out_csv_file=args.output,
                    )
    elif args.parcellation.endswith('.pkl'):
        calc_tractometry(point_label_file_name=args.parcellation,
                         parameter_map_file_name=args.map,
                         out_csv_file=args.output,
                         )


if __name__ == '__main__':
    main()
