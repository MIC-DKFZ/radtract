# Copyright © 2023 German Cancer Research Center (DKFZ), Division of Medical Image Computing
#
# SPDX-License-Identifier: Apache-2.0

import radiomics
import pandas as pd
import os
import nibabel as nib
import numpy as np
import argparse
import joblib
from scipy.ndimage import map_coordinates
from nibabel.affines import apply_affine
import SimpleITK as sitk
import sys
import scipy
import sys


def get_version():
    """
    Get version of radtract package
    """
    path = os.path.dirname(os.path.abspath(__file__))
    with open(path + '/__init__.py', 'r') as file:
        content = file.read()
        version = content.split('__version__ = ')[1].split('\n')[0].replace("'", '')
        return version


class Extractor:
    """
    Base class for feature extractors
    """

    def __init__(self, num_parcels: int = None) -> None:
        """
        :param num_parcels: optional to ensure that all labels are present in the parcellation
        """
        self.num_parcels = num_parcels
        self.init_features()
        
    def init_features(self):
        print('Initializing features dataframe ...')
        self.features = pd.DataFrame({'map': [], 
                                      'parcellation': [], 
                                      'parcel': [], 
                                      'filter': [], 
                                      'feature_type': [], 
                                      'feature': [], 
                                      'value': [], 
                                      'extractor': [], 
                                      'extractor_version': [],
                                      'radtract_version': []})

    def calc_features(self, parcellation_file_name: str, parameter_map_file_name: str):
        """
        Calculate features for a given parcellation and parameter map. Has to be implemented by the subclass.
        """
        raise NotImplementedError
    
    def flatten_features(feature_df: pd.DataFrame, expected_parcels: int = None):
        """
        Flatten feature dataframe with one column per feature
        
        :param feature_df: feature dataframe as returned by calc_features
        :param expected_parcels: optional to ensure that all labels are present in the parcellation
        :return: flattened feature dataframe with one column per feature
        """
        # check for nan values
        if feature_df.isna().values.any():
            print('Feature frame contains NaN values:')

        parcels = feature_df['parcel'].unique()
        if expected_parcels is not None and len(parcels) != expected_parcels + 1:  # +1 accounts for tract-global features
            raise Exception('ERROR: Feature frame does not contain ' + str(expected_parcels) + ' parcels')

        # drop parcellation, extractor, extractor_version and radtract_version columns
        feature_df = feature_df.drop(columns=['parcellation', 'extractor', 'extractor_version', 'radtract_version'])

        # remove file ending from map names
        feature_df['map'] = feature_df['map'].str.replace('.nii.gz', '')

        # split map names at '/' and keep last part
        feature_df['map'] = feature_df['map'].str.split('/').str[-1]

        # in all columns, replace '_' with '-'
        feature_df = feature_df.replace('_', '-', regex=True)

        # creat new column 'f' concatenating all other columns but the values using '_' as separator in the following order: map, parcel, filter, feature
        feature_df['f'] = feature_df['map'] + '_' + feature_df['parcel'] + '_' + feature_df['filter'] + '_' + feature_df['feature_type'] + '_' + feature_df['feature']

        # drop all columns but 'f' and 'value'
        feature_df = feature_df[['f', 'value']]
        # rename 'f' to 'feature'
        feature_df = feature_df.rename(columns={'f': 'feature'})
        # transpose to get one column per feature
        feature_df = feature_df.transpose()
        # set first row as column names (feature names)
        feature_df.columns = feature_df.iloc[0]
        # drop first row (feature names)
        feature_df = feature_df.drop(feature_df.index[0])
        # reset index
        feature_df = feature_df.reset_index(drop=True)

        return feature_df


class PyradiomicsExtractor(Extractor):
    """
    Extract features using the pyradiomics package (https://pyradiomics.readthedocs.io/en/latest/)
    """

    def __init__(self, num_parcels: int = None, pyrad_params=None) -> None:
        super().__init__(num_parcels = num_parcels)
        if pyrad_params is not None:
            self.extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(pyrad_params)
        else:
            self.extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(os.path.dirname(__file__) + '/pyrad.yaml')

    def map_check(self, data: np.ndarray, extractor_settings: dict = None):
        """
        :param map: Path to the map
        :param extractor_settings: Settings for the pyradiomics feature extractor
        """
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
        f['feature_type'] = []
        f['feature'] = []
        f['value'] = []
        for key, value in features.items():
            filter = key.split('_')[0]
            feature = key.replace(filter + '_', '')
            f['filter'].append(filter)
            ftype = feature.split('_')[0]
            if ftype != 'shape' and ftype != 'firstorder':
                ftype = 'texture'
            else:
                feature = feature.replace(ftype + '_', '')
            f['feature_type'].append(ftype)
            f['feature'].append(feature)
            f['value'].append(value)

        features_df = pd.DataFrame(f)
        features_df['extractor'] = 'pyradiomics'
        features_df['extractor_version'] = radiomics.__version__
        features_df['radtract_version'] = get_version()
        features_df['map'] = parameter_map_file_name
        features_df['parcellation'] = parcellation_file_name
        features_df['parcel'] = parcel
        self.features = pd.concat([self.features, features_df], axis=0)


    def calc_features(self, parcellation_file_name: str, parameter_map_file_name: str):
            
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

            # drop where filter == 'diagnostic'
            self.features = self.features[self.features['filter'] != 'diagnostics']
            # reset index
            self.features = self.features.reset_index(drop=True)

            return self.features
    

if sys.version_info >= (3, 11): # mirp only works with python >= 3.11
    import mirp 

    class MirpExtractor(Extractor):
        """
        Extract features using the mirp package (https://github.com/oncoray/mirp/)
        """

        def __init__(self, num_parcels: int = None) -> None:
            super().__init__(num_parcels = num_parcels)

        def calc_features(self, parcellation_file_name: str, parameter_map_file_name: str):

            self.init_features()

            features = {'map': [], 
                        'parcellation': [], 
                        'parcel': [], 
                        'filter': [], 
                        'feature_type': [],
                        'feature': [], 
                        'value': [], 
                        'extractor': [], 
                        'extractor_version': [],
                        'radtract_version': []}

            
            parcellation_data = nib.load(parcellation_file_name).get_fdata()

            labels = np.unique(parcellation_data)
            print('Found labels', labels)
            if self.num_parcels is not None:
                assert self.num_parcels == len(labels) - 1, 'Number of parcels does not match number of labels in ' + parcellation_file_name

            # to-do: get global features
                

            # get features for each label
            np.random.seed(0)
            print('RANDOM SEED SET TO 0 FOR MIRP')
            print('\033[91mWARNING: MIRP is automatically normalizing inside the mask. This is aknown issue and has to be fixed.\033[0m')
            mirp_output = mirp.extract_features(image=parameter_map_file_name, 
                                            mask=parcellation_file_name, 
                                            base_discretisation_method="fixed_bin_number", 
                                            base_discretisation_n_bins=32, 
                                            export_features=True)
            np.random.seed(None)
            mirp_output = mirp_output[0]

            # drop column 'sample_name'
            mirp_output = mirp_output.drop(columns=['sample_name'])

            # drop all columns where column name starts with 'image_'
            mirp_output = mirp_output.loc[:, ~mirp_output.columns.str.startswith('image_')]

            features = {'map': [], 
                        'parcellation': [], 
                        'parcel': [], 
                        'filter': [], 
                        'feature_type': [],
                        'feature': [], 
                        'value': [], 
                        'extractor': [], 
                        'extractor_version': [],
                        'radtract_version': []}

            # fill output dataframe
            for index, row in mirp_output.iterrows():
                for column in mirp_output.columns:
                    parcel = 'P' + str(index)
                    filter = 'original'
                    feature = column
                    feature_type = '???'
                    tmp = feature.split('_')[0]
                    if tmp == 'morph':
                        feature_type = 'shape'
                    elif tmp in ['stat', 'ivh', 'ih', 'loc']:
                        feature_type = 'firstorder'
                    elif tmp in ['cm', 'rlm', 'szm', 'dzm', 'ngl']:
                        feature_type = 'texture'
                    
                    features['map'].append(parameter_map_file_name)
                    features['parcellation'].append(parcellation_file_name)
                    features['parcel'].append(parcel)
                    features['filter'].append(filter)
                    features['extractor'].append('mirp')
                    features['extractor_version'].append('???')
                    features['radtract_version'].append(get_version())
                    features['feature_type'].append(feature_type)
                    features['feature'].append(feature)
                    features['value'].append(row[column])
            features = pd.DataFrame(features)
            self.features = self.features.reset_index(drop=True)

            print('mirp finished processing')

            return features
        

#class MitkExtractor(Extractor):



class TractometryExtractor(Extractor):
    """
    Extract features using tractometry
    """

    def __init__(self, num_parcels: int = None) -> None:
        super().__init__(num_parcels = num_parcels)

    def calc_features(self,
                      parcellation_file_name: str,
                      parameter_map_file_name: str):
        """
        Calculate tractometry features using points and corresponding parcel labels
        :param parcellation_file_name: parcellation file in .pkl format (points and labels)
        :param parameter_map_file_name: parameter map file in .nii.gz format (e.g. fractional anisotropy)
        :return: tractometry features
        """

        self.init_features()

        features = {'map': [], 
                    'parcellation': [], 
                    'parcel': [], 
                    'filter': [], 
                    'feature_type': [],
                    'feature': [], 
                    'value': [], 
                    'extractor': [], 
                    'extractor_version': [],
                    'radtract_version': []}

        streamline_point_parcels = joblib.load(parcellation_file_name)
        map = nib.load(parameter_map_file_name)
        map_data = map.get_fdata()

        print('Calculating tractometry ...')
        points = apply_affine(np.linalg.inv(map.affine), streamline_point_parcels['points'])
        values = map_coordinates(map_data, points.T, order=1)
        vals_per_parcel = dict()
        points_per_parcel = dict()
        if self.num_parcels is not None:
            assert self.num_parcels == np.unique(streamline_point_parcels['parcels']).shape[0], 'Number of parcels does not match number of labels in ' + parcellation_file_name

        for parcel, val, p in zip(streamline_point_parcels['parcels'], values, streamline_point_parcels['points']):
            if parcel not in vals_per_parcel.keys():
                vals_per_parcel[parcel] = []
            vals_per_parcel[parcel].append(val)
            if parcel not in points_per_parcel.keys():
                points_per_parcel[parcel] = []
            points_per_parcel[parcel].append(p)

        measures = [('mean', np.nanmean), ('median', np.nanmedian), ('std', np.nanstd), ('min', np.nanmin), ('max', np.nanmax), ('sum', np.nansum), ('count', len),
                    ('range', lambda x: np.nanmax(x) - np.nanmin(x)), ('skew', lambda x: scipy.stats.skew(x, nan_policy='omit')), ('kurtosis', lambda x: scipy.stats.kurtosis(x, nan_policy='omit')),
                    ('percentile-5', lambda x: np.nanpercentile(x, 5)), ('percentile-25', lambda x: np.nanpercentile(x, 25)), ('percentile-75', lambda x: np.nanpercentile(x, 75)), ('percentile-95', lambda x: np.nanpercentile(x, 95)),
                    ('iqr', lambda x: scipy.stats.iqr(x, nan_policy='omit')), ('entropy', lambda x: scipy.stats.entropy(x)), ('variation', lambda x: scipy.stats.variation(x, nan_policy='omit')),
                    ('median-abs-dev', lambda x: scipy.stats.median_abs_deviation(x, nan_policy='omit')),
                    ]
        
        # tract-global features
        for measure, func in measures:
            features['map'].append(parameter_map_file_name)
            features['parcellation'].append(parcellation_file_name)
            features['parcel'].append('GLOBAL')
            features['filter'].append('original')
            features['extractor'].append('tractometry')
            features['extractor_version'].append(get_version())
            features['radtract_version'].append(get_version())
            features['feature_type'].append('firstorder')
            features['feature'].append(measure)
            features['value'].append(func(values))

        # per label features
        for parcel in sorted(vals_per_parcel.keys()):

            for measure, func in measures:
                features['map'].append(parameter_map_file_name)
                features['parcellation'].append(parcellation_file_name)
                features['parcel'].append('P' + str(parcel))
                features['filter'].append('original')
                features['extractor'].append('tractometry')
                features['extractor_version'].append(get_version())
                features['radtract_version'].append(get_version())
                features['feature_type'].append('firstorder')
                features['feature'].append(measure)
                features['value'].append(func(vals_per_parcel[parcel]))

        self.features = pd.DataFrame(features)
        self.features = self.features.reset_index(drop=True)
        
        print('tractometry finished processing')

        return self.features


def main():

    parser = argparse.ArgumentParser(description='RadTract Feature Calculation')
    parser.add_argument('--parcellation', type=str, help='Input parcellation file')
    parser.add_argument('--map', type=str, help='Parameter map file (e.g. fractional anisotropy)')
    parser.add_argument('--pyrad_params', type=str, help='Pyradiomics parameter file (e.g. pyrad.yaml)', default=None)
    parser.add_argument('--output', type=str, help='Output feature file (.csv or .pkl)')

    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()

    features = None

    if args.parcellation.endswith('.nii.gz'):
        extractor = PyradiomicsExtractor(pyrad_params=args.pyrad_params)
        features = extractor.calc_features(parcellation_file_name=args.parcellation,
                                           parameter_map_file_name=args.map)
        
    elif args.parcellation.endswith('.pkl'):
        extractor = TractometryExtractor()
        features = extractor.calc_features(parcellation_file_name=args.parcellation,
                                           parameter_map_file_name=args.map)
    
    else:
        raise Exception('ERROR: Parcellation file must be in .nii.gz or .pkl format')
    
    if features is not None:
        if args.output.endswith('.csv'):
            features.to_csv(args.output)
        elif args.output.endswith('.pkl'):
            features.to_pickle(args.output)
        else:
            raise Exception('ERROR: Output file must be in .csv or .pkl format')


if __name__ == '__main__':
    main()
