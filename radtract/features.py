from radiomics import featureextractor
import pandas as pd
import os
import nibabel as nib
import numpy as np
import argparse


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
        assert num_parcels == len(labels)-1, 'Number of parcels does not match number of labels in parcellation file'
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
