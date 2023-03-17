from radiomics import featureextractor
import pandas as pd
import os
import nibabel as nib
import numpy as np


def calc_radiomics(parcellation_file_name: str,
                   parameter_map_file_name: str,
                   out_csv_file: str,
                   num_parcels: int = None,
                   features: dict = None,
                   pyrad_params=None):
    """
    Calculate radiomics features for a parcellation and a matching parameter map using pyradiomcs
    :param parcellation_file_name:
    :param parameter_map_file_name:
    :param out_csv_file: save features to this file
    :param num_parcels: optional to ensure that all labels are present in the parcellation
    :param features: append new features to this dict, if none, create empty dict
    :param pyrad_params: if none, use default parameter file (designed for FA maps)
    :return:
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
        print('pyradiomics saving results ...')
        features = pd.DataFrame(features)
        features.to_csv(out_csv_file, index=False)
    print('pyradiomics finished processing')

    return features
