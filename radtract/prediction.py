from radtract.features import load_features
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import pandas as pd
from cmdint.Utils import ProgressBar


def univariate_feature_selection(X_train, y_train, X_test, k):
    # https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
    if k <= 0 or k > X_train.shape[1]:
        return X_train, X_test

    X_train_np = X_train.to_numpy()
    new_f = np.add(X_train_np, -np.min(X_train_np, axis=0))

    if np.issubdtype(y_train.dtype, np.integer):
        selector = SelectKBest(k=k)
    else:
        selector = SelectKBest(k=k, score_func=f_regression)

    selector.fit_transform(new_f, y_train)
    mask = selector._get_support_mask()
    X_train = X_train.loc[:, mask]
    X_test = X_test.loc[:, mask]

    return X_train, X_test


def remove_low_variance_features(features_df):
    print('Removing low variance features (< 1.0e-10)')
    features_df = features_df.loc[:, features_df.var() > 1.0e-10]
    print('Num. features', features_df.shape[1])
    return features_df


def remove_correlated_features(features_df):
    print('Removing correlated features')
    if features_df.shape[1] > 10000:
        parcels = features_df.columns.tolist()
        i =  0
        for i in range(len(parcels)):
            parcels[i] = parcels[i].split('_')[0]
        parcels = np.unique(parcels)
        proc_dfs = []
        for parcel in parcels:
            df_temp = features_df.loc[:, features_df.columns.str.startswith(parcel)]
            corr_matrix = np.triu(np.fabs(np.corrcoef(df_temp.to_numpy(), rowvar=False)), k=1)
            to_drop = np.unique(np.where(corr_matrix > 0.95)[1])
            df_temp = df_temp.drop(df_temp.columns[to_drop], axis=1)
            proc_dfs.append(df_temp)
        features_df = pd.concat(proc_dfs, axis=1)

    corr_matrix = np.triu(np.fabs(np.corrcoef(features_df.to_numpy(), rowvar=False)), k=1)
    to_drop = np.unique(np.where(corr_matrix > 0.95)[1])
    features_df = features_df.drop(features_df.columns[to_drop], axis=1)
    print('Num. features', features_df.shape[1])
    return features_df


def normalize_features(features_df):
    print('Normalizing features')
    features_df = (features_df - features_df.mean()) / features_df.std()
    print('Done')
    return features_df


def drop_nan_features(features_df):
    print('Dropping features with NaN values')
    features_df = features_df.dropna(axis=1)
    print('Num. features', features_df.shape[1])
    return features_df


def run_cv_experiment(feature_files, targets, remove_map_substrings=[], n_jobs=-1, select = [], drop = [], remove_low_variance=True, remove_correlated=True, kbest_features=0, folds=0):
    '''
    Runs a cross-validation experiment using random forest classifier/regressor.
    :param feature_files: list of feature files
    :param targets: numpy array of targets
    :param remove_map_substrings: list of substrings to remove from feature names (feature names stored in the classifiers will be shorter)
    :param n_jobs: number of jobs of the RandomForestClassifier/RandomForestRegressor
    :param select: features containing any of these substrings will be included in the experiment
    :param drop: features containing any of these substrings will be excluded from the experiment (after inclusion)
    :param remove_low_variance: if True, features with variance < 1.0e-10 will be removed
    :param remove_correlated: if True, correlated features will be removed (Pearson r > 0.95)
    :param kbest_features: if > 0, univariate feature selection will be performed, keeping only the k best features
    :param folds: number of folds for cross-validation (if <= 1, leave-one-out cross-validation will be performed)
    '''

    features_df = load_features(feature_files, verbose=True, remove_map_substrings=remove_map_substrings, select=select, drop=drop)

    features_df = drop_nan_features(features_df)

    if remove_low_variance:
        features_df = remove_low_variance_features(features_df)

    if remove_correlated:
        features_df = remove_correlated_features(features_df)

    features_df = normalize_features(features_df)

    is_classification = True
    if not np.issubdtype(targets.dtype, np.integer):
        print('Targets are not integer. Interpreting as regression problem.')
        is_classification = False
        print('Starting regression experiment')
    else:
        print('Starting classification experiment')

    if folds > 1:
        cv = StratifiedKFold(n_splits=folds)
        print('Using {}-fold stratified cross-validation'.format(folds))
    else:
        cv = LeaveOneOut()
        print('Using leave-one-out cross-validation')

    predictions = []
    ground_truth = []
    classifiers = []
    bar = ProgressBar(len(feature_files) * 10)
    for seed in range(10):
        for train_idxs, test_idxs in cv.split(features_df, targets):

            x_train = features_df.iloc[train_idxs, :]
            y_train = targets[train_idxs]
            x_test = features_df.iloc[test_idxs, :]
            y_test = targets[test_idxs]

            if kbest_features > 0:
                x_train, x_test = univariate_feature_selection(x_train, y_train, x_test, kbest_features)

            if is_classification:
                model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=seed, n_jobs=n_jobs)
            else:
                model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=seed, n_jobs=n_jobs)
            model.fit(x_train, y_train)

            if is_classification:
                y_pred = model.predict_proba(x_test)
            else:
                y_pred = model.predict(x_test)
            predictions.append(y_pred)
            ground_truth.append(y_test)
            classifiers.append(model)
            bar.next()

    predictions = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truth)
    print('Done')

    return predictions, ground_truth, classifiers


def main():
    pass


if __name__ == '__main__':
    main()
