from radtract.features import load_features
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def univariate_feature_selection(X_train, y_train, X_test, k):
    # https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
    X_train_np = X_train.to_numpy()
    new_f = np.add(X_train_np, -np.min(X_train_np, axis=0))
    selector = SelectKBest(k=k)
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
    corr_matrix = np.triu(np.fabs(np.corrcoef(features_df.to_numpy(), rowvar=False)), k=1)
    to_drop = np.unique(np.where(corr_matrix > 0.95)[1])
    features_df = features_df.drop(features_df.columns[to_drop], axis=1)
    print('Num. features', features_df.shape[1])
    return features_df


def normalize_features(features_df):
    print('Normalizing features')
    features_df = (features_df - features_df.mean()) / features_df.std()
    return features_df


def classification_experiment(feature_files, targets, remove_map_substrings=[], n_jobs=-1):
    features_df = load_features(feature_files, verbose=True, remove_map_substrings=[])
    features_df = remove_low_variance_features(features_df)
    features_df = remove_correlated_features(features_df)
    features_df = normalize_features(features_df)

    cv = LeaveOneOut()

    predictions = []
    ground_truth = []
    classifiers = []
    for seed in range(10):
        for train_idxs, test_idxs in cv.split(features_df, targets):

            x_train = features_df.iloc[train_idxs, :]
            y_train = targets[train_idxs]
            x_test = features_df.iloc[test_idxs, :]
            y_test = targets[test_idxs]

            clf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=n_jobs)
            clf.fit(x_train, y_train)
            y_pred = clf.predict_proba(x_test)
            predictions.append(y_pred)
            ground_truth.append(y_test)
            classifiers.append(clf)

    predictions = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truth)

    return predictions, ground_truth, classifiers


def main():
    pass


if __name__ == '__main__':
    main()
