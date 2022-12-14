import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score as f1
from sklearn.metrics import roc_auc_score as aucroc
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder


def train_score_model(model, data, splits=5, **model_param):
    X_reg, y_reg = data
    X_reg = np.array(X_reg)
    y_reg = np.array(y_reg)

    kf = KFold(n_splits=splits)
    scores_aucroc = []
    scores_f1 = []
    for train_idx, test_idx in kf.split(X_reg, y_reg):
        X_train, y_train = X_reg[train_idx], y_reg[train_idx]
        X_test, y_test = X_reg[test_idx], y_reg[test_idx]

        clf_model = model(**model_param)
        clf_model.fit(X_train, y_train)
        y_pred = clf_model.predict(X_test)
        scores_aucroc.append(aucroc(y_pred, y_test))
        scores_f1.append(f1(y_pred, y_test))

    return np.array(scores_aucroc).mean(), np.array(scores_f1).mean()


def check_on_test_data(model, data_train, data_test, **model_param):
    X_train, y_train = data_train
    X_test, y_test = data_test
    clf_model = model(**model_param)
    clf_model.fit(X_train, y_train)
    y_pred = clf_model.predict(X_test)
    return aucroc(y_pred, y_test), f1(y_pred, y_test)


def preprocess_data(clf_data):
    print("Transforming categorical features to numerical one...")
    bitrate_state_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    transformed = bitrate_state_encoder.fit_transform(
        clf_data["auto_bitrate_state"].to_numpy().reshape(-1, 1)
    )
    bitrate_state_df = pd.DataFrame(
        transformed,
        columns=[
            f"bitrate_{category}" for category in bitrate_state_encoder.categories_[0]
        ],
    )

    fec_state_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    transformed = fec_state_encoder.fit_transform(
        clf_data["auto_fec_state"].to_numpy().reshape(-1, 1)
    )
    fec_state_df = pd.DataFrame(
        transformed,
        columns=[f"fec_{category}" for category in fec_state_encoder.categories_[0]],
    )

    y_true = clf_data[["stream_quality"]].to_numpy()
    clf_data = clf_data[
        [
            "fps_mean",
            "fps_std",
            "fps_lags",
            "rtt_mean",
            "rtt_std",
            "dropped_frames_mean",
            "dropped_frames_std",
            "dropped_frames_max",
            "auto_fec_mean",
        ]
    ]

    clf_data.reset_index(inplace=True)
    clf_data.drop(["index"], axis=1, inplace=True, errors="ignore")
    fec_state_df.reset_index(inplace=True)
    fec_state_df.drop(["index"], axis=1, inplace=True, errors="ignore")
    bitrate_state_df.reset_index(inplace=True)
    bitrate_state_df.drop(["index"], axis=1, inplace=True, errors="ignore")

    clf_data = pd.concat([clf_data, fec_state_df, bitrate_state_df], axis=1)

    return clf_data, y_true


def clf_pipeline():
    print("Starting classification tasks...")
    clf_data = pd.read_csv("data/stream_quality_data/train_data.csv")
    test_data = pd.read_csv("data/stream_quality_data/test_data.csv")

    print("Removing outliers...")
    clf_data = clf_data[clf_data["fps_mean"] < 80]
    clf_data = clf_data[clf_data["fps_std"] < 60]
    clf_data = clf_data[clf_data["rtt_mean"] < 8000]

    X_train, y_train = preprocess_data(clf_data)
    X_test, y_test = preprocess_data(test_data)

    print("Train linear classification model...")
    aucroc, f1 = train_score_model(SGDClassifier, (X_train, y_train), penalty="l2")
    print(f"Train AUC ROC: {aucroc}, F1: {f1}")
    aucroc, f1 = check_on_test_data(
        SGDClassifier, (X_train, y_train), (X_test, y_test), penalty="l2"
    )
    print(f"Test AUC ROC: {aucroc}, F1: {f1}")

    print("Resampling train data...")
    cc = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = cc.fit_resample(X_train, y_train)

    print("Train on resampled data...")
    aucroc, f1 = train_score_model(
        SGDClassifier, (X_resampled, y_resampled), penalty="l2"
    )
    print(f"Train score after resampling. AUC ROC: {aucroc}, F1: {f1}")
    aucroc, f1 = check_on_test_data(
        SGDClassifier, (X_resampled, y_resampled), (X_test, y_test), penalty="l2"
    )
    print(f"Test score after resampling. AUC ROC: {aucroc}, F1: {f1}")
