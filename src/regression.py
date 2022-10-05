import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.svm import SVR
from tqdm import tqdm
import warnings


def train_score_model(model, data, splits=5, **model_param):
    X_reg, y_reg = data
    X_reg = np.array(X_reg)
    y_reg = np.array(y_reg)

    kf = KFold(n_splits=splits)
    scores_rmse = []
    scores_r2 = []
    for train_idx, test_idx in kf.split(X_reg, y_reg):
        X_train, y_train = X_reg[train_idx], y_reg[train_idx]
        X_test, y_test = X_reg[test_idx], y_reg[test_idx]

        reg_model = model(**model_param)
        reg_model.fit(X_train, y_train)
        y_pred = reg_model.predict(X_test)
        scores_rmse.append(mse(y_pred, y_test, squared=False))
        scores_r2.append(r2_score(y_pred, y_test))

    return np.array(scores_rmse).mean(), np.array(scores_r2).mean()


def reg_pipeline():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print("Starting regression tasks...")

    reg_data = pd.read_csv("data/bitrate_prediction/bitrate_train.csv")

    print("Removing outliers...")
    print(f"Dataset shape before removing outliers: {reg_data.shape}")
    reg_data = reg_data[reg_data["fps_mean"] < 65]
    reg_data = reg_data[reg_data["fps_std"] < 40.0]
    reg_data = reg_data[reg_data["rtt_std"] < 500]
    reg_data = reg_data[reg_data["dropped_frames_mean"] < 100]
    reg_data = reg_data[reg_data["dropped_frames_std"] < 75]
    reg_data = reg_data[reg_data["dropped_frames_max"] < 75]
    reg_data = reg_data[reg_data["rtt_std"] < 300]
    reg_data = reg_data[reg_data["bitrate_mean"] < 30000]
    reg_data = reg_data[reg_data["bitrate_std"] < 10000]

    print(f"Dataset shape after removing outliers: {reg_data.shape}")

    X_reg = reg_data[
        [
            "fps_mean",
            "fps_std",
            "rtt_mean",
            "rtt_std",
            "dropped_frames_mean",
            "dropped_frames_std",
            "dropped_frames_max",
            "bitrate_mean",
            "bitrate_std",
        ]
    ]
    y_reg = reg_data[["target"]]

    print("Training linear regression on raw data...")
    rmse, r2 = train_score_model(LinearRegression, (X_reg, y_reg))
    print(f"RMSE: {rmse}, R2: {r2}")

    print("Training linear model on important features...")
    rmse, r2 = train_score_model(
        LinearRegression,
        (reg_data[["bitrate_mean", "bitrate_std"]], reg_data[["target"]]),
    )
    print(f"RMSE: {rmse}, R2: {r2}")

    print("Scaling data...")
    scaler = MinMaxScaler()
    reg_data_norm = scaler.fit_transform(X_reg)
    X_reg_norm = pd.DataFrame(X_reg, columns=X_reg.columns)
    rmse, r2 = train_score_model(LinearRegression, (X_reg_norm, y_reg))
    print(f"RMSE: {rmse}, R2: {r2}")

    print("Add polynomial features...")
    poly = PolynomialFeatures(2)
    X_reg_norm = poly.fit_transform(X_reg_norm[["bitrate_mean", "bitrate_std"]])

    print("Training Ridge model...")
    scores = {}
    alphas = [0.0, 0.1, 0.5, 0.6, 0.7, 1.0, 1.5, 2.0, 100.0]

    for alpha in alphas:
        scores[alpha] = train_score_model(Ridge, (X_reg_norm, y_reg), alpha=alpha)

    for alpha, score in scores.items():
        print(f"Alpha: {alpha}, RMSE: {score[0]}, R2: {score[1]}")

    print("Training Lasso model...")
    scores = {}
    alphas = [0.0, 0.1, 0.5, 0.6, 0.7, 1.0, 1.5, 2.0, 100.0, 500.0, 1000.0]

    for alpha in tqdm(alphas):
        scores[alpha] = train_score_model(Lasso, (X_reg_norm, y_reg), alpha=alpha)

    for alpha, score in scores.items():
        print(f"Alpha: {alpha}, RMSE: {score[0]}, R2: {score[1]}")

    print("Training SVR model...")

    rmse, r2 = train_score_model(SVR, (X_reg_norm, y_reg), kernel="poly", max_iter=2000)
    print(f"RMSE: {rmse}, R2: {r2}")
