import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV, LassoCV


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if {"dteday", "hr"}.issubset(df.columns):
        df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
    return df


def _add_cyc(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    x = df[col].astype(float).to_numpy()
    return pd.DataFrame({
        f"{col}_sin": np.sin(2*np.pi*x/period),
        f"{col}_cos": np.cos(2*np.pi*x/period),
    }, index=df.index)


def time_split(df: pd.DataFrame, train_ratio: float = 0.7):
    n = len(df)
    k = int(n * train_ratio)
    return df.iloc[:k].copy(), df.iloc[k:].copy()


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # 为可视化保留时间索引
    if {"dteday", "hr"}.issubset(train_df.columns):
        try:
            train_time = pd.to_datetime(train_df["dteday"]) + pd.to_timedelta(train_df["hr"], unit="h")
            test_time = pd.to_datetime(test_df["dteday"]) + pd.to_timedelta(test_df["hr"], unit="h")
        except Exception:
            train_time = pd.RangeIndex(len(train_df))
            test_time = pd.RangeIndex(len(test_df))
    else:
        train_time = pd.RangeIndex(len(train_df))
        test_time = pd.RangeIndex(len(test_df))

    # 删除泄漏/无意义列（但已获取时间后可删除 dteday）
    drop_cols = [c for c in ["casual", "registered", "instant", "dteday"] if c in train_df.columns]
    for split in (train_df, test_df):
        split.drop(columns=drop_cols, inplace=True, errors="ignore")

    target_col = "cnt"
    y_tr = train_df[target_col].astype(float)
    y_te = test_df[target_col].astype(float)
    Xtr_raw = train_df.drop(columns=[target_col]).copy()
    Xte_raw = test_df.drop(columns=[target_col]).copy()

    # 数值列
    keep_numeric = [c for c in ["temp", "atemp", "hum", "windspeed", "yr", "holiday", "workingday"] if c in Xtr_raw.columns]

    # 周期特征
    cyc_tr = []
    cyc_te = []
    if "hr" in Xtr_raw:      cyc_tr.append(_add_cyc(Xtr_raw, "hr", 24));      cyc_te.append(_add_cyc(Xte_raw, "hr", 24))
    if "mnth" in Xtr_raw:    cyc_tr.append(_add_cyc(Xtr_raw, "mnth", 12));    cyc_te.append(_add_cyc(Xte_raw, "mnth", 12))
    if "weekday" in Xtr_raw: cyc_tr.append(_add_cyc(Xtr_raw, "weekday", 7));  cyc_te.append(_add_cyc(Xte_raw, "weekday", 7))

    # 类别 One-Hot
    oh_cols = [c for c in ["season", "weathersit"] if c in Xtr_raw.columns]
    Xtr_oh = pd.get_dummies(Xtr_raw[oh_cols].astype("category"), drop_first=False)
    Xte_oh = pd.get_dummies(Xte_raw[oh_cols].astype("category"), drop_first=False)
    Xtr_oh, Xte_oh = Xtr_oh.align(Xte_oh, join="outer", axis=1, fill_value=0)
    Xtr_oh = Xtr_oh.astype(float)
    Xte_oh = Xte_oh.astype(float)

    # 数值标准化（仅用训练集统计）
    Xtr_num_base = Xtr_raw[keep_numeric].astype(float)
    Xte_num_base = Xte_raw[keep_numeric].astype(float)
    mu = Xtr_num_base.mean(axis=0)
    sigma = Xtr_num_base.std(axis=0).replace(0, 1.0)
    Xtr_num = (Xtr_num_base - mu) / sigma
    Xte_num = (Xte_num_base - mu) / sigma

    # 二次项
    quad_cols = [c for c in ["temp", "atemp", "hum", "windspeed"] if c in Xtr_num.columns]
    Xtr_quad = (Xtr_num[quad_cols] ** 2).add_suffix("^2")
    Xte_quad = (Xte_num[quad_cols] ** 2).add_suffix("^2")

    # 合并
    X_tr_df = pd.concat([Xtr_num, Xtr_quad, Xtr_oh] + cyc_tr, axis=1)
    X_te_df = pd.concat([Xte_num, Xte_quad, Xte_oh] + cyc_te, axis=1)

    feature_names = X_tr_df.columns.tolist()
    X_tr = X_tr_df.to_numpy(dtype=float)
    X_te = X_te_df.to_numpy(dtype=float)
    y_tr = y_tr.to_numpy(dtype=float)
    y_te = y_te.to_numpy(dtype=float)
    return X_tr, y_tr, X_te, y_te, feature_names, train_time, test_time


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def fit_and_diagnose(X_tr, y_tr, X_te, y_te, feature_names, test_time):
    cv = TimeSeriesSplit(n_splits=5)
    alphas = np.logspace(-3, 3, 13)

    # RidgeCV
    ridge = RidgeCV(alphas=alphas, cv=cv, scoring='neg_mean_squared_error', fit_intercept=True)
    ridge.fit(X_tr, y_tr)
    y_pred_ridge = ridge.predict(X_te)

    # LassoCV
    lasso = LassoCV(alphas=alphas, cv=cv, max_iter=20000, random_state=42, fit_intercept=True)
    lasso.fit(X_tr, y_tr)
    y_pred_lasso = lasso.predict(X_te)

    # 评估
    print("Ridge 最优 alpha:", ridge.alpha_)
    print("Ridge Test MSE:", mse(y_te, y_pred_ridge), "MAE:", mae(y_te, y_pred_ridge))
    print("LASSO 最优 alpha:", lasso.alpha_)
    print("LASSO Test MSE:", mse(y_te, y_pred_lasso), "MAE:", mae(y_te, y_pred_lasso))

    # 特征重要性与筛选
    coef_ridge = pd.Series(ridge.coef_, index=feature_names)
    coef_lasso = pd.Series(lasso.coef_, index=feature_names)
    print("\nRidge |coef| Top 15:")
    print(coef_ridge.reindex(coef_ridge.abs().sort_values(ascending=False).index[:15]))
    nz = (coef_lasso != 0)
    print(f"\nLASSO 非零特征数: {nz.sum()} / {len(coef_lasso)}")
    print(coef_lasso[nz].reindex(coef_lasso[nz].abs().sort_values(ascending=False).index[:15]))

    # 诊断：残差与正态性
    residuals = y_te - y_pred_ridge
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=50)
    plt.title('Ridge Residuals Histogram')
    plt.subplot(1, 2, 2)
    try:
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Ridge Residuals QQ-Plot')
    except Exception:
        plt.text(0.5, 0.5, 'Install scipy for QQ-plot', ha='center')
    plt.tight_layout()
    plt.show()

    # 时间序列可视化（测试集）
    plt.figure(figsize=(14, 5))
    plt.plot(test_time, y_te, label='Actual')
    plt.plot(test_time, y_pred_ridge, label='Ridge Pred', alpha=0.8)
    plt.plot(test_time, y_pred_lasso, label='LASSO Pred', alpha=0.8)
    plt.xlabel('Time')
    plt.ylabel('cnt')
    plt.title('Test Set: Actual vs Predicted')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    df = load_data('bike_sharing_hour.csv')
    train_df, test_df = time_split(df)
    X_tr, y_tr, X_te, y_te, feature_names, train_time, test_time = build_features(train_df, test_df)
    fit_and_diagnose(X_tr, y_tr, X_te, y_te, feature_names, test_time)


if __name__ == '__main__':
    main()


