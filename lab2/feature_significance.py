import numpy as np
import pandas as pd


def _add_cyc(df, col, period):
    x = df[col].astype(float).to_numpy()
    return pd.DataFrame({
        f"{col}_sin": np.sin(2*np.pi*x/period),
        f"{col}_cos": np.cos(2*np.pi*x/period),
    }, index=df.index)


def load_data(path):
    df = pd.read_csv(path)
    if {"dteday", "hr"}.issubset(df.columns):
        df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
    return df


def preprocess_with_names(train_df, test_df):
    drop_cols = [c for c in ["casual", "registered", "instant", "dteday"] if c in train_df.columns]
    for split in (train_df, test_df):
        split.drop(columns=drop_cols, inplace=True, errors="ignore")

    target_col = "cnt"
    y_tr = train_df[target_col].astype(float)
    y_te = test_df[target_col].astype(float)
    Xtr_raw = train_df.drop(columns=[target_col]).copy()
    Xte_raw = test_df.drop(columns=[target_col]).copy()

    keep_numeric = [c for c in ["temp", "atemp", "hum", "windspeed", "yr", "holiday", "workingday"] if c in Xtr_raw.columns]

    cyc_tr = []
    cyc_te = []
    if "hr" in Xtr_raw:      cyc_tr.append(_add_cyc(Xtr_raw, "hr", 24));      cyc_te.append(_add_cyc(Xte_raw, "hr", 24))
    if "mnth" in Xtr_raw:    cyc_tr.append(_add_cyc(Xtr_raw, "mnth", 12));    cyc_te.append(_add_cyc(Xte_raw, "mnth", 12))
    if "weekday" in Xtr_raw: cyc_tr.append(_add_cyc(Xtr_raw, "weekday", 7));  cyc_te.append(_add_cyc(Xte_raw, "weekday", 7))

    oh_cols = [c for c in ["season", "weathersit"] if c in Xtr_raw.columns]
    Xtr_oh = pd.get_dummies(Xtr_raw[oh_cols].astype("category"), drop_first=False)
    Xte_oh = pd.get_dummies(Xte_raw[oh_cols].astype("category"), drop_first=False)
    Xtr_oh, Xte_oh = Xtr_oh.align(Xte_oh, join="outer", axis=1, fill_value=0)
    Xtr_oh = Xtr_oh.astype(float)
    Xte_oh = Xte_oh.astype(float)

    Xtr_num_base = Xtr_raw[keep_numeric].astype(float)
    Xte_num_base = Xte_raw[keep_numeric].astype(float)
    mu = Xtr_num_base.mean(axis=0)
    sigma = Xtr_num_base.std(axis=0).replace(0, 1.0)
    Xtr_num = (Xtr_num_base - mu) / sigma
    Xte_num = (Xte_num_base - mu) / sigma

    quad_cols = [c for c in ["temp", "atemp", "hum", "windspeed"] if c in Xtr_num.columns]
    Xtr_quad = (Xtr_num[quad_cols] ** 2).add_suffix("^2")
    Xte_quad = (Xte_num[quad_cols] ** 2).add_suffix("^2")

    X_tr_df = pd.concat([Xtr_num, Xtr_quad, Xtr_oh] + cyc_tr, axis=1)
    X_te_df = pd.concat([Xte_num, Xte_quad, Xte_oh] + cyc_te, axis=1)
    X_tr_df.insert(0, "bias", 1.0)
    X_te_df.insert(0, "bias", 1.0)

    feature_names = X_tr_df.columns.tolist()
    return (
        X_tr_df.to_numpy(dtype=float),
        y_tr.to_numpy(dtype=float),
        X_te_df.to_numpy(dtype=float),
        y_te.to_numpy(dtype=float),
        feature_names,
    )


def ols_with_significance(X, y):
    # beta = (X^T X)^-1 X^T y （用 pinv 更稳健）
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)

    # 残差与噪声方差估计（无偏）：sigma^2 = RSS / (n - p)
    n, p = X.shape
    residuals = y - X @ beta
    rss = float(residuals.T @ residuals)
    dof = max(1, n - p)
    sigma2 = rss / dof

    # 系数协方差：sigma^2 * (X^T X)^-1
    cov_beta = sigma2 * XtX_inv
    se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))

    # t 值与 p 值（正态近似）
    with np.errstate(divide='ignore', invalid='ignore'):
        t_values = np.divide(beta, se_beta, out=np.zeros_like(beta), where=se_beta>0)
    # 双侧 p 值，正态近似：p = 2 * (1 - Phi(|t|))
    # 这里用数值误差小的近似：1 - Phi(x) ≈ 0.5 * erfc(x / sqrt(2))
    from math import erfc, sqrt
    p_values = 2.0 * 0.5 * np.array([erfc(abs(t)/sqrt(2.0)) for t in t_values])

    return beta, se_beta, t_values, p_values


def main():
    df = load_data('bike_sharing_hour.csv')
    n = len(df)
    split = int(n * 0.7)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    X_tr, y_tr, X_te, y_te, feature_names = preprocess_with_names(train_df, test_df)

    beta, se, tvals, pvals = ols_with_significance(X_tr, y_tr)

    # 排序并打印（跳过 bias）
    idx = np.arange(len(feature_names))
    non_bias = idx[1:]
    order = non_bias[np.argsort(-np.abs(tvals[1:]))]

    print("按显著性排序（|t| 从大到小）：")
    topk = min(30, len(order))
    for j in order[:topk]:
        print(f"{feature_names[j]}\tcoef={beta[j]:.4f}\tse={se[j]:.4f}\tt={tvals[j]:.2f}\tp={pvals[j]:.3g}")


if __name__ == '__main__':
    main()


