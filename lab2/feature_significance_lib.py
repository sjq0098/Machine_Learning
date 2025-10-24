import numpy as np
import pandas as pd
import statsmodels.formula.api as smf



def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if {"dteday", "hr"}.issubset(df.columns):
        df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
    return df


def add_cyc_features(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    x = df[col].astype(float).to_numpy()
    return pd.DataFrame({
        f"{col}_sin": np.sin(2 * np.pi * x / period),
        f"{col}_cos": np.cos(2 * np.pi * x / period),
    }, index=df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 删除泄漏或无意义列
    for c in ["casual", "registered", "instant", "dteday"]:
        if c in out.columns:
            out.drop(columns=c, inplace=True)

    # 周期特征
    if "hr" in out:      out = pd.concat([out, add_cyc_features(out, "hr", 24)], axis=1)
    if "mnth" in out:    out = pd.concat([out, add_cyc_features(out, "mnth", 12)], axis=1)
    if "weekday" in out: out = pd.concat([out, add_cyc_features(out, "weekday", 7)], axis=1)

    # 二次项（用公式 I(x**2) 亦可，这里也预先生成方便复用）
    for c in ["temp", "atemp", "hum", "windspeed"]:
        if c in out.columns:
            out[f"{c}_sq"] = out[c].astype(float) ** 2

    return out


def time_split(df: pd.DataFrame, train_ratio: float = 0.7):
    n = len(df)
    k = int(n * train_ratio)
    return df.iloc[:k].copy(), df.iloc[k:].copy()


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def main():
    df = load_data('bike_sharing_hour.csv')
    train_df, test_df = time_split(df)
    train_df = build_features(train_df)
    test_df = build_features(test_df)

    # 公式：包含数值、二次项、类别及周期特征（拦截默认包含）
    # C() 指定类别处理；二次项用已生成的 *_sq，亦可使用 I(temp**2)
    formula_terms = [
        'temp', 'temp_sq', 'atemp', 'atemp_sq',
        'hum', 'hum_sq', 'windspeed', 'windspeed_sq',
        'yr', 'holiday', 'workingday',
        'C(season)', 'C(weathersit)',
        'hr_sin', 'hr_cos', 'mnth_sin', 'mnth_cos', 'weekday_sin', 'weekday_cos'
    ]
    formula = 'cnt ~ ' + ' + '.join([t for t in formula_terms if t in train_df.columns])

    # 拟合 OLS 并打印显著性摘要
    model = smf.ols(formula=formula, data=train_df).fit()
    print(model.summary())

    # 按 |t| 排序输出前 30 项（跳过拦截）
    params = model.params.drop(labels=['Intercept'], errors='ignore')
    tvals = model.tvalues.drop(labels=['Intercept'], errors='ignore')
    pvals = model.pvalues.drop(labels=['Intercept'], errors='ignore')
    order = np.argsort(-np.abs(tvals.values))
    names = tvals.index.to_list()
    print("\n按显著性排序（|t| 从大到小）前 30：")
    for i in order[:min(30, len(order))]:
        name = names[i]
        print(f"{name}\tcoef={params[name]:.4f}\tt={tvals[name]:.2f}\tp={pvals[name]:.3g}")

    # 测试集评估
    y_te = test_df['cnt'].astype(float).to_numpy()
    y_pred = model.predict(test_df).to_numpy(dtype=float)
    print("\nTest MSE:", mse(y_te, y_pred), "Test MAE:", mae(y_te, y_pred))


if __name__ == '__main__':
    main()


