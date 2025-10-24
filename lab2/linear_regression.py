import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#读数据
def load_data(file_path):
    df = pd.read_csv(file_path)
    # 按时间排序，确保无信息穿越
    if {"dteday","hr"}.issubset(df.columns):
        df = df.sort_values(["dteday","hr"]).reset_index(drop=True)
    return df

#按照时间划分数据
def time_split(df, train_ratio=0.7):
    n = len(df)
    k = int(n * train_ratio)
    return df.iloc[:k].copy(), df.iloc[k:].copy()


#数据预处理，删除缺失值并处理到可以使用多元线性回归
def _add_cyc(df, col, period):
    x = df[col].astype(float).to_numpy()
    return pd.DataFrame({
        f"{col}_sin": np.sin(2*np.pi*x/period),
        f"{col}_cos": np.cos(2*np.pi*x/period),
    }, index=df.index)

def preprocess(train_df, test_df):
    drop_cols = [c for c in ["casual","registered","instant","dteday"] if c in train_df.columns]
    for split in (train_df, test_df):
        split.drop(columns=drop_cols, inplace=True, errors="ignore")

    target_col = "cnt"
    y_tr = train_df[target_col].astype(float)
    y_te = test_df[target_col].astype(float)
    Xtr_raw = train_df.drop(columns=[target_col]).copy()
    Xte_raw = test_df.drop(columns=[target_col]).copy()

    # 保持数值/二值列
    # 保留主要数值列（恢复 atemp），后续通过多项式抑制简单线性不足
    keep_numeric = [c for c in ["temp","atemp","hum","windspeed","yr","holiday","workingday"] if c in Xtr_raw.columns]

    # === 周期编码替代 One-Hot：hr(24), mnth(12), weekday(7) ===
    cyc_tr = []
    cyc_te = []
    if "hr" in Xtr_raw:      cyc_tr.append(_add_cyc(Xtr_raw,"hr",24));      cyc_te.append(_add_cyc(Xte_raw,"hr",24))
    if "mnth" in Xtr_raw:    cyc_tr.append(_add_cyc(Xtr_raw,"mnth",12));    cyc_te.append(_add_cyc(Xte_raw,"mnth",12))
    if "weekday" in Xtr_raw: cyc_tr.append(_add_cyc(Xtr_raw,"weekday",7));  cyc_te.append(_add_cyc(Xte_raw,"weekday",7))

    # 仍然做 One-Hot 的少量非周期多类（season, weathersit）
    oh_cols = [c for c in ["season","weathersit"] if c in Xtr_raw.columns]
    Xtr_oh = pd.get_dummies(Xtr_raw[oh_cols].astype("category"), drop_first=False)
    Xte_oh = pd.get_dummies(Xte_raw[oh_cols].astype("category"), drop_first=False)
    Xtr_oh, Xte_oh = Xtr_oh.align(Xte_oh, join="outer", axis=1, fill_value=0)
    # 统一为浮点类型，避免后续 numpy 转换变为 object dtype
    Xtr_oh = Xtr_oh.astype(float)
    Xte_oh = Xte_oh.astype(float)

    # 数值标准化（仅用训练集统计）+ 简单二次多项式扩展
    Xtr_num_base = Xtr_raw[keep_numeric].astype(float)
    Xte_num_base = Xte_raw[keep_numeric].astype(float)
    mu = Xtr_num_base.mean(axis=0); sigma = Xtr_num_base.std(axis=0).replace(0, 1.0)
    Xtr_num = (Xtr_num_base - mu) / sigma
    Xte_num = (Xte_num_base - mu) / sigma

    # 二次项（不包含 yr/holiday/workingday 这类接近二值/低取值范围列）
    quad_cols = [c for c in ["temp","atemp","hum","windspeed"] if c in Xtr_num.columns]
    Xtr_quad = (Xtr_num[quad_cols] ** 2).add_suffix("^2")
    Xte_quad = (Xte_num[quad_cols] ** 2).add_suffix("^2")

    # 合并
    X_tr = pd.concat([Xtr_num, Xtr_quad, Xtr_oh] + cyc_tr, axis=1)
    X_te = pd.concat([Xte_num, Xte_quad, Xte_oh] + cyc_te, axis=1)
    X_tr.insert(0, "bias", 1.0); X_te.insert(0, "bias", 1.0)
    
    # 强制转换为 float，避免 object 数组导致 ufunc 报错
    return X_tr.to_numpy(dtype=float), y_tr.to_numpy(dtype=float), X_te.to_numpy(dtype=float), y_te.to_numpy(dtype=float)


#拿到数据集开始实现多元线性回归和梯度下降

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """将小时级数据按天聚合：
    - cnt, casual, registered 按天求和
    - temp, atemp, hum, windspeed 按天求均值
    - yr 取当天首个
    - holiday, workingday 取当天众数（若并列，选择最小值）
    - 从 dteday 派生 mnth 与 weekday
    保留列风格与下游 preprocess 兼容（无 hr）。
    """
    dfc = df.copy()
    if "dteday" not in dfc.columns:
        raise ValueError("缺少 dteday 列，无法按天聚合")
    dfc["dteday"] = pd.to_datetime(dfc["dteday"])  # 确保为日期
    grp = dfc.groupby("dteday", sort=True)

    def mode_safe(s: pd.Series):
        vc = s.value_counts()
        if len(vc) == 0:
            return np.nan
        # 众数有并列时取值最小者以稳定
        maxc = vc.max()
        candidates = vc[vc == maxc].index
        try:
            return np.min(candidates.astype(float))
        except Exception:
            return candidates.min()

    daily = pd.DataFrame({
        "cnt": grp["cnt"].sum(),
        "casual": grp["casual"].sum() if "casual" in dfc.columns else grp["cnt"].sum()*0,
        "registered": grp["registered"].sum() if "registered" in dfc.columns else grp["cnt"].sum()*0,
        "temp": grp["temp"].mean(),
        "atemp": grp["atemp"].mean() if "atemp" in dfc.columns else grp["temp"].mean(),
        "hum": grp["hum"].mean(),
        "windspeed": grp["windspeed"].mean(),
        "yr": grp["yr"].first() if "yr" in dfc.columns else 0,
        "holiday": grp["holiday"].apply(mode_safe) if "holiday" in dfc.columns else 0,
        "workingday": grp["workingday"].apply(mode_safe) if "workingday" in dfc.columns else 0,
        "season": grp["season"].apply(mode_safe) if "season" in dfc.columns else np.nan,
        "weathersit": grp["weathersit"].apply(mode_safe) if "weathersit" in dfc.columns else np.nan,
    }).reset_index()

    # 从日期派生月份与星期（与原数据字段保持一致命名）
    daily["mnth"] = daily["dteday"].dt.month
    daily["weekday"] = daily["dteday"].dt.weekday
    # 将日期列转回字符串以与其他流程一致，后续 preprocess 会丢弃它
    daily["dteday"] = daily["dteday"].dt.strftime("%Y-%m-%d")
    return daily

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def nmse(y_true, y_pred):
    var = np.var(y_true)
    if var <= 0:
        return np.nan
    return mse(y_true, y_pred) / var

def minibatch_gradient_descent(X, y, learning_rate=0.01, epochs=1000, batch_size=32):
    n,p=X.shape
    beta = np.zeros(p)
    losses = []
    for _ in range(epochs):
        for i in range(0, n, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            gradient = 2 * X_batch.T @ (X_batch @ beta - y_batch) / max(1, len(X_batch))
            beta = beta - learning_rate * gradient
        # epoch 结束后记录整体训练损失
        losses.append(mse(y, X @ beta))
    return beta, losses

def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    n,p=X.shape
    beta = np.zeros(p)
    losses = []
    for _ in range(epochs):
        for i in range(n):
            gradient = 2 * X[i] @ (X[i] @ beta - y[i])
            beta = beta - learning_rate * gradient
        losses.append(mse(y, X @ beta))
    return beta, losses

def batch_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    n,p=X.shape
    beta = np.zeros(p)
    losses = []
    for _ in range(epochs):
        gradient = 2 * X.T @ (X @ beta - y) / n
        beta = beta - learning_rate * gradient
        losses.append(mse(y, X @ beta))
    return beta, losses

def linear_regression(X_tr, y_tr, X_te, y_te, learning_rate=0.001, epochs=500, batch_size=32):
    # 支持对数目标变换
    log_target = False

    if log_target:
        y_tr_used = np.log1p(y_tr)
        y_te_used = np.log1p(y_te)
    else:
        y_tr_used = y_tr
        y_te_used = y_te

    beta, losses = minibatch_gradient_descent(X_tr, y_tr_used, learning_rate, epochs, batch_size)
    #beta, losses = stochastic_gradient_descent(X_tr, y_tr, lr, epochs)
    #beta, losses = batch_gradient_descent(X_tr, y_tr, lr, epochs)

    # 计算训练集和测试集的MSE和MAE
    y_pred_tr_used = X_tr @ beta
    y_pred_te_used = X_te @ beta

    if log_target:
        y_pred_tr = np.expm1(y_pred_tr_used)
        y_pred_te = np.expm1(y_pred_te_used)
    else:
        y_pred_tr = y_pred_tr_used
        y_pred_te = y_pred_te_used

    print(f"训练集MSE: {mse(y_tr, y_pred_tr)}, 训练集MAE: {mae(y_tr, y_pred_tr)}, 训练集NMSE: {nmse(y_tr, y_pred_tr)}")
    print(f"测试集MSE: {mse(y_te, y_pred_te)}, 测试集MAE: {mae(y_te, y_pred_te)}, 测试集NMSE: {nmse(y_te, y_pred_te)}")

    # 画出损失收敛曲线（训练损失）
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE (log target)" if log_target else "Train MSE")
    plt.title("Loss Convergence")
    plt.tight_layout()
    plt.show()

    return {
        "beta": beta,
        "losses": losses,
        "y_pred_tr": y_pred_tr,
        "y_pred_te": y_pred_te,
    }

def plot_multi_lr_curves(X_tr, y_tr, learning_rates, epochs, title, save_path):
    plt.figure()
    for lr in learning_rates:
        _, losses = batch_gradient_descent(X_tr, y_tr, learning_rate=lr, epochs=epochs)
        plt.plot(losses, label=f"lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=160)
    except Exception:
        pass
    plt.show()

def baseline_closed_form(X_tr, y_tr, X_te, y_te):
    # 与训练函数一致的目标变换设置
    log_target = False

    if log_target:
        y_tr_used = np.log1p(y_tr)
    else:
        y_tr_used = y_tr

    # 闭式解（普通最小二乘）：beta = pinv(X) @ y
    beta_ols = np.linalg.pinv(X_tr) @ y_tr_used

    # Ridge 闭式解（不惩罚 bias），beta = (X^T X + λI*)^{-1} X^T y
    lam = 1.0
    XtX = X_tr.T @ X_tr
    I = np.eye(XtX.shape[0])
    I[0,0] = 0.0  # 不惩罚偏置项
    beta_ridge = np.linalg.pinv(XtX + lam * I) @ (X_tr.T @ y_tr_used)

    # OLS 预测
    y_pred_tr_used_ols = X_tr @ beta_ols
    y_pred_te_used_ols = X_te @ beta_ols
    # Ridge 预测
    y_pred_tr_used_ridge = X_tr @ beta_ridge
    y_pred_te_used_ridge = X_te @ beta_ridge

    if log_target:
        y_pred_tr_ols = np.expm1(y_pred_tr_used_ols)
        y_pred_te_ols = np.expm1(y_pred_te_used_ols)
        y_pred_tr_ridge = np.expm1(y_pred_tr_used_ridge)
        y_pred_te_ridge = np.expm1(y_pred_te_used_ridge)
    else:
        y_pred_tr_ols = y_pred_tr_used_ols
        y_pred_te_ols = y_pred_te_used_ols
        y_pred_tr_ridge = y_pred_tr_used_ridge
        y_pred_te_ridge = y_pred_te_used_ridge

    print("[OLS] 训练集MSE:", mse(y_tr, y_pred_tr_ols), "训练集MAE:", mae(y_tr, y_pred_tr_ols))
    print("[OLS] 测试集MSE:", mse(y_te, y_pred_te_ols), "测试集MAE:", mae(y_te, y_pred_te_ols))
    print("[Ridge] 训练集MSE:", mse(y_tr, y_pred_tr_ridge), "训练集MAE:", mae(y_tr, y_pred_tr_ridge))
    print("[Ridge] 测试集MSE:", mse(y_te, y_pred_te_ridge), "测试集MAE:", mae(y_te, y_pred_te_ridge))
    print("beta OLS（含bias在第1位）:", beta_ols[:10], " ...")
    print("beta Ridge（含bias在第1位）:", beta_ridge[:10], " ...")

def main():
    df = load_data('bike_sharing_hour.csv')

    # ==== 不聚合：小时级 ====
    train_df, test_df = time_split(df)
    X_tr, y_tr, X_te, y_te = preprocess(train_df, test_df)
    # 多学习率收敛曲线（小时级）
    lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    plot_multi_lr_curves(X_tr, y_tr, lrs, epochs=400, title="GD Loss Curves (Hourly, Non-Aggregated)", save_path="pic/gd_loss_curves_nonagg.png")
    # 基础训练并打印 NMSE
    _ = linear_regression(X_tr, y_tr, X_te, y_te, learning_rate=1e-3, epochs=500, batch_size=64)

    print("--------------------------------")

    # ==== 聚合：按天 ====
    df_daily = aggregate_daily(df)
    train_d, test_d = time_split(df_daily)
    X_tr_d, y_tr_d, X_te_d, y_te_d = preprocess(train_d, test_d)
    plot_multi_lr_curves(X_tr_d, y_tr_d, lrs, epochs=400, title="GD Loss Curves (Daily Aggregated)", save_path="pic/gd_loss_curves_daily.png")
    _ = linear_regression(X_tr_d, y_tr_d, X_te_d, y_te_d, learning_rate=1e-3, epochs=500, batch_size=32)

    print("--------------------------------")
    # 仍保留闭式解作为参考（基于小时级特征）
    baseline_closed_form(X_tr, y_tr, X_te, y_te)
if __name__ == "__main__":
    main()



