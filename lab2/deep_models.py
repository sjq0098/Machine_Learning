import math
import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader
except Exception as e:
    raise SystemExit("需要安装 PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu") from e


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if {"dteday", "hr"}.issubset(df.columns):
        df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame, train_ratio: float = 0.7):
    n = len(df)
    k = int(n * train_ratio)
    return df.iloc[:k].copy(), df.iloc[k:].copy()


def _add_cyc(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    x = df[col].astype(float).to_numpy()
    return pd.DataFrame({
        f"{col}_sin": np.sin(2*np.pi*x/period),
        f"{col}_cos": np.cos(2*np.pi*x/period),
    }, index=df.index)


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # 删除泄漏/无意义列
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

    X_tr = X_tr_df.to_numpy(dtype=np.float32)
    X_te = X_te_df.to_numpy(dtype=np.float32)
    y_tr = y_tr.to_numpy(dtype=np.float32)
    y_te = y_te.to_numpy(dtype=np.float32)
    return X_tr, y_tr, X_te, y_te, feature_names


def make_val_split_time(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.1):
    n = len(X)
    k = max(1, int(n * (1 - val_ratio)))
    return (X[:k], y[:k]), (X[k:], y[k:])


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_tr, y_tr, X_te, y_te, epochs: int = 50, batch_size: int = 512, lr: float = 1e-3):
    device = torch.device("cpu")
    (X_train, y_train), (X_val, y_val) = make_val_split_time(X_tr, y_tr, val_ratio=0.1)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MLPRegressor(in_dim=X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = math.inf
    best_state = None
    patience = 8
    bad = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_trues = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = model(xb).cpu().numpy()
                val_preds.append(pred)
                val_trues.append(yb.numpy())
            val_pred = np.concatenate(val_preds)
            val_true = np.concatenate(val_trues)
            val_mse = mse(val_true, val_pred)
        if val_mse < best_val - 1e-6:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 测试
    model.eval()
    with torch.no_grad():
        test_preds = []
        test_trues = []
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            test_preds.append(pred)
            test_trues.append(yb.numpy())
        y_pred = np.concatenate(test_preds)
        y_true = np.concatenate(test_trues)
    return {
        "model": model,
        "test_mse": mse(y_true, y_pred),
        "test_mae": mae(y_true, y_pred),
    }


class LSTMRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: [B, T, D]
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        y = self.head(last).squeeze(-1)
        return y


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    assert len(X) == len(y)
    if len(X) <= seq_len:
        return np.empty((0, seq_len, X.shape[1]), dtype=X.dtype), np.empty((0,), dtype=y.dtype)
    xs = []
    ys = []
    for i in range(seq_len, len(X)):
        xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.stack(xs, axis=0), np.array(ys)


def make_sequences_with_ar(X: np.ndarray, z: np.ndarray, seq_len: int):
    """构造带自回归通道的序列：将过去 seq_len 个标准化目标 z 作为额外通道拼接到特征上。
    返回 (X_seq_aug, z_target)。
    """
    assert len(X) == len(z)
    if len(X) <= seq_len:
        return np.empty((0, seq_len, X.shape[1] + 1), dtype=X.dtype), np.empty((0,), dtype=z.dtype)
    xs = []
    ys = []
    for i in range(seq_len, len(X)):
        x_win = X[i - seq_len:i]                 # [T, D]
        z_hist = z[i - seq_len:i].reshape(-1, 1) # [T, 1]
        x_aug = np.concatenate([x_win, z_hist], axis=1)  # [T, D+1]
        xs.append(x_aug)
        ys.append(z[i])                          # 预测标准化后的下一个值
    return np.stack(xs, axis=0), np.array(ys)


def train_lstm(X_tr, y_tr, X_te, y_te, seq_len: int = 24, epochs: int = 60, batch_size: int = 256, lr: float = 1e-3):
    device = torch.device("cpu")
    # 仅在各自分割内部构造序列，避免泄漏
    (X_train_raw, y_train_raw), (X_val_raw, y_val_raw) = make_val_split_time(X_tr, y_tr, val_ratio=0.1)
    # 目标标准化（仅用训练集统计）
    y_mu = float(y_tr.mean())
    y_sigma = float(max(1e-6, y_tr.std()))
    z_train_raw = (y_train_raw - y_mu) / y_sigma
    z_val_raw = (y_val_raw - y_mu) / y_sigma
    z_test_raw = (y_te - y_mu) / y_sigma

    # 带自回归通道的序列
    X_train, z_train = make_sequences_with_ar(X_train_raw, z_train_raw, seq_len)
    X_val, z_val = make_sequences_with_ar(X_val_raw, z_val_raw, seq_len)
    X_test, z_test = make_sequences_with_ar(X_te, z_test_raw, seq_len)

    train_ds = TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(z_train.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(X_val.astype(np.float32)), torch.from_numpy(z_val.astype(np.float32)))
    test_ds = TensorDataset(torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(z_test.astype(np.float32)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = LSTMRegressor(in_dim=X_tr.shape[1] + 1, hidden=96, num_layers=1, dropout=0.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = math.inf
    best_state = None
    patience = 10
    bad = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_trues = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = model(xb).cpu().numpy()
                val_preds.append(pred)
                val_trues.append(yb.numpy())
            val_pred = np.concatenate(val_preds) if val_preds else np.array([])
            val_true = np.concatenate(val_trues) if val_trues else np.array([])
            val_mse = mse(val_true, val_pred) if len(val_true) else math.inf
        if val_mse < best_val - 1e-6:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 测试
    model.eval()
    with torch.no_grad():
        test_preds = []
        test_trues = []
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            test_preds.append(pred)
            test_trues.append(yb.numpy())
        z_pred = np.concatenate(test_preds) if test_preds else np.array([])
        z_true = np.concatenate(test_trues) if test_trues else np.array([])
        # 反标准化到原尺度
        y_pred = z_pred * y_sigma + y_mu
        y_true = z_true * y_sigma + y_mu
    return {
        "model": model,
        "test_mse": mse(y_true, y_pred) if len(y_true) else math.inf,
        "test_mae": mae(y_true, y_pred) if len(y_true) else math.inf,
    }


def main():
    set_seed(42)
    df = load_data('bike_sharing_hour.csv')
    train_df, test_df = time_split(df)
    X_tr, y_tr, X_te, y_te, feature_names = build_features(train_df, test_df)

    print("[MLP] 训练中...")
    mlp_res = train_mlp(X_tr, y_tr, X_te, y_te, epochs=60, batch_size=512, lr=1e-3)
    print("[MLP] Test MSE:", mlp_res["test_mse"], "MAE:", mlp_res["test_mae"])

    print("[LSTM] 训练中...")
    lstm_res = train_lstm(X_tr, y_tr, X_te, y_te, seq_len=48, epochs=40, batch_size=256, lr=1e-3)
    print("[LSTM] Test MSE:", lstm_res["test_mse"], "MAE:", lstm_res["test_mae"])


if __name__ == '__main__':
    main()


