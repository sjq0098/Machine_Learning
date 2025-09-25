import numpy as np
import torch
from torch.utils.data import TensorDataset
import torchvision.transforms.functional as TF
import random
from collections import Counter
from sklearn.metrics import normalized_mutual_info_score


# ========= 数据加载 =========
def load_raw_data(file_path="semeion.data"):
    data = np.loadtxt(file_path, dtype=np.float32)
    X = data[:, :-10].reshape(-1, 1, 16, 16)   # (N,1,16,16)
    y = np.argmax(data[:, -10:], axis=1)       # (N,)
    return X, y


# ========= 数据增强 =========
def augment_dataset(X, y):
    X_list, y_list = [], []
    for i in range(len(X)):
        img = torch.tensor(X[i])   # (1,16,16)
        label = y[i]

        # 原始
        X_list.append(img)
        y_list.append(label)

        # 左上旋转（-10°到-5°）
        angle1 = random.uniform(-10, -5)
        X_rot1 = TF.rotate(img, angle1, interpolation=TF.InterpolationMode.BILINEAR)
        X_list.append(X_rot1)
        y_list.append(label)

        # 左下旋转（+5°到+10°）
        angle2 = random.uniform(5, 10)
        X_rot2 = TF.rotate(img, angle2, interpolation=TF.InterpolationMode.BILINEAR)
        X_list.append(X_rot2)
        y_list.append(label)

    X_tensor = torch.stack(X_list)        # (N*3,1,16,16)
    y_tensor = torch.tensor(y_list)       # (N*3,)
    return X_tensor.numpy(), y_tensor.numpy()


# ========= 评价指标 =========
def cen(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / len(y_true)

def nmi(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred)


# ========= 手写 KNN 留一法 =========
def knn(X, y, k=5):
    n = X.shape[0]
    correct = 0
    preds = []
    for i in range(n):
        test_vec = X[i]
        test_label = y[i]

        train_vecs = np.delete(X, i, axis=0)
        train_labels = np.delete(y, i, axis=0)

        distances = np.linalg.norm(train_vecs - test_vec, axis=1)
        sorted_idx = np.argsort(distances)
        k_nearest = train_labels[sorted_idx[:k]]

        pred = Counter(k_nearest).most_common(1)[0][0]
        preds.append(pred)

        if pred == test_label:
            correct += 1

    acc = correct / n
    nmi_val = nmi(y, preds)
    cen_val = cen(y, preds)
    return acc, nmi_val, cen_val


# ========= 主程序 =========
if __name__ == "__main__":
    # 选择原始 or 增强
    use_aug = True   # 改成 False 就跑原始

    # 加载数据
    X, y = load_raw_data()

    if use_aug:
        X, y = augment_dataset(X, y)
        print("使用增强数据:", X.shape, y.shape)
    else:
        X = X.reshape(X.shape[0], -1)   # (N,256)
        print("使用原始数据:", X.shape, y.shape)

    # 如果增强，记得 reshape
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    # 跑不同 k
    for k in [5, 7, 9, 13]:
        acc, nmi_val, cen_val = knn(X, y, k)
        print(f"KNN (k={k}) | ACC={acc:.4f} | NMI={nmi_val:.4f} | CEN={cen_val:.4f}")

