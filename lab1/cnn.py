import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import KFold #用于k折检验

#加载数据并处理
def load_data(file_path):
    # 文件中是类似 0.0000/1.0000 的浮点数，用 numpy 读取为浮点
    data = np.loadtxt(file_path, dtype=np.float32)
    return data

data=load_data('semeion.data')

def preprocessing(data):
    X = data[:, :-10]   # 图像数据
    y = data[:, -10:]   # one-hot 标签

    X = X.reshape(-1, 1, 16, 16)      # (N, 1, 16, 16)
    y = np.argmax(y, axis=1)          # (N,)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return dataset

dataset=preprocessing(data)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # [N,32,8,8]
        x = self.pool(self.relu(self.conv2(x)))  # [N,64,4,4]
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train_and_evaluate(train_idx, val_idx, dataset, device, epochs=15, batch_size=64):
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # 验证
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total


# ========== k-fold 交叉验证 ==========
def k_fold_cross_val(dataset, k=5, epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    acc_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        acc = train_and_evaluate(train_idx, val_idx, dataset, device, epochs=epochs)
        acc_scores.append(acc)
        print(f"Fold {fold+1}: Acc={acc:.4f}")

    print(f"\nAverage Accuracy over {k}-folds: {np.mean(acc_scores):.4f}")

# 运行
k_fold_cross_val(dataset, k=5, epochs=15)

