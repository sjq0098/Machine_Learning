import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torchvision.transforms.functional as TF

# ========== 数据加载 ==========
def load_data(file_path):
    data = np.loadtxt(file_path, dtype=np.float32)
    return data

def preprocessing(data):
    X = data[:, :-10]
    y = data[:, -10:]

    X = X.reshape(-1, 1, 16, 16)  # (N,1,16,16)
    y = np.argmax(y, axis=1)      # (N,)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return dataset

# ========== 数据增强 ==========
def augment_dataset(dataset):
    X_list, y_list = [], []
    for X, y in dataset:
        X_list.append(X)  # 原始
        y_list.append(y)

        # 左上旋转
        angle1 = random.uniform(-10, -5)
        X_rot1 = TF.rotate(X, angle1, interpolation=TF.InterpolationMode.BILINEAR)
        X_list.append(X_rot1)
        y_list.append(y)

        # 左下旋转
        angle2 = random.uniform(5, 10)
        X_rot2 = TF.rotate(X, angle2, interpolation=TF.InterpolationMode.BILINEAR)
        X_list.append(X_rot2)
        y_list.append(y)

    X_tensor = torch.stack(X_list)
    y_tensor = torch.tensor(y_list, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)

# ========== CNN 模型 ==========
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

# ========== 训练 + 验证函数 ==========
def train_and_evaluate(train_idx, val_idx, dataset, device, epochs=15, batch_size=64):
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(correct / total)

        # ---- Validation ----
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        val_losses.append(running_loss / len(val_loader))
        val_accs.append(correct / total)

    return model, train_losses, val_losses, train_accs, val_accs, val_loader

# ========== 可视化函数 ==========
def plot_curves(train_losses, val_losses, train_accs, val_accs, fold):
    epochs = range(1, len(train_losses)+1)

    fig, axs = plt.subplots(1, 2, figsize=(12,5))

    axs[0].plot(epochs, train_losses, label="Train Loss")
    axs[0].plot(epochs, val_losses, label="Val Loss")
    axs[0].set_title(f"Loss Curve (Fold {fold})")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot(epochs, train_accs, label="Train Acc")
    axs[1].plot(epochs, val_accs, label="Val Acc")
    axs[1].set_title(f"Accuracy Curve (Fold {fold})")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, dataloader, device):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = outputs.max(1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# ========== K-fold 主程序 ==========
def k_fold_cross_val(dataset, k=5, epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    acc_scores = []
    last_model, last_val_loader = None, None

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        model, train_losses, val_losses, train_accs, val_accs, val_loader = \
            train_and_evaluate(train_idx, val_idx, dataset, device, epochs=epochs)

        acc_scores.append(val_accs[-1])
        print(f"Fold {fold+1}: Final Val Acc = {val_accs[-1]:.4f}")

        # 每折画一次曲线
        plot_curves(train_losses, val_losses, train_accs, val_accs, fold+1)

        # 记录最后一折的模型和验证集
        last_model, last_val_loader = model, val_loader

    print(f"\nAverage Accuracy over {k}-folds: {np.mean(acc_scores):.4f}")

    # 画最后一折的混淆矩阵
    plot_confusion_matrix(last_model, last_val_loader, device)

# ========== 运行 ==========
if __name__ == "__main__":
    data = load_data("semeion.data")
    dataset = preprocessing(data)
    aug_dataset = augment_dataset(dataset)

    print("原始数据大小:", len(dataset))
    print("增强后数据大小:", len(aug_dataset))

    k_fold_cross_val(aug_dataset, k=5, epochs=15)
