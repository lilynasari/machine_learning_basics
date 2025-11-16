import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子保证可重现性
torch.manual_seed(42)
np.random.seed(42)

# ---------------------- 1. 数据准备 ----------------------
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)


# ---------------------- 2. 定义主干模型 ----------------------
class BaseNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=2):
        super(BaseNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)
        return x


# 创建模型实例
input_dim = X_train.shape[1]
model = BaseNeuralNetwork(input_dim=input_dim, hidden_dims=[128, 64], output_dim=2)

# ---------------------- 3. 训练配置 ----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------- 4. 训练过程 ----------------------
print("开始训练基准模型（无任何正则化）")
print("=" * 50)

epochs = 200
for epoch in range(epochs):
    # 训练
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每40轮输出一次进度
    if epoch % 40 == 0 or epoch == epochs - 1:
        model.eval()
        with torch.no_grad():
            # 训练集评估
            train_outputs = model(X_train_tensor)
            train_loss = criterion(train_outputs, y_train_tensor)
            train_pred = torch.argmax(train_outputs, dim=1)
            train_acc = accuracy_score(y_train, train_pred.numpy())

            # 测试集评估
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            test_pred = torch.argmax(test_outputs, dim=1)
            test_acc = accuracy_score(y_test, test_pred.numpy())

            # 计算权重范数
            total_norm = 0
            for param in model.parameters():
                if param.requires_grad:
                    total_norm += param.norm(2).item() ** 2
            weight_norm = total_norm ** 0.5

            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss.item():.4f} | "
                  f"Test Loss: {test_loss.item():.4f} | "
                  f"Weight Norm: {weight_norm:.2f}")

# ---------------------- 5. 最终结果输出 ----------------------
print("\n" + "=" * 50)
print("基准模型最终结果")
print("=" * 50)

model.eval()
with torch.no_grad():
    # 最终训练集评估
    train_outputs = model(X_train_tensor)
    train_loss = criterion(train_outputs, y_train_tensor)
    train_pred = torch.argmax(train_outputs, dim=1)
    train_acc = accuracy_score(y_train, train_pred.numpy())

    # 最终测试集评估
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    test_pred = torch.argmax(test_outputs, dim=1)
    test_acc = accuracy_score(y_test, test_pred.numpy())

    # 最终权重范数
    total_norm = 0
    for param in model.parameters():
        if param.requires_grad:
            total_norm += param.norm(2).item() ** 2
    weight_norm = total_norm ** 0.5

print(f"训练集交叉熵损失: {train_loss.item():.6f}")
print(f"测试集交叉熵损失: {test_loss.item():.6f}")
print(f"权重L2范数: {weight_norm:.6f}")
print(f"训练集准确率: {train_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")

# 计算过拟合程度
overfitting_gap = train_acc - test_acc
print(f"过拟合程度: {overfitting_gap:.4f}")