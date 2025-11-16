import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子保证可重现性
torch.manual_seed(42)
np.random.seed(42)

# ---------------------- 1. 数据准备 ----------------------
print("加载Fashion-MNIST数据集...")
from torchvision import datasets, transforms

# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # 展平为1D向量
])

# 下载并加载数据
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 转换为numpy数组进行处理
X_train = train_dataset.data.numpy().reshape(-1, 784) / 255.0  # 归一化
y_train = train_dataset.targets.numpy()
X_test = test_dataset.data.numpy().reshape(-1, 784) / 255.0
y_test = test_dataset.targets.numpy()

print(f"数据集信息:")
print(f"- 训练样本: {X_train.shape[0]}")
print(f"- 测试样本: {X_test.shape[0]}")
print(f"- 特征数: {X_train.shape[1]}")
print(f"- 类别数: 10")
print(f"- 类别分布: {np.bincount(y_train)}")

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)


# ---------------------- 2. 定义基准模型（无Dropout） ----------------------
class BaseNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], output_dim=10):
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


# ---------------------- 3. 训练基准模型 ----------------------
print("\n" + "=" * 50)
print("训练基准模型（无Dropout）")
print("=" * 50)

input_dim = X_train.shape[1]
base_model = BaseNeuralNetwork(input_dim=input_dim, hidden_dims=[512, 256, 128], output_dim=10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(base_model.parameters(), lr=0.001)

# 训练过程
epochs = 100
for epoch in range(epochs):
    base_model.train()
    outputs = base_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == epochs - 1:
        base_model.eval()
        with torch.no_grad():
            train_outputs = base_model(X_train_tensor)
            train_loss = criterion(train_outputs, y_train_tensor)
            train_pred = torch.argmax(train_outputs, dim=1)
            train_acc = accuracy_score(y_train, train_pred.numpy())

            test_outputs = base_model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            test_pred = torch.argmax(test_outputs, dim=1)
            test_acc = accuracy_score(y_test, test_pred.numpy())

            total_norm = 0
            for param in base_model.parameters():
                if param.requires_grad:
                    total_norm += param.norm(2).item() ** 2
            weight_norm = total_norm ** 0.5

            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss.item():.4f} | "
                  f"Test Loss: {test_loss.item():.4f} | "
                  f"Weight Norm: {weight_norm:.2f}")

# 基准模型最终结果
base_model.eval()
with torch.no_grad():
    train_outputs = base_model(X_train_tensor)
    base_train_loss = criterion(train_outputs, y_train_tensor).item()
    train_pred = torch.argmax(train_outputs, dim=1)
    base_train_acc = accuracy_score(y_train, train_pred.numpy())

    test_outputs = base_model(X_test_tensor)
    base_test_loss = criterion(test_outputs, y_test_tensor).item()
    test_pred = torch.argmax(test_outputs, dim=1)
    base_test_acc = accuracy_score(y_test, test_pred.numpy())

    total_norm = 0
    for param in base_model.parameters():
        if param.requires_grad:
            total_norm += param.norm(2).item() ** 2
    base_weight_norm = total_norm ** 0.5

print("\n基准模型最终结果:")
print(f"训练集交叉熵损失: {base_train_loss:.6f}")
print(f"测试集交叉熵损失: {base_test_loss:.6f}")
print(f"权重L2范数: {base_weight_norm:.6f}")
print(f"训练集准确率: {base_train_acc:.4f}")
print(f"测试集准确率: {base_test_acc:.4f}")
print(f"过拟合程度: {base_train_acc - base_test_acc:.4f}")


# ---------------------- 4. 定义带Dropout的模型 ----------------------
class DropoutNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], output_dim=10, dropout_rate=0.5):
        super(DropoutNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x


# ---------------------- 5. 测试不同Dropout率 ----------------------
print("\n" + "=" * 50)
print("测试不同Dropout率")
print("=" * 50)

dropout_rates = [0.1, 0.3, 0.5, 0.7]
best_dropout_rate = None
best_test_loss = float('inf')
best_results = {}

for dropout_rate in dropout_rates:
    print(f"\nDropout率 = {dropout_rate}")
    print("-" * 40)

    model = DropoutNeuralNetwork(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        output_dim=10,
        dropout_rate=dropout_rate
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train_tensor)
                train_loss = criterion(train_outputs, y_train_tensor)
                train_pred = torch.argmax(train_outputs, dim=1)
                train_acc = accuracy_score(y_train, train_pred.numpy())

                test_outputs = model(X_test_tensor)
                test_loss = criterion(test_outputs, y_test_tensor)
                test_pred = torch.argmax(test_outputs, dim=1)
                test_acc = accuracy_score(y_test, test_pred.numpy())

                total_norm = 0
                for param in model.parameters():
                    if param.requires_grad:
                        total_norm += param.norm(2).item() ** 2
                weight_norm = total_norm ** 0.5

                print(
                    f"最终: 训练损失={train_loss.item():.4f}, 测试损失={test_loss.item():.4f}, 测试准确率={test_acc:.4f}")

                # 记录结果
                results = {
                    'train_loss': train_loss.item(),
                    'test_loss': test_loss.item(),
                    'weight_norm': weight_norm,
                    'train_acc': train_acc,
                    'test_acc': test_acc
                }

                if test_loss.item() < best_test_loss:
                    best_test_loss = test_loss.item()
                    best_dropout_rate = dropout_rate
                    best_results = results

# ---------------------- 6. 最终结果对比 ----------------------
print("\n" + "=" * 60)
print("Dropout实验结果总结")
print("=" * 60)

print(f"最佳Dropout率: {best_dropout_rate}")
print(f"\n基准模型 vs 最佳Dropout模型:")
print("-" * 50)
print(f"{'指标':<15} {'基准模型':<12} {'Dropout模型':<12} {'改善':<12}")
print("-" * 50)
print(
    f"{'训练损失':<15} {base_train_loss:.4f}    {best_results['train_loss']:.4f}    {base_train_loss - best_results['train_loss']:+.4f}")
print(
    f"{'测试损失':<15} {base_test_loss:.4f}    {best_results['test_loss']:.4f}    {base_test_loss - best_results['test_loss']:+.4f}")
print(
    f"{'训练准确率':<15} {base_train_acc:.4f}    {best_results['train_acc']:.4f}    {best_results['train_acc'] - base_train_acc:+.4f}")
print(
    f"{'测试准确率':<15} {base_test_acc:.4f}    {best_results['test_acc']:.4f}    {best_results['test_acc'] - base_test_acc:+.4f}")
print(
    f"{'权重范数':<15} {base_weight_norm:.2f}    {best_results['weight_norm']:.2f}    {best_results['weight_norm'] - base_weight_norm:+.2f}")

# 分析改善效果
test_loss_improvement = base_test_loss - best_results['test_loss']
test_acc_improvement = best_results['test_acc'] - base_test_acc
overfitting_reduction = (base_train_acc - base_test_acc) - (best_results['train_acc'] - best_results['test_acc'])

print(f"\n改善分析:")
print(f"- 测试损失改善: {test_loss_improvement:+.4f}")
print(f"- 测试准确率改善: {test_acc_improvement:+.4f}")
print(f"- 过拟合减少: {overfitting_reduction:+.4f}")

if test_loss_improvement > 0:
    print("✅ Dropout成功降低了测试损失，提高了泛化能力！")
else:
    print("❌ Dropout未能改善测试损失")