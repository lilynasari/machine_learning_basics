# 从库中导入
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 1. 数据准备 ----------------------
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ---------------------- 2. 训练无正则化模型 ----------------------
# 使用无正则化的逻辑回归
lr = LogisticRegression(penalty=None, max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# 保存原始权重
original_weights = lr.coef_[0].copy()
original_intercept = lr.intercept_[0].copy()

# 计算原始模型的损失
y_train_proba = lr.predict_proba(X_train)[:, 1]
y_test_proba = lr.predict_proba(X_test)[:, 1]
train_loss = log_loss(y_train, y_train_proba)
test_loss = log_loss(y_test, y_test_proba)
weight_norm = np.linalg.norm(original_weights)

print("原始无正则化模型结果：")
print(f"权重L2范数: {weight_norm:.6f}")
print(f"训练集交叉熵损失: {train_loss:.6f}")
print(f"测试集交叉熵损失: {test_loss:.6f}")

# ---------------------- 3. 权重加噪实验 ----------------------
noise_stds = np.logspace(-3, 1, 20)  # 噪声标准差从0.001到10
noised_train_losses = []
noised_test_losses = []
noised_weight_norms = []

print("\n权重加噪实验结果：")
print("=" * 60)


# 修正后的自定义预测函数
def predict_with_noisy_weights(X, weights, intercept, noise_std, random_seed=42):
    """使用加噪的权重进行预测（数值稳定的版本）"""
    np.random.seed(random_seed)
    weight_noise = np.random.normal(0, noise_std, weights.shape)
    intercept_noise = np.random.normal(0, noise_std)

    noisy_weights = weights + weight_noise
    noisy_intercept = intercept + intercept_noise

    # 手动计算逻辑回归的预测概率（数值稳定版本）
    z = np.dot(X, noisy_weights) + noisy_intercept

    # 数值稳定的sigmoid计算
    # 防止exp溢出
    z = np.clip(z, -500, 500)  # 限制z的范围

    # 计算概率
    probabilities = 1 / (1 + np.exp(-z))

    # 确保概率在[0,1]范围内
    probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)

    return probabilities


for i, noise_std in enumerate(noise_stds):
    # 使用加噪权重进行预测
    y_train_proba_noised = predict_with_noisy_weights(X_train, original_weights, original_intercept, noise_std, 42 + i)
    y_test_proba_noised = predict_with_noisy_weights(X_test, original_weights, original_intercept, noise_std, 42 + i)

    train_loss_noised = log_loss(y_train, y_train_proba_noised)
    test_loss_noised = log_loss(y_test, y_test_proba_noised)

    # 计算加噪后的权重范数
    np.random.seed(42 + i)
    weight_noise = np.random.normal(0, noise_std, original_weights.shape)
    weight_norm_noised = np.linalg.norm(original_weights + weight_noise)

    noised_train_losses.append(train_loss_noised)
    noised_test_losses.append(test_loss_noised)
    noised_weight_norms.append(weight_norm_noised)

    # 输出每个噪声水平的结果
    if i % 5 == 0:  # 每5个输出一次，避免输出太多
        print(f"噪声σ={noise_std:.4f}: 测试损失={test_loss_noised:.6f}, 权重范数={weight_norm_noised:.4f}")


# 找到最优权重噪声水平
best_idx = np.argmin(noised_test_losses)
best_noise_std = noise_stds[best_idx]
best_test_loss = noised_test_losses[best_idx]
best_train_loss = noised_train_losses[best_idx]  # 新增：最优噪声时的训练损失
best_weight_norm = noised_weight_norms[best_idx]  # 新增：最优噪声时的权重范数

print("=" * 60)
print(f"最优权重噪声标准差: {best_noise_std:.6f}")
print(f"最优测试损失: {best_test_loss:.6f}")
print(f"最优训练损失: {best_train_loss:.6f}")  # 新增输出
print(f"最优权重范数: {best_weight_norm:.6f}")  # 新增输出
print(f"原始测试损失: {test_loss:.6f}")
print(f"原始训练损失: {train_loss:.6f}")  # 新增输出
print(f"原始权重范数: {weight_norm:.6f}")  # 新增输出
print(f"测试损失改善程度: {test_loss - best_test_loss:.6f}")


# ---------------------- 4. 可视化权重加噪效果 ----------------------
plt.figure(figsize=(12, 8))

# 子图1：权重噪声水平与损失的关系
plt.subplot(2, 1, 1)
plt.semilogx(noise_stds, noised_train_losses, marker='o', label='加噪训练集损失', color='#2ca02c', linewidth=2)
plt.semilogx(noise_stds, noised_test_losses, marker='s', label='加噪测试集损失', color='#d62728', linewidth=2)
plt.axhline(y=train_loss, color='#2ca02c', linestyle='--', alpha=0.7, label='原始训练损失')
plt.axhline(y=test_loss, color='#d62728', linestyle='--', alpha=0.7, label='原始测试损失')
plt.semilogx(best_noise_std, best_test_loss, 'ro', markersize=10, markeredgewidth=3,
             markerfacecolor='none', label=f'最优点: σ={best_noise_std:.3f}')
plt.xlabel('权重噪声标准差 σ')
plt.ylabel('交叉熵损失')
plt.title('权重加噪：噪声水平 vs 交叉熵损失')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：权重噪声水平与权重范数的关系
plt.subplot(2, 1, 2)
plt.semilogx(noise_stds, noised_weight_norms, marker='o', color='#1f77b4', linewidth=2, label='加噪权重范数')
plt.axhline(y=weight_norm, color='#1f77b4', linestyle='--', alpha=0.7, label='原始权重范数')
plt.semilogx(best_noise_std, noised_weight_norms[best_idx], 'ro', markersize=10, markeredgewidth=3,
             markerfacecolor='none', label=f'最优点')
plt.xlabel('权重噪声标准差 σ')
plt.ylabel('权重L2范数')
plt.title('权重加噪：噪声水平 vs 权重范数')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

