# 从库中导入
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# 设置全局随机种子，确保结果可重现
np.random.seed(42)  # 固定numpy的随机数生成器

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 1. 数据准备 ----------------------
cancer = load_breast_cancer()
X_original = cancer.data
y = cancer.target

# 特征标准化
scaler = StandardScaler()
X_scaled_original = scaler.fit_transform(X_original)

# 划分训练集和测试集（固定随机种子）
X_train_original, X_test_original, y_train, y_test = train_test_split(
    X_scaled_original, y, test_size=0.3, random_state=42, stratify=y
)

# ---------------------- 2. 预生成所有噪声数据 ----------------------
noise_stds = np.logspace(-3, 1, 20)  # 噪声标准差从0.001到10

# 预生成所有噪声矩阵，确保每次运行使用相同的噪声
noise_matrices = []
for i, noise_std in enumerate(noise_stds):
    # 为每个噪声水平设置不同的但固定的种子
    np.random.seed(42 + i)  # 使用不同的种子生成不同的但固定的噪声
    noise = np.random.normal(0, noise_std, X_train_original.shape)
    noise_matrices.append(noise)

# 恢复全局种子
np.random.seed(42)

train_losses = []
test_losses = []
weight_norms = []

# 用于保存最优模型
best_model = None
best_noise_std = None
best_train_loss = None
min_test_loss = float('inf')
best_weight_norm = None
best_X_train_noisy = None

for i, noise_std in enumerate(noise_stds):
    # 使用预生成的固定噪声
    noise = noise_matrices[i]
    X_train_noisy = X_train_original + noise

    # 使用无正则化的逻辑回归（固定随机种子）
    lr = LogisticRegression(penalty=None, max_iter=5000, random_state=42, solver='lbfgs')
    lr.fit(X_train_noisy, y_train)

    # 计算权重L2范数
    weight_norm = np.linalg.norm(lr.coef_[0])
    weight_norms.append(weight_norm)

    # 计算交叉熵损失
    y_train_proba = lr.predict_proba(X_train_noisy)[:, 1]
    y_test_proba = lr.predict_proba(X_test_original)[:, 1]

    train_loss = log_loss(y_train, y_train_proba)
    test_loss = log_loss(y_test, y_test_proba)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    # 保存最优模型
    if test_loss < min_test_loss:
        min_test_loss = test_loss
        best_model = lr
        best_noise_std = noise_std
        best_train_loss = train_loss
        best_weight_norm = weight_norm
        best_X_train_noisy = X_train_noisy.copy()  # 保存副本

# ---------------------- 3. 输出最优结果 ----------------------
print("=" * 50)
print("高斯噪声优化结果（无正则化）- 可重现版本")
print("=" * 50)
print(f"最优噪声标准差: {best_noise_std:.6f}")
print(f"最小测试集交叉熵损失: {min_test_loss:.6f}")
print(f"对应的训练集交叉熵损失: {best_train_loss:.6f}")
print(f"对应的权重L2范数: {best_weight_norm:.6f}")
print(f"训练集准确率: {best_model.score(best_X_train_noisy, y_train):.4f}")
print(f"测试集准确率: {best_model.score(X_test_original, y_test):.4f}")

# ---------------------- 4. 绘制噪声水平与损失的关系 ----------------------
plt.figure(figsize=(12, 10))

# 子图1：噪声水平与损失的关系
plt.subplot(2, 1, 1)
plt.semilogx(noise_stds, train_losses, marker='o', label='训练集交叉熵', color='#2ca02c', linewidth=2)
plt.semilogx(noise_stds, test_losses, marker='s', label='测试集交叉熵', color='#d62728', linewidth=2)
plt.semilogx(best_noise_std, min_test_loss, 'ro', markersize=10, markeredgewidth=3,
             markerfacecolor='none', label=f'最优点: σ={best_noise_std:.3f}')
plt.xlabel('高斯噪声标准差 σ')
plt.ylabel('交叉熵损失')
plt.title('乳腺癌分类：噪声水平 vs 交叉熵损失（无正则化）- 可重现结果')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：噪声水平与权重范数的关系
plt.subplot(2, 1, 2)
plt.semilogx(noise_stds, weight_norms, marker='o', color='#1f77b4', linewidth=2, label='权重范数')
plt.semilogx(best_noise_std, best_weight_norm, 'ro', markersize=10, markeredgewidth=3,
             markerfacecolor='none', label=f'最优点')
plt.xlabel('高斯噪声标准差 σ')
plt.ylabel('权重L2范数')
plt.title('乳腺癌分类：噪声水平 vs 权重范数')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reproducible_noise_optimization.png', dpi=300, bbox_inches='tight')
plt.show()

