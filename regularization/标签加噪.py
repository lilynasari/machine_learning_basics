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

# ---------------------- 2. 标签加噪实验 ----------------------
noise_probs = np.linspace(0, 0.5, 20)  # 标签翻转概率从0%到50%
train_losses = []
test_losses = []
weight_norms = []
train_accuracies = []
test_accuracies = []

print("标签加噪实验结果：")
print("=" * 60)

# 保存原始标签
y_train_original = y_train.copy()

for noise_prob in noise_probs:
    # 复制原始训练标签
    y_train_noisy = y_train_original.copy()

    # 随机选择一部分标签进行翻转
    np.random.seed(42)
    n_samples = len(y_train_noisy)
    n_flip = int(noise_prob * n_samples)

    # 随机选择要翻转的样本索引
    flip_indices = np.random.choice(n_samples, size=n_flip, replace=False)

    # 翻转标签 (0->1, 1->0)
    y_train_noisy[flip_indices] = 1 - y_train_noisy[flip_indices]

    # 使用带噪声标签训练模型
    lr_noisy = LogisticRegression(penalty=None, max_iter=1000, random_state=42)
    lr_noisy.fit(X_train, y_train_noisy)

    # 计算权重L2范数
    weight_norm = np.linalg.norm(lr_noisy.coef_[0])
    weight_norms.append(weight_norm)

    # 计算交叉熵损失（使用原始标签评估！）
    y_train_proba = lr_noisy.predict_proba(X_train)[:, 1]
    y_test_proba = lr_noisy.predict_proba(X_test)[:, 1]

    train_loss = log_loss(y_train_original, y_train_proba)  # 使用原始标签
    test_loss = log_loss(y_test, y_test_proba)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    # 计算准确率
    train_acc = lr_noisy.score(X_train, y_train_original)  # 使用原始标签
    test_acc = lr_noisy.score(X_test, y_test)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    # 输出每个噪声水平的结果
    if noise_prob == 0 or noise_prob == 0.1 or noise_prob == 0.2 or noise_prob == 0.3 or noise_prob >= 0.4:
        print(f"噪声概率={noise_prob:.3f}: 翻转了{len(flip_indices)}个标签")
        print(f"  测试损失={test_loss:.6f}, 测试准确率={test_acc:.4f}")

# 找到最优标签噪声水平
best_idx = np.argmin(test_losses)
best_noise_prob = noise_probs[best_idx]
best_test_loss = test_losses[best_idx]

print("=" * 60)
print(f"最优标签噪声概率: {best_noise_prob:.6f}")
print(f"最优测试损失: {best_test_loss:.6f}")
print(f"原始测试损失: {test_losses[0]:.6f}")  # noise_prob=0时的损失
print(f"改善程度: {test_losses[0] - best_test_loss:.6f}")

# ---------------------- 3. 训练原始模型作为对比基准 ----------------------
print("\n原始无噪声模型结果：")
lr_original = LogisticRegression(penalty=None, max_iter=1000, random_state=42)
lr_original.fit(X_train, y_train_original)

y_train_proba_original = lr_original.predict_proba(X_train)[:, 1]
y_test_proba_original = lr_original.predict_proba(X_test)[:, 1]

train_loss_original = log_loss(y_train_original, y_train_proba_original)
test_loss_original = log_loss(y_test, y_test_proba_original)
weight_norm_original = np.linalg.norm(lr_original.coef_[0])

print(f"权重L2范数: {weight_norm_original:.6f}")
print(f"训练集交叉熵损失: {train_loss_original:.6f}")
print(f"测试集交叉熵损失: {test_loss_original:.6f}")
print(f"训练集准确率: {lr_original.score(X_train, y_train_original):.4f}")
print(f"测试集准确率: {lr_original.score(X_test, y_test):.4f}")

# ---------------------- 4. 可视化标签加噪效果 ----------------------
plt.figure(figsize=(15, 10))

# 子图1：标签噪声概率与损失的关系
plt.subplot(2, 2, 1)
plt.plot(noise_probs, train_losses, marker='o', label='训练集损失(原始标签)', color='#2ca02c', linewidth=2)
plt.plot(noise_probs, test_losses, marker='s', label='测试集损失', color='#d62728', linewidth=2)
plt.axhline(y=train_loss_original, color='#2ca02c', linestyle='--', alpha=0.7, label='原始训练损失')
plt.axhline(y=test_loss_original, color='#d62728', linestyle='--', alpha=0.7, label='原始测试损失')
plt.plot(best_noise_prob, best_test_loss, 'ro', markersize=10, markeredgewidth=3,
         markerfacecolor='none', label=f'最优点: p={best_noise_prob:.3f}')
plt.xlabel('标签翻转概率')
plt.ylabel('交叉熵损失')
plt.title('标签加噪：噪声概率 vs 交叉熵损失')
plt.legend()
plt.grid(True, alpha=0.3)



# 子图3：标签噪声概率与权重范数的关系
plt.subplot(2, 2, 3)
plt.plot(noise_probs, weight_norms, marker='o', color='#1f77b4', linewidth=2, label='权重范数')
plt.axhline(y=weight_norm_original, color='#1f77b4', linestyle='--', alpha=0.7, label='原始权重范数')
plt.plot(best_noise_prob, weight_norms[best_idx], 'ro', markersize=10, markeredgewidth=3,
         markerfacecolor='none', label=f'最优点')
plt.xlabel('标签翻转概率')
plt.ylabel('权重L2范数')
plt.title('标签加噪：噪声概率 vs 权重范数')
plt.legend()
plt.grid(True, alpha=0.3)

# 创建文本表格显示关键指标
table_data = [
    ["指标", "原始模型", "最优加噪模型", "改善"],
    ["-"*40, "-"*10, "-"*12, "-"*8],
    [f"权重范数", f"{weight_norm_original:.4f}", f"{weight_norms[best_idx]:.4f}", f"{weight_norms[best_idx] - weight_norm_original:+.4f}"],
    [f"训练损失", f"{train_loss_original:.4f}", f"{train_losses[best_idx]:.4f}", f"{train_loss_original - train_losses[best_idx]:+.4f}"],
    [f"测试损失", f"{test_loss_original:.4f}", f"{best_test_loss:.4f}", f"{test_loss_original - best_test_loss:+.4f}"],
    [f"测试准确率", f"{lr_original.score(X_test, y_test):.4f}", f"{test_accuracies[best_idx]:.4f}", f"{test_accuracies[best_idx] - lr_original.score(X_test, y_test):+.4f}"]
]

# 在图上添加表格
table = plt.table(cellText=table_data,
                  cellLoc='center',
                  loc='center',
                  bbox=[0.1, 0.1, 0.8, 0.8])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

plt.title('最优标签加噪模型性能对比', fontsize=12, pad=20)

plt.tight_layout()
plt.show()

# ---------------------- 5. 最终结果比较 ----------------------
print("\n最终模型比较：")
print(f"{'指标':<20} {'原始模型':<12} {'最优加噪模型':<12} {'改善':<12}")
print("-" * 60)
print(
    f"{'训练损失':<20} {train_loss_original:.6f}    {train_losses[best_idx]:.6f}    {train_loss_original - train_losses[best_idx]:.6f}")
print(f"{'测试损失':<20} {test_loss_original:.6f}    {best_test_loss:.6f}    {test_loss_original - best_test_loss:.6f}")
print(
    f"{'训练准确率':<20} {lr_original.score(X_train, y_train_original):.4f}    {train_accuracies[best_idx]:.4f}    {train_accuracies[best_idx] - lr_original.score(X_train, y_train_original):.4f}")
print(
    f"{'测试准确率':<20} {lr_original.score(X_test, y_test):.4f}    {test_accuracies[best_idx]:.4f}    {test_accuracies[best_idx] - lr_original.score(X_test, y_test):.4f}")

