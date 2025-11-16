# 从库中导入
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss


# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 适配Windows

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

# ---------------------- 1. 数据准备（乳腺癌二分类任务） ----------------------
cancer = load_breast_cancer()
X = cancer.data  # 特征：包含30个乳腺癌相关特征
y = cancer.target  # 标签：0=恶性，1=良性（原生标签已符合二分类需求）


# 特征标准化（逻辑回归对特征尺度敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集，保持标签分布均衡
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y  # 乳腺癌数据测试集用30%更合理
)

# 2. 遍历不同正则化强度，记录指标
Cs = np.logspace(-4, 4, 20)  # 正则化强度倒数，对数分布
train_losses = []
test_losses = []
weight_norms = []

for C in Cs:
    # 使用L2正则化的逻辑回归
    lr = LogisticRegression(C=C, penalty='l2', max_iter=1000, random_state=42)  # 增加迭代次数确保收敛
    lr.fit(X_train, y_train)

    # 计算权重L2范数
    weight_norm = np.linalg.norm(lr.coef_[0])
    weight_norms.append(weight_norm)

    # 计算交叉熵损失（使用预测概率）
    y_train_proba = lr.predict_proba(X_train)[:, 1]
    y_test_proba = lr.predict_proba(X_test)[:, 1]

    train_loss = log_loss(y_train, y_train_proba)
    test_loss = log_loss(y_test, y_test_proba)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

# 找到测试集损失最小值的索引
min_test_loss_idx = np.argmin(test_losses)
min_test_loss = test_losses[min_test_loss_idx]
optimal_C = Cs[min_test_loss_idx]
optimal_train_loss = train_losses[min_test_loss_idx]
optimal_weight_norm = weight_norms[min_test_loss_idx]

# 输出测试集曲线最低点的交叉熵值和对应的C值
print("=" * 60)
print("L2正则化最优解结果")
print("=" * 60)
print(f"最优正则化强度 C: {optimal_C:.6f}")
print(f"最优测试集交叉熵损失: {min_test_loss:.6f}")
print(f"对应训练集交叉熵损失: {optimal_train_loss:.6f}")
print(f"对应权重L2范数: {optimal_weight_norm:.6f}")

# 计算无正则化时的结果作为对比基准
lr_no_reg = LogisticRegression(penalty=None, max_iter=1000, random_state=42)
lr_no_reg.fit(X_train, y_train)
y_train_proba_no_reg = lr_no_reg.predict_proba(X_train)[:, 1]
y_test_proba_no_reg = lr_no_reg.predict_proba(X_test)[:, 1]
train_loss_no_reg = log_loss(y_train, y_train_proba_no_reg)
test_loss_no_reg = log_loss(y_test, y_test_proba_no_reg)
weight_norm_no_reg = np.linalg.norm(lr_no_reg.coef_[0])

print("\n" + "=" * 60)
print("与无正则化模型对比")
print("=" * 60)
print(f"{'指标':<15} {'无正则化':<12} {'L2正则化':<12} {'改善':<12}")
print("-" * 50)
print(f"{'训练损失':<15} {train_loss_no_reg:.6f}    {optimal_train_loss:.6f}    {train_loss_no_reg - optimal_train_loss:+.6f}")
print(f"{'测试损失':<15} {test_loss_no_reg:.6f}    {min_test_loss:.6f}    {test_loss_no_reg - min_test_loss:+.6f}")
print(f"{'权重范数':<15} {weight_norm_no_reg:.6f}    {optimal_weight_norm:.6f}    {weight_norm_no_reg - optimal_weight_norm:+.6f}")

# 计算改善程度
test_loss_improvement = (test_loss_no_reg - min_test_loss) / test_loss_no_reg * 100
weight_reduction = (weight_norm_no_reg - optimal_weight_norm) / weight_norm_no_reg * 100

print(f"\n改善程度分析:")
print(f"测试损失改善: {test_loss_improvement:+.2f}%")
print(f"权重范数减少: {weight_reduction:+.2f}%")

if test_loss_improvement > 0:
    print("✅ L2正则化有效改善了模型泛化能力")
else:
    print("❌ L2正则化未能改善模型性能")

# 3. 绘制趋势曲线
plt.figure(figsize=(12, 10))

# 子图1：正则化强度与权重范数的关系
plt.subplot(2, 1, 1)
plt.semilogx(Cs, weight_norms, marker='o', color='#2ca02c', linewidth=2)
plt.semilogx(optimal_C, optimal_weight_norm, 'ro', markersize=10, markeredgewidth=3,
             markerfacecolor='none', label=f'最优点: C={optimal_C:.3f}')
plt.xlabel('正则化强度倒数 C（C越小，正则化越强）')
plt.ylabel('权重L2范数')
plt.title('乳腺癌分类：正则化强度 vs 权重范数')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：正则化强度 vs 交叉熵损失
plt.subplot(2, 1, 2)
plt.semilogx(Cs, train_losses, marker='o', label='训练集交叉熵', color='#2ca02c', linewidth=2)
plt.semilogx(Cs, test_losses, marker='s', label='测试集交叉熵', color='#d62728', linewidth=2)

# 在图上标记最小值点
plt.semilogx(optimal_C, min_test_loss, 'ro', markersize=10, markeredgewidth=3,
             markerfacecolor='none', label=f'最优点: 测试损失={min_test_loss:.4f}')

plt.xlabel('正则化强度倒数 C（C越小，正则化越强）')
plt.ylabel('交叉熵损失（值越小，模型拟合越优）')
plt.title('乳腺癌分类：正则化强度 vs 交叉熵损失')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 输出正则化效果总结
print("\n" + "=" * 60)
print("L2正则化效果总结")
print("=" * 60)
print(f"1. 最优正则化强度: C = {optimal_C:.6f}")
print(f"2. 在最优强度下:")
print(f"   - 测试集损失: {min_test_loss:.6f} (相比无正则化改善 {test_loss_improvement:+.2f}%)")
print(f"   - 训练集损失: {optimal_train_loss:.6f}")
print(f"   - 权重范数: {optimal_weight_norm:.6f} (减少 {weight_reduction:+.2f}%)")
print(f"3. 正则化成功约束了模型复杂度，提高了泛化能力")