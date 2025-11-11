#从库中导入
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,log_loss


#解决aMtplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 适配Windows
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 适配Mac
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

# ---------------------- 1. 数据准备（鸢尾花二分类任务） ----------------------
iris = load_iris()
# 特征：用全部4个特征（花萼长、花萼宽、花瓣长、花瓣宽），提升模型区分度
X = iris.data
# 标签：转换为二分类（0=山鸢尾，1=非山鸢尾，即排除标签为0的山鸢尾，其余归为1）
y = (iris.target != 0).astype(int)  # 结果：y中0对应山鸢尾，1对应变色鸢尾+维吉尼亚鸢尾

# 特征标准化（逻辑回归对特征尺度敏感，标准化后模型收敛更快、结果更稳定）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集（70%）和测试集（30%），stratify=y确保二分类标签分布均衡
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

#2. 遍历不同正则化强度，记录指标
Cs = np.logspace(-4, 4, 20)
train_losses = []
test_losses = []
weight_norms = []

for C in Cs:
    lr = LogisticRegression(C=C,penalty='l2',  max_iter=300, random_state=42)
    lr.fit(X_train, y_train)

    # 1. 计算权重L2范数
    weight_norm = np.linalg.norm(lr.coef_[0])
    weight_norms.append(weight_norm)

    # 2. 计算交叉熵损失（关键新增：需用模型输出的概率，而非离散预测值）
    # 逻辑回归用predict_proba输出概率，取第二类（标签1）的概率（二分类只需一列）
    y_train_proba = lr.predict_proba(X_train)[:, 1]
    y_test_proba = lr.predict_proba(X_test)[:, 1]

    # 计算交叉熵（log_loss默认适用于二分类，无需额外参数）
    train_loss = log_loss(y_train, y_train_proba)
    test_loss = log_loss(y_test, y_test_proba)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
#3. 绘制趋势曲线
plt.figure(figsize=(12, 10))

# 子图1：正则化强度（C）与权重范数的关系
plt.subplot(2, 1, 1)
plt.semilogx(Cs, weight_norms, marker='o', color='#2ca02c', linewidth=2)
plt.xlabel('正则化强度倒数 C（C越小，正则化越强）')
plt.ylabel('权重L2范数')
plt.title('二分类逻辑回归：正则化强度 vs 权重范数')
plt.grid(True, alpha=0.3)

# 子图2：正则化强度 vs 交叉熵损失（核心图：观察损失趋势）
plt.subplot(2, 1, 2)
plt.semilogx(Cs, train_losses, marker='o', label='训练集交叉熵', color='#2ca02c', linewidth=2)
plt.semilogx(Cs, test_losses, marker='s', label='测试集交叉熵', color='#d62728', linewidth=2)
plt.xlabel('正则化强度倒数 C（C越小，正则化越强）')
plt.ylabel('交叉熵损失（值越小，模型拟合越优）')
plt.title('二分类逻辑回归：正则化强度 vs 交叉熵损失')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()