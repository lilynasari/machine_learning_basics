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

# 添加高斯噪声（模拟现实数据噪声）
noise = np.random.normal(0, 1.0, X.shape)  # 乳腺癌数据特征尺度较小，调整噪声标准差
X = X + noise

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

# 3. 绘制趋势曲线
plt.figure(figsize=(12, 10))

# 子图1：正则化强度与权重范数的关系
plt.subplot(2, 1, 1)
plt.semilogx(Cs, weight_norms, marker='o', color='#2ca02c', linewidth=2)
plt.xlabel('正则化强度倒数 C（C越小，正则化越强）')
plt.ylabel('权重L2范数')
plt.title('乳腺癌分类：正则化强度 vs 权重范数')
plt.grid(True, alpha=0.3)

# 子图2：正则化强度 vs 交叉熵损失
plt.subplot(2, 1, 2)
plt.semilogx(Cs, train_losses, marker='o', label='训练集交叉熵', color='#2ca02c', linewidth=2)
plt.semilogx(Cs, test_losses, marker='s', label='测试集交叉熵', color='#d62728', linewidth=2)
plt.xlabel('正则化强度倒数 C（C越小，正则化越强）')
plt.ylabel('交叉熵损失（值越小，模型拟合越优）')
plt.title('乳腺癌分类：正则化强度 vs 交叉熵损失')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()