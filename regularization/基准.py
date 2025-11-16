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

# 计算权重L2范数
weight_norm = np.linalg.norm(lr.coef_[0])

# 计算交叉熵损失
y_train_proba = lr.predict_proba(X_train)[:, 1]
y_test_proba = lr.predict_proba(X_test)[:, 1]

train_loss = log_loss(y_train, y_train_proba)
test_loss = log_loss(y_test, y_test_proba)

# ---------------------- 3. 直接输出结果 ----------------------
print("无正则化逻辑回归结果：")
print(f"权重L2范数: {weight_norm:.6f}")
print(f"训练集交叉熵损失: {train_loss:.6f}")
print(f"测试集交叉熵损失: {test_loss:.6f}")
print(f"训练集准确率: {lr.score(X_train, y_train):.4f}")
print(f"测试集准确率: {lr.score(X_test, y_test):.4f}")