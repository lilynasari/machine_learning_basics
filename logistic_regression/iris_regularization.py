import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体为微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常问题
#导入鸢尾花数据集并改为二分类小数据集

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
iris = load_iris()
X = iris.data[:, :2]
y = (iris.target == 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#遍历不同正则化强度并记录

lambdas = np.logspace(-5, 5, 20)
weight_norms = []
train_errors = []
test_errors = []
for lam in lambdas:
    # 初始化Ridge模型（L2正则化，alpha=λ）
    ridge = Ridge(alpha=lam, random_state=42)
    ridge.fit(X_train, y_train)

    # 计算权重范数（L2范数）
    weight_norm = np.linalg.norm(ridge.coef_)
    weight_norms.append(weight_norm)

    # 计算训练误差
    y_train_pred = ridge.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_errors.append(train_mse)

    # 计算测试误差
    y_test_pred = ridge.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_errors.append(test_mse)

#展示曲线
# 绘制子图1：正则化强度λ与权重范数的关系
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.semilogx(lambdas, weight_norms, marker='o')
plt.xlabel('正则化强度λ (alpha)')
plt.ylabel('权重L2范数')
plt.title('正则化强度λ vs 权重范数')
plt.grid(True)

# 绘制子图2：正则化强度λ与训练/测试误差的关系
plt.subplot(2, 1, 2)
plt.semilogx(lambdas, train_errors, marker='o', label='训练误差')
plt.semilogx(lambdas, test_errors, marker='s', label='测试误差')
plt.xlabel('正则化强度λ (alpha)')
plt.ylabel('均方误差(MSE)')
plt.title('正则化强度λ vs 训练/测试误差')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()